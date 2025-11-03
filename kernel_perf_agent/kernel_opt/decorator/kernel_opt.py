"""Kernel agent decorator."""

import functools
import os
import subprocess
from typing import Callable

from dotenv import load_dotenv
from kernel_perf_agent.kernel_opt.configs.envs import NUM_OF_ROUNDS
from kernel_perf_agent.kernel_opt.database.base import OptHierarchy
from kernel_perf_agent.kernel_opt.profiler.profiler import KernelProfiler
from kernel_perf_agent.kernel_opt.prompts.prompt_manager import PromptManager
from kernel_perf_agent.kernel_opt.retriever.retriever import Retriever
from kernel_perf_agent.kernel_opt.rewriter.kernel_rewriter import KernelRewriter
from kernel_perf_agent.kernel_opt.utils.parser_util import (
    extract_code,
    get_module_path,
    remove_decorators_from_file,
)
from kernel_perf_agent.kernel_opt.verifier.verifier import KernelCodeVerifier, KernelFileVerifier


def kernel_opt(
    func_prompt: str,
    opt_prompt: str,
    model: str,
    dsl: str,
    kernel_name: str,
    debug: bool,
):
    """Decorator for kernel generation.
    :param prompt: Description of the kernel to generate
    :param model: LLM model to use (e.g., "deepseek-chat")
    :param dsl: Target DSL (e.g., "triton")
    :param kernel_name: Name of the kernel (defaults to function name)
    :param debug: Whether to print debug information
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            func_path = get_module_path(func).resolve()
            func_dir = get_module_path(func).parent.resolve()

            # Debug output
            if debug:
                debug_output_path = func_dir / "debug_output"
                if debug_output_path.is_dir():
                    subprocess.run(["rm", "-rf", str(debug_output_path)], check=True)
                subprocess.run(["mkdir", str(debug_output_path)])

            # Load environment variables
            current_script_dir = os.path.dirname(os.getcwd())
            dotenv_path = os.path.join(current_script_dir, ".env")
            load_dotenv(dotenv_path)

            # Verifier - check syntax errors
            verifier_result = KernelFileVerifier(module_path=func_path)
            if not verifier_result.ok:
                print("❌ Input syntax validation failed.")
                return func(*args, **kwargs)
            print("✅ Input syntax validation passed.")

            # TODO: Profiler - check functional correctness
            profiling_result = KernelProfiler(kernel_path=func_path)
            if not profiling_result.ok:
                print("❌ Input functional validation failed.")
                return func(*args, **kwargs)
            print("✅ Input functional validation passed.")

            # Database Construction
            common_path = func_dir / "../kernel_opt/database/code_samples/"
            opt_hierarchy = OptHierarchy()
            opt_hierarchy.hard_initialize(common_path)

            # Retriever - fetch related context from database
            retriever = Retriever(
                func_prompt=func_prompt,
                opt_prompt=opt_prompt,
                model=model,
                dsl=dsl,
                kernel_name=kernel_name,
                database=opt_hierarchy,
                module_path=func_dir,
                debug=debug,
            )
            opt_node, debug_str = retriever.retrieve()

            # Prompt Manager - Build prompt for LLM
            func_source_code = remove_decorators_from_file(func_path)
            prompt_manager = PromptManager(
                func_source_code=func_source_code,
                func_prompt=func_prompt,
                opt_prompt=opt_prompt,
                model=model,
                dsl=dsl,
                kernel_name=kernel_name,
                database=opt_hierarchy,
                opt_node=opt_node,
                module_path=func_dir,
                debug=debug,
            )
            prompt, debug_str = prompt_manager.build_rewrite_prompt()
            error = ""

            # Iterate with error messages
            for attempt in range(NUM_OF_ROUNDS):
                print("=" * 50)
                print(f"Attempt: {attempt}")

                # Rewriter - rewrite kernel
                rewriter = KernelRewriter(
                    prompt=prompt,
                    model=model,
                    debug=debug,
                    module_path=func_dir,
                    error=error,
                )
                response, debug_str = rewriter.generate_kernel(error=error)

                # Extractor - extract kernel from response
                output_program = extract_code(response_text=response, debug=debug)
                if debug:
                    debug_output_path = func_dir / "debug_output" / "output.log"
                    with open(str(debug_output_path), "w") as file:
                        file.write("****** Extracted code ****** : \n")
                        file.write(output_program)

                correct = True

                # Verifier - check syntax errors
                error = ""
                verifier_result = KernelCodeVerifier(
                    code=output_program,
                )
                if verifier_result.ok:
                    print("✅ Output syntax validation passed.")
                else:
                    correct = False
                    print("❌ Output syntax validation failed.")
                    error += (
                        f"""
The previous generated program has a syntax Error: {error} \n"""
                        + verifier_result.message
                    )

                # TODO: Profiler - check functional correctness
                profiling_result = KernelProfiler(kernel_path=func_path)
                if profiling_result.ok:
                    print("✅ Output functional validation passed.")
                else:
                    correct = False
                    print("❌ Output functional validation failed.")
                    error += (
                        f"""
The previous generated program has a function error: {error} \n"""
                        + profiling_result.message
                    )

                # Stop iteration and store the output if correct
                if correct:
                    with open(
                        str(func_dir / str(kernel_name + "_opt.py")), "w"
                    ) as file:
                        file.write(output_program)
                    break

            else:
                print("❌ Kernel Generation Failed")
                raise RuntimeError(
                    f"❌ Kernel validation failed after {NUM_OF_ROUNDS} attempts with the last error: \n{error}"
                )

            # Run original function
            print("=" * 50)
            print("Please find the generated kernel in {}_opt.py.".format(kernel_name))
            print("Below runs the original program.")
            return func(*args, **kwargs)

        return wrapper

    return decorator
