import re
from pathlib import Path
from typing import Tuple

from fastapi import status

from kernel_opt.configs.envs import NUM_OF_ROUNDS
from kernel_opt.database.base import OptHierarchy, OptNode
from kernel_opt.profiler.profiler import KernelProfiler
from kernel_opt.prompts.prompt_manager import PromptManager
from kernel_opt.retriever.retriever import Retriever
from kernel_opt.rewriter.kernel_rewriter import KernelRewriter
from kernel_opt.utils.parser_util import extract_code, get_module_path
from kernel_opt.verifier.verifier import KernelCodeVerifier, KernelFileVerifier


def KernelAgent(
    model: str,
    dsl: str,
    kernel_name: str,
    func_prompt: str,
    func: str,
    opt_prompt: str,
    debug: bool,
) -> Tuple[str, str, str, str]:
    """Agent wrapper for kernel optimization."""

    status_msg = ""
    debug_msg = ""
    session_info = ""
    func_out = ""

    # Verifier - check syntax errors
    verifier_result = KernelCodeVerifier(
        code=func,
    )
    if not verifier_result.ok:
        status_msg += "❌ Input syntax validation failed.\n"
        session_info += "❌ Input syntax validation failed.\n"
        debug_msg = verifier_result.message
        return status_msg, func, debug_msg, ""

    session_info += "✅ Input syntax validation passed.\n"

    # TODO: Profiler - check functional correctness
    profiling_result = KernelProfiler(kernel_path=Path())
    if not profiling_result.ok:
        status_msg += "❌ Input functional validation failed.\n"
        session_info += "❌ Input functional validation failed.\n"
        debug_msg = profiling_result.message
        return status_msg, func, debug_msg, ""
    session_info += "✅ Input functional validation passed.\n"

    # Database Construction
    common_path = Path("kernel_opt/database/code_samples/")
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
        module_path=Path(),
        debug=debug,
    )
    opt_node, debug_str = retriever.retrieve()
    debug_msg += debug_str

    # Prompt Manager - Build prompt for LLM
    prompt_manager = PromptManager(
        func_source_code=func,
        func_prompt=func_prompt,
        opt_prompt=opt_prompt,
        model=model,
        dsl=dsl,
        kernel_name=kernel_name,
        database=opt_hierarchy,
        opt_node=opt_node,
        module_path=Path(),
        debug=debug,
    )
    prompt, debug_str = prompt_manager.build_rewrite_prompt()
    debug_msg += debug_str
    error = ""

    # Iterate with error messages
    for attempt in range(NUM_OF_ROUNDS):
        session_info += "=" * 30 + "\n"
        session_info += f"Attempt: {attempt}" + "\n"

        # Rewriter - rewrite kernel
        rewriter = KernelRewriter(
            prompt=prompt,
            model=model,
            debug=debug,
            module_path=Path(),
            error=error,
        )
        response, debug_str = rewriter.generate_kernel(error)
        debug_msg += debug_str

        # Extractor - extract kernel from response
        output_program = extract_code(response_text=response, debug=debug)
        # if debug:
        #     debug_msg += "****** Extracted code ****** : \n"
        #     debug_msg += output_program

        correct = True
        # Verifier - check syntax errors
        verifier_result = KernelCodeVerifier(
            code=output_program,
        )
        if verifier_result.ok:
            session_info += "✅ Output syntax validation passed.\n"
        else:
            correct = False
            session_info += "❌ Output syntax validation failed.\n"
            session_info += (
                f"""
The previous generated program has a syntax Error: {error} """
                + "\n"
                + verifier_result.message
            )

        # TODO: Profiler - check functional correctness
        profiling_result = KernelProfiler(kernel_path=Path())
        if profiling_result.ok:
            session_info += "✅ Output functional validation passed.\n"
        else:
            correct = False
            session_info += "❌ Output functional validation failed.\n"
            session_info += (
                f"""
The previous generated program has a function error: {error} """
                + "\n"
                + profiling_result.message
            )
        if correct:
            func_out = output_program
            break

    else:
        status_msg = "❌ Kernel Generation Failed. {\n}"
        session_info += f"""
❌ Kernel validation failed after {NUM_OF_ROUNDS} attempts with the last error:
{error}"""

    return status_msg, func_out, debug_msg, session_info
