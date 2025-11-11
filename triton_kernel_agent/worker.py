"""
Verification Worker for testing and refining individual kernels.
"""

import sys
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import multiprocessing as mp
from collections import deque

from .prompt_manager import PromptManager
from .providers import get_model_provider

from kernel_perf_agent.kernel_opt.database.base import OptHierarchy
from kernel_perf_agent.kernel_opt.retriever.retriever import Retriever
from kernel_perf_agent.kernel_opt.prompts.prompt_manager import (
    PromptManager as PerfPromptManager,
)


class VerificationWorker:
    """Worker that verifies and refines a single kernel implementation."""

    def __init__(
        self,
        worker_id: int,
        workdir: Path,
        log_dir: Path,
        max_rounds: int = 10,
        history_size: int = 8,
        openai_api_key: Optional[str] = None,
        openai_model: str = "o3-2025-04-16",
        high_reasoning_effort: bool = True,
        enable_optimization: bool = True,  # Flag to enable optimization phase
        optimization_hint: Optional[str] = None,  # e.g., "use persistent programming"
        optimization_database_path: Optional[Path] = None,  # Path to code_samples
    ):
        """
        Initialize a verification worker.

        Args:
            worker_id: Unique identifier for this worker
            workdir: Working directory for this worker
            log_dir: Directory for logging
            max_rounds: Maximum refinement rounds
            history_size: Number of recent rounds to keep
            openai_api_key: OpenAI API key for refinement
            openai_model: Model name for refinement
            high_reasoning_effort: Whether to use high reasoning effort for OpenAI models
        """
        self.worker_id = worker_id
        self.workdir = Path(workdir)
        self.log_dir = Path(log_dir)
        self.max_rounds = max_rounds
        self.history_size = history_size
        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort

        # Setup files
        self.kernel_file = self.workdir / "kernel.py"
        self.test_file = self.workdir / "test_kernel.py"

        # History for LLM context
        self.history = deque(maxlen=history_size)

        # Setup logging FIRST (before any logger.xxx() calls)
        self._setup_logging()

        # Initialize provider
        self.provider = None
        try:
            self.provider = get_model_provider(self.openai_model)
        except ValueError as e:
            # Provider not available, will use mock mode
            self.logger.warning(f"Provider not available: {e}")

        # Initialize prompt manager
        self.prompt_manager = PromptManager()

        # NEW: Optimization setup
        self.enable_optimization = enable_optimization
        self.optimization_hint = optimization_hint or "optimize for performance"

        # Initialize optimization database if enabled
        self.opt_hierarchy = None
        if self.enable_optimization:
            self.opt_hierarchy = OptHierarchy()
            db_path = optimization_database_path or (
                Path(__file__).parent.parent
                / "kernel_perf_agent/kernel_opt/database/code_samples"
            )
            self.opt_hierarchy.hard_initialize(db_path)
            self.logger.info("Initialized optimization database")

    def _setup_logging(self):
        """Setup worker-specific logging."""
        log_file = self.log_dir / f"worker_{self.worker_id}.log"
        self.logger = logging.getLogger(f"worker_{self.worker_id}")
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)

    def _extract_code_from_response(
        self, response_text: str, language: str = "python"
    ) -> Optional[str]:
        """
        Extract code from LLM response text.

        Args:
            response_text: The full LLM response text
            language: The expected language (default: python)

        Returns:
            Extracted code or None if no valid code block found
        """
        if not response_text:
            return None

        # First, try to find code blocks with language markers
        # Pattern matches ```python or ```language_name
        pattern = rf"```{language}\s*\n(.*?)```"
        matches = re.findall(pattern, response_text, re.DOTALL)

        if matches:
            # Return the first match (largest code block)
            return matches[0].strip()

        # Try generic code blocks without language marker
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, response_text, re.DOTALL)

        if matches:
            # Return the first match
            return matches[0].strip()

        # If no code blocks found, check if the entire response looks like code
        # This is a fallback for cases where LLM doesn't use code blocks
        lines = response_text.strip().split("\n")

        # Simple heuristic: if response contains import statements or function definitions
        code_indicators = ["import ", "from ", "def ", "class ", "@", '"""', "'''"]
        if any(
            line.strip().startswith(indicator)
            for line in lines
            for indicator in code_indicators
        ):
            # Likely the entire response is code
            return response_text.strip()

        # No code found
        self.logger.warning("No code block found in LLM response")
        return None

    def _write_kernel(self, kernel_code: str):
        """Write only the kernel code to file."""
        self.kernel_file.write_text(kernel_code)
        self.logger.info("Updated kernel file")

    def _write_files(self, kernel_code: str, test_code: str):
        """Write kernel and test code to files.

        Note: The test code should import the kernel function from the kernel file:
            from kernel import kernel_function

        Both files are written to the same directory (workdir).
        """
        self.kernel_file.write_text(kernel_code)
        self.test_file.write_text(test_code)
        self.logger.info("Wrote kernel and test files")

    def _run_test(self) -> Tuple[bool, str, str]:
        """
        Run the test script and capture results.

        Returns:
            Tuple of (success, stdout, stderr)
        """
        cmd = [sys.executable, str(self.test_file)]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
            )

            success = result.returncode == 0
            self.logger.info(
                f"Test {'passed' if success else 'failed'} with code {result.returncode}"
            )

            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            self.logger.error("Test timed out")
            return False, "", "Test execution timed out after 30 seconds"
        except Exception as e:
            self.logger.error(f"Test execution error: {e}")
            return False, "", str(e)

    def _call_llm(self, messages: list, **kwargs) -> str:
        """
        Call the LLM provider for the configured model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the API call

        Returns:
            Generated response text
        """
        if not self.provider:
            raise RuntimeError(f"No provider available for model {self.openai_model}")

        # Add high_reasoning_effort to kwargs if set
        if self.high_reasoning_effort:
            kwargs["high_reasoning_effort"] = True

        response = self.provider.get_response(self.openai_model, messages, **kwargs)
        return response.content

    def _refine_kernel(
        self,
        kernel_code: str,
        error_info: Dict[str, str],
        problem_description: str,
        test_code: str,
        additional_code: Optional[str] = None,
    ) -> str:
        """
        Refine kernel based on error information using OpenAI API.

        Uses multi-turn dialogue by incorporating history of previous attempts.
        """
        if self.provider:
            try:
                self.logger.info(f"Refining kernel using {self.openai_model}")

                # Build context from history
                history_context = ""
                if self.history:
                    history_context = "\n\nPREVIOUS ATTEMPTS:\n"
                    for i, round_data in enumerate(self.history):
                        history_context += f"\nAttempt {i + 1}:\n"
                        history_context += f"Kernel code:\n```python\n{round_data['kernel_code'][:500]}...\n```\n"
                        if round_data.get("stderr"):
                            history_context += f"Error: {round_data['stderr'][:200]}\n"
                        if round_data.get("stdout"):
                            history_context += f"Output: {round_data['stdout'][:200]}\n"

                # Create refinement prompt using template
                prompt = self.prompt_manager.render_kernel_refinement_prompt(
                    problem_description=problem_description,
                    test_code=test_code,
                    kernel_code=kernel_code,
                    error_info=error_info,
                    history_context=history_context,
                    additional_code=additional_code,
                )

                # Call LLM API
                messages = [{"role": "user", "content": prompt}]
                response_text = self._call_llm(messages, max_tokens=24576)

                # Extract refined kernel from response
                refined_kernel = self._extract_code_from_response(response_text)

                if refined_kernel:
                    self.logger.info(
                        f"Successfully refined kernel using {self.openai_model}"
                    )
                    return refined_kernel
                else:
                    self.logger.error("Failed to extract valid code from LLM response")
                    # Return original kernel if extraction fails
                    return kernel_code

            except Exception as e:
                self.logger.error(f"Error refining kernel with LLM API: {e}")
                # Fall back to mock refinement

        # Mock refinement (fallback)
        self.logger.info("Refining kernel (mock implementation)")

        # For testing, make a simple modification
        if "error" in error_info.get("stderr", "").lower():
            # Add a comment to show refinement happened
            return f"# Refinement attempt {len(self.history) + 1}\n{kernel_code}"

        return kernel_code

    def _optimize_kernel(
        self,
        kernel_code: str,
        problem_description: str,
        test_code: str,
        max_opt_rounds: int = 3,
        additional_code: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Optimize a working kernel using RAG-based pattern retrieval.

        Args:
            kernel_code: Working kernel to optimize
            problem_description: Original problem description
            test_code: Test code to verify correctness
            max_opt_rounds: Maximum optimization attempts

        Returns:
            Tuple of (success, optimized_kernel_code)
        """
        if not self.enable_optimization or not self.opt_hierarchy:
            return False, kernel_code

        self.logger.info("Starting optimization phase")

        self.dsl = "triton"
        self.kernel_name = "triton_kernel"

        try:
            # Step 1: RAG Retrieval
            self.logger.info("Step 1: Retrieving optimization pattern from database")
            retriever = Retriever(
                func_prompt=problem_description,
                opt_prompt=self.optimization_hint,
                model=self.openai_model,
                dsl=self.dsl,
                kernel_name=self.kernel_name,
                database=self.opt_hierarchy,
                module_path=self.workdir,
                debug=True,
            )

            opt_node, debug_info = retriever.retrieve()
            self.logger.info(
                f"Retrieved optimization pattern: {opt_node.opt_desc[:100]}..."
            )
            # Step 2: Build optimization prompt using PerfPromptManager
            self.logger.info("Step 2: Building optimization prompt")
            perf_prompt_manager = PerfPromptManager(
                func_source_code=kernel_code,
                func_prompt=problem_description,
                opt_prompt=self.optimization_hint,
                model=self.openai_model,
                dsl=self.dsl,
                kernel_name=self.kernel_name,
                database=self.opt_hierarchy,
                opt_node=opt_node,
                module_path=self.workdir,
                debug=True,
            )

            opt_prompt, debug_str = perf_prompt_manager.build_rewrite_prompt()
            self.logger.info(
                f"Optimization prompt built successfully: {opt_prompt[:100]}..."
            )

            # Step 3: Try optimization with multiple rounds
            best_kernel = kernel_code
            best_perf = None
            error_feedback = ""

            for opt_round in range(max_opt_rounds):
                self.logger.info(f"Optimization round {opt_round + 1}/{max_opt_rounds}")

                # Build current prompt with error feedback if available
                current_prompt = opt_prompt
                if error_feedback:
                    current_prompt = f"""{error_feedback}

Please fix the issues in the previous attempt and generate a corrected optimized kernel.

{opt_prompt}
"""

                # Step 3a: Call LLM (same pattern as _refine_kernel)
                messages = [{"role": "user", "content": current_prompt}]
                try:
                    response_text = self._call_llm(messages, max_tokens=24576)
                except Exception as e:
                    self.logger.error(f"LLM call failed: {e}")
                    error_feedback = (
                        f"Previous attempt failed: LLM call error - {str(e)}"
                    )
                    continue

                # Step 3b: Extract code (same pattern as _refine_kernel)
                optimized_kernel = self._extract_code_from_response(response_text)

                if not optimized_kernel or len(optimized_kernel) < 100:
                    self.logger.warning(
                        f"Failed to extract valid optimized kernel (length: {len(optimized_kernel) if optimized_kernel else 0})"
                    )
                    error_feedback = "Previous attempt failed: No valid kernel code extracted from LLM response. Please provide complete Triton kernel code wrapped in ```python code blocks."
                    continue

                # Step 4: Verify correctness by running tests
                self.logger.info("Testing optimized kernel...")
                self._write_kernel(optimized_kernel)
                success, stdout, stderr = self._run_test()

                if not success:
                    self.logger.warning(
                        f"Optimized kernel failed tests: {stderr[:200]}"
                    )
                    error_feedback = f"""Previous optimization attempt FAILED with error:
{stderr[:500]}

The kernel must:
1. Pass all correctness tests
2. Maintain the same interface as the original kernel
3. Be syntactically valid Python/Triton code
"""
                    continue

                # Step 5: Passed tests! Extract performance metrics
                self.logger.info("✅ Optimized kernel passed tests!")
                perf_metrics = self._extract_performance_metrics(stdout)

                if perf_metrics:
                    speedup = perf_metrics.get("speedup", 0)
                    self.logger.info(f"Performance metrics: {perf_metrics}")

                    # Update best if this is better
                    if best_perf is None or speedup > best_perf.get("speedup", 0):
                        best_kernel = optimized_kernel
                        best_perf = perf_metrics
                        self.logger.info(f"🎉 New best speedup: {speedup:.2f}x")
                        error_feedback = ""  # Clear error for next iteration
                    else:
                        self.logger.info(
                            f"Speedup {speedup:.2f}x not better than best {best_perf.get('speedup', 0):.2f}x"
                        )
                else:
                    # No metrics available, accept first working optimization
                    self.logger.info(
                        "No performance metrics found, accepting optimized kernel"
                    )
                    best_kernel = optimized_kernel
                    break

            # After all rounds, restore original kernel file
            self._write_kernel(kernel_code)

            # Return best result
            if best_kernel != kernel_code:
                self.logger.info("✅ Optimization successful!")
                if best_perf:
                    self.logger.info(
                        f"Final speedup: {best_perf.get('speedup', 'N/A'):.2f}x"
                    )
                return True, best_kernel
            else:
                self.logger.info("No improvement found, keeping original")
                return False, kernel_code

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False, kernel_code

    def _extract_performance_metrics(self, stdout: str) -> Optional[Dict[str, float]]:
        """
        Extract performance metrics from test output.
        Looks for: PERF_METRICS:{"triton_ms": X, "pytorch_ms": Y, "speedup": Z}
        """
        try:
            match = re.search(r"PERF_METRICS:(\{[^}]+\})", stdout)
            if match:
                return json.loads(match.group(1))
        except Exception as e:
            self.logger.warning(f"Failed to extract metrics: {e}")
        return None

    def _log_round(
        self, round_num: int, success: bool, kernel_code: str, stdout: str, stderr: str
    ):
        """Log the results of a verification round."""
        round_data = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "kernel_code": kernel_code,
            "stdout": stdout,
            "stderr": stderr,
        }

        # Save to log file
        round_log_file = self.log_dir / f"round_{round_num}.json"
        with open(round_log_file, "w") as f:
            json.dump(round_data, f, indent=2)

        # Add to history
        self.history.append(round_data)

    def run(
        self,
        kernel_code: str,
        test_code: str,
        problem_description: str,
        success_event: mp.Event,
        additional_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run verification and refinement loop.

        Args:
            kernel_code: Initial kernel implementation
            test_code: Test code to verify kernel
            problem_description: Problem description for context
            success_event: Shared event to check if another worker succeeded
            additional_code: Optional additional code (reference implementation)

        Returns:
            Dictionary with results
        """
        self.logger.info(f"Starting verification for worker {self.worker_id}")

        current_kernel = kernel_code

        # PHASE 1: Generation & Correctness (existing code)
        for round_num in range(self.max_rounds):
            # Check if another worker has succeeded
            if success_event.is_set():
                self.logger.info("Another worker succeeded, stopping")
                return {
                    "worker_id": self.worker_id,
                    "success": False,
                    "stopped_early": True,
                    "rounds": round_num,
                }

            self.logger.info(f"Round {round_num + 1}/{self.max_rounds}")

            # Write files - test only on first round, kernel every round
            if round_num == 0:
                # First round: write both kernel and test
                self._write_files(current_kernel, test_code)
            else:
                # Subsequent rounds: only update kernel, test remains unchanged
                self._write_kernel(current_kernel)

            # Run test
            success, stdout, stderr = self._run_test()

            # Log round
            self._log_round(round_num + 1, success, current_kernel, stdout, stderr)

            if success:
                self.logger.info(
                    f"Success! Kernel passed test in round {round_num + 1}"
                )

                # PHASE 2: Optimization if enabled
                if self.enable_optimization:
                    self.logger.info("Entering optimization phase...")
                    opt_success, optimized_kernel = self._optimize_kernel(
                        kernel_code=current_kernel,
                        problem_description=problem_description,
                        test_code=test_code,
                        additional_code=additional_code,
                    )

                    if opt_success:
                        current_kernel = optimized_kernel
                        self.logger.info("Using optimized kernel")
                    else:
                        self.logger.info("Using original working kernel")

                    return {
                        "worker_id": self.worker_id,
                        "success": True,
                        "kernel_code": current_kernel,
                        "rounds": round_num + 1,
                        "optimized": self.enable_optimization and opt_success,  # NEW
                        "history": list(self.history),
                    }

                return {
                    "worker_id": self.worker_id,
                    "success": True,
                    "kernel_code": current_kernel,
                    "rounds": round_num + 1,
                    "history": list(self.history),
                }

            # Refine kernel for next round
            error_info = {
                "stdout": stdout,
                "stderr": stderr,
                "history": list(self.history),
            }

            current_kernel = self._refine_kernel(
                current_kernel,
                error_info,
                problem_description,
                test_code,
                additional_code,
            )

        # Max rounds reached without success
        self.logger.warning(f"Max rounds ({self.max_rounds}) reached without success")
        return {
            "worker_id": self.worker_id,
            "success": False,
            "max_rounds_reached": True,
            "rounds": self.max_rounds,
            "history": list(self.history),
        }
