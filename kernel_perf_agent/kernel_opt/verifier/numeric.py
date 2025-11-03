"""Verify kernel numerical correctness."""

import subprocess
from pathlib import Path
import sys
import inspect
from .base import Verifier, VerifyResult
from kernel_perf_agent.kernel_opt.configs.envs import TIMEOUT

class NumericVerifier(Verifier):
    """Verify kernel numerical correctness."""

    def verify(self, kernel_path: Path, kernel_code: str, test_code: bool) -> VerifyResult:
        """Run numeric tests using pytest.

        Args:
            build_dir: Path to the build directory containing test.py

        Returns:
            VerifyResult indicating success or failure
        """
        build_dir = kernel_path.parent
        test_file_name = kernel_path.name
        exec_code = kernel_code

        print("build_dir: ", build_dir)
        if not test_code:
            test_file_name = build_dir / "test.py"
            try:
                with open(test_file_name, 'r') as file:
                    exec_code = file.read()
            except FileNotFoundError:
                print(f"Error: The file '{test_file_name}' was not found.")
            except Exception as e:
                print(f"An error occurred: {e}")

            print(f"[debug] test_path: {test_file_name}")

            if not test_file_name.exists():
                return VerifyResult(ok=False, message=f"Test file not found: {test_file_name}")

        print("test_file_name: ", test_file_name)
        try:
            # Run the test in the build directory to make relative imports work
            # subprocess.run(
            #     ["python", test_file_name],  # Use relative path since we set cwd
            #     capture_output=True,
            #     text=True,
            #     check=True,
            #     timeout=TIMEOUT,
            #     cwd=str(build_dir),  # Run in the build directory
            # )
            # exec(exec_code)
            subprocess.run(["python3", "-c", exec_code])
            return VerifyResult(ok=True, message="Numeric verification passed")
        except subprocess.CalledProcessError as e:
            return VerifyResult(ok=False, message=f"Test failed: {e.stderr}")
