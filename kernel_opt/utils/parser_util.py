"""Utility functions for parsing and extracting code from text."""

import re
import ast
import inspect
from typing import Callable
from pathlib import Path

def get_unwrapped_source(func):
    # Get the full source (including decorator)
    full_source = inspect.getsource(func)

    # Parse the AST
    tree = ast.parse(full_source)

    # Get the function node (should be the first statement)
    func_node = tree.body[0]

    # Remove decorators
    func_node.decorator_list = []

    # Convert back to source code
    return ast.unparse(func_node)

def get_module_path(torch_fn: Callable) -> Path:
    module = inspect.getmodule(torch_fn)
    if module is None:
        raise ValueError("Could not determine module for function")
    module_path = Path(module.__file__)
    return module_path

def remove_decorators_from_file(filepath: Path):
    # with open(filepath, 'r') as f:
    #     source_code = f.read()
    source_code = filepath.read_text()

    # Parse the AST
    tree = ast.parse(source_code)

    # Remove kernel_opt decorators from all functions and classes
    for node in ast.walk(tree):
        # if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        #     node.decorator_list = []  # Clear the list of decorators
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "kernel_opt":
                    # print(f"  - {decorator.id}")
                    node.decorator_list.remove(decorator)
                elif isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name) and decorator.func.id == "kernel_opt":
                        node.decorator_list.remove(decorator)
                        # print(f"  - {decorator.func.id} (with arguments)")

    # Convert back to source code
    modified_code = ast.unparse(tree)

    return modified_code

def extract_code(response_text: str, debug: bool) -> str:
    """Extract code from response text with proper error handling.

    Args:
        response_text: The text containing potential code blocks

    Returns:
        The extracted code if found, otherwise the original text stripped
    """
    code = ""
    # Look for python code blocks
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, response_text, re.DOTALL)

    if matches:
        # Return first code block
        code = matches[0].strip()
    else:
        # No markdown found, return original text
        code = response_text.strip()

    return code
