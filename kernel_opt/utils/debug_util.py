"""Utility functions for debugging."""

from kernel_opt.configs.envs import DEBUG_MODE


def debug_print(prompt: str, response: str, conversation_length: int):
    """Print debug information."""
    if DEBUG_MODE:
        print("=" * 20 + " [DEBUG_MODE] START " + "=" * 20)
        print(f"Prompt: \n{prompt}\n")
        print(f"Response: \n{response}\n")
        print(f"Conversation length: {conversation_length}")
        print("=" * 20 + " [DEBUG_MODE] END " + "=" * 20)
