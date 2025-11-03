"""Utility functions for logging configuration and operations."""

import logging


def setup_basic_logging() -> None:
    """Configure basic logging with a modern format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def log_interaction(
    logger: logging.Logger, prompt: str, response: str, conversation_length: int
) -> None:
    """Log an interaction with the model.

    Args:
        logger: Logger instance to use
        prompt: The prompt sent to the model
        response: The response from the model
        conversation_length: Current length of the conversation
    """
    logger.info(
        f"Generated response: prompt={len(prompt)}, "
        f"response={len(response)}, "
        f"conversation_length={conversation_length}"
    )
