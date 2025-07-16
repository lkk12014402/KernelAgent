import asyncio
import os
import uuid
from abc import ABC, abstractmethod
from typing import List

from kernel_opt.configs.envs import MAX_TOKENS
from anthropic import Anthropic
from openai import OpenAI

class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_response(self, model_name: str, prompt: str) -> str:
        """Get response from the LLM provider."""
        pass


class AnthropicProvider(BaseProvider):
    def __init__(self):
        self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def get_response(self, model_name: str, prompt: str) -> str:
        response = self._client.messages.create(
            model=model_name,
            max_tokens=MAX_TOKENS,
            temperature=0.5,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIProvider(BaseProvider):
    def __init__(self):
        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1"
        )

    def get_response(self, model_name: str, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=model_name,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class DeepSeekProvider(BaseProvider):
    def __init__(self):
        self._client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    def get_response(self, model_name: str, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=model_name,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

class GoogleProvider(BaseProvider):
    def __init__(self):
        self._client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def get_response(self, model_name: str, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=model_name,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
