from dataclasses import dataclass
from typing import List

from kernel_opt.models.providers import (
    BaseProvider,
    AnthropicProvider,
    OpenAIProvider,
    DeepSeekProvider,
    GoogleProvider,
)


@dataclass
class Model:
    """Model = Model Name + Provider"""

    name: str
    provider: BaseProvider


AVAILABLE_MODELS = [
    Model(name="claude-sonnet-4-20250514", provider=AnthropicProvider),
    Model(name="gpt-3.5-turbo", provider=OpenAIProvider),
    Model(name="deepseek-reasoner", provider=DeepSeekProvider),
    Model(name="deepseek-chat", provider=DeepSeekProvider),
    Model(name="gemini-2.5-flash", provider=GoogleProvider),
]

MODEL_NAME_TO_MODEL = {model.name: model for model in AVAILABLE_MODELS}

AVAILABLE_MODEL_NAMES = [model.name for model in AVAILABLE_MODELS]


def get_model(model_name: str) -> Model:
    return MODEL_NAME_TO_MODEL[model_name]


def check_model_is_available(model_name: str) -> bool:
    return model_name in MODEL_NAME_TO_MODEL


def get_available_model_names() -> List[str]:
    return AVAILABLE_MODEL_NAMES
