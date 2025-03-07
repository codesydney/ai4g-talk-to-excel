import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="gpt-4o")


class OllamaSettings(LLMSettings):
    """Ollama specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OLLAMA_API_KEY"))
    base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL"))
    default_model: str = Field(default="deepseek-r1:8b")


class BedrockSettings(LLMSettings):
    """Bedrock specific settings extending LLMSettings."""

    access_key_id: str = Field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID"))
    secret_access_key: str = Field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY"))
    session_token: str = Field(default_factory=lambda: os.getenv("AWS_SESSION_TOKEN"))
    default_region: str = Field(default_factory=lambda: os.getenv("AWS_DEFAULT_REGION"))
    default_model: str = Field(default="anthropic.claude-3-5-sonnet-20241022-v2:0")
    max_tokens: Optional[int] = 1024
