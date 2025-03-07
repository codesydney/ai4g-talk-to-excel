from functools import lru_cache
from config.app_logging import setup_logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from config.llm_settings import (
    AnthropicSettings,
    BedrockSettings,
    OllamaSettings,
    OpenAISettings,
)


class AppSettings(BaseSettings):
    """Main settings class combining all sub-settings."""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    bedrock: BedrockSettings = Field(default_factory=BedrockSettings)

    model_config = SettingsConfigDict(
        env_file="../.env",  # Load environment variables from the .env file
        case_sensitive=False,
    )


@lru_cache()
def get_settings() -> AppSettings:
    """Create and return a cached instance of the AppSettings."""
    settings = AppSettings()
    setup_logging()
    return settings
