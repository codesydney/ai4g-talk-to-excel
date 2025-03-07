from functools import lru_cache
import os
from config.app_logging import setup_logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class AppSettings(BaseSettings):
    # Define your settings with types and defaults
    aws_access_key_id: str = Field(..., env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    aws_session_token: str = Field(..., env="AWS_SESSION_TOKEN")
    aws_default_region: str = Field(..., env="AWS_DEFAULT_REGION")
    
    # Configure settings behavior
    model_config = SettingsConfigDict(
        env_file="../.env",
        case_sensitive=False
    )

@lru_cache()
def get_settings() -> AppSettings:
    """Create and return a cached instance of the Settings."""
    settings = AppSettings()
    setup_logging()
    return settings   
