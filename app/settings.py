from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables."""

    model_dir: Path = Path("models")
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
