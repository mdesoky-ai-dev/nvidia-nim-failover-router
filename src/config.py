"""Centralized, environment-driven configuration.

Everything that differs between dev / edge box / cloud lives here so the
router and clients stay free of magic strings. Values are read from the
environment (and an optional `.env` file) via pydantic-settings.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Primary lane: local NVIDIA NIM container (OpenAI-compatible) ---
    nim_base_url: str = "http://localhost:8000"
    nim_chat_path: str = "/v1/chat/completions"
    nim_health_path: str = "/v1/health/ready"
    nim_model: str = "meta/llama-3.1-8b-instruct"
    nim_api_key: str | None = None  # local NIM usually needs none; set if required
    health_timeout_ms: int = 200  # hard sub-second budget for the readiness probe

    # --- Failover lane: AWS Bedrock (Claude 3.5 Sonnet) ---
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    # --- Generation + transport defaults ---
    default_max_tokens: int = 1024
    default_temperature: float = 0.2
    request_timeout_s: float = 60.0
    connect_timeout_s: float = 2.0

    @property
    def nim_chat_url(self) -> str:
        return f"{self.nim_base_url}{self.nim_chat_path}"

    @property
    def nim_health_url(self) -> str:
        return f"{self.nim_base_url}{self.nim_health_path}"

    @property
    def health_timeout_s(self) -> float:
        return self.health_timeout_ms / 1000.0


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the env is parsed once per process."""
    return Settings()