"""
Configuration management using pydantic-settings and python-dotenv.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # LLM Configuration
    llm_provider: str = Field(default="openai", description="LLM provider")
    llm_api_key: str = Field(..., description="API key for LLM provider")
    llm_model: str = Field(default="gpt-4o", description="Model name")
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for LLM API"
    )

    # OpenAI Embeddings Configuration
    openai_embedding_api_key: str = Field(
        ...,
        description="OpenAI API key for embeddings"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name"
    )

    # Supabase Configuration
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_key: str = Field(..., description="Supabase anon key")

    # Application Configuration
    ado_project_name: str = Field(
        default="NorthStar",
        description="Azure DevOps project name"
    )
    artifact_registry_path: str = Field(
        ...,
        description="File system path for artifact storage"
    )

    # Application Settings
    app_env: str = Field(default="development", description="Environment")
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")
    max_retries: int = Field(default=3, description="Max retry attempts")
    timeout_seconds: int = Field(default=30, description="Default timeout")

    @field_validator("llm_api_key", "openai_embedding_api_key", "supabase_url", "supabase_key")
    @classmethod
    def validate_required_keys(cls, v):
        """Ensure required keys are not empty."""
        if not v or v.strip() == "":
            raise ValueError("Required API key or URL cannot be empty")
        return v

    @field_validator("app_env")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"app_env must be one of {valid_envs}")
        return v


def load_settings() -> Settings:
    """Load settings with proper error handling."""
    try:
        return Settings()
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        if "api_key" in str(e).lower():
            error_msg += "\nMake sure to set all required API keys in your .env file"
        raise ValueError(error_msg) from e


# Global settings instance
try:
    settings = load_settings()
except Exception:
    # For testing, create settings with dummy values
    import os
    os.environ.setdefault("LLM_API_KEY", "test_key")
    os.environ.setdefault("OPENAI_EMBEDDING_API_KEY", "test_key")
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "test_key")
    os.environ.setdefault("ARTIFACT_REGISTRY_PATH", "/tmp/artifacts")
    settings = Settings()
