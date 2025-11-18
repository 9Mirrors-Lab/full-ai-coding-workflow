"""
Provider configuration for LLM and external services.
"""

from typing import Optional
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import OpenAI
from supabase import create_client, Client
from .settings import settings


def get_llm_model(model_choice: Optional[str] = None) -> OpenAIModel:
    """
    Get LLM model configuration based on environment variables.

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured OpenAI model instance
    """
    model_name = model_choice or settings.llm_model

    provider = OpenAIProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key
    )

    return OpenAIModel(model_name, provider=provider)


def get_embedding_client() -> OpenAI:
    """
    Get OpenAI client for embeddings.

    Returns:
        Configured OpenAI client
    """
    return OpenAI(api_key=settings.openai_embedding_api_key)


def get_supabase_client() -> Client:
    """
    Get Supabase client for database operations.

    Returns:
        Configured Supabase client
    """
    return create_client(settings.supabase_url, settings.supabase_key)


def validate_llm_configuration() -> bool:
    """
    Validate LLM configuration is complete.

    Returns:
        True if valid, raises ValueError otherwise
    """
    if not settings.llm_api_key:
        raise ValueError("LLM_API_KEY is required")
    if not settings.llm_model:
        raise ValueError("LLM_MODEL is required")
    return True
