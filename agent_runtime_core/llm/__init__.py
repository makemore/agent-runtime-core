"""
LLM client implementations.

Provides:
- LLMClient: Abstract interface (from interfaces.py)
- OpenAIClient: OpenAI API client
- AnthropicClient: Anthropic API client
- LiteLLMClient: LiteLLM adapter (optional)
- get_llm_client: Factory with auto-detection from model name
- get_llm_client_for_model: Get client for a specific model
"""

from typing import Optional

from agent_runtime_core.interfaces import LLMClient, LLMResponse, LLMStreamChunk
from agent_runtime_core.llm.models_config import (
    ModelInfo,
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
    get_model_info,
    get_provider_for_model,
    list_models_for_ui,
)

__all__ = [
    # Interfaces
    "LLMClient",
    "LLMResponse",
    "LLMStreamChunk",
    # Factory functions
    "get_llm_client",
    "get_llm_client_for_model",
    # Model config
    "ModelInfo",
    "SUPPORTED_MODELS",
    "DEFAULT_MODEL",
    "get_model_info",
    "get_provider_for_model",
    "list_models_for_ui",
    # Exceptions
    "OpenAIConfigurationError",
    "AnthropicConfigurationError",
]


class OpenAIConfigurationError(Exception):
    """Raised when OpenAI API key is not configured."""
    pass


class AnthropicConfigurationError(Exception):
    """Raised when Anthropic API key is not configured."""
    pass


def get_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to get an LLM client.

    Can auto-detect provider from model name if model is provided.

    Args:
        provider: "openai", "anthropic", "litellm", etc. (optional if model provided)
        model: Model ID - if provided, auto-detects provider
        **kwargs: Provider-specific configuration (e.g., api_key, default_model)

    Returns:
        LLMClient instance

    Raises:
        OpenAIConfigurationError: If OpenAI is selected but API key is not configured
        AnthropicConfigurationError: If Anthropic is selected but API key is not configured
        ValueError: If an unknown provider is specified

    Example:
        # Auto-detect from model name (recommended)
        llm = get_llm_client(model="claude-sonnet-4-20250514")
        llm = get_llm_client(model="gpt-4o")

        # Using config
        from agent_runtime_core.config import configure
        configure(model_provider="openai", openai_api_key="sk-...")
        llm = get_llm_client()

        # Or with explicit API key
        llm = get_llm_client(api_key='sk-...')

        # Or with a different provider
        llm = get_llm_client(provider='anthropic', api_key='sk-ant-...')
    """
    from agent_runtime_core.config import get_config

    config = get_config()

    # Auto-detect provider from model name if not explicitly provided
    if provider is None and model:
        detected_provider = get_provider_for_model(model)
        if detected_provider:
            provider = detected_provider

    # Fall back to config
    provider = provider or config.model_provider

    if provider == "openai":
        from agent_runtime_core.llm.openai import OpenAIClient
        return OpenAIClient(**kwargs)

    elif provider == "anthropic":
        from agent_runtime_core.llm.anthropic import AnthropicClient
        return AnthropicClient(**kwargs)

    elif provider == "litellm":
        from agent_runtime_core.llm.litellm_client import LiteLLMClient
        return LiteLLMClient(**kwargs)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}\n\n"
            f"Supported providers: 'openai', 'anthropic', 'litellm'\n"
            f"Set model_provider in your configuration."
        )


def get_llm_client_for_model(model: str, **kwargs) -> LLMClient:
    """
    Get an LLM client configured for a specific model.

    This is a convenience function that auto-detects the provider
    and sets the default model.

    Args:
        model: Model ID (e.g., "gpt-4o", "claude-sonnet-4-20250514")
        **kwargs: Additional client configuration

    Returns:
        LLMClient configured for the specified model

    Raises:
        ValueError: If model provider cannot be determined

    Example:
        llm = get_llm_client_for_model("claude-sonnet-4-20250514")
        response = await llm.generate(messages)  # Uses claude-sonnet-4-20250514
    """
    provider = get_provider_for_model(model)
    if not provider:
        raise ValueError(
            f"Cannot determine provider for model: {model}\n"
            f"Known models: {', '.join(SUPPORTED_MODELS.keys())}"
        )

    return get_llm_client(provider=provider, default_model=model, **kwargs)
