"""
LLM client implementations.

Provides:
- LLMClient: Abstract interface (from interfaces.py)
- OpenAIClient: OpenAI API client
- AnthropicClient: Anthropic API client
- LiteLLMClient: LiteLLM adapter (optional)
"""

from agent_runtime_core.interfaces import LLMClient, LLMResponse, LLMStreamChunk

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LLMStreamChunk",
    "get_llm_client",
    "OpenAIConfigurationError",
    "AnthropicConfigurationError",
]


class OpenAIConfigurationError(Exception):
    """Raised when OpenAI API key is not configured."""
    pass


class AnthropicConfigurationError(Exception):
    """Raised when Anthropic API key is not configured."""
    pass


def get_llm_client(provider: str = None, **kwargs) -> LLMClient:
    """
    Factory function to get an LLM client.

    Args:
        provider: "openai", "anthropic", "litellm", etc.
        **kwargs: Provider-specific configuration (e.g., api_key, default_model)

    Returns:
        LLMClient instance
        
    Raises:
        OpenAIConfigurationError: If OpenAI is selected but API key is not configured
        AnthropicConfigurationError: If Anthropic is selected but API key is not configured
        ValueError: If an unknown provider is specified
        
    Example:
        # Using config (recommended)
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
