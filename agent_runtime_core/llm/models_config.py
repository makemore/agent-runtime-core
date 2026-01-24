"""
Supported LLM models configuration.

Provides a central registry of supported models with their providers,
capabilities, and metadata. Used for:
- Auto-detecting provider from model name
- Populating model selectors in UI
- Validating model choices
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    """Information about a supported model."""
    id: str  # Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514")
    name: str  # Display name (e.g., "GPT-4o", "Claude Sonnet 4")
    provider: str  # "openai" or "anthropic"
    context_window: int  # Max context in tokens
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    description: str = ""


# Registry of supported models
SUPPORTED_MODELS: dict[str, ModelInfo] = {
    # OpenAI Models
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        context_window=128000,
        supports_vision=True,
        description="Most capable OpenAI model, multimodal",
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        context_window=128000,
        supports_vision=True,
        description="Fast and affordable, good for most tasks",
    ),
    "gpt-4-turbo": ModelInfo(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider="openai",
        context_window=128000,
        supports_vision=True,
        description="Previous generation flagship",
    ),
    "o1": ModelInfo(
        id="o1",
        name="o1",
        provider="openai",
        context_window=200000,
        supports_tools=False,
        description="Advanced reasoning model",
    ),
    "o1-mini": ModelInfo(
        id="o1-mini",
        name="o1 Mini",
        provider="openai",
        context_window=128000,
        supports_tools=False,
        description="Fast reasoning model",
    ),
    "o3-mini": ModelInfo(
        id="o3-mini",
        name="o3 Mini",
        provider="openai",
        context_window=200000,
        supports_tools=True,
        description="Latest reasoning model with tool use",
    ),
    
    # Anthropic Models - Claude 4.5 (latest)
    "claude-sonnet-4-5-20250929": ModelInfo(
        id="claude-sonnet-4-5-20250929",
        name="Claude Sonnet 4.5",
        provider="anthropic",
        context_window=200000,
        supports_vision=True,
        description="Best balance of speed and capability for agents and coding",
    ),
    "claude-opus-4-5-20251101": ModelInfo(
        id="claude-opus-4-5-20251101",
        name="Claude Opus 4.5",
        provider="anthropic",
        context_window=200000,
        supports_vision=True,
        description="Premium model - maximum intelligence with practical performance",
    ),
    "claude-haiku-4-5-20251001": ModelInfo(
        id="claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5",
        provider="anthropic",
        context_window=200000,
        supports_vision=True,
        description="Fastest model with near-frontier intelligence",
    ),
    # Anthropic Models - Claude 4 (previous generation)
    "claude-sonnet-4-20250514": ModelInfo(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        provider="anthropic",
        context_window=200000,
        supports_vision=True,
        description="Previous generation Sonnet",
    ),
    "claude-opus-4-20250514": ModelInfo(
        id="claude-opus-4-20250514",
        name="Claude Opus 4",
        provider="anthropic",
        context_window=200000,
        supports_vision=True,
        description="Previous generation Opus",
    ),
    # Anthropic Models - Claude 3.5 (legacy)
    "claude-3-5-sonnet-20241022": ModelInfo(
        id="claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        context_window=200000,
        supports_vision=True,
        description="Legacy model, still excellent",
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        id="claude-3-5-haiku-20241022",
        name="Claude 3.5 Haiku",
        provider="anthropic",
        context_window=200000,
        supports_vision=True,
        description="Legacy fast model",
    ),
}

# Default model to use
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Get info for a model by ID."""
    return SUPPORTED_MODELS.get(model_id)


def get_provider_for_model(model_id: str) -> Optional[str]:
    """
    Detect the provider for a model ID.
    
    Returns "openai", "anthropic", or None if unknown.
    """
    # Check registry first
    if model_id in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_id].provider
    
    # Fallback heuristics for unlisted models
    if model_id.startswith("gpt-") or model_id.startswith("o1") or model_id.startswith("o3"):
        return "openai"
    if model_id.startswith("claude"):
        return "anthropic"
    
    return None


def list_models_for_ui() -> list[dict]:
    """Get list of models formatted for UI dropdowns."""
    return [
        {
            "id": m.id,
            "name": m.name,
            "provider": m.provider,
            "description": m.description,
        }
        for m in SUPPORTED_MODELS.values()
    ]

