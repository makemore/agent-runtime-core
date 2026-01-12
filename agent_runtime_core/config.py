"""
Configuration system for agent_runtime.

Supports:
- Programmatic configuration via configure()
- Environment variables
- Default values
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RuntimeConfig:
    """
    Configuration for the agent runtime.
    
    All settings can be overridden via environment variables with
    AGENT_RUNTIME_ prefix (e.g., AGENT_RUNTIME_MODEL_PROVIDER).
    """
    
    # LLM Provider
    model_provider: str = "openai"  # openai, anthropic, litellm
    default_model: str = "gpt-4o"
    
    # API Keys (loaded from env if not set)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Queue backend
    queue_backend: str = "memory"  # memory, redis, sqlite
    
    # Event bus backend
    event_bus_backend: str = "memory"  # memory, redis, sqlite
    
    # State store backend
    state_store_backend: str = "memory"  # memory, redis, sqlite
    
    # Tracing backend
    tracing_backend: Optional[str] = None  # noop, langfuse
    
    # Redis settings
    redis_url: Optional[str] = None
    
    # SQLite settings
    sqlite_path: Optional[str] = None
    
    # Langfuse settings
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: Optional[str] = None
    
    # Runner settings
    run_timeout_seconds: int = 300
    heartbeat_interval_seconds: int = 30
    lease_ttl_seconds: int = 60
    max_retries: int = 3
    retry_backoff_base: int = 2
    retry_backoff_max: int = 300
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from config or environment."""
        return self.openai_api_key or os.environ.get("OPENAI_API_KEY")
    
    def get_anthropic_api_key(self) -> Optional[str]:
        """Get Anthropic API key from config or environment."""
        return self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")


# Global configuration instance
_config: Optional[RuntimeConfig] = None


def configure(**kwargs) -> RuntimeConfig:
    """
    Configure the agent runtime.
    
    Args:
        **kwargs: Configuration options (see RuntimeConfig)
        
    Returns:
        The configured RuntimeConfig instance
        
    Example:
        from agent_runtime_core import configure
        
        configure(
            model_provider="openai",
            openai_api_key="sk-...",
            queue_backend="redis",
            redis_url="redis://localhost:6379",
        )
    """
    global _config
    
    # Start with defaults
    config = RuntimeConfig()
    
    # Apply environment variables
    _apply_env_vars(config)
    
    # Apply explicit kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration option: {key}")
    
    _config = config
    return config


def get_config() -> RuntimeConfig:
    """
    Get the current configuration.
    
    If not configured, returns default configuration with env vars applied.
    
    Returns:
        RuntimeConfig instance
    """
    global _config
    
    if _config is None:
        _config = RuntimeConfig()
        _apply_env_vars(_config)
    
    return _config


def reset_config() -> None:
    """Reset configuration to defaults. Useful for testing."""
    global _config
    _config = None


def _apply_env_vars(config: RuntimeConfig) -> None:
    """Apply environment variables to config."""
    env_mapping = {
        "AGENT_RUNTIME_MODEL_PROVIDER": "model_provider",
        "AGENT_RUNTIME_DEFAULT_MODEL": "default_model",
        "AGENT_RUNTIME_QUEUE_BACKEND": "queue_backend",
        "AGENT_RUNTIME_EVENT_BUS_BACKEND": "event_bus_backend",
        "AGENT_RUNTIME_STATE_STORE_BACKEND": "state_store_backend",
        "AGENT_RUNTIME_TRACING_BACKEND": "tracing_backend",
        "AGENT_RUNTIME_REDIS_URL": "redis_url",
        "AGENT_RUNTIME_SQLITE_PATH": "sqlite_path",
        "AGENT_RUNTIME_LANGFUSE_PUBLIC_KEY": "langfuse_public_key",
        "AGENT_RUNTIME_LANGFUSE_SECRET_KEY": "langfuse_secret_key",
        "AGENT_RUNTIME_LANGFUSE_HOST": "langfuse_host",
    }
    
    int_fields = {
        "AGENT_RUNTIME_RUN_TIMEOUT_SECONDS": "run_timeout_seconds",
        "AGENT_RUNTIME_HEARTBEAT_INTERVAL_SECONDS": "heartbeat_interval_seconds",
        "AGENT_RUNTIME_LEASE_TTL_SECONDS": "lease_ttl_seconds",
        "AGENT_RUNTIME_MAX_RETRIES": "max_retries",
        "AGENT_RUNTIME_RETRY_BACKOFF_BASE": "retry_backoff_base",
        "AGENT_RUNTIME_RETRY_BACKOFF_MAX": "retry_backoff_max",
    }
    
    for env_var, attr in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            setattr(config, attr, value)
    
    for env_var, attr in int_fields.items():
        value = os.environ.get(env_var)
        if value is not None:
            setattr(config, attr, int(value))
