"""
Tracing implementations for observability.

Provides:
- TraceSink: Abstract interface
- NoopTraceSink: No-op implementation
- LangfuseTraceSink: Langfuse integration
"""

from agent_runtime_core.tracing.noop import NoopTraceSink

__all__ = [
    "NoopTraceSink",
    "get_trace_sink",
]


def get_trace_sink(backend: str = None, **kwargs):
    """
    Factory function to get a trace sink.
    
    Args:
        backend: "noop" or "langfuse"
        **kwargs: Backend-specific configuration
        
    Returns:
        TraceSink instance
    """
    from agent_runtime_core.config import get_config
    from agent_runtime_core.interfaces import TraceSink
    
    config = get_config()
    backend = backend or config.tracing_backend or "noop"
    
    if backend == "noop":
        return NoopTraceSink()
    
    elif backend == "langfuse":
        from agent_runtime_core.tracing.langfuse import LangfuseTraceSink
        return LangfuseTraceSink(
            public_key=kwargs.get("public_key") or config.langfuse_public_key,
            secret_key=kwargs.get("secret_key") or config.langfuse_secret_key,
            host=kwargs.get("host") or config.langfuse_host,
        )
    
    else:
        raise ValueError(f"Unknown tracing backend: {backend}")
