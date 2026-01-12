"""
Agent runtime registry.

Provides a global registry for agent runtimes, allowing them to be
looked up by key.
"""

from typing import Optional

from agent_runtime_core.interfaces import AgentRuntime


# Global registry
_runtimes: dict[str, AgentRuntime] = {}


def register_runtime(runtime: AgentRuntime) -> None:
    """
    Register an agent runtime.
    
    Args:
        runtime: The runtime to register
        
    Raises:
        ValueError: If a runtime with the same key is already registered
    """
    key = runtime.key
    if key in _runtimes:
        raise ValueError(f"Runtime already registered: {key}")
    _runtimes[key] = runtime


def get_runtime(key: str) -> Optional[AgentRuntime]:
    """
    Get a registered runtime by key.
    
    Args:
        key: The runtime key
        
    Returns:
        The runtime, or None if not found
    """
    return _runtimes.get(key)


def list_runtimes() -> list[str]:
    """
    List all registered runtime keys.
    
    Returns:
        List of runtime keys
    """
    return list(_runtimes.keys())


def unregister_runtime(key: str) -> bool:
    """
    Unregister a runtime.
    
    Args:
        key: The runtime key
        
    Returns:
        True if unregistered, False if not found
    """
    if key in _runtimes:
        del _runtimes[key]
        return True
    return False


def clear_registry() -> None:
    """Clear all registered runtimes. Useful for testing."""
    _runtimes.clear()
