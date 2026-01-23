"""
Agent runtime registry.

Provides a global registry for agent runtimes, allowing them to be
looked up by key.

Supports:
- Manual registration via register_runtime()
- Factory functions for lazy instantiation
- Class registration (auto-instantiation)
"""

import logging
from typing import Callable, Optional, Type, Union

from agent_runtime_core.interfaces import AgentRuntime

logger = logging.getLogger(__name__)

# Global registry
_runtimes: dict[str, AgentRuntime] = {}
_runtime_factories: dict[str, Callable[[], AgentRuntime]] = {}


def _is_agent_runtime(obj) -> bool:
    """Check if an object is an AgentRuntime instance."""
    return isinstance(obj, AgentRuntime)


def _is_agent_runtime_class(cls) -> bool:
    """Check if a class is an AgentRuntime subclass."""
    if not isinstance(cls, type):
        return False
    try:
        return issubclass(cls, AgentRuntime)
    except TypeError:
        return False


def register_runtime(
    runtime: Union[AgentRuntime, Type[AgentRuntime], Callable[[], AgentRuntime]],
    key: Optional[str] = None,
) -> None:
    """
    Register an agent runtime.

    Args:
        runtime: Runtime instance, class, or factory function
        key: Optional key override (uses runtime.key if not provided)

    Examples:
        # Register an instance
        register_runtime(MyRuntime())

        # Register a class (will be instantiated)
        register_runtime(MyRuntime)

        # Register with custom key
        register_runtime(MyRuntime(), key="custom-key")

        # Register a factory
        register_runtime(lambda: MyRuntime(config=get_config()), key="my-runtime")

    Raises:
        ValueError: If key is required but not provided
        TypeError: If runtime is not a valid type
    """
    if _is_agent_runtime(runtime):
        # Instance provided
        runtime_key = key or runtime.key
        _runtimes[runtime_key] = runtime
        logger.info(f"Registered agent runtime: {runtime_key}")

    elif _is_agent_runtime_class(runtime):
        # Class provided - instantiate it
        instance = runtime()
        runtime_key = key or instance.key
        _runtimes[runtime_key] = instance
        logger.info(f"Registered agent runtime: {runtime_key}")

    elif callable(runtime):
        # Factory function provided
        if not key:
            raise ValueError("key is required when registering a factory function")
        _runtime_factories[key] = runtime
        logger.info(f"Registered agent runtime factory: {key}")

    else:
        raise TypeError(
            f"runtime must be AgentRuntime instance, class, or callable, got {type(runtime)}"
        )


def get_runtime(key: str) -> AgentRuntime:
    """
    Get a registered runtime by key.

    Args:
        key: The runtime key

    Returns:
        The runtime instance

    Raises:
        KeyError: If runtime not found
    """
    # Check instances first
    if key in _runtimes:
        return _runtimes[key]

    # Check factories
    if key in _runtime_factories:
        instance = _runtime_factories[key]()
        _runtimes[key] = instance
        return instance

    raise KeyError(f"Agent runtime not found: {key}. Available: {list_runtimes()}")


def list_runtimes() -> list[str]:
    """
    List all registered runtime keys.

    Returns:
        List of runtime keys
    """
    return list(set(_runtimes.keys()) | set(_runtime_factories.keys()))


def unregister_runtime(key: str) -> bool:
    """
    Unregister a runtime.

    Args:
        key: The runtime key

    Returns:
        True if unregistered, False if not found
    """
    removed = False
    if key in _runtimes:
        del _runtimes[key]
        removed = True
    if key in _runtime_factories:
        del _runtime_factories[key]
        removed = True
    return removed


def clear_registry() -> None:
    """Clear all registered runtimes. Useful for testing."""
    _runtimes.clear()
    _runtime_factories.clear()
