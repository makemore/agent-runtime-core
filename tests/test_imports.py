"""Test that all imports work correctly."""

import pytest


def test_version():
    """Test that version is accessible."""
    import agent_runtime_core
    # Just check that version is a string in semver format
    assert isinstance(agent_runtime_core.__version__, str)
    parts = agent_runtime_core.__version__.split(".")
    assert len(parts) >= 2  # At least major.minor


def test_core_imports():
    """Test that core interfaces can be imported."""
    from agent_runtime_core import (
        AgentRuntime,
        RunContext,
        RunResult,
        Tool,
        ToolRegistry,
        Message,
        EventType,
        ErrorInfo,
    )
    
    assert AgentRuntime is not None
    assert RunContext is not None
    assert RunResult is not None
    assert Tool is not None


def test_config_imports():
    """Test that config can be imported."""
    from agent_runtime_core import (
        RuntimeConfig,
        configure,
        get_config,
    )
    
    assert RuntimeConfig is not None
    assert configure is not None
    assert get_config is not None


def test_registry_imports():
    """Test that registry can be imported."""
    from agent_runtime_core import (
        register_runtime,
        get_runtime,
        list_runtimes,
        unregister_runtime,
        clear_registry,
    )
    
    assert register_runtime is not None
    assert get_runtime is not None


def test_runner_imports():
    """Test that runner can be imported."""
    from agent_runtime_core import (
        AgentRunner,
        RunnerConfig,
        RunContextImpl,
    )
    
    assert AgentRunner is not None
    assert RunnerConfig is not None


def test_submodule_imports():
    """Test that submodules can be imported."""
    from agent_runtime_core.state import get_state_store, InMemoryStateStore
    from agent_runtime_core.queue import get_queue, InMemoryQueue
    from agent_runtime_core.events import get_event_bus, InMemoryEventBus
    from agent_runtime_core.tracing import get_trace_sink, NoopTraceSink
    
    assert get_state_store is not None
    assert get_queue is not None
    assert get_event_bus is not None
    assert get_trace_sink is not None
