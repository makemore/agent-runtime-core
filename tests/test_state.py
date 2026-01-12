"""Test state store implementations."""

import pytest
from uuid import uuid4

from agent_runtime_core.state import InMemoryStateStore


@pytest.fixture
def state_store():
    """Create a fresh state store for each test."""
    store = InMemoryStateStore()
    yield store
    store.clear()


@pytest.mark.asyncio
async def test_save_and_get_checkpoint(state_store):
    """Test saving and retrieving checkpoints."""
    run_id = uuid4()
    state = {"step": 1, "data": "test"}
    
    await state_store.save_checkpoint(run_id, state)
    
    result = await state_store.get_checkpoint(run_id)
    assert result == state


@pytest.mark.asyncio
async def test_get_checkpoint_returns_latest(state_store):
    """Test that get_checkpoint returns the latest checkpoint."""
    run_id = uuid4()
    
    await state_store.save_checkpoint(run_id, {"step": 1})
    await state_store.save_checkpoint(run_id, {"step": 2})
    await state_store.save_checkpoint(run_id, {"step": 3})
    
    result = await state_store.get_checkpoint(run_id)
    assert result == {"step": 3}


@pytest.mark.asyncio
async def test_get_checkpoint_returns_none_for_unknown_run(state_store):
    """Test that get_checkpoint returns None for unknown runs."""
    run_id = uuid4()
    
    result = await state_store.get_checkpoint(run_id)
    assert result is None


@pytest.mark.asyncio
async def test_update_and_get_status(state_store):
    """Test updating and retrieving run status."""
    run_id = uuid4()
    
    await state_store.update_run_status(run_id, "running")
    
    result = await state_store.get_run_status(run_id)
    assert result == "running"


@pytest.mark.asyncio
async def test_get_status_returns_none_for_unknown_run(state_store):
    """Test that get_run_status returns None for unknown runs."""
    run_id = uuid4()
    
    result = await state_store.get_run_status(run_id)
    assert result is None
