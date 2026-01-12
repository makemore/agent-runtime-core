"""Test event bus implementations."""

import pytest
import asyncio
from uuid import uuid4

from agent_runtime_core.events import InMemoryEventBus


@pytest.fixture
def event_bus():
    """Create a fresh event bus for each test."""
    bus = InMemoryEventBus()
    yield bus
    bus.clear()


@pytest.mark.asyncio
async def test_publish_and_get_events(event_bus):
    """Test publishing and retrieving events."""
    run_id = uuid4()
    
    await event_bus.publish(run_id, "test.event", {"data": "test"})
    
    events = await event_bus.get_events(run_id)
    
    assert len(events) == 1
    assert events[0].event_type == "test.event"
    assert events[0].payload == {"data": "test"}


@pytest.mark.asyncio
async def test_get_events_since_sequence(event_bus):
    """Test getting events since a sequence number."""
    run_id = uuid4()
    
    await event_bus.publish(run_id, "event.1", {})
    await event_bus.publish(run_id, "event.2", {})
    await event_bus.publish(run_id, "event.3", {})
    
    events = await event_bus.get_events(run_id, since_sequence=1)
    
    assert len(events) == 2
    assert events[0].event_type == "event.2"
    assert events[1].event_type == "event.3"


@pytest.mark.asyncio
async def test_subscribe_receives_events(event_bus):
    """Test that subscribers receive published events."""
    run_id = uuid4()
    received_events = []
    
    async def subscriber():
        async for event in event_bus.subscribe(run_id):
            received_events.append(event)
            if len(received_events) >= 2:
                break
    
    # Start subscriber in background
    task = asyncio.create_task(subscriber())
    
    # Give subscriber time to start
    await asyncio.sleep(0.1)
    
    # Publish events
    await event_bus.publish(run_id, "event.1", {"n": 1})
    await event_bus.publish(run_id, "event.2", {"n": 2})
    
    # Wait for subscriber to receive
    await asyncio.wait_for(task, timeout=1.0)
    
    assert len(received_events) == 2
    assert received_events[0].event_type == "event.1"
    assert received_events[1].event_type == "event.2"
