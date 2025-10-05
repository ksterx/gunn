"""Unit tests for ActionCompletionTracker.

Tests cover completion tracking, timeout handling, and error scenarios
as specified in task 28.
"""

import asyncio

import pytest

from gunn.core.completion_tracker import ActionCompletionTracker
from gunn.schemas.types import Effect


@pytest.fixture
def tracker():
    """Create a completion tracker for testing."""
    return ActionCompletionTracker(default_timeout=1.0)


@pytest.mark.asyncio
async def test_register_action(tracker):
    """Test registering an action for tracking."""
    future = tracker.register_action("req_1")

    assert isinstance(future, asyncio.Future)
    assert tracker.is_pending("req_1")
    assert tracker.get_pending_count() == 1


@pytest.mark.asyncio
async def test_register_duplicate_action(tracker):
    """Test that registering duplicate action raises error."""
    tracker.register_action("req_1")

    with pytest.raises(ValueError, match="already registered"):
        tracker.register_action("req_1")


@pytest.mark.asyncio
async def test_register_empty_req_id(tracker):
    """Test that empty req_id raises error."""
    with pytest.raises(ValueError, match="cannot be empty"):
        tracker.register_action("")

    with pytest.raises(ValueError, match="cannot be empty"):
        tracker.register_action("   ")


@pytest.mark.asyncio
async def test_complete_action(tracker):
    """Test completing an action successfully."""
    future = tracker.register_action("req_1")

    effect: Effect = {
        "uuid": "test-uuid",
        "kind": "Move",
        "payload": {"to": [1.0, 2.0, 3.0]},
        "global_seq": 1,
        "sim_time": 0.0,
        "source_id": "agent_1",
        "schema_version": "1.0.0",
        "req_id": "req_1",
        "duration_ms": None,
        "apply_at": None,
    }

    await tracker.complete_action("req_1", effect)

    assert not tracker.is_pending("req_1")
    assert tracker.get_pending_count() == 0
    assert future.done()
    assert future.result() == effect


@pytest.mark.asyncio
async def test_fail_action(tracker):
    """Test failing an action with error."""
    future = tracker.register_action("req_1")

    error = RuntimeError("Test error")
    await tracker.fail_action("req_1", error)

    assert not tracker.is_pending("req_1")
    assert tracker.get_pending_count() == 0
    assert future.done()

    with pytest.raises(RuntimeError, match="Test error"):
        future.result()


@pytest.mark.asyncio
async def test_wait_for_completion_success(tracker):
    """Test waiting for action completion successfully."""
    tracker.register_action("req_1")

    effect: Effect = {
        "uuid": "test-uuid",
        "kind": "Speak",
        "payload": {"text": "Hello"},
        "global_seq": 1,
        "sim_time": 0.0,
        "source_id": "agent_1",
        "schema_version": "1.0.0",
        "req_id": "req_1",
        "duration_ms": None,
        "apply_at": None,
    }

    # Complete action in background
    async def complete_later():
        await asyncio.sleep(0.1)
        await tracker.complete_action("req_1", effect)

    asyncio.create_task(complete_later())

    # Wait for completion
    result = await tracker.wait_for_completion("req_1", timeout=2.0)
    assert result == effect


@pytest.mark.asyncio
async def test_wait_for_completion_timeout(tracker):
    """Test timeout when waiting for action completion."""
    tracker.register_action("req_1")

    with pytest.raises(asyncio.TimeoutError):
        await tracker.wait_for_completion("req_1", timeout=0.1)


@pytest.mark.asyncio
async def test_wait_for_completion_not_registered(tracker):
    """Test waiting for non-registered action raises error."""
    with pytest.raises(ValueError, match="not registered"):
        await tracker.wait_for_completion("req_1")


@pytest.mark.asyncio
async def test_action_timeout(tracker):
    """Test that action times out automatically."""
    future = tracker.register_action("req_1", timeout=0.2)

    # Wait for timeout
    await asyncio.sleep(0.3)

    assert future.done()
    assert not tracker.is_pending("req_1")

    with pytest.raises(asyncio.TimeoutError, match="timed out"):
        future.result()


@pytest.mark.asyncio
async def test_complete_action_cancels_timeout(tracker):
    """Test that completing action cancels timeout task."""
    future = tracker.register_action("req_1", timeout=1.0)

    effect: Effect = {
        "uuid": "test-uuid",
        "kind": "Move",
        "payload": {"to": [1.0, 2.0, 3.0]},
        "global_seq": 1,
        "sim_time": 0.0,
        "source_id": "agent_1",
        "schema_version": "1.0.0",
        "req_id": "req_1",
        "duration_ms": None,
        "apply_at": None,
    }

    await tracker.complete_action("req_1", effect)

    # Wait to ensure timeout doesn't fire
    await asyncio.sleep(1.2)

    assert future.done()
    assert future.result() == effect


@pytest.mark.asyncio
async def test_multiple_actions(tracker):
    """Test tracking multiple actions simultaneously."""
    tracker.register_action("req_1")
    tracker.register_action("req_2")
    tracker.register_action("req_3")

    assert tracker.get_pending_count() == 3
    assert tracker.is_pending("req_1")
    assert tracker.is_pending("req_2")
    assert tracker.is_pending("req_3")

    effect1: Effect = {
        "uuid": "uuid-1",
        "kind": "Move",
        "payload": {},
        "global_seq": 1,
        "sim_time": 0.0,
        "source_id": "agent_1",
        "schema_version": "1.0.0",
        "req_id": "req_1",
        "duration_ms": None,
        "apply_at": None,
    }

    await tracker.complete_action("req_1", effect1)

    assert tracker.get_pending_count() == 2
    assert not tracker.is_pending("req_1")
    assert tracker.is_pending("req_2")
    assert tracker.is_pending("req_3")


@pytest.mark.asyncio
async def test_cancel_all(tracker):
    """Test cancelling all pending actions."""
    future1 = tracker.register_action("req_1")
    future2 = tracker.register_action("req_2")
    future3 = tracker.register_action("req_3")

    tracker.cancel_all()

    assert tracker.get_pending_count() == 0
    assert future1.cancelled()
    assert future2.cancelled()
    assert future3.cancelled()


@pytest.mark.asyncio
async def test_complete_already_completed_action(tracker):
    """Test completing an already completed action is safe."""
    future = tracker.register_action("req_1")

    effect: Effect = {
        "uuid": "test-uuid",
        "kind": "Move",
        "payload": {},
        "global_seq": 1,
        "sim_time": 0.0,
        "source_id": "agent_1",
        "schema_version": "1.0.0",
        "req_id": "req_1",
        "duration_ms": None,
        "apply_at": None,
    }

    await tracker.complete_action("req_1", effect)

    # Complete again - should be safe
    await tracker.complete_action("req_1", effect)

    assert future.result() == effect


@pytest.mark.asyncio
async def test_custom_timeout(tracker):
    """Test using custom timeout override."""
    future = tracker.register_action("req_1", timeout=0.1)

    # Wait for timeout
    await asyncio.sleep(0.2)

    assert future.done()
    with pytest.raises(asyncio.TimeoutError):
        future.result()


@pytest.mark.asyncio
async def test_wait_with_custom_timeout(tracker):
    """Test wait_for_completion with custom timeout."""
    tracker.register_action("req_1", timeout=10.0)  # Long default timeout

    # But use short timeout for wait
    with pytest.raises(asyncio.TimeoutError):
        await tracker.wait_for_completion("req_1", timeout=0.1)


@pytest.mark.asyncio
async def test_concurrent_completions(tracker):
    """Test handling concurrent action completions."""
    # Register multiple actions
    for i in range(10):
        tracker.register_action(f"req_{i}")

    # Complete them concurrently
    async def complete_action(req_id: str, delay: float):
        await asyncio.sleep(delay)
        effect: Effect = {
            "uuid": f"uuid-{req_id}",
            "kind": "Move",
            "payload": {},
            "global_seq": int(req_id.split("_")[1]),
            "sim_time": 0.0,
            "source_id": "agent_1",
            "schema_version": "1.0.0",
            "req_id": req_id,
            "duration_ms": None,
            "apply_at": None,
        }
        await tracker.complete_action(req_id, effect)

    tasks = [
        asyncio.create_task(complete_action(f"req_{i}", i * 0.01)) for i in range(10)
    ]

    await asyncio.gather(*tasks)

    assert tracker.get_pending_count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
