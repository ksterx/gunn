"""Integration tests for Orchestrator action completion tracking.

Tests the full flow of intent submission, effect application, and completion
confirmation as specified in task 28.
"""

import asyncio

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig
from gunn.schemas.types import Intent


@pytest.fixture
async def orchestrator():
    """Create an orchestrator for testing."""
    config = OrchestratorConfig(
        max_agents=10,
        use_in_memory_dedup=True,
        action_completion_timeout=5.0,
    )
    orch = Orchestrator(config, world_id="test_completion")
    await orch.initialize()
    yield orch
    await orch.shutdown()


@pytest.fixture
async def agent_handle(orchestrator):
    """Create an agent handle for testing."""
    # Add agent to world state first
    orchestrator.world_state.entities["agent_1"] = {"position": [0.0, 0.0, 0.0]}
    orchestrator.world_state.spatial_index["agent_1"] = (0.0, 0.0, 0.0)

    policy_config = PolicyConfig(distance_limit=100.0)
    policy = DefaultObservationPolicy(policy_config)
    handle = await orchestrator.register_agent("agent_1", policy)
    return handle


@pytest.mark.asyncio
async def test_intent_completion_tracking(orchestrator, agent_handle):
    """Test that intents are tracked through to effect application."""
    intent: Intent = {
        "kind": "Move",
        "payload": {"to": [10.0, 20.0, 30.0], "agent_id": "agent_1"},
        "context_seq": 0,
        "req_id": "test_req_1",
        "agent_id": "agent_1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    # Submit intent
    req_id = await agent_handle.submit_intent(intent)
    assert req_id == "test_req_1"

    # Give processing loop time to process the intent
    await asyncio.sleep(0.2)

    # Wait for completion
    effect = await orchestrator.wait_effect_applied(req_id, timeout=2.0)

    assert effect is not None
    assert effect["req_id"] == req_id
    assert effect["kind"] == "Move"
    assert effect["source_id"] == "agent_1"


@pytest.mark.asyncio
async def test_wait_for_action_completion(orchestrator, agent_handle):
    """Test AgentHandle._wait_for_action_completion method."""
    intent: Intent = {
        "kind": "Speak",
        "payload": {"text": "Hello world", "agent_id": "agent_1"},
        "context_seq": 0,
        "req_id": "test_req_2",
        "agent_id": "agent_1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    # Submit intent
    req_id = await agent_handle.submit_intent(intent)

    # Wait for action completion (internal method)
    await agent_handle._wait_for_action_completion(req_id)

    # Verify effect was applied
    effect = await orchestrator.wait_effect_applied(req_id, timeout=0.1)
    assert effect["kind"] == "Speak"


@pytest.mark.asyncio
async def test_completion_timeout(orchestrator):
    """Test timeout when waiting for non-existent action."""
    with pytest.raises((asyncio.TimeoutError, ValueError)):
        await orchestrator.wait_effect_applied("nonexistent_req", timeout=0.1)


@pytest.mark.asyncio
async def test_multiple_concurrent_intents(orchestrator, agent_handle):
    """Test tracking multiple concurrent intent completions."""
    intents = [
        Intent(
            kind="Move",
            payload={"to": [float(i), float(i), 0.0], "agent_id": "agent_1"},
            context_seq=0,
            req_id=f"req_{i}",
            agent_id="agent_1",
            priority=0,
            schema_version="1.0.0",
        )
        for i in range(5)
    ]

    # Submit all intents
    req_ids = []
    for intent in intents:
        req_id = await agent_handle.submit_intent(intent)
        req_ids.append(req_id)

    # Wait for all completions
    effects = await asyncio.gather(
        *[orchestrator.wait_effect_applied(req_id, timeout=5.0) for req_id in req_ids]
    )

    assert len(effects) == 5
    for i, effect in enumerate(effects):
        assert effect["req_id"] == f"req_{i}"
        assert effect["kind"] == "Move"


@pytest.mark.asyncio
async def test_completion_after_effect_applied(orchestrator, agent_handle):
    """Test that completion is signaled after effect is applied to world state."""
    intent: Intent = {
        "kind": "Move",
        "payload": {"to": [5.0, 5.0, 0.0], "agent_id": "agent_1"},
        "context_seq": 0,
        "req_id": "test_req_3",
        "agent_id": "agent_1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    # Submit intent
    req_id = await agent_handle.submit_intent(intent)

    # Wait for completion
    effect = await orchestrator.wait_effect_applied(req_id, timeout=2.0)

    # Verify world state was updated
    assert "agent_1" in orchestrator.world_state.spatial_index
    position = orchestrator.world_state.spatial_index["agent_1"]
    assert position == (5.0, 5.0, 0.0)


@pytest.mark.asyncio
async def test_effect_has_req_id_fields(orchestrator, agent_handle):
    """Test that effects include req_id, duration_ms, and apply_at fields."""
    intent: Intent = {
        "kind": "Speak",
        "payload": {"text": "Test message", "agent_id": "agent_1"},
        "context_seq": 0,
        "req_id": "test_req_4",
        "agent_id": "agent_1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    req_id = await agent_handle.submit_intent(intent)
    effect = await orchestrator.wait_effect_applied(req_id, timeout=2.0)

    # Verify new fields are present
    assert "req_id" in effect
    assert effect["req_id"] == req_id
    assert "duration_ms" in effect
    assert "apply_at" in effect


@pytest.mark.asyncio
async def test_completion_on_validation_failure(orchestrator, agent_handle):
    """Test that failed validation doesn't leave pending completions."""
    # Create an invalid intent (this will depend on validator implementation)
    intent: Intent = {
        "kind": "InvalidKind",
        "payload": {},
        "context_seq": 0,
        "req_id": "test_req_5",
        "agent_id": "agent_1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    try:
        await agent_handle.submit_intent(intent)
    except Exception:
        pass  # Expected to fail

    # Verify no pending completions
    assert orchestrator._completion_tracker.get_pending_count() == 0


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_completions(orchestrator, agent_handle):
    """Test that shutdown cancels all pending action completions."""
    intent: Intent = {
        "kind": "Move",
        "payload": {"to": [1.0, 1.0, 0.0], "agent_id": "agent_1"},
        "context_seq": 0,
        "req_id": "test_req_6",
        "agent_id": "agent_1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    # Submit intent but don't wait
    req_id = await agent_handle.submit_intent(intent)

    # Shutdown immediately
    await orchestrator.shutdown()

    # Verify completion tracker was cleaned up
    assert orchestrator._completion_tracker.get_pending_count() == 0


@pytest.mark.asyncio
async def test_broadcast_event_without_req_id(orchestrator):
    """Test that broadcast_event works without req_id (backward compatibility)."""
    from gunn.schemas.types import EffectDraft

    draft: EffectDraft = {
        "kind": "EnvironmentChanged",
        "payload": {"temperature": 25.0},
        "source_id": "environment",
        "schema_version": "1.0.0",
    }

    # Should not raise error
    await orchestrator.broadcast_event(draft)

    # Verify effect was created without req_id
    latest_entry = orchestrator.event_log._entries[-1]
    assert latest_entry.effect["req_id"] is None


@pytest.mark.asyncio
async def test_completion_tracker_timeout_config(orchestrator):
    """Test that completion tracker uses configured timeout."""
    # Verify tracker was initialized with config timeout
    assert orchestrator._completion_tracker._default_timeout == 5.0


@pytest.mark.asyncio
async def test_wait_effect_applied_already_completed(orchestrator, agent_handle):
    """Test waiting for an already completed action."""
    intent: Intent = {
        "kind": "Move",
        "payload": {"to": [3.0, 3.0, 0.0], "agent_id": "agent_1"},
        "context_seq": 0,
        "req_id": "test_req_7",
        "agent_id": "agent_1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    req_id = await agent_handle.submit_intent(intent)

    # Wait once
    effect1 = await orchestrator.wait_effect_applied(req_id, timeout=2.0)

    # Wait again - should return immediately from event log
    effect2 = await orchestrator.wait_effect_applied(req_id, timeout=0.1)

    assert effect1 == effect2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
