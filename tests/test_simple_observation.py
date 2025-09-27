"""Simple test for observation distribution debugging."""

import asyncio

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig
from gunn.schemas.types import EffectDraft


@pytest.mark.asyncio
async def test_simple_observation_distribution():
    """Test basic observation distribution."""
    # Create orchestrator
    config = OrchestratorConfig(max_agents=5, use_in_memory_dedup=True)
    orchestrator = Orchestrator(config, world_id="test")
    await orchestrator.initialize()

    # Create observation policy
    policy = DefaultObservationPolicy(PolicyConfig(distance_limit=100.0))

    # Register agent
    agent_handle = await orchestrator.register_agent("agent_1", policy)

    # Position agent
    orchestrator.world_state.spatial_index["agent_1"] = (0.0, 0.0, 0.0)
    orchestrator.world_state.spatial_index["speaker"] = (10.0, 10.0, 0.0)

    # Broadcast event
    draft: EffectDraft = {
        "kind": "Speak",
        "payload": {"text": "Hello!"},
        "source_id": "speaker",
        "schema_version": "1.0.0",
    }

    await orchestrator.broadcast_event(draft)

    # Try to get observation
    try:
        observation = await asyncio.wait_for(
            agent_handle.next_observation(), timeout=1.0
        )
        print(f"Received observation: {observation}")
        assert observation is not None
        assert observation["view_seq"] == 1
    except TimeoutError:
        pytest.fail("Should have received observation")

    await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(test_simple_observation_distribution())
