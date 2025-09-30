"""Unit tests for the Orchestrator class.

Tests cover agent registration, basic orchestration functionality,
deterministic ordering, and sim_time authority handling.
"""

import asyncio
from typing import cast

import pytest

from gunn.core.orchestrator import (
    AgentHandle,
    DefaultEffectValidator,
    EffectValidator,
    Orchestrator,
    OrchestratorConfig,
)
from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig
from gunn.schemas.messages import WorldState
from gunn.schemas.types import CancelToken, EffectDraft, Intent


class TestOrchestratorConfig:
    """Test OrchestratorConfig initialization and defaults."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = OrchestratorConfig()

        assert config.max_agents == 100
        assert config.staleness_threshold == 0
        assert config.debounce_ms == 100.0
        assert config.deadline_ms == 5000.0
        assert config.token_budget == 1000
        assert config.backpressure_policy == "defer"
        assert config.default_priority == 0
        assert config.processing_idle_shutdown_ms == 250.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = OrchestratorConfig(
            max_agents=50,
            staleness_threshold=5,
            debounce_ms=200.0,
            deadline_ms=10000.0,
            token_budget=2000,
            backpressure_policy="shed_oldest",
            default_priority=10,
            processing_idle_shutdown_ms=0.0,
        )

        assert config.max_agents == 50
        assert config.staleness_threshold == 5
        assert config.debounce_ms == 200.0
        assert config.deadline_ms == 10000.0
        assert config.token_budget == 2000
        assert config.backpressure_policy == "shed_oldest"
        assert config.default_priority == 10
        assert config.processing_idle_shutdown_ms == 0.0


class TestDefaultEffectValidator:
    """Test DefaultEffectValidator implementation."""

    def test_validate_intent_with_proper_setup(self) -> None:
        """Test that default validator returns True with proper setup."""
        validator = DefaultEffectValidator()

        # Set up permissions for the agent
        validator.set_agent_permissions("agent_1", {"submit_intent", "intent:speak"})

        # Set up world state with the agent
        world_state = WorldState(entities={"agent_1": {"name": "Test Agent"}})

        intent: Intent = {
            "kind": "Speak",
            "payload": {"message": "Hello"},  # Changed from "text" to "message"
            "context_seq": 1,
            "req_id": "test_req_1",
            "agent_id": "agent_1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        result = validator.validate_intent(intent, world_state)
        assert result is True

    def test_validate_intent_fails_without_permissions(self) -> None:
        """Test that default validator fails without proper permissions."""
        from gunn.utils.errors import ValidationError

        validator = DefaultEffectValidator()
        world_state = WorldState()

        intent: Intent = {
            "kind": "Speak",
            "payload": {"message": "Hello"},
            "context_seq": 1,
            "req_id": "test_req_1",
            "agent_id": "agent_1",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        with pytest.raises(ValidationError):
            validator.validate_intent(intent, world_state)


class TestAgentHandle:
    """Test AgentHandle functionality."""

    @pytest.fixture
    async def orchestrator(self) -> Orchestrator:
        """Create orchestrator for testing."""
        config = OrchestratorConfig(max_agents=10)
        orchestrator = Orchestrator(config, world_id="test_world")
        await orchestrator.initialize()
        return orchestrator

    @pytest.fixture
    def agent_handle(self, orchestrator: Orchestrator) -> AgentHandle:
        """Create agent handle for testing."""
        return AgentHandle("test_agent", orchestrator)

    def test_initialization(self, agent_handle: AgentHandle) -> None:
        """Test AgentHandle initialization."""
        assert agent_handle.agent_id == "test_agent"
        assert agent_handle.view_seq == 0
        assert isinstance(agent_handle.orchestrator, Orchestrator)

    def test_get_view_seq(self, agent_handle: AgentHandle) -> None:
        """Test view sequence tracking."""
        assert agent_handle.get_view_seq() == 0

        # Simulate view sequence update
        agent_handle.view_seq = 42
        assert agent_handle.get_view_seq() == 42

    @pytest.mark.asyncio
    async def test_next_observation_unregistered_agent(
        self, agent_handle: AgentHandle
    ) -> None:
        """Test next_observation raises error for unregistered agent."""
        with pytest.raises(RuntimeError, match="Agent test_agent is not registered"):
            await agent_handle.next_observation()

    @pytest.mark.asyncio
    async def test_submit_intent(self, orchestrator: Orchestrator) -> None:
        """Test intent submission through agent handle."""
        # Register agent first
        policy = DefaultObservationPolicy(PolicyConfig())
        handle = await orchestrator.register_agent("test_agent", policy)

        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 1,
            "req_id": "test_req_1",
            "agent_id": "test_agent",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        req_id = await handle.submit_intent(intent)
        assert req_id == "test_req_1"

    @pytest.mark.asyncio
    async def test_cancel_intent(self, orchestrator: Orchestrator) -> None:
        """Test intent cancellation through agent handle."""
        # Register agent
        policy = DefaultObservationPolicy(PolicyConfig())
        handle = await orchestrator.register_agent("test_agent", policy)

        # Issue cancel token
        token = orchestrator.issue_cancel_token("test_agent", "test_req_1")
        assert not token.cancelled

        # Cancel through handle
        await handle.cancel("test_req_1")
        assert token.cancelled
        assert token.reason == "user_requested"  # type: ignore[unreachable]


class TestOrchestrator:
    """Test Orchestrator core functionality."""

    @pytest.fixture
    def config(self) -> OrchestratorConfig:
        """Create test configuration."""
        return OrchestratorConfig(
            max_agents=5,
            staleness_threshold=2,
            dedup_warmup_minutes=0,  # Disable warmup for tests
            use_in_memory_dedup=True,  # Use in-memory dedup for tests
        )

    @pytest.fixture
    async def orchestrator(self, config: OrchestratorConfig) -> Orchestrator:
        """Create orchestrator for testing."""
        orchestrator = Orchestrator(config, world_id="test_world")
        await orchestrator.initialize()
        return orchestrator

    @pytest.fixture
    def observation_policy(self) -> DefaultObservationPolicy:
        """Create observation policy for testing."""
        return DefaultObservationPolicy(PolicyConfig(distance_limit=50.0))

    def test_initialization(
        self, orchestrator: Orchestrator, config: OrchestratorConfig
    ) -> None:
        """Test Orchestrator initialization."""
        assert orchestrator.world_id == "test_world"
        assert orchestrator.config == config
        assert isinstance(orchestrator.event_log, type(orchestrator.event_log))
        assert isinstance(orchestrator.world_state, WorldState)
        assert isinstance(orchestrator.effect_validator, DefaultEffectValidator)
        assert len(orchestrator.agent_handles) == 0
        assert orchestrator.get_agent_count() == 0
        assert orchestrator.get_latest_seq() == 0

    def test_custom_effect_validator(self) -> None:
        """Test orchestrator with custom effect validator."""

        class CustomValidator(EffectValidator):
            def validate_intent(self, intent: Intent, world_state: WorldState) -> bool:
                return False

            def set_agent_permissions(
                self, agent_id: str, permissions: set[str]
            ) -> None:
                pass

        config = OrchestratorConfig()
        validator = CustomValidator()
        orchestrator = Orchestrator(config, effect_validator=validator)

        assert orchestrator.effect_validator == validator

    def test_sim_time_authority(self, orchestrator: Orchestrator) -> None:
        """Test sim_time authority management."""
        assert orchestrator._sim_time_authority == "none"

        orchestrator.set_sim_time_authority("unity")
        assert orchestrator._sim_time_authority == "unity"

        orchestrator.set_sim_time_authority("unreal")
        assert orchestrator._sim_time_authority == "unreal"

        orchestrator.set_sim_time_authority("none")
        assert orchestrator._sim_time_authority == "none"

    @pytest.mark.asyncio
    async def test_register_agent_success(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test successful agent registration."""
        handle = await orchestrator.register_agent("agent_1", observation_policy)

        assert isinstance(handle, AgentHandle)
        assert handle.agent_id == "agent_1"
        assert orchestrator.get_agent_count() == 1
        assert "agent_1" in orchestrator.agent_handles
        assert "agent_1" in orchestrator.observation_policies
        assert "agent_1" in orchestrator._per_agent_queues

    @pytest.mark.asyncio
    async def test_register_agent_empty_id(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test agent registration with empty ID fails."""
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            await orchestrator.register_agent("", observation_policy)

        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            await orchestrator.register_agent("   ", observation_policy)

    @pytest.mark.asyncio
    async def test_register_agent_duplicate(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test duplicate agent registration fails."""
        await orchestrator.register_agent("agent_1", observation_policy)

        with pytest.raises(ValueError, match="Agent agent_1 is already registered"):
            await orchestrator.register_agent("agent_1", observation_policy)

    @pytest.mark.asyncio
    async def test_register_agent_max_exceeded(
        self, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test agent registration fails when max agents exceeded."""
        config = OrchestratorConfig(max_agents=2)
        orchestrator = Orchestrator(config)
        await orchestrator.initialize()

        # Register up to max
        await orchestrator.register_agent("agent_1", observation_policy)
        await orchestrator.register_agent("agent_2", observation_policy)

        # Third registration should fail
        with pytest.raises(RuntimeError, match="Maximum agents \\(2\\) exceeded"):
            await orchestrator.register_agent("agent_3", observation_policy)

    @pytest.mark.asyncio
    async def test_multiple_agent_registration(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test registering multiple agents."""
        handles = []
        for i in range(3):
            handle = await orchestrator.register_agent(f"agent_{i}", observation_policy)
            handles.append(handle)

        assert orchestrator.get_agent_count() == 3
        assert len(orchestrator.agent_handles) == 3
        assert len(orchestrator.observation_policies) == 3
        assert len(orchestrator._per_agent_queues) == 3

        # Verify each handle is unique
        agent_ids = [h.agent_id for h in handles]
        assert len(set(agent_ids)) == 3

    def test_next_seq_monotonic(self, orchestrator: Orchestrator) -> None:
        """Test that _next_seq() returns monotonically increasing values."""
        seq1 = orchestrator._next_seq()
        seq2 = orchestrator._next_seq()
        seq3 = orchestrator._next_seq()

        assert seq1 == 1
        assert seq2 == 2
        assert seq3 == 3
        assert seq1 < seq2 < seq3

    def test_current_sim_time(self, orchestrator: Orchestrator) -> None:
        """Test _current_sim_time() returns valid time."""
        time1 = orchestrator._current_sim_time()
        time2 = orchestrator._current_sim_time()

        assert isinstance(time1, float)
        assert isinstance(time2, float)
        assert time2 >= time1  # Time should not go backwards

    @pytest.mark.asyncio
    async def test_broadcast_event_success(self, orchestrator: Orchestrator) -> None:
        """Test successful event broadcasting."""
        draft: EffectDraft = {
            "kind": "TestEvent",
            "payload": {"data": "test"},
            "source_id": "test_source",
            "schema_version": "1.0.0",
        }

        initial_seq = orchestrator.get_latest_seq()
        await orchestrator.broadcast_event(draft)

        # Verify sequence incremented
        assert orchestrator.get_latest_seq() == initial_seq + 1

        # Verify event was logged
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1

        effect = entries[0].effect
        assert effect["kind"] == "TestEvent"
        assert effect["payload"]["data"] == "test"
        assert (
            effect["payload"]["priority"] == orchestrator.config.default_priority
        )  # Priority completion
        assert effect["source_id"] == "test_source"
        assert effect["schema_version"] == "1.0.0"
        assert "uuid" in effect
        assert "global_seq" in effect
        assert "sim_time" in effect

    @pytest.mark.asyncio
    async def test_broadcast_event_missing_fields(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test broadcast_event validation for missing fields."""
        # Missing kind
        from gunn.schemas.types import EffectDraft as EffectDraftType

        with pytest.raises(ValueError, match="EffectDraft must have 'kind' field"):
            invalid_effect = cast(
                EffectDraftType,
                {
                    "payload": {},
                    "source_id": "test",
                    "schema_version": "1.0.0",
                    # Missing kind
                },
            )
            await orchestrator.broadcast_event(invalid_effect)

        # Missing source_id
        with pytest.raises(ValueError, match="EffectDraft must have 'source_id' field"):
            invalid_effect = cast(
                EffectDraftType,
                {
                    "kind": "Test",
                    "payload": {},
                    "schema_version": "1.0.0",
                    # Missing source_id
                },
            )
            await orchestrator.broadcast_event(invalid_effect)

        # Missing schema_version
        with pytest.raises(
            ValueError, match="EffectDraft must have 'schema_version' field"
        ):
            invalid_effect = cast(
                EffectDraftType,
                {
                    "kind": "Test",
                    "payload": {},
                    "source_id": "test",
                    # Missing schema_version
                },
            )
            await orchestrator.broadcast_event(invalid_effect)

    @pytest.mark.asyncio
    async def test_submit_intent_success(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test successful intent submission."""
        # Register agent first
        await orchestrator.register_agent("test_agent", observation_policy)

        # Set up permissions for the agent
        orchestrator.effect_validator.set_agent_permissions(
            "test_agent", {"submit_intent", "intent:speak"}
        )

        # Add agent to world state
        orchestrator.world_state.entities["test_agent"] = {"name": "Test Agent"}

        intent: Intent = {
            "kind": "Speak",
            "payload": {"message": "Hello"},  # Changed from "text" to "message"
            "context_seq": 1,
            "req_id": "test_req_1",
            "agent_id": "test_agent",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        req_id = await orchestrator.submit_intent(intent)
        assert req_id == "test_req_1"

        # Wait a bit for async processing to complete
        await asyncio.sleep(0.1)

        # Verify effect was created
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1

        effect = entries[0].effect
        assert effect["kind"] == "Speak"
        assert (
            effect["payload"]["message"] == "Hello"
        )  # Changed from "text" to "message"
        assert effect["payload"]["priority"] == 0  # Default priority should be added
        assert effect["source_id"] == "test_agent"

    @pytest.mark.asyncio
    async def test_submit_intent_idempotency(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test intent idempotency - duplicate req_id should not create duplicate effects."""
        await orchestrator.register_agent("test_agent", observation_policy)

        # Set up permissions for the agent
        orchestrator.effect_validator.set_agent_permissions(
            "test_agent", {"submit_intent", "intent:speak"}
        )

        # Add agent to world state
        orchestrator.world_state.entities["test_agent"] = {"name": "Test Agent"}

        intent1: Intent = {
            "kind": "Speak",
            "payload": {"message": "Hello"},  # Changed from "text" to "message"
            "context_seq": 1,
            "req_id": "test_req_1",
            "agent_id": "test_agent",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        intent2: Intent = {
            "kind": "Speak",
            "payload": {"message": "Hello"},  # Changed from "text" to "message"
            "context_seq": 1,
            "req_id": "test_req_1",  # Same req_id for idempotency test
            "agent_id": "test_agent",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Submit same intent twice
        req_id1 = await orchestrator.submit_intent(intent1)
        req_id2 = await orchestrator.submit_intent(intent2)

        assert req_id1 == req_id2 == "test_req_1"

        # Wait a bit for async processing to complete
        await asyncio.sleep(0.5)

        # Should only have one effect in log
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_submit_intent_validation_failure(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test intent submission with validation failure."""

        # Create validator that always fails
        class FailingValidator(EffectValidator):
            def validate_intent(self, intent: Intent, world_state: WorldState) -> bool:
                return False

            def set_agent_permissions(
                self, agent_id: str, permissions: set[str]
            ) -> None:
                pass

        config = OrchestratorConfig()
        orchestrator_with_failing_validator = Orchestrator(
            config, effect_validator=FailingValidator()
        )
        await orchestrator_with_failing_validator.initialize()

        await orchestrator_with_failing_validator.register_agent(
            "test_agent", observation_policy
        )

        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 1,
            "req_id": "test_req_1",
            "agent_id": "test_agent",
            "priority": 0,
            "schema_version": "1.0.0",
        }

        # Submit intent (should be enqueued but fail during processing)
        req_id = await orchestrator_with_failing_validator.submit_intent(intent)
        assert req_id == "test_req_1"

        # Wait for processing
        await asyncio.sleep(0.1)

        # Intent should be enqueued but processing should fail
        # No effect should be created in the event log
        entries = orchestrator_with_failing_validator.event_log.get_all_entries()
        assert len(entries) == 0

        # Check that intent was enqueued
        stats = orchestrator_with_failing_validator.get_processing_stats()
        assert stats["scheduler"]["total_enqueued"] == 1

    @pytest.mark.asyncio
    async def test_submit_intent_missing_fields(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test submit_intent validation for missing fields."""
        # Missing req_id
        from gunn.schemas.types import Intent as IntentType

        with pytest.raises(ValueError, match="Intent must have 'req_id' field"):
            invalid_intent = cast(
                IntentType,
                {
                    "kind": "Speak",
                    "payload": {},
                    "agent_id": "test",
                    "context_seq": 1,
                    "priority": 0,
                    "schema_version": "1.0.0",
                    # Missing req_id
                },
            )
            await orchestrator.submit_intent(invalid_intent)

        # Missing agent_id
        with pytest.raises(ValueError, match="Intent must have 'agent_id' field"):
            invalid_intent = cast(
                IntentType,
                {
                    "kind": "Speak",
                    "payload": {},
                    "req_id": "test",
                    "context_seq": 1,
                    "priority": 0,
                    "schema_version": "1.0.0",
                    # Missing agent_id
                },
            )
            await orchestrator.submit_intent(invalid_intent)

        # Missing kind
        with pytest.raises(ValueError, match="Intent must have 'kind' field"):
            invalid_intent = cast(
                IntentType,
                {
                    "payload": {},
                    "req_id": "test",
                    "agent_id": "test",
                    "context_seq": 1,
                    "priority": 0,
                    "schema_version": "1.0.0",
                    # Missing kind
                },
            )
            await orchestrator.submit_intent(invalid_intent)

    def test_issue_cancel_token(self, orchestrator: Orchestrator) -> None:
        """Test cancel token issuance."""
        token = orchestrator.issue_cancel_token("test_agent", "test_req")

        assert isinstance(token, CancelToken)
        assert token.agent_id == "test_agent"
        assert token.req_id == "test_req"
        assert not token.cancelled

        # Verify token is stored
        key = ("test_world", "test_agent", "test_req")
        assert key in orchestrator._cancel_tokens
        assert orchestrator._cancel_tokens[key] == token

    @pytest.mark.asyncio
    async def test_cancel_if_stale_no_token(self, orchestrator: Orchestrator) -> None:
        """Test cancel_if_stale with no existing token."""
        result = await orchestrator.cancel_if_stale("test_agent", "test_req", 10)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_if_stale_already_cancelled(
        self, orchestrator: Orchestrator
    ) -> None:
        """Test cancel_if_stale with already cancelled token."""
        token = orchestrator.issue_cancel_token("test_agent", "test_req")
        token.cancel("already_cancelled")

        result = await orchestrator.cancel_if_stale("test_agent", "test_req", 10)
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_if_stale_no_agent(self, orchestrator: Orchestrator) -> None:
        """Test cancel_if_stale with unregistered agent."""
        orchestrator.issue_cancel_token("test_agent", "test_req")

        result = await orchestrator.cancel_if_stale("test_agent", "test_req", 10)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_if_stale_threshold_not_exceeded(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test cancel_if_stale when staleness threshold is not exceeded."""
        # Register agent
        handle = await orchestrator.register_agent("test_agent", observation_policy)
        handle.view_seq = 5

        # Issue token
        token = orchestrator.issue_cancel_token("test_agent", "test_req")

        # Check staleness (new_seq=7, current=5, staleness=2, threshold=2)
        result = await orchestrator.cancel_if_stale("test_agent", "test_req", 7)
        assert result is False
        assert not token.cancelled

    @pytest.mark.asyncio
    async def test_cancel_if_stale_threshold_exceeded(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test cancel_if_stale when staleness threshold is exceeded."""
        # Register agent
        handle = await orchestrator.register_agent("test_agent", observation_policy)
        handle.view_seq = 5

        # Issue token
        token = orchestrator.issue_cancel_token("test_agent", "test_req")

        # Check staleness (new_seq=10, current=5, staleness=5, threshold=2)
        result = await orchestrator.cancel_if_stale("test_agent", "test_req", 10)
        assert result is True
        assert token.cancelled
        assert token.reason is not None and "stale_due_to_seq_gap_5" in token.reason

    def test_get_world_state(self, orchestrator: Orchestrator) -> None:
        """Test world state access."""
        world_state = orchestrator.get_world_state()
        assert isinstance(world_state, WorldState)
        assert world_state == orchestrator.world_state

    @pytest.mark.asyncio
    async def test_shutdown(
        self, orchestrator: Orchestrator, observation_policy: DefaultObservationPolicy
    ) -> None:
        """Test orchestrator shutdown."""
        # Register some agents
        await orchestrator.register_agent("agent_1", observation_policy)
        await orchestrator.register_agent("agent_2", observation_policy)

        # Issue some tokens
        orchestrator.issue_cancel_token("agent_1", "req_1")
        orchestrator.issue_cancel_token("agent_2", "req_2")

        # Verify state before shutdown
        assert len(orchestrator.agent_handles) == 2
        assert len(orchestrator._per_agent_queues) == 2
        assert len(orchestrator._cancel_tokens) == 2

        # Shutdown
        await orchestrator.shutdown()

        # Verify cleanup
        assert len(orchestrator.agent_handles) == 0
        assert len(orchestrator.observation_policies) == 0
        assert len(orchestrator._per_agent_queues) == 0
        assert len(orchestrator._cancel_tokens) == 0
        assert len(orchestrator._quota_tracker) == 0


class TestDeterministicOrdering:
    """Test deterministic ordering requirements."""

    @pytest.fixture
    async def orchestrator(self) -> Orchestrator:
        """Create orchestrator for ordering tests."""
        config = OrchestratorConfig()
        orchestrator = Orchestrator(config, world_id="ordering_test")
        await orchestrator.initialize()
        return orchestrator

    @pytest.mark.asyncio
    async def test_effect_ordering_fields(self, orchestrator: Orchestrator) -> None:
        """Test that effects have all required ordering fields."""
        draft: EffectDraft = {
            "kind": "TestEvent",
            "payload": {"data": "test"},
            "source_id": "test_source",
            "schema_version": "1.0.0",
        }

        await orchestrator.broadcast_event(draft)

        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 1

        effect = entries[0].effect

        # Verify all ordering fields are present
        assert "sim_time" in effect
        assert "global_seq" in effect
        assert "source_id" in effect
        assert "uuid" in effect

        # Verify types
        assert isinstance(effect["sim_time"], float)
        assert isinstance(effect["global_seq"], int)
        assert isinstance(effect["source_id"], str)
        assert isinstance(effect["uuid"], str)

    @pytest.mark.asyncio
    async def test_global_seq_ordering(self, orchestrator: Orchestrator) -> None:
        """Test that global_seq provides deterministic ordering."""
        # Create multiple effects
        drafts = [
            {
                "kind": f"Event_{i}",
                "payload": {"index": i},
                "source_id": f"source_{i}",
                "schema_version": "1.0.0",
            }
            for i in range(5)
        ]

        # Broadcast all effects
        for draft in drafts:
            from gunn.schemas.types import EffectDraft

            typed_draft = cast(EffectDraft, draft)
            await orchestrator.broadcast_event(typed_draft)

        # Verify ordering
        entries = orchestrator.event_log.get_all_entries()
        assert len(entries) == 5

        for i, entry in enumerate(entries):
            assert entry.effect["global_seq"] == i + 1
            assert entry.effect["payload"]["index"] == i

    @pytest.mark.asyncio
    async def test_uuid_uniqueness(self, orchestrator: Orchestrator) -> None:
        """Test that each effect gets a unique UUID."""
        # Create multiple effects with same content
        draft: EffectDraft = {
            "kind": "TestEvent",
            "payload": {"data": "same"},
            "source_id": "same_source",
            "schema_version": "1.0.0",
        }

        # Broadcast multiple times
        for _ in range(10):
            await orchestrator.broadcast_event(draft)

        # Verify all UUIDs are unique
        entries = orchestrator.event_log.get_all_entries()
        uuids = [entry.effect["uuid"] for entry in entries]

        assert len(uuids) == 10
        assert len(set(uuids)) == 10  # All unique

    def test_sim_time_consistency(self, orchestrator: Orchestrator) -> None:
        """Test that sim_time is consistent within ordering requirements."""
        # Test multiple calls to _current_sim_time()
        times = [orchestrator._current_sim_time() for _ in range(10)]

        # All times should be valid floats
        assert all(isinstance(t, float) for t in times)

        # Times should be non-decreasing (monotonic)
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1]


if __name__ == "__main__":
    pytest.main([__file__])
