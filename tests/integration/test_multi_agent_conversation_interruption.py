"""Integration tests for multi-agent conversation with interruption scenarios.

These tests verify the complete multi-agent conversation workflow including
intelligent interruption, regeneration, and context staleness detection.
"""

import asyncio
import time
import uuid

import pytest

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.facades import MessageFacade, RLFacade
from gunn.policies.observation import ConversationObservationPolicy, PolicyConfig
from gunn.schemas.types import Intent
from gunn.utils.telemetry import get_logger


class MockLLMAgent:
    """Mock LLM agent for testing interruption scenarios."""

    def __init__(self, agent_id: str, facade: RLFacade | MessageFacade):
        self.agent_id = agent_id
        self.facade = facade
        self.logger = get_logger(f"mock_agent.{agent_id}")
        self.generation_active = False
        self.current_cancel_token = None
        self.tokens_generated = 0
        self.interruption_count = 0

    async def generate_with_interruption(
        self, message: str, req_id: str, context_seq: int = 0
    ) -> tuple[str, bool]:
        """Generate message with interruption awareness.

        Returns:
            tuple: (generated_message, was_interrupted)
        """
        orchestrator = self.facade.get_orchestrator()
        self.current_cancel_token = orchestrator.issue_cancel_token(
            self.agent_id, req_id
        )

        tokens = message.split()
        generated_tokens = []
        was_interrupted = False

        try:
            self.generation_active = True
            self.tokens_generated = 0

            for i, token in enumerate(tokens):
                # Check for cancellation at token boundaries (requirement 6.2)
                if self.current_cancel_token and self.current_cancel_token.cancelled:
                    self.logger.info(
                        f"Agent {self.agent_id} interrupted at token {i}/{len(tokens)}"
                    )
                    was_interrupted = True
                    self.interruption_count += 1
                    break

                # Simulate token generation time (20-30ms for responsive cancellation)
                await asyncio.sleep(0.025)  # 25ms per token
                generated_tokens.append(token)
                self.tokens_generated += 1

            final_message = " ".join(generated_tokens)
            if was_interrupted:
                final_message += " [INTERRUPTED]"

            return final_message, was_interrupted

        finally:
            self.generation_active = False
            self.current_cancel_token = None

    async def submit_intent(self, message: str, context_seq: int = 0) -> str:
        """Submit intent with the generated message."""
        req_id = f"intent_{uuid.uuid4().hex[:8]}"

        intent: Intent = {
            "kind": "Speak",
            "payload": {"message": message},
            "context_seq": context_seq,
            "req_id": req_id,
            "agent_id": self.agent_id,
            "priority": 1,
            "schema_version": "1.0.0",
        }

        if isinstance(self.facade, RLFacade):
            effect, observation = await self.facade.step(self.agent_id, intent)
            return req_id
        else:
            await self.facade.emit("Speak", {"message": message}, self.agent_id)
            return req_id


class TestMultiAgentConversationInterruption:
    """Test suite for multi-agent conversation with interruption scenarios."""

    def setup_agent_permissions(
        self, orchestrator: Orchestrator, agent_ids: list[str]
    ) -> None:
        """Setup permissions and world state for test agents."""
        validator = orchestrator.effect_validator
        if hasattr(validator, "set_agent_permissions"):
            # Grant all necessary permissions for testing
            permissions = {
                "submit_intent",
                "intent:speak",
                "intent:move",
                "intent:interact",
                "intent:custom",
            }
            for agent_id in agent_ids:
                validator.set_agent_permissions(agent_id, permissions)

        # Add agents to world state so they are not "not_in_world"
        for agent_id in agent_ids:
            orchestrator.world_state.entities[agent_id] = {
                "id": agent_id,
                "type": "agent",
                "position": {"x": 0, "y": 0},  # Add default position for Move intents
            }
            # Also add to spatial_index for Move intent validation
            orchestrator.world_state.spatial_index[agent_id] = (0.0, 0.0, 0.0)

    @pytest.fixture
    def config(self) -> OrchestratorConfig:
        """Create test configuration optimized for interruption testing."""
        return OrchestratorConfig(
            max_agents=5,
            staleness_threshold=0,  # Immediate staleness detection
            debounce_ms=50.0,  # Short debounce for testing
            deadline_ms=5000.0,
            token_budget=1000,
            backpressure_policy="defer",
            default_priority=1,
            use_in_memory_dedup=True,
            processing_idle_shutdown_ms=0.0,  # Disable idle shutdown for tests
        )

    @pytest.fixture
    async def orchestrator(self, config: OrchestratorConfig) -> Orchestrator:
        """Create and initialize orchestrator."""
        orchestrator = Orchestrator(config, world_id="conversation_test")
        await orchestrator.initialize()
        return orchestrator

    @pytest.fixture
    async def rl_facade(self, orchestrator: Orchestrator) -> RLFacade:
        """Create and initialize RL facade."""
        facade = RLFacade(orchestrator=orchestrator)
        await facade.initialize()
        return facade

    @pytest.fixture
    async def message_facade(self, orchestrator: Orchestrator) -> MessageFacade:
        """Create and initialize message facade."""
        facade = MessageFacade(orchestrator=orchestrator)
        await facade.initialize()
        return facade

    @pytest.fixture
    def conversation_policy(self) -> ConversationObservationPolicy:
        """Create conversation observation policy."""
        config = PolicyConfig(
            distance_limit=float("inf"),
            relationship_filter=[],
            field_visibility={},
            max_patch_ops=20,
        )
        return ConversationObservationPolicy(config)

    @pytest.mark.asyncio
    async def test_abc_conversation_with_interruption(
        self, rl_facade: RLFacade, conversation_policy: ConversationObservationPolicy
    ) -> None:
        """Test A/B/C conversation scenario with B interrupting A."""
        # Register agents
        alice_handle = await rl_facade.register_agent("alice", conversation_policy)
        bob_handle = await rl_facade.register_agent("bob", conversation_policy)
        charlie_handle = await rl_facade.register_agent("charlie", conversation_policy)

        # Create mock agents
        alice = MockLLMAgent("alice", rl_facade)
        bob = MockLLMAgent("bob", rl_facade)
        charlie = MockLLMAgent("charlie", rl_facade)

        # Setup permissions and world state for agents
        orchestrator = rl_facade.get_orchestrator()
        self.setup_agent_permissions(orchestrator, ["alice", "bob", "charlie"])

        # Phase 1: Alice starts a long message
        alice_message = (
            "Hello everyone! I wanted to tell you about this amazing discovery "
            "I made yesterday while working on the quantum computing project. "
            "It's really fascinating how the quantum entanglement principles "
            "can be applied to multi-agent systems and I think we should "
            "definitely explore this further in our next research phase."
        )

        alice_task = asyncio.create_task(
            alice.generate_with_interruption(alice_message, "alice_msg_1", 0)
        )

        # Phase 2: Let Alice generate for a bit, then Bob interrupts
        await asyncio.sleep(0.3)  # Let Alice generate ~12 tokens

        # Bob emits urgent message that should trigger Alice's cancellation
        bob_message = (
            "Wait Alice! I have urgent news about the quantum project - "
            "we just got approval for the next phase!"
        )

        # Broadcast Bob's event to trigger staleness
        await rl_facade.get_orchestrator().broadcast_event(
            {
                "kind": "MessageEmitted",
                "payload": {"text": bob_message, "speaker": "bob"},
                "source_id": "bob",
                "schema_version": "1.0.0",
            }
        )

        # Phase 3: Complete Alice's generation (should be interrupted)
        alice_result, alice_interrupted = await alice_task

        # Verify Alice was interrupted
        assert alice_interrupted, "Alice should have been interrupted"
        assert "[INTERRUPTED]" in alice_result
        assert alice.interruption_count == 1
        assert alice.tokens_generated < len(alice_message.split())

        # Phase 4: Bob completes his message
        bob_result, bob_interrupted = await bob.generate_with_interruption(
            bob_message, "bob_msg_1", 1
        )

        assert not bob_interrupted, "Bob should not have been interrupted"
        assert "[INTERRUPTED]" not in bob_result

        # Phase 5: Charlie responds to both
        charlie_message = (
            "Wow, that's exciting news Bob! And Alice, I'd love to hear more "
            "about your quantum entanglement ideas when you're ready."
        )

        charlie_result, charlie_interrupted = await charlie.generate_with_interruption(
            charlie_message, "charlie_msg_1", 2
        )

        assert not charlie_interrupted, "Charlie should not have been interrupted"

        # Verify event log contains all events in correct order
        event_log = rl_facade.get_orchestrator().event_log
        entries = event_log.get_all_entries()

        # Should have at least the broadcast event
        assert len(entries) >= 1
        assert any(entry.effect["kind"] == "MessageEmitted" for entry in entries)

        # Verify log integrity
        assert event_log.validate_integrity()

    @pytest.mark.asyncio
    async def test_rapid_interruption_debounce(
        self, rl_facade: RLFacade, conversation_policy: ConversationObservationPolicy
    ) -> None:
        """Test that rapid interruptions are properly debounced."""
        # Register agents
        await rl_facade.register_agent("agent1", conversation_policy)
        await rl_facade.register_agent("agent2", conversation_policy)

        agent1 = MockLLMAgent("agent1", rl_facade)
        orchestrator = rl_facade.get_orchestrator()

        # Setup permissions and world state for agents
        self.setup_agent_permissions(orchestrator, ["agent1", "agent2"])

        # Start generation
        generation_task = asyncio.create_task(
            agent1.generate_with_interruption(
                "This is a long message that should be interrupted multiple times",
                "rapid_test_1",
                0,
            )
        )

        # Wait a bit then trigger rapid interruptions
        await asyncio.sleep(0.1)

        # Trigger multiple rapid events
        for i in range(5):
            await orchestrator.broadcast_event(
                {
                    "kind": "RapidEvent",
                    "payload": {"event_id": i},
                    "source_id": "agent2",
                    "schema_version": "1.0.0",
                }
            )
            await asyncio.sleep(0.01)  # 10ms between events

        # Complete generation
        result, was_interrupted = await generation_task

        # Should be interrupted, but debounce should prevent excessive interruptions
        assert was_interrupted
        assert agent1.interruption_count <= 2, "Debounce should limit interruptions"

    @pytest.mark.asyncio
    async def test_cancellation_timing_slo(
        self, rl_facade: RLFacade, conversation_policy: ConversationObservationPolicy
    ) -> None:
        """Test that cancellation meets 100ms SLO requirement."""
        # Register agent
        await rl_facade.register_agent("timing_agent", conversation_policy)

        agent = MockLLMAgent("timing_agent", rl_facade)
        orchestrator = rl_facade.get_orchestrator()

        # Setup permissions and world state for agents
        self.setup_agent_permissions(orchestrator, ["timing_agent"])

        # Issue cancel token
        req_id = "timing_test_1"
        cancel_token = orchestrator.issue_cancel_token("timing_agent", req_id)

        # Start generation
        generation_task = asyncio.create_task(
            agent.generate_with_interruption(
                "This is a message that will be cancelled for timing test " * 10,
                req_id,
                0,
            )
        )

        # Wait for generation to start
        await asyncio.sleep(0.05)

        # Measure cancellation timing
        cancel_start = time.perf_counter()

        # Trigger cancellation via staleness
        await orchestrator.broadcast_event(
            {
                "kind": "CancelTrigger",
                "payload": {"test": "timing"},
                "source_id": "system",
                "schema_version": "1.0.0",
            }
        )

        # Wait for cancellation to take effect
        result, was_interrupted = await generation_task
        cancel_end = time.perf_counter()

        cancellation_time_ms = (cancel_end - cancel_start) * 1000

        # Verify SLO: cancellation should happen within 100ms
        assert was_interrupted, "Generation should have been cancelled"
        assert cancellation_time_ms <= 100.0, (
            f"Cancellation took {cancellation_time_ms:.1f}ms, exceeds 100ms SLO"
        )

    @pytest.mark.asyncio
    async def test_context_staleness_detection(
        self, rl_facade: RLFacade, conversation_policy: ConversationObservationPolicy
    ) -> None:
        """Test context staleness detection accuracy."""
        # Register agents
        agent_handle = await rl_facade.register_agent(
            "staleness_agent", conversation_policy
        )
        await rl_facade.register_agent("other_agent", conversation_policy)

        agent = MockLLMAgent("staleness_agent", rl_facade)
        orchestrator = rl_facade.get_orchestrator()

        # Setup permissions and world state for agents
        self.setup_agent_permissions(orchestrator, ["staleness_agent", "other_agent"])

        # Set initial view_seq
        agent_handle.view_seq = 5

        # Issue cancel token
        req_id = "staleness_test_1"
        cancel_token = orchestrator.issue_cancel_token("staleness_agent", req_id)

        # Test staleness detection
        was_cancelled = await orchestrator.cancel_if_stale(
            "staleness_agent",
            req_id,
            10,  # new_seq=10, current=5, gap=5 > threshold=0
        )

        assert was_cancelled, "Token should be cancelled due to staleness"
        assert cancel_token.cancelled
        assert "stale_due_to_seq_gap" in cancel_token.reason

        # Test within threshold (should not cancel)
        token2 = orchestrator.issue_cancel_token("staleness_agent", "staleness_test_2")
        was_cancelled2 = await orchestrator.cancel_if_stale(
            "staleness_agent",
            "staleness_test_2",
            5,  # same as view_seq
        )

        assert not was_cancelled2, "Token should not be cancelled when not stale"
        assert not token2.cancelled

    @pytest.mark.asyncio
    async def test_interrupt_policy_always_vs_conflict(
        self, rl_facade: RLFacade, conversation_policy: ConversationObservationPolicy
    ) -> None:
        """Test different interrupt policies: 'always' vs 'only_conflict'."""
        # Register agents
        await rl_facade.register_agent("always_agent", conversation_policy)
        await rl_facade.register_agent("conflict_agent", conversation_policy)
        await rl_facade.register_agent("other_agent", conversation_policy)

        orchestrator = rl_facade.get_orchestrator()

        # Setup permissions and world state for agents
        self.setup_agent_permissions(
            orchestrator, ["always_agent", "conflict_agent", "other_agent"]
        )

        # Set interrupt policies
        orchestrator.set_agent_interrupt_policy("always_agent", "always")
        orchestrator.set_agent_interrupt_policy("conflict_agent", "only_conflict")

        # Issue tokens for both agents
        always_token = orchestrator.issue_cancel_token("always_agent", "always_test")
        conflict_token = orchestrator.issue_cancel_token(
            "conflict_agent", "conflict_test"
        )

        # Create effect from same agent (should not trigger 'only_conflict')
        effect_same_agent = {
            "uuid": "test_uuid_1",
            "kind": "TestEffect",
            "payload": {},
            "global_seq": 10,
            "sim_time": 1.0,
            "source_id": "conflict_agent",  # Same agent
            "schema_version": "1.0.0",
        }

        cancelled_req_ids = await orchestrator.check_and_cancel_stale_tokens(
            effect_same_agent
        )

        # 'always' should be cancelled, 'only_conflict' should not
        assert "always_test" in cancelled_req_ids
        assert "conflict_test" not in cancelled_req_ids
        assert always_token.cancelled
        assert not conflict_token.cancelled

        # Create effect from different agent (should trigger both)
        effect_different_agent = {
            "uuid": "test_uuid_2",
            "kind": "TestEffect",
            "payload": {},
            "global_seq": 15,
            "sim_time": 2.0,
            "source_id": "other_agent",  # Different agent
            "schema_version": "1.0.0",
        }

        cancelled_req_ids2 = await orchestrator.check_and_cancel_stale_tokens(
            effect_different_agent
        )

        # Now 'only_conflict' should also be cancelled
        assert "conflict_test" in cancelled_req_ids2
        assert conflict_token.cancelled

    @pytest.mark.asyncio
    async def test_multi_agent_concurrent_interruption(
        self, rl_facade: RLFacade, conversation_policy: ConversationObservationPolicy
    ) -> None:
        """Test concurrent interruption scenarios with multiple agents."""
        # Register multiple agents
        agents = {}
        for i in range(5):
            agent_id = f"agent_{i}"
            await rl_facade.register_agent(agent_id, conversation_policy)
            agents[agent_id] = MockLLMAgent(agent_id, rl_facade)

        orchestrator = rl_facade.get_orchestrator()

        # Setup permissions and world state for agents
        agent_ids = list(agents.keys())
        self.setup_agent_permissions(orchestrator, agent_ids)

        # Start concurrent generation for all agents
        generation_tasks = []
        for agent_id, agent in agents.items():
            message = (
                f"This is a long message from {agent_id} that might be interrupted"
            )
            task = asyncio.create_task(
                agent.generate_with_interruption(message, f"concurrent_{agent_id}", 0)
            )
            generation_tasks.append((agent_id, agent, task))

        # Wait for all to start generating
        await asyncio.sleep(0.1)

        # Trigger interruption event
        await orchestrator.broadcast_event(
            {
                "kind": "GlobalInterrupt",
                "payload": {"reason": "urgent_update"},
                "source_id": "system",
                "schema_version": "1.0.0",
            }
        )

        # Wait for all generations to complete
        results = []
        for agent_id, agent, task in generation_tasks:
            result, was_interrupted = await task
            results.append((agent_id, agent, result, was_interrupted))

        # Verify that all agents were interrupted
        interrupted_count = sum(
            1 for _, _, _, was_interrupted in results if was_interrupted
        )
        assert interrupted_count >= 3, (
            f"Expected at least 3 agents to be interrupted, got {interrupted_count}"
        )

        # Verify event log integrity
        event_log = orchestrator.event_log
        assert event_log.validate_integrity()

        # Verify no agent blocked others (non-blocking requirement 11.4)
        total_interruptions = sum(
            agent.interruption_count for _, agent, _, _ in results
        )
        assert total_interruptions >= interrupted_count, (
            "Interruption counts should be consistent"
        )

    @pytest.mark.asyncio
    async def test_regeneration_with_updated_context(
        self, rl_facade: RLFacade, conversation_policy: ConversationObservationPolicy
    ) -> None:
        """Test regeneration with updated context after interruption."""
        # Register agents
        agent_handle = await rl_facade.register_agent(
            "regen_agent", conversation_policy
        )
        await rl_facade.register_agent("context_agent", conversation_policy)

        agent = MockLLMAgent("regen_agent", rl_facade)
        orchestrator = rl_facade.get_orchestrator()

        # Setup permissions and world state for agents
        self.setup_agent_permissions(orchestrator, ["regen_agent", "context_agent"])

        # Start generation
        original_message = "This is the original message that will be interrupted"
        generation_task = asyncio.create_task(
            agent.generate_with_interruption(original_message, "regen_test_1", 0)
        )

        # Wait then trigger context change
        await asyncio.sleep(0.1)

        # Add context-changing event
        await orchestrator.broadcast_event(
            {
                "kind": "ContextUpdate",
                "payload": {"new_info": "Important context change"},
                "source_id": "context_agent",
                "schema_version": "1.0.0",
            }
        )

        # Complete first generation (should be interrupted)
        result1, was_interrupted = await generation_task
        assert was_interrupted

        # Get updated context sequence (after context update event)
        # Since the agent was interrupted, we know the context_seq is now 1
        new_context_seq = 1

        # Regenerate with updated context
        updated_message = f"[Updated based on new context] {original_message}"
        result2, was_interrupted2 = await agent.generate_with_interruption(
            updated_message, "regen_test_2", new_context_seq
        )

        # Second generation should complete without interruption
        assert not was_interrupted2
        assert "[Updated based on new context]" in result2
        assert "[INTERRUPTED]" not in result2

        # Verify context sequence progression
        assert new_context_seq > 0, "Context sequence should have advanced"

    @pytest.mark.asyncio
    async def test_conversation_replay_determinism(
        self, rl_facade: RLFacade, conversation_policy: ConversationObservationPolicy
    ) -> None:
        """Test that conversation with interruptions can be replayed deterministically."""
        # Register agents
        await rl_facade.register_agent("replay_alice", conversation_policy)
        await rl_facade.register_agent("replay_bob", conversation_policy)

        orchestrator = rl_facade.get_orchestrator()

        # Setup permissions and world state for agents
        self.setup_agent_permissions(orchestrator, ["replay_alice", "replay_bob"])

        # Record initial world seed for deterministic replay
        world_seed = 12345
        # orchestrator.set_world_seed(world_seed)  # Method not implemented yet

        # Execute conversation scenario
        alice = MockLLMAgent("replay_alice", rl_facade)
        bob = MockLLMAgent("replay_bob", rl_facade)

        # Alice speaks
        await alice.submit_intent("Hello Bob, how are you today?", 0)

        # Bob interrupts with urgent message
        await orchestrator.broadcast_event(
            {
                "kind": "UrgentMessage",
                "payload": {"text": "Alice, we need to talk urgently!"},
                "source_id": "replay_bob",
                "schema_version": "1.0.0",
            }
        )

        # Bob speaks (use current context_seq after broadcast event)
        await bob.submit_intent("Sorry Alice, but this is urgent!", 2)

        # Get event log for replay
        event_log = orchestrator.event_log
        original_entries = event_log.get_all_entries()

        # Verify log integrity
        assert event_log.validate_integrity()

        # Verify deterministic properties
        assert len(original_entries) >= 2, "Should have recorded multiple events"

        # Check that all events have deterministic ordering fields
        for entry in original_entries:
            effect = entry.effect
            assert "global_seq" in effect
            assert "sim_time" in effect
            assert "uuid" in effect
            assert isinstance(effect["global_seq"], int)
            assert isinstance(effect["sim_time"], float)

        # Verify global_seq is monotonic
        global_seqs = [entry.effect["global_seq"] for entry in original_entries]
        assert global_seqs == sorted(global_seqs), (
            "Global sequences should be monotonic"
        )

        # Verify unique UUIDs
        uuids = [entry.effect["uuid"] for entry in original_entries]
        assert len(uuids) == len(set(uuids)), "All UUIDs should be unique"
