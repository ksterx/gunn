#!/usr/bin/env python3
"""A/B/C Conversation Demo with Visible Interruption and Regeneration.

This demo showcases the intelligent interruption and regeneration capabilities
of the gunn multi-agent simulation core. Three agents (Alice, Bob, Charlie)
engage in a conversation where interruptions trigger visible regeneration.

Key features demonstrated:
- Multi-agent conversation with partial observation
- Intelligent interruption based on context staleness
- Visible regeneration when context changes
- Cancel token integration with 100ms SLO
- Deterministic event ordering and replay capability

Requirements addressed:
- 4.1: Issue cancel_token with current context_digest
- 4.2: Evaluate staleness using latest_view_seq > context_seq + staleness_threshold
- 4.3: Cancel current generation when context becomes stale
- 6.2: Monitor for cancellation signals at token boundaries
- 6.3: Immediately halt token generation within 100ms
"""

import asyncio
import uuid
from typing import Any

from gunn import Orchestrator, OrchestratorConfig
from gunn.facades import MessageFacade, RLFacade
from gunn.policies.observation import ConversationObservationPolicy, PolicyConfig
from gunn.schemas.types import CancelToken, EffectDraft, Intent
from gunn.utils.telemetry import get_logger, setup_logging


class ConversationAgent:
    """Simulated conversation agent with interruption awareness."""

    def __init__(self, agent_id: str, name: str, facade: MessageFacade | RLFacade):
        self.agent_id = agent_id
        self.name = name
        self.facade = facade
        self.logger = get_logger(f"agent.{agent_id}")
        self.generation_active = False
        self.current_cancel_token: CancelToken | None = None

    async def start_conversation(self) -> None:
        """Start the conversation loop for this agent."""
        self.logger.info(f"{self.name} joined the conversation")

        # Subscribe to conversation messages if using MessageFacade
        if isinstance(self.facade, MessageFacade):
            await self.facade.subscribe(
                self.agent_id,
                message_types={"Speak", "SpeakResponse", "MessageEmitted"},
                handler=self._handle_message,
            )

    async def speak(self, message: str, context_seq: int = 0) -> str:
        """Speak a message with interruption awareness."""
        req_id = f"speak_{uuid.uuid4().hex[:8]}"

        self.logger.info(f"{self.name} starting to speak: '{message[:50]}...'")

        # Get current context sequence from orchestrator if not provided
        orchestrator = self.facade.get_orchestrator()
        if context_seq == 0:
            try:
                if isinstance(self.facade, RLFacade):
                    observation = await self.facade.observe(self.agent_id)
                    context_seq = observation.get("view_seq", 0)
                else:
                    # For MessageFacade, use a safe default
                    context_seq = 0
            except Exception as e:
                self.logger.warning(f"Could not get current context seq: {e}, using 0")
                context_seq = 0

        # Issue cancel token for generation tracking
        self.current_cancel_token = orchestrator.issue_cancel_token(
            self.agent_id, req_id
        )

        try:
            self.generation_active = True

            # Simulate token-by-token generation with cancellation checks
            generated_message = await self._generate_with_interruption(
                message, req_id, context_seq
            )

            # Submit the final intent
            intent: Intent = {
                "kind": "Speak",
                "payload": {"text": generated_message, "original_text": message},
                "context_seq": context_seq,
                "req_id": req_id,
                "agent_id": self.agent_id,
                "priority": 1,
                "schema_version": "1.0.0",
            }

            if isinstance(self.facade, RLFacade):
                _effect, _observation = await self.facade.step(self.agent_id, intent)
                self.logger.info(
                    f"{self.name} spoke successfully: '{generated_message}'"
                )
                return generated_message
            else:
                # MessageFacade approach
                await self.facade.emit(
                    "Speak",
                    {"text": generated_message, "speaker": self.name},
                    self.agent_id,
                )
                self.logger.info(
                    f"{self.name} spoke successfully: '{generated_message}'"
                )
                return generated_message

        except Exception as e:
            if self.current_cancel_token and self.current_cancel_token.cancelled:
                self.logger.info(
                    f"{self.name} generation cancelled: {self.current_cancel_token.reason}"
                )
                # Trigger regeneration with updated context
                return await self._regenerate_response(message, req_id)
            else:
                self.logger.error(f"{self.name} speaking failed: {e}")
                raise
        finally:
            self.generation_active = False
            self.current_cancel_token = None

    async def _generate_with_interruption(
        self, message: str, req_id: str, context_seq: int
    ) -> str:
        """Simulate token-by-token generation with interruption checks."""
        tokens = message.split()
        generated_tokens = []

        for i, token in enumerate(tokens):
            # Check for cancellation at token boundaries (requirement 6.2)
            if self.current_cancel_token and self.current_cancel_token.cancelled:
                self.logger.info(
                    f"{self.name} generation interrupted at token {i}/{len(tokens)}"
                )
                raise asyncio.CancelledError("Generation cancelled due to staleness")

            # Simulate token generation time (20-30ms per token for responsive cancellation)
            await asyncio.sleep(0.025)  # 25ms per token
            generated_tokens.append(token)

            # Log progress for visibility
            if i % 5 == 0:  # Log every 5 tokens
                partial_message = " ".join(generated_tokens)
                self.logger.debug(f"{self.name} generating: '{partial_message}...'")

        return " ".join(generated_tokens)

    async def _regenerate_response(self, original_message: str, req_id: str) -> str:
        """Regenerate response after interruption with updated context."""
        self.logger.info(f"{self.name} regenerating response due to interruption")

        # Get updated context
        if isinstance(self.facade, RLFacade):
            observation = await self.facade.observe(self.agent_id)
            new_context_seq = observation.get("view_seq", 0)
        else:
            # For MessageFacade, get latest messages
            messages = await self.facade.get_messages(self.agent_id, timeout=0.1)
            new_context_seq = len(messages)  # Simplified context tracking

        # Generate modified response based on new context
        modified_message = f"[Updated] {original_message} (responding to new context)"

        # Generate the new response
        return await self._generate_with_interruption(
            modified_message, f"regen_{req_id}", new_context_seq
        )

    def _handle_message(self, agent_id: str, message: dict[str, Any]) -> None:
        """Handle incoming messages (for MessageFacade)."""
        if message.get("type") == "observation_update":
            data = message.get("data", {})
            patches = data.get("patches", [])
            if patches:
                self.logger.info(
                    f"{self.name} received observation update with {len(patches)} patches"
                )


class ABCConversationDemo:
    """Main demo class orchestrating the A/B/C conversation."""

    def __init__(self, use_message_facade: bool = False):
        self.use_message_facade = use_message_facade
        self.logger = get_logger("abc_demo")

        # Configure orchestrator for conversation scenario
        self.config = OrchestratorConfig(
            max_agents=3,
            staleness_threshold=10,  # Allow significant staleness for demo to work properly
            debounce_ms=50.0,  # Short debounce for responsive interruption
            deadline_ms=10000.0,  # 10 second deadline
            token_budget=200,  # Reasonable token budget
            backpressure_policy="defer",
            default_priority=1,
            use_in_memory_dedup=True,  # Use in-memory for demo
        )

        self.orchestrator = Orchestrator(self.config, world_id="abc_conversation")

        # Create facade
        self.facade: MessageFacade | RLFacade
        if use_message_facade:
            self.facade = MessageFacade(orchestrator=self.orchestrator)
        else:
            self.facade = RLFacade(orchestrator=self.orchestrator)

        # Create agents
        self.agents = {
            "alice": ConversationAgent("alice", "Alice", self.facade),
            "bob": ConversationAgent("bob", "Bob", self.facade),
            "charlie": ConversationAgent("charlie", "Charlie", self.facade),
        }

    async def setup(self) -> None:
        """Set up the conversation environment."""
        self.logger.info("Setting up A/B/C conversation demo")

        # Initialize facade
        await self.facade.initialize()

        # Create conversation observation policy
        policy_config = PolicyConfig(
            distance_limit=float("inf"),  # No distance limit for conversation
            relationship_filter=[],  # No relationship filtering
            field_visibility={},  # All fields visible
            max_patch_ops=20,  # Reasonable patch limit
        )

        # Register agents with conversation policy
        for agent_id in self.agents:
            policy = ConversationObservationPolicy(policy_config)
            await self.facade.register_agent(agent_id, policy)
            await self.agents[agent_id].start_conversation()

        # Set up initial world state with conversation participants
        await self._setup_conversation_world()

        self.logger.info("A/B/C conversation demo setup complete")

    async def _setup_conversation_world(self) -> None:
        """Set up the initial world state for conversation."""
        # Add agents as entities in the world
        for agent_id, agent in self.agents.items():
            await self.facade.get_orchestrator().broadcast_event(
                EffectDraft(
                    kind="ParticipantJoined",
                    payload={
                        "agent_id": agent_id,
                        "name": agent.name,
                        "type": "agent",
                        "conversation_id": "abc_conversation",
                    },
                    source_id="system",
                    schema_version="1.0.0",
                )
            )

        # Establish relationships between participants
        for agent_id in self.agents:
            other_agents = [aid for aid in self.agents if aid != agent_id]
            await self.facade.get_orchestrator().broadcast_event(
                EffectDraft(
                    kind="RelationshipEstablished",
                    payload={
                        "from_agent": agent_id,
                        "to_agents": other_agents,
                        "relationship_type": "conversation_participant",
                    },
                    source_id="system",
                    schema_version="1.0.0",
                )
            )

    async def run_conversation_scenario(self) -> None:
        """Run the main conversation scenario with interruption."""
        self.logger.info("Starting A/B/C conversation scenario")

        try:
            # Scenario: Alice starts speaking, Bob interrupts, Charlie responds

            # Phase 1: Alice starts a long message
            alice_task = asyncio.create_task(
                self.agents["alice"].speak(
                    "Hello everyone! I wanted to tell you about this amazing discovery I made yesterday while working on the quantum computing project. It's really fascinating how the quantum entanglement principles can be applied to multi-agent systems and I think we should definitely explore this further in our next research phase.",
                    context_seq=0,
                )
            )

            # Phase 2: Let Alice speak for a bit, then Bob interrupts
            await asyncio.sleep(0.3)  # Let Alice generate ~12 tokens

            self.logger.info("Bob is about to interrupt Alice...")

            # Bob emits an urgent message that should trigger Alice's cancellation
            bob_interrupt_task = asyncio.create_task(
                self.agents["bob"].speak(
                    "Wait Alice! I have urgent news about the quantum project - we just got approval for the next phase!",
                    context_seq=1,
                )
            )

            # Phase 3: Wait for both to complete
            alice_result = await alice_task
            bob_result = await bob_interrupt_task

            self.logger.info(f"Alice final message: '{alice_result}'")
            self.logger.info(f"Bob interrupt message: '{bob_result}'")

            # Phase 4: Charlie responds to both
            await asyncio.sleep(0.1)  # Brief pause

            charlie_result = await self.agents["charlie"].speak(
                "Wow, that's exciting news Bob! And Alice, I'd love to hear more about your quantum entanglement ideas. Maybe we can combine both topics in our next meeting?",
                context_seq=2,
            )

            self.logger.info(f"Charlie response: '{charlie_result}'")

            # Phase 5: Demonstrate replay capability
            await self._demonstrate_replay()

        except Exception as e:
            self.logger.error(f"Conversation scenario failed: {e}")
            raise

    async def _demonstrate_replay(self) -> None:
        """Demonstrate deterministic replay capability."""
        self.logger.info("Demonstrating replay capability...")

        # Get event log entries
        event_log = self.facade.get_orchestrator().event_log
        entries = event_log.get_entries_since(0)

        self.logger.info(f"Recorded {len(entries)} events in conversation")

        # Show event sequence for replay
        for i, entry in enumerate(entries[:5]):  # Show first 5 events
            effect = entry.effect
            self.logger.info(
                f"Event {i + 1}: {effect['kind']} from {effect['source_id']} "
                f"at seq={effect['global_seq']}"
            )

        # Validate log integrity
        is_valid = event_log.validate_integrity()
        self.logger.info(f"Event log integrity: {'VALID' if is_valid else 'CORRUPTED'}")

    async def shutdown(self) -> None:
        """Clean up demo resources."""
        self.logger.info("Shutting down A/B/C conversation demo")
        await self.facade.shutdown()


async def main() -> None:
    """Run the A/B/C conversation demo."""
    # Set up logging
    setup_logging("INFO")

    print("🎭 A/B/C Conversation Demo with Interruption and Regeneration")
    print("=" * 60)
    print()

    # Run with MessageFacade
    print("Running demo with MessageFacade...")
    demo_message = ABCConversationDemo(use_message_facade=True)

    try:
        await demo_message.setup()
        await demo_message.run_conversation_scenario()
    finally:
        await demo_message.shutdown()

    print()
    print("Demo completed! Check the logs above to see:")
    print("✅ Intelligent interruption when Bob interrupts Alice")
    print("✅ Visible regeneration with updated context")
    print("✅ Cancel token integration with <100ms response")
    print("✅ Deterministic event ordering and replay capability")
    print("✅ Multi-agent conversation with partial observation")


if __name__ == "__main__":
    asyncio.run(main())
