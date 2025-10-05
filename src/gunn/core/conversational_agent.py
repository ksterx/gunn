"""Conversational agent implementation for multi-agent simulation.

This module provides the ConversationalAgent class that extends AsyncAgentLogic
to create agents capable of natural conversation, context building, and LLM-driven
decision making in multi-agent environments.
"""

import asyncio
import uuid
from typing import Any, Protocol

from gunn.core.agent_logic import AsyncAgentLogic
from gunn.schemas.messages import View
from gunn.schemas.types import Intent


class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    async def generate_response(
        self,
        context: str,
        personality: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
    ) -> "LLMResponse":
        """Generate a response based on context and personality.

        Args:
            context: Context string for the LLM
            personality: Personality description for the agent
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation

        Returns:
            LLMResponse with action type and content
        """
        ...


class LLMResponse:
    """Response from LLM client containing action and content."""

    def __init__(
        self,
        action_type: str,
        text: str | None = None,
        target_position: list[float] | None = None,
        target_agent: str | None = None,
        reasoning: str | None = None,
    ):
        """Initialize LLM response.

        Args:
            action_type: Type of action ("speak", "move", "interact", "wait")
            text: Text content for speak actions
            target_position: Target position for move actions
            target_agent: Target agent for interact actions
            reasoning: Reasoning behind the action
        """
        self.action_type = action_type
        self.text = text
        self.target_position = target_position
        self.target_agent = target_agent
        self.reasoning = reasoning


class MockLLMClient:
    """Mock LLM client for testing and demonstration purposes."""

    def __init__(self, response_delay: float = 0.1):
        """Initialize mock LLM client.

        Args:
            response_delay: Simulated response delay in seconds
        """
        self.response_delay = response_delay
        self._response_count = 0

    async def generate_response(
        self,
        context: str,
        personality: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a mock response with simulated delay."""
        await asyncio.sleep(self.response_delay)
        self._response_count += 1

        # Simple mock logic based on context content
        context_lower = context.lower()

        if "hello" in context_lower and self._response_count % 3 == 1:
            return LLMResponse(
                action_type="speak",
                text=f"Hello there! Nice to meet you. ({personality} speaking)",
                reasoning="Responding to greeting",
            )
        elif "move" in context_lower and self._response_count % 4 == 2:
            import random

            return LLMResponse(
                action_type="move",
                target_position=[
                    random.uniform(-20, 20),
                    random.uniform(-20, 20),
                    0.0,
                ],
                reasoning="Moving to explore",
            )
        elif self._response_count % 5 == 0:
            return LLMResponse(
                action_type="speak",
                text=f"I'm thinking... ({self._response_count} thoughts so far)",
                reasoning="Sharing thoughts",
            )
        else:
            return LLMResponse(action_type="wait", reasoning="Observing and waiting")


class ConversationMemory:
    """Memory system for tracking conversation history and context."""

    def __init__(self, max_messages: int = 20, max_age_seconds: float = 300.0):
        """Initialize conversation memory.

        Args:
            max_messages: Maximum number of messages to remember
            max_age_seconds: Maximum age of messages to keep in seconds
        """
        self.max_messages = max_messages
        self.max_age_seconds = max_age_seconds
        self.messages: list[dict[str, Any]] = []
        self.agent_positions: dict[str, tuple[float, float, float]] = {}
        self.last_interaction_time: dict[str, float] = {}

    def add_message(
        self,
        speaker_id: str,
        text: str,
        timestamp: float,
        speaker_position: tuple[float, float, float] | None = None,
    ) -> None:
        """Add a message to conversation memory.

        Args:
            speaker_id: ID of the speaking agent
            text: Message text
            timestamp: Message timestamp
            speaker_position: Position of speaker when message was sent
        """
        message = {
            "speaker_id": speaker_id,
            "text": text,
            "timestamp": timestamp,
            "speaker_position": speaker_position,
        }

        self.messages.append(message)
        self.last_interaction_time[speaker_id] = timestamp

        # Clean up old messages
        self._cleanup_old_messages(timestamp)

    def update_agent_position(
        self, agent_id: str, position: tuple[float, float, float]
    ) -> None:
        """Update an agent's position in memory.

        Args:
            agent_id: Agent identifier
            position: New position as (x, y, z) tuple
        """
        self.agent_positions[agent_id] = position

    def get_recent_messages(
        self, since_timestamp: float | None = None, max_count: int | None = None
    ) -> list[dict[str, Any]]:
        """Get recent messages from memory.

        Args:
            since_timestamp: Only return messages after this timestamp
            max_count: Maximum number of messages to return

        Returns:
            List of message dictionaries
        """
        messages = self.messages

        if since_timestamp is not None:
            messages = [m for m in messages if m["timestamp"] > since_timestamp]

        if max_count is not None:
            messages = messages[-max_count:]

        return messages

    def get_nearby_agents(
        self,
        agent_position: tuple[float, float, float],
        max_distance: float = 10.0,
    ) -> list[tuple[str, tuple[float, float, float], float]]:
        """Get agents within a certain distance.

        Args:
            agent_position: Position to search from
            max_distance: Maximum distance to consider

        Returns:
            List of (agent_id, position, distance) tuples
        """
        nearby = []

        for agent_id, position in self.agent_positions.items():
            distance = (
                sum(
                    (a - b) ** 2 for a, b in zip(agent_position, position, strict=False)
                )
                ** 0.5
            )

            if distance <= max_distance:
                nearby.append((agent_id, position, distance))

        # Sort by distance
        nearby.sort(key=lambda x: x[2])
        return nearby

    def _cleanup_old_messages(self, current_timestamp: float) -> None:
        """Remove old messages from memory.

        Args:
            current_timestamp: Current timestamp for age calculation
        """
        cutoff_time = current_timestamp - self.max_age_seconds

        # Remove old messages
        self.messages = [m for m in self.messages if m["timestamp"] > cutoff_time]

        # Keep only the most recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]


class ConversationalAgent(AsyncAgentLogic):
    """Conversational agent implementation with LLM integration.

    This agent can engage in natural conversations, build context from observations,
    and make decisions using an LLM. It maintains conversation memory and can
    coordinate with other agents through observed actions and communication.

    Requirements addressed:
    - 14.6: Observes message content and speaker identity from other agents
    - 14.7: Observes nearby agents and can engage in conversation
    - 14.8: Can choose to wait when no new information is received
    - 14.9: Coordinates through observed actions and communication
    """

    def __init__(
        self,
        llm_client: LLMClient,
        personality: str = "helpful and friendly",
        name: str | None = None,
        conversation_distance: float = 15.0,
        response_probability: float = 0.7,
        movement_probability: float = 0.2,
        max_context_length: int = 1000,
    ):
        """Initialize conversational agent.

        Args:
            llm_client: LLM client for generating responses
            personality: Personality description for the agent
            name: Optional name for the agent (defaults to agent_id)
            conversation_distance: Maximum distance to engage in conversation
            response_probability: Probability of responding to messages
            movement_probability: Probability of moving when idle
            max_context_length: Maximum length of context string for LLM
        """
        self.llm_client = llm_client
        self.personality = personality
        self.name = name
        self.conversation_distance = conversation_distance
        self.response_probability = response_probability
        self.movement_probability = movement_probability
        self.max_context_length = max_context_length

        # Memory and state tracking
        self.memory = ConversationMemory()
        self.last_observation_time = 0.0
        self.last_action_time = 0.0
        self.current_position: tuple[float, float, float] | None = None
        self.observations_processed = 0
        self.messages_sent = 0
        self.moves_made = 0

        # Context building state
        self._last_context_hash = ""
        self._context_change_threshold = 0.1  # Minimum change to trigger action

    async def process_observation(
        self, observation: View, agent_id: str
    ) -> Intent | None:
        """Process observation and generate intent based on LLM decision.

        This method extracts nearby agents and recent messages from the observation,
        builds context for the LLM, and generates appropriate intents based on the
        LLM's response.

        Args:
            observation: Current view of the world state
            agent_id: ID of this agent

        Returns:
            Intent to execute, or None if no action should be taken
        """
        import time

        current_time = time.time()
        self.observations_processed += 1
        self.last_observation_time = current_time

        # Use agent name if not set
        if self.name is None:
            self.name = agent_id

        # Extract information from observation
        nearby_agents = self._extract_nearby_agents(observation, agent_id)
        recent_messages = self._extract_recent_messages(observation, current_time)
        self._update_agent_positions(observation)

        # Update our own position
        if agent_id in observation.visible_entities:
            entity_data = observation.visible_entities[agent_id]
            if "position" in entity_data:
                pos = entity_data["position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    self.current_position = (
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2]) if len(pos) > 2 else 0.0,
                    )
                    self.memory.update_agent_position(agent_id, self.current_position)

        # Build context for LLM
        context = self._build_context(nearby_agents, recent_messages, agent_id)

        # Check if context has changed significantly
        context_hash = str(hash(context))
        context_changed = context_hash != self._last_context_hash
        self._last_context_hash = context_hash

        # Decide whether to take action based on context changes and probabilities
        should_act = self._should_take_action(
            context_changed, recent_messages, nearby_agents, current_time
        )

        if not should_act:
            return None  # Wait and observe

        try:
            # Generate response using LLM
            response = await self.llm_client.generate_response(
                context, self.personality
            )

            # Convert LLM response to intent
            intent = self._create_intent_from_response(
                response, observation.view_seq, agent_id
            )

            if intent:
                self.last_action_time = current_time
                if intent["kind"] == "Speak":
                    self.messages_sent += 1
                    # Add our own message to memory
                    if self.current_position:
                        self.memory.add_message(
                            agent_id,
                            intent["payload"]["text"],
                            current_time,
                            self.current_position,
                        )
                elif intent["kind"] == "Move":
                    self.moves_made += 1

            return intent

        except Exception as e:
            # Log error but don't crash the agent
            print(f"Error generating response for {agent_id}: {e}")
            return None

    def _extract_nearby_agents(
        self, observation: View, agent_id: str
    ) -> list[dict[str, Any]]:
        """Extract information about nearby agents from observation.

        Args:
            observation: Current observation
            agent_id: This agent's ID

        Returns:
            List of nearby agent information dictionaries
        """
        nearby_agents = []

        for entity_id, entity_data in observation.visible_entities.items():
            # Skip self and non-agent entities
            if entity_id == agent_id or not entity_id.startswith("agent_"):
                continue

            # Extract position if available
            position = None
            if "position" in entity_data:
                pos = entity_data["position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    position = (
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2]) if len(pos) > 2 else 0.0,
                    )

            # Calculate distance if both positions are known
            distance = None
            if position and self.current_position:
                distance = (
                    sum(
                        (a - b) ** 2
                        for a, b in zip(self.current_position, position, strict=False)
                    )
                    ** 0.5
                )

            agent_info = {
                "agent_id": entity_id,
                "name": entity_data.get("name", entity_id),
                "position": position,
                "distance": distance,
                "data": entity_data,
            }

            # Only include agents within conversation distance
            if distance is None or distance <= self.conversation_distance:
                nearby_agents.append(agent_info)

        # Sort by distance (closest first)
        nearby_agents.sort(key=lambda x: x["distance"] or float("inf"))
        return nearby_agents

    def _extract_recent_messages(
        self, observation: View, current_time: float
    ) -> list[dict[str, Any]]:
        """Extract recent messages from observation.

        Args:
            observation: Current observation
            current_time: Current timestamp

        Returns:
            List of recent message dictionaries
        """
        messages = []

        # Look for message events in the observation
        # This is a simplified implementation - in a real system,
        # messages would be tracked through the event system
        for entity_id, entity_data in observation.visible_entities.items():
            if "recent_message" in entity_data:
                message_data = entity_data["recent_message"]
                if isinstance(message_data, dict):
                    message = {
                        "speaker_id": entity_id,
                        "text": message_data.get("text", ""),
                        "timestamp": message_data.get("timestamp", current_time),
                        "speaker_position": entity_data.get("position"),
                    }
                    messages.append(message)

                    # Add to memory
                    speaker_pos = None
                    if "position" in entity_data:
                        pos = entity_data["position"]
                        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                            speaker_pos = (
                                float(pos[0]),
                                float(pos[1]),
                                float(pos[2]) if len(pos) > 2 else 0.0,
                            )

                    self.memory.add_message(
                        entity_id, message["text"], message["timestamp"], speaker_pos
                    )

        # Also get messages from memory
        memory_messages = self.memory.get_recent_messages(
            since_timestamp=current_time - 30.0,  # Last 30 seconds
            max_count=10,
        )
        messages.extend(memory_messages)

        # Remove duplicates and sort by timestamp
        seen = set()
        unique_messages = []
        for msg in messages:
            key = (msg["speaker_id"], msg["text"], msg["timestamp"])
            if key not in seen:
                seen.add(key)
                unique_messages.append(msg)

        unique_messages.sort(key=lambda x: x["timestamp"])
        return unique_messages[-10:]  # Keep only the 10 most recent

    def _update_agent_positions(self, observation: View) -> None:
        """Update agent positions in memory from observation.

        Args:
            observation: Current observation
        """
        for entity_id, entity_data in observation.visible_entities.items():
            if entity_id.startswith("agent_") and "position" in entity_data:
                pos = entity_data["position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    position = (
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2]) if len(pos) > 2 else 0.0,
                    )
                    self.memory.update_agent_position(entity_id, position)

    def _build_context(
        self,
        nearby_agents: list[dict[str, Any]],
        recent_messages: list[dict[str, Any]],
        agent_id: str,
    ) -> str:
        """Build context string for LLM input.

        Args:
            nearby_agents: List of nearby agent information
            recent_messages: List of recent messages
            agent_id: This agent's ID

        Returns:
            Context string for LLM
        """
        context_parts = []

        # Agent identity and personality
        context_parts.append(f"You are {self.name} (ID: {agent_id}).")
        context_parts.append(f"Your personality: {self.personality}")

        # Current situation
        if self.current_position:
            x, y, z = self.current_position
            context_parts.append(
                f"You are currently at position ({x:.1f}, {y:.1f}, {z:.1f})."
            )

        # Nearby agents
        if nearby_agents:
            context_parts.append(f"\nNearby agents ({len(nearby_agents)}):")
            for agent in nearby_agents[:5]:  # Limit to 5 closest agents
                name = agent["name"]
                distance = agent["distance"]
                if distance is not None:
                    context_parts.append(f"- {name} at distance {distance:.1f}")
                else:
                    context_parts.append(f"- {name} (distance unknown)")
        else:
            context_parts.append("\nNo other agents are nearby.")

        # Recent conversation
        if recent_messages:
            context_parts.append(
                f"\nRecent conversation ({len(recent_messages)} messages):"
            )
            for msg in recent_messages[-5:]:  # Last 5 messages
                speaker = msg["speaker_id"]
                text = msg["text"]
                context_parts.append(f"- {speaker}: {text}")
        else:
            context_parts.append("\nNo recent conversation.")

        # Action guidance
        context_parts.append(
            "\nYou can: speak (to communicate), move (to change position), "
            "interact (with nearby agents), or wait (to observe)."
        )
        context_parts.append(
            "Choose an action based on the situation. Be natural and engaging."
        )

        # Join and truncate if necessary
        context = "\n".join(context_parts)
        if len(context) > self.max_context_length:
            context = context[: self.max_context_length] + "..."

        return context

    def _should_take_action(
        self,
        context_changed: bool,
        recent_messages: list[dict[str, Any]],
        nearby_agents: list[dict[str, Any]],
        current_time: float,
    ) -> bool:
        """Determine whether the agent should take action.

        Args:
            context_changed: Whether the context has changed significantly
            recent_messages: List of recent messages
            nearby_agents: List of nearby agents
            current_time: Current timestamp

        Returns:
            True if agent should take action, False to wait
        """
        import random

        # Always respond to direct messages (messages mentioning this agent)
        for msg in recent_messages[-3:]:  # Check last 3 messages
            if self.name and self.name.lower() in msg["text"].lower():
                return True

        # Respond to new messages with some probability
        if recent_messages and context_changed:
            if random.random() < self.response_probability:
                return True

        # Take action if context changed significantly
        if context_changed and nearby_agents:
            if random.random() < 0.5:  # 50% chance when context changes
                return True

        # Occasionally take action even without changes (to avoid being too passive)
        time_since_last_action = current_time - self.last_action_time
        if time_since_last_action > 10.0:  # Been idle for 10+ seconds
            if random.random() < 0.3:  # 30% chance to act
                return True

        # Move occasionally when idle
        if not recent_messages and not nearby_agents:
            if random.random() < self.movement_probability:
                return True

        return False  # Wait and observe

    def _create_intent_from_response(
        self, response: LLMResponse, context_seq: int, agent_id: str
    ) -> Intent | None:
        """Create an intent from LLM response.

        Args:
            response: LLM response object
            context_seq: Current context sequence number
            agent_id: This agent's ID

        Returns:
            Intent object or None if no valid intent
        """
        if response.action_type == "speak" and response.text:
            return {
                "kind": "Speak",
                "payload": {"text": response.text, "agent_id": agent_id},
                "context_seq": context_seq,
                "req_id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }

        elif response.action_type == "move" and response.target_position:
            return {
                "kind": "Move",
                "payload": {
                    "from": list(self.current_position)
                    if self.current_position
                    else [0.0, 0.0, 0.0],
                    "to": response.target_position,
                    "agent_id": agent_id,
                },
                "context_seq": context_seq,
                "req_id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }

        elif response.action_type == "interact" and response.target_agent:
            return {
                "kind": "Interact",
                "payload": {
                    "target_id": response.target_agent,
                    "interaction_type": "talk",
                    "agent_id": agent_id,
                },
                "context_seq": context_seq,
                "req_id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }

        # For "wait" or invalid responses, return None
        return None

    async def on_loop_start(self, agent_id: str) -> None:
        """Called when the agent's async loop starts."""
        print(f"🗣️  {self.name or agent_id} started conversational loop")
        print(f"   Personality: {self.personality}")

    async def on_loop_stop(self, agent_id: str) -> None:
        """Called when the agent's async loop stops."""
        print(f"🔇 {self.name or agent_id} stopped conversational loop")
        print(
            f"   Stats: {self.messages_sent} messages, {self.moves_made} moves, {self.observations_processed} observations"
        )

    async def on_error(self, agent_id: str, error: Exception) -> bool:
        """Handle errors during processing."""
        print(f"❌ {self.name or agent_id} encountered error: {error}")
        return True  # Continue on error

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dictionary with agent statistics
        """
        return {
            "name": self.name,
            "personality": self.personality,
            "observations_processed": self.observations_processed,
            "messages_sent": self.messages_sent,
            "moves_made": self.moves_made,
            "current_position": self.current_position,
            "memory_messages": len(self.memory.messages),
            "known_agents": len(self.memory.agent_positions),
        }
