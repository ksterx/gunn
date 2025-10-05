"""Collaborative agent implementation with enhanced coordination capabilities.

This module extends the ConversationalAgent to include collaborative behavior
detection and coordination patterns for multi-agent task coordination.
"""

import time
from typing import Any

from gunn.core.collaborative_behavior import (
    CollaborativeBehaviorManager,
    create_collaborative_intent,
)
from gunn.core.conversational_agent import ConversationalAgent, LLMClient
from gunn.schemas.messages import View
from gunn.schemas.types import Intent


class CollaborativeAgent(ConversationalAgent):
    """Enhanced conversational agent with collaborative behavior capabilities.

    This agent extends ConversationalAgent to detect collaboration opportunities
    and coordinate with other agents through observed actions and communication.

    Requirements addressed:
    - 3.6: Immediately re-observes world state for changes from other agents
    - 4.6: Observes and coordinates when collaborative opportunities arise
    - 14.9: Coordinates through observed actions and communication
    """

    def __init__(
        self,
        llm_client: LLMClient,
        personality: str = "helpful and collaborative",
        name: str | None = None,
        conversation_distance: float = 15.0,
        response_probability: float = 0.7,
        movement_probability: float = 0.2,
        max_context_length: int = 1000,
        collaboration_threshold: float = 0.6,
        enable_following: bool = True,
        enable_helping: bool = True,
        enable_task_coordination: bool = True,
    ):
        """Initialize collaborative agent.

        Args:
            llm_client: LLM client for generating responses
            personality: Personality description for the agent
            name: Optional name for the agent (defaults to agent_id)
            conversation_distance: Maximum distance to engage in conversation
            response_probability: Probability of responding to messages
            movement_probability: Probability of moving when idle
            max_context_length: Maximum length of context string for LLM
            collaboration_threshold: Minimum confidence to act on collaboration opportunities
            enable_following: Whether to enable following behavior
            enable_helping: Whether to enable helping behavior
            enable_task_coordination: Whether to enable task coordination
        """
        super().__init__(
            llm_client=llm_client,
            personality=personality,
            name=name,
            conversation_distance=conversation_distance,
            response_probability=response_probability,
            movement_probability=movement_probability,
            max_context_length=max_context_length,
        )

        # Collaborative behavior settings
        self.collaboration_threshold = collaboration_threshold
        self.enable_following = enable_following
        self.enable_helping = enable_helping
        self.enable_task_coordination = enable_task_coordination

        # Collaboration manager and memory
        self.collaboration_manager = CollaborativeBehaviorManager()
        self.collaboration_memory: dict[str, Any] = {}

        # Statistics
        self.collaborative_actions_taken = 0
        self.opportunities_detected = 0
        self.coordination_patterns_participated = 0

    async def process_observation(
        self, observation: View, agent_id: str
    ) -> Intent | None:
        """Process observation with collaborative behavior detection.

        This method extends the base conversational agent to detect collaboration
        opportunities and coordinate with other agents when appropriate.

        Args:
            observation: Current view of the world state
            agent_id: ID of this agent

        Returns:
            Intent to execute, prioritizing collaborative actions when detected
        """
        current_time = time.time()
        self.observations_processed += 1
        self.last_observation_time = current_time

        # Use agent name if not set
        if self.name is None:
            self.name = agent_id

        # Update collaboration patterns
        self.collaboration_manager.update_coordination_patterns(observation, agent_id)

        # Detect collaboration opportunities
        opportunities = self.collaboration_manager.detect_collaboration_opportunities(
            observation, agent_id, self.collaboration_memory
        )

        # Filter opportunities based on settings and threshold
        filtered_opportunities = self._filter_opportunities(opportunities)

        if filtered_opportunities:
            self.opportunities_detected += len(filtered_opportunities)

            # Prioritize collaboration opportunities
            best_opportunity = filtered_opportunities[0]  # Already sorted by confidence

            if best_opportunity.confidence >= self.collaboration_threshold:
                # Create collaborative intent
                collaborative_intent = create_collaborative_intent(
                    best_opportunity, agent_id, observation.view_seq
                )

                if collaborative_intent:
                    self.collaborative_actions_taken += 1
                    self.last_action_time = current_time

                    # Track the collaborative action
                    self._track_collaborative_action(
                        best_opportunity, collaborative_intent
                    )

                    return collaborative_intent

        # Fall back to normal conversational behavior if no collaboration opportunities
        return await super().process_observation(observation, agent_id)

    def _filter_opportunities(self, opportunities: list) -> list:
        """Filter collaboration opportunities based on agent settings.

        Args:
            opportunities: List of detected collaboration opportunities

        Returns:
            Filtered list of opportunities based on agent configuration
        """
        filtered = []

        for opportunity in opportunities:
            # Check if this type of collaboration is enabled
            if (
                opportunity.collaboration_type.value == "following"
                and not self.enable_following
            ):
                continue
            elif (
                opportunity.collaboration_type.value == "helping"
                and not self.enable_helping
            ):
                continue
            elif (
                opportunity.collaboration_type.value == "task_coordination"
                and not self.enable_task_coordination
            ):
                continue

            # Check confidence threshold
            if opportunity.confidence >= self.collaboration_threshold:
                filtered.append(opportunity)

        return filtered

    def _track_collaborative_action(self, opportunity, intent: Intent) -> None:
        """Track collaborative actions for analysis and learning.

        Args:
            opportunity: The collaboration opportunity that triggered the action
            intent: The intent that was created in response
        """
        if "collaborative_actions" not in self.collaboration_memory:
            self.collaboration_memory["collaborative_actions"] = []

        action_record = {
            "timestamp": time.time(),
            "opportunity_type": opportunity.collaboration_type.value,
            "target_agents": opportunity.target_agents,
            "confidence": opportunity.confidence,
            "intent_kind": intent["kind"],
            "reasoning": opportunity.reasoning,
        }

        self.collaboration_memory["collaborative_actions"].append(action_record)

        # Keep only recent actions (last 50)
        if len(self.collaboration_memory["collaborative_actions"]) > 50:
            self.collaboration_memory["collaborative_actions"] = (
                self.collaboration_memory["collaborative_actions"][-50:]
            )

    def _build_context(
        self,
        nearby_agents: list[dict[str, Any]],
        recent_messages: list[dict[str, Any]],
        agent_id: str,
    ) -> str:
        """Build enhanced context including collaboration information.

        Args:
            nearby_agents: List of nearby agent information
            recent_messages: List of recent messages
            agent_id: This agent's ID

        Returns:
            Enhanced context string for LLM including collaboration context
        """
        # Get base context from parent class
        base_context = super()._build_context(nearby_agents, recent_messages, agent_id)

        # Add collaboration context
        collaboration_parts = []

        # Add active coordination patterns
        active_patterns = self.collaboration_manager.get_active_coordination_patterns(
            agent_id
        )
        if active_patterns:
            collaboration_parts.append(
                f"\nActive coordination patterns ({len(active_patterns)}):"
            )
            for pattern in active_patterns[:3]:  # Show top 3
                other_agents = [a for a in pattern.involved_agents if a != agent_id]
                collaboration_parts.append(
                    f"- {pattern.pattern_type} with {', '.join(other_agents)} "
                    f"(strength: {pattern.strength:.1f})"
                )

        # Add recent collaborative actions
        if "collaborative_actions" in self.collaboration_memory:
            recent_actions = self.collaboration_memory["collaborative_actions"][-3:]
            if recent_actions:
                collaboration_parts.append("\nRecent collaborative actions:")
                for action in recent_actions:
                    collaboration_parts.append(
                        f"- {action['opportunity_type']} with {', '.join(action['target_agents'])}"
                    )

        # Add collaboration guidance
        if collaboration_parts:
            collaboration_parts.append(
                "\nYou are collaborative and look for opportunities to help, "
                "coordinate, and work together with other agents."
            )

        # Combine contexts
        if collaboration_parts:
            enhanced_context = base_context + "\n" + "\n".join(collaboration_parts)
        else:
            enhanced_context = base_context

        # Truncate if necessary
        if len(enhanced_context) > self.max_context_length:
            enhanced_context = enhanced_context[: self.max_context_length] + "..."

        return enhanced_context

    def get_collaboration_stats(self) -> dict[str, Any]:
        """Get collaboration-specific statistics.

        Returns:
            Dictionary with collaboration statistics
        """
        active_patterns = self.collaboration_manager.get_active_coordination_patterns(
            self.name or "unknown"
        )

        return {
            "collaborative_actions_taken": self.collaborative_actions_taken,
            "opportunities_detected": self.opportunities_detected,
            "active_coordination_patterns": len(active_patterns),
            "coordination_patterns_participated": len(
                self.collaboration_manager.pattern_history
            ),
            "collaboration_threshold": self.collaboration_threshold,
            "collaboration_settings": {
                "following_enabled": self.enable_following,
                "helping_enabled": self.enable_helping,
                "task_coordination_enabled": self.enable_task_coordination,
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive agent statistics including collaboration data.

        Returns:
            Dictionary with all agent statistics
        """
        base_stats = super().get_stats()
        collaboration_stats = self.get_collaboration_stats()

        # Merge the statistics
        base_stats.update(collaboration_stats)
        return base_stats

    async def on_loop_start(self, agent_id: str) -> None:
        """Called when the agent's async loop starts."""
        print(f"🤝 {self.name or agent_id} started collaborative loop")
        print(f"   Personality: {self.personality}")
        print(
            f"   Collaboration settings: Following={self.enable_following}, "
            f"Helping={self.enable_helping}, TaskCoord={self.enable_task_coordination}"
        )

    async def on_loop_stop(self, agent_id: str) -> None:
        """Called when the agent's async loop stops."""
        stats = self.get_stats()
        print(f"🔇 {self.name or agent_id} stopped collaborative loop")
        print(
            f"   Messages: {stats['messages_sent']}, Moves: {stats['moves_made']}, "
            f"Observations: {stats['observations_processed']}"
        )
        print(
            f"   Collaborative actions: {stats['collaborative_actions_taken']}, "
            f"Opportunities: {stats['opportunities_detected']}"
        )
        print(
            f"   Coordination patterns: {stats['active_coordination_patterns']} active, "
            f"{stats['coordination_patterns_participated']} total"
        )


class SpecializedCollaborativeAgent(CollaborativeAgent):
    """Specialized collaborative agent with role-specific behavior.

    This agent can be configured for specific collaborative roles like
    leader, follower, helper, or coordinator.
    """

    def __init__(self, llm_client: LLMClient, role: str = "general", **kwargs):
        """Initialize specialized collaborative agent.

        Args:
            llm_client: LLM client for generating responses
            role: Specialized role ("leader", "follower", "helper", "coordinator")
            **kwargs: Additional arguments passed to CollaborativeAgent
        """
        # Adjust settings based on role
        role_configs = {
            "leader": {
                "personality": "confident leader who initiates group activities and coordinates tasks",
                "response_probability": 0.9,
                "collaboration_threshold": 0.4,
                "enable_task_coordination": True,
            },
            "follower": {
                "personality": "supportive follower who prefers to assist and follow others",
                "response_probability": 0.6,
                "collaboration_threshold": 0.3,
                "enable_following": True,
            },
            "helper": {
                "personality": "helpful assistant who looks for opportunities to help others",
                "response_probability": 0.8,
                "collaboration_threshold": 0.5,
                "enable_helping": True,
            },
            "coordinator": {
                "personality": "organized coordinator who facilitates group work and communication",
                "response_probability": 0.7,
                "collaboration_threshold": 0.6,
                "enable_task_coordination": True,
            },
        }

        # Apply role-specific configuration
        role_config = role_configs.get(role, {})
        for key, value in role_config.items():
            if key not in kwargs:
                kwargs[key] = value

        super().__init__(llm_client=llm_client, **kwargs)
        self.role = role

    async def on_loop_start(self, agent_id: str) -> None:
        """Called when the agent's async loop starts."""
        print(f"🎭 {self.name or agent_id} started as {self.role}")
        print(f"   Personality: {self.personality}")
        print(
            f"   Collaboration settings: Following={self.enable_following}, "
            f"Helping={self.enable_helping}, TaskCoord={self.enable_task_coordination}"
        )
