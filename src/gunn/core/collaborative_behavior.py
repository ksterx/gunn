"""Collaborative behavior patterns for multi-agent coordination.

This module provides helper methods and patterns for detecting collaboration
opportunities and implementing coordination behaviors without explicit synchronization.
Agents coordinate through observed actions and communication.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from gunn.schemas.messages import View
from gunn.schemas.types import Intent


class CollaborationType(Enum):
    """Types of collaborative behaviors that can be detected."""

    FOLLOWING = "following"
    HELPING = "helping"
    GROUP_CONVERSATION = "group_conversation"
    TASK_COORDINATION = "task_coordination"
    SPATIAL_CLUSTERING = "spatial_clustering"
    SYNCHRONIZED_MOVEMENT = "synchronized_movement"


@dataclass
class CollaborationOpportunity:
    """Represents a detected collaboration opportunity."""

    collaboration_type: CollaborationType
    target_agents: list[str]
    context: dict[str, Any]
    confidence: float  # 0.0 to 1.0
    suggested_action: str | None = None
    reasoning: str | None = None


@dataclass
class CoordinationPattern:
    """Represents a detected coordination pattern between agents."""

    pattern_type: str
    involved_agents: list[str]
    pattern_data: dict[str, Any]
    start_time: float
    last_update: float
    strength: float  # 0.0 to 1.0


class CollaborationDetector(ABC):
    """Abstract base class for detecting specific types of collaboration opportunities."""

    @abstractmethod
    def detect_opportunities(
        self, observation: View, agent_id: str, agent_memory: dict[str, Any]
    ) -> list[CollaborationOpportunity]:
        """Detect collaboration opportunities from current observation.

        Args:
            observation: Current view of the world state
            agent_id: ID of the observing agent
            agent_memory: Agent's memory/context data

        Returns:
            List of detected collaboration opportunities
        """
        pass


class FollowingDetector(CollaborationDetector):
    """Detects opportunities for following behavior."""

    def __init__(self, follow_distance: float = 5.0, movement_threshold: float = 2.0):
        """Initialize following detector.

        Args:
            follow_distance: Preferred distance to maintain when following
            movement_threshold: Minimum movement to trigger following behavior
        """
        self.follow_distance = follow_distance
        self.movement_threshold = movement_threshold

    def detect_opportunities(
        self, observation: View, agent_id: str, agent_memory: dict[str, Any]
    ) -> list[CollaborationOpportunity]:
        """Detect agents that are moving and could be followed."""
        opportunities = []

        # Get agent's current position
        agent_pos = self._get_agent_position(observation, agent_id)
        if not agent_pos:
            return opportunities

        # Look for agents that have moved recently
        for entity_id, entity_data in observation.visible_entities.items():
            if entity_id == agent_id or not entity_id.startswith("agent_"):
                continue

            target_pos = self._get_agent_position(observation, entity_id)
            if not target_pos:
                continue

            # Check if target agent has moved recently
            if self._has_moved_recently(entity_id, target_pos, agent_memory):
                distance = self._calculate_distance(agent_pos, target_pos)

                # If target is moving away and we're close enough to follow
                if distance > self.follow_distance and distance < 15.0:
                    confidence = min(1.0, (15.0 - distance) / 10.0)

                    opportunities.append(
                        CollaborationOpportunity(
                            collaboration_type=CollaborationType.FOLLOWING,
                            target_agents=[entity_id],
                            context={
                                "target_position": target_pos,
                                "current_distance": distance,
                                "preferred_distance": self.follow_distance,
                            },
                            confidence=confidence,
                            suggested_action="move",
                            reasoning=f"Agent {entity_id} is moving, could follow at distance {self.follow_distance}",
                        )
                    )

        return opportunities

    def _get_agent_position(
        self, observation: View, agent_id: str
    ) -> tuple[float, float, float] | None:
        """Extract agent position from observation."""
        if agent_id not in observation.visible_entities:
            return None

        entity_data = observation.visible_entities[agent_id]
        pos = entity_data.get("position")

        if not pos or not isinstance(pos, (list, tuple)) or len(pos) < 2:
            return None

        return (float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0)

    def _has_moved_recently(
        self,
        agent_id: str,
        current_pos: tuple[float, float, float],
        agent_memory: dict[str, Any],
    ) -> bool:
        """Check if agent has moved recently based on memory."""
        if "agent_positions" not in agent_memory:
            agent_memory["agent_positions"] = {}

        positions = agent_memory["agent_positions"]

        if agent_id not in positions:
            positions[agent_id] = {"position": current_pos, "timestamp": time.time()}
            return False

        prev_data = positions[agent_id]
        prev_pos = prev_data["position"]

        # Calculate movement distance
        movement = self._calculate_distance(prev_pos, current_pos)

        # Update position
        positions[agent_id] = {"position": current_pos, "timestamp": time.time()}

        return movement > self.movement_threshold

    def _calculate_distance(
        self, pos1: tuple[float, float, float], pos2: tuple[float, float, float]
    ) -> float:
        """Calculate Euclidean distance between two positions."""
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2, strict=False)) ** 0.5


class GroupConversationDetector(CollaborationDetector):
    """Detects opportunities for group conversation participation."""

    def __init__(self, conversation_radius: float = 10.0, min_participants: int = 2):
        """Initialize group conversation detector.

        Args:
            conversation_radius: Maximum distance to participate in conversation
            min_participants: Minimum number of agents for a group conversation
        """
        self.conversation_radius = conversation_radius
        self.min_participants = min_participants

    def detect_opportunities(
        self, observation: View, agent_id: str, agent_memory: dict[str, Any]
    ) -> list[CollaborationOpportunity]:
        """Detect ongoing group conversations that agent could join."""
        opportunities = []

        # Get agent's current position
        agent_pos = self._get_agent_position(observation, agent_id)
        if not agent_pos:
            return opportunities

        # Find recent messages and their speakers
        recent_speakers = self._find_recent_speakers(observation, agent_memory)

        if len(recent_speakers) >= self.min_participants:
            # Check if speakers are clustered spatially (indicating group conversation)
            conversation_groups = self._find_conversation_clusters(
                recent_speakers, observation, agent_pos
            )

            for group in conversation_groups:
                if len(group["participants"]) >= self.min_participants:
                    # Calculate confidence based on proximity and recent activity
                    avg_distance = sum(group["distances"]) / len(group["distances"])
                    confidence = max(
                        0.0,
                        min(
                            1.0,
                            (self.conversation_radius - avg_distance)
                            / self.conversation_radius,
                        ),
                    )

                    if confidence > 0.3:  # Only suggest if reasonably confident
                        opportunities.append(
                            CollaborationOpportunity(
                                collaboration_type=CollaborationType.GROUP_CONVERSATION,
                                target_agents=group["participants"],
                                context={
                                    "conversation_center": group["center"],
                                    "participants": group["participants"],
                                    "recent_messages": group["recent_messages"],
                                    "average_distance": avg_distance,
                                },
                                confidence=confidence,
                                suggested_action="speak",
                                reasoning=f"Group conversation detected with {len(group['participants'])} participants",
                            )
                        )

        return opportunities

    def _get_agent_position(
        self, observation: View, agent_id: str
    ) -> tuple[float, float, float] | None:
        """Extract agent position from observation."""
        if agent_id not in observation.visible_entities:
            return None

        entity_data = observation.visible_entities[agent_id]
        pos = entity_data.get("position")

        if not pos or not isinstance(pos, (list, tuple)) or len(pos) < 2:
            return None

        return (float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0)

    def _find_recent_speakers(
        self, observation: View, agent_memory: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Find agents who have spoken recently."""
        current_time = time.time()
        speakers = []

        # Look for recent message indicators in observation
        for entity_id, entity_data in observation.visible_entities.items():
            if entity_id.startswith("agent_") and "recent_message" in entity_data:
                message_data = entity_data["recent_message"]
                if isinstance(message_data, dict):
                    message_time = message_data.get("timestamp", current_time)

                    # Only consider messages from last 30 seconds
                    if current_time - message_time < 30.0:
                        speakers.append(
                            {
                                "agent_id": entity_id,
                                "message": message_data.get("text", ""),
                                "timestamp": message_time,
                                "position": self._get_agent_position(
                                    observation, entity_id
                                ),
                            }
                        )

        return speakers

    def _find_conversation_clusters(
        self,
        speakers: list[dict[str, Any]],
        observation: View,
        agent_pos: tuple[float, float, float],
    ) -> list[dict[str, Any]]:
        """Find spatial clusters of speakers indicating group conversations."""
        clusters = []

        # Simple clustering: group speakers within conversation radius
        for speaker in speakers:
            speaker_pos = speaker["position"]
            if not speaker_pos:
                continue

            # Find other speakers near this one
            cluster_participants = [speaker["agent_id"]]
            cluster_positions = [speaker_pos]
            cluster_messages = [speaker["message"]]
            distances_to_agent = [self._calculate_distance(agent_pos, speaker_pos)]

            for other_speaker in speakers:
                if other_speaker["agent_id"] == speaker["agent_id"]:
                    continue

                other_pos = other_speaker["position"]
                if not other_pos:
                    continue

                distance = self._calculate_distance(speaker_pos, other_pos)
                if distance <= self.conversation_radius:
                    cluster_participants.append(other_speaker["agent_id"])
                    cluster_positions.append(other_pos)
                    cluster_messages.append(other_speaker["message"])
                    distances_to_agent.append(
                        self._calculate_distance(agent_pos, other_pos)
                    )

            if len(cluster_participants) >= self.min_participants:
                # Calculate cluster center
                center = (
                    sum(pos[0] for pos in cluster_positions) / len(cluster_positions),
                    sum(pos[1] for pos in cluster_positions) / len(cluster_positions),
                    sum(pos[2] for pos in cluster_positions) / len(cluster_positions),
                )

                clusters.append(
                    {
                        "participants": cluster_participants,
                        "center": center,
                        "recent_messages": cluster_messages,
                        "distances": distances_to_agent,
                    }
                )

        return clusters

    def _calculate_distance(
        self, pos1: tuple[float, float, float], pos2: tuple[float, float, float]
    ) -> float:
        """Calculate Euclidean distance between two positions."""
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2, strict=False)) ** 0.5


class TaskCoordinationDetector(CollaborationDetector):
    """Detects opportunities for task coordination and helping behavior."""

    def __init__(self, help_keywords: list[str] | None = None):
        """Initialize task coordination detector.

        Args:
            help_keywords: Keywords that indicate need for help or coordination
        """
        self.help_keywords = help_keywords or [
            "help",
            "assist",
            "together",
            "collaborate",
            "work with",
            "need",
            "stuck",
            "problem",
            "difficult",
            "challenge",
        ]

    def detect_opportunities(
        self, observation: View, agent_id: str, agent_memory: dict[str, Any]
    ) -> list[CollaborationOpportunity]:
        """Detect requests for help or coordination opportunities."""
        opportunities = []

        # Look for help requests in recent messages
        for entity_id, entity_data in observation.visible_entities.items():
            if entity_id == agent_id or not entity_id.startswith("agent_"):
                continue

            if "recent_message" in entity_data:
                message_data = entity_data["recent_message"]
                if isinstance(message_data, dict):
                    message_text = message_data.get("text", "").lower()

                    # Check for help keywords
                    help_score = sum(
                        1 for keyword in self.help_keywords if keyword in message_text
                    )

                    if help_score > 0:
                        confidence = min(1.0, help_score / 3.0)  # Normalize to 0-1

                        opportunities.append(
                            CollaborationOpportunity(
                                collaboration_type=CollaborationType.HELPING,
                                target_agents=[entity_id],
                                context={
                                    "help_request": message_data.get("text", ""),
                                    "requester_position": entity_data.get("position"),
                                    "keywords_found": [
                                        kw
                                        for kw in self.help_keywords
                                        if kw in message_text
                                    ],
                                },
                                confidence=confidence,
                                suggested_action="speak",
                                reasoning=f"Agent {entity_id} appears to need help based on their message",
                            )
                        )

        # Look for task coordination patterns (agents working on similar things)
        task_opportunities = self._detect_task_coordination(
            observation, agent_id, agent_memory
        )
        opportunities.extend(task_opportunities)

        return opportunities

    def _detect_task_coordination(
        self, observation: View, agent_id: str, agent_memory: dict[str, Any]
    ) -> list[CollaborationOpportunity]:
        """Detect opportunities for task coordination."""
        opportunities = []

        # Look for agents performing similar actions or in similar states
        agent_actions = self._extract_agent_actions(observation, agent_memory)

        # Find agents with similar recent actions
        similar_agents = []
        agent_recent_actions = agent_actions.get(agent_id, [])

        for other_agent_id, other_actions in agent_actions.items():
            if other_agent_id == agent_id:
                continue

            # Calculate action similarity
            similarity = self._calculate_action_similarity(
                agent_recent_actions, other_actions
            )

            if similarity > 0.5:  # Threshold for considering coordination
                similar_agents.append(
                    {
                        "agent_id": other_agent_id,
                        "similarity": similarity,
                        "actions": other_actions,
                    }
                )

        if similar_agents:
            # Sort by similarity
            similar_agents.sort(key=lambda x: x["similarity"], reverse=True)

            # Create coordination opportunity
            target_agents = [agent["agent_id"] for agent in similar_agents[:3]]  # Top 3
            avg_similarity = sum(
                agent["similarity"] for agent in similar_agents[:3]
            ) / len(similar_agents[:3])

            opportunities.append(
                CollaborationOpportunity(
                    collaboration_type=CollaborationType.TASK_COORDINATION,
                    target_agents=target_agents,
                    context={
                        "similar_agents": similar_agents[:3],
                        "coordination_type": "similar_actions",
                        "average_similarity": avg_similarity,
                    },
                    confidence=avg_similarity,
                    suggested_action="speak",
                    reasoning=f"Found {len(similar_agents)} agents with similar recent actions",
                )
            )

        return opportunities

    def _extract_agent_actions(
        self, observation: View, agent_memory: dict[str, Any]
    ) -> dict[str, list[str]]:
        """Extract recent actions for each agent."""
        if "agent_actions" not in agent_memory:
            agent_memory["agent_actions"] = {}

        actions = agent_memory["agent_actions"]
        current_time = time.time()

        # Update actions based on current observation
        for entity_id, entity_data in observation.visible_entities.items():
            if entity_id.startswith("agent_"):
                if entity_id not in actions:
                    actions[entity_id] = []

                # Infer actions from entity state changes
                # This is simplified - in a real system you'd track actual actions
                if "recent_message" in entity_data:
                    actions[entity_id].append("speak")

                # Clean old actions (keep only last 5 minutes)
                actions[entity_id] = [
                    action
                    for action in actions[entity_id][-10:]  # Keep last 10 actions
                ]

        return actions

    def _calculate_action_similarity(
        self, actions1: list[str], actions2: list[str]
    ) -> float:
        """Calculate similarity between two action sequences."""
        if not actions1 or not actions2:
            return 0.0

        # Simple similarity: ratio of common actions
        common_actions = set(actions1) & set(actions2)
        total_unique_actions = set(actions1) | set(actions2)

        if not total_unique_actions:
            return 0.0

        return len(common_actions) / len(total_unique_actions)


class CollaborativeBehaviorManager:
    """Manages collaborative behavior detection and coordination patterns."""

    def __init__(self):
        """Initialize collaborative behavior manager."""
        self.detectors: list[CollaborationDetector] = [
            FollowingDetector(),
            GroupConversationDetector(),
            TaskCoordinationDetector(),
        ]
        self.active_patterns: dict[str, CoordinationPattern] = {}
        self.pattern_history: list[CoordinationPattern] = []

    def detect_collaboration_opportunities(
        self, observation: View, agent_id: str, agent_memory: dict[str, Any]
    ) -> list[CollaborationOpportunity]:
        """Detect all types of collaboration opportunities.

        Args:
            observation: Current view of the world state
            agent_id: ID of the observing agent
            agent_memory: Agent's memory/context data

        Returns:
            List of all detected collaboration opportunities
        """
        all_opportunities = []

        for detector in self.detectors:
            try:
                opportunities = detector.detect_opportunities(
                    observation, agent_id, agent_memory
                )
                all_opportunities.extend(opportunities)
            except Exception as e:
                # Log error but continue with other detectors
                print(f"Error in collaboration detector {type(detector).__name__}: {e}")

        # Sort by confidence (highest first)
        all_opportunities.sort(key=lambda x: x.confidence, reverse=True)

        return all_opportunities

    def update_coordination_patterns(
        self, observation: View, agent_id: str, recent_action: Intent | None = None
    ) -> None:
        """Update tracking of coordination patterns.

        Args:
            observation: Current view of the world state
            agent_id: ID of the agent
            recent_action: Recent action taken by the agent
        """
        current_time = time.time()

        # Update existing patterns
        patterns_to_remove = []
        for pattern_id, pattern in self.active_patterns.items():
            # Check if pattern is still active (agents still coordinating)
            if self._is_pattern_still_active(pattern, observation, current_time):
                pattern.last_update = current_time
                # Update pattern strength based on continued coordination
                pattern.strength = min(1.0, pattern.strength + 0.1)
            else:
                # Pattern has ended, move to history
                patterns_to_remove.append(pattern_id)
                self.pattern_history.append(pattern)

        # Remove inactive patterns
        for pattern_id in patterns_to_remove:
            del self.active_patterns[pattern_id]

        # Detect new coordination patterns
        new_patterns = self._detect_new_patterns(
            observation, agent_id, recent_action, current_time
        )
        for pattern in new_patterns:
            pattern_id = (
                f"{pattern.pattern_type}_{hash(tuple(sorted(pattern.involved_agents)))}"
            )
            self.active_patterns[pattern_id] = pattern

    def get_active_coordination_patterns(
        self, agent_id: str
    ) -> list[CoordinationPattern]:
        """Get coordination patterns involving the specified agent.

        Args:
            agent_id: ID of the agent

        Returns:
            List of active coordination patterns involving the agent
        """
        return [
            pattern
            for pattern in self.active_patterns.values()
            if agent_id in pattern.involved_agents
        ]

    def _is_pattern_still_active(
        self, pattern: CoordinationPattern, observation: View, current_time: float
    ) -> bool:
        """Check if a coordination pattern is still active."""
        # Pattern expires after 60 seconds of no updates
        if current_time - pattern.last_update > 60.0:
            return False

        # Check if involved agents are still visible and coordinating
        visible_agents = [
            agent_id
            for agent_id in pattern.involved_agents
            if agent_id in observation.visible_entities
        ]

        # Pattern is inactive if less than half the agents are visible
        return len(visible_agents) >= len(pattern.involved_agents) / 2

    def _detect_new_patterns(
        self,
        observation: View,
        agent_id: str,
        recent_action: Intent | None,
        current_time: float,
    ) -> list[CoordinationPattern]:
        """Detect new coordination patterns."""
        new_patterns = []

        # Detect spatial clustering pattern
        clustering_pattern = self._detect_spatial_clustering(observation, current_time)
        if clustering_pattern:
            new_patterns.append(clustering_pattern)

        # Detect synchronized movement pattern
        if recent_action and recent_action["kind"] == "Move":
            movement_pattern = self._detect_synchronized_movement(
                observation, agent_id, recent_action, current_time
            )
            if movement_pattern:
                new_patterns.append(movement_pattern)

        return new_patterns

    def _detect_spatial_clustering(
        self, observation: View, current_time: float
    ) -> CoordinationPattern | None:
        """Detect agents clustering spatially."""
        agent_positions = {}

        # Extract agent positions
        for entity_id, entity_data in observation.visible_entities.items():
            if entity_id.startswith("agent_") and "position" in entity_data:
                pos = entity_data["position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    agent_positions[entity_id] = (
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2]) if len(pos) > 2 else 0.0,
                    )

        if len(agent_positions) < 2:
            return None

        # Find clusters of agents within close proximity
        cluster_threshold = 8.0  # Distance threshold for clustering
        clusters = []

        for agent_id, position in agent_positions.items():
            # Find nearby agents
            nearby_agents = [agent_id]
            for other_id, other_pos in agent_positions.items():
                if other_id != agent_id:
                    distance = (
                        sum(
                            (a - b) ** 2
                            for a, b in zip(position, other_pos, strict=False)
                        )
                        ** 0.5
                    )
                    if distance <= cluster_threshold:
                        nearby_agents.append(other_id)

            if len(nearby_agents) >= 3:  # Minimum cluster size
                clusters.append(nearby_agents)

        # Find the largest cluster
        if clusters:
            largest_cluster = max(clusters, key=len)

            return CoordinationPattern(
                pattern_type="spatial_clustering",
                involved_agents=largest_cluster,
                pattern_data={
                    "cluster_size": len(largest_cluster),
                    "cluster_threshold": cluster_threshold,
                },
                start_time=current_time,
                last_update=current_time,
                strength=min(
                    1.0, len(largest_cluster) / 5.0
                ),  # Normalize by max expected cluster size
            )

        return None

    def _detect_synchronized_movement(
        self,
        observation: View,
        agent_id: str,
        recent_action: Intent,
        current_time: float,
    ) -> CoordinationPattern | None:
        """Detect synchronized movement patterns."""
        # This is a simplified implementation
        # In a real system, you'd track movement vectors and timing

        # Look for other agents that might be moving in similar directions
        agent_target = recent_action["payload"].get("to")
        if not agent_target:
            return None

        synchronized_agents = [agent_id]

        # Check if other agents have similar recent movement (this would require more state tracking)
        # For now, just detect if multiple agents are in motion
        for entity_id, entity_data in observation.visible_entities.items():
            if entity_id != agent_id and entity_id.startswith("agent_"):
                # In a real implementation, you'd check for recent movement actions
                # For now, just include agents that are nearby and might be coordinating
                pass

        # Return pattern only if we detect actual coordination
        # This is a placeholder - real implementation would need movement history
        return None


def create_collaborative_intent(
    opportunity: CollaborationOpportunity, agent_id: str, context_seq: int
) -> Intent | None:
    """Create an intent based on a collaboration opportunity.

    Args:
        opportunity: Detected collaboration opportunity
        agent_id: ID of the agent creating the intent
        context_seq: Current context sequence number

    Returns:
        Intent to execute the collaborative behavior, or None if no action needed
    """
    import uuid

    if opportunity.suggested_action == "speak":
        # Generate collaborative message based on opportunity type
        if opportunity.collaboration_type == CollaborationType.GROUP_CONVERSATION:
            text = "I'd like to join the conversation! What are you all discussing?"
        elif opportunity.collaboration_type == CollaborationType.HELPING:
            text = "I heard you might need some help. How can I assist?"
        elif opportunity.collaboration_type == CollaborationType.TASK_COORDINATION:
            text = "It looks like we're working on similar things. Want to coordinate?"
        else:
            text = "Hello! I noticed we might be able to work together."

        return {
            "kind": "Speak",
            "payload": {
                "text": text,
                "agent_id": agent_id,
                "collaboration_context": {
                    "type": opportunity.collaboration_type.value,
                    "target_agents": opportunity.target_agents,
                    "confidence": opportunity.confidence,
                },
            },
            "context_seq": context_seq,
            "req_id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "priority": 1,  # Higher priority for collaborative actions
            "schema_version": "1.0.0",
        }

    elif opportunity.suggested_action == "move":
        # Generate movement intent for following or spatial coordination
        if opportunity.collaboration_type == CollaborationType.FOLLOWING:
            target_pos = opportunity.context.get("target_position")
            preferred_distance = opportunity.context.get("preferred_distance", 5.0)

            if target_pos:
                # Calculate position to maintain preferred distance
                # This is simplified - real implementation would consider movement vectors
                follow_pos = [
                    target_pos[0] - preferred_distance,
                    target_pos[1],
                    target_pos[2] if len(target_pos) > 2 else 0.0,
                ]

                return {
                    "kind": "Move",
                    "payload": {
                        "to": follow_pos,
                        "agent_id": agent_id,
                        "collaboration_context": {
                            "type": opportunity.collaboration_type.value,
                            "following_agent": opportunity.target_agents[0]
                            if opportunity.target_agents
                            else None,
                        },
                    },
                    "context_seq": context_seq,
                    "req_id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "priority": 1,
                    "schema_version": "1.0.0",
                }

        elif opportunity.collaboration_type == CollaborationType.SPATIAL_CLUSTERING:
            # Move towards the cluster center
            cluster_center = opportunity.context.get("cluster_center")
            if cluster_center:
                return {
                    "kind": "Move",
                    "payload": {
                        "to": list(cluster_center),
                        "agent_id": agent_id,
                        "collaboration_context": {
                            "type": opportunity.collaboration_type.value,
                            "cluster_agents": opportunity.target_agents,
                        },
                    },
                    "context_seq": context_seq,
                    "req_id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "priority": 1,
                    "schema_version": "1.0.0",
                }

    return None


# Helper functions for collaborative behavior patterns


def detect_emergent_behaviors(
    agents_data: dict[str, Any], time_window: float = 30.0
) -> list[dict[str, Any]]:
    """Detect emergent collaborative behaviors from agent data.

    Args:
        agents_data: Dictionary of agent data with positions, actions, and timing
        time_window: Time window in seconds to analyze for emergent patterns

    Returns:
        List of detected emergent behavior patterns
    """
    current_time = time.time()
    emergent_patterns = []

    # Detect flocking behavior (agents moving in similar directions)
    flocking_groups = _detect_flocking_behavior(agents_data, current_time, time_window)
    emergent_patterns.extend(flocking_groups)

    # Detect conversation chains (agents responding to each other in sequence)
    conversation_chains = _detect_conversation_chains(
        agents_data, current_time, time_window
    )
    emergent_patterns.extend(conversation_chains)

    # Detect helping chains (agents helping others who are helping others)
    helping_chains = _detect_helping_chains(agents_data, current_time, time_window)
    emergent_patterns.extend(helping_chains)

    return emergent_patterns


def _detect_flocking_behavior(
    agents_data: dict[str, Any], current_time: float, time_window: float
) -> list[dict[str, Any]]:
    """Detect flocking behavior patterns."""
    flocking_groups = []

    # Group agents by similar movement vectors
    movement_groups = {}

    for agent_id, data in agents_data.items():
        recent_positions = data.get("position_history", [])
        if len(recent_positions) < 2:
            continue

        # Calculate movement vector
        recent_pos = recent_positions[-1]
        prev_pos = recent_positions[-2]

        if recent_pos["timestamp"] - prev_pos["timestamp"] > time_window:
            continue

        movement_vector = (
            recent_pos["position"][0] - prev_pos["position"][0],
            recent_pos["position"][1] - prev_pos["position"][1],
        )

        # Normalize and discretize movement direction
        magnitude = (movement_vector[0] ** 2 + movement_vector[1] ** 2) ** 0.5
        if magnitude > 0.1:  # Minimum movement threshold
            direction = (
                round(movement_vector[0] / magnitude, 1),
                round(movement_vector[1] / magnitude, 1),
            )

            if direction not in movement_groups:
                movement_groups[direction] = []
            movement_groups[direction].append(
                {
                    "agent_id": agent_id,
                    "position": recent_pos["position"],
                    "speed": magnitude,
                }
            )

    # Find groups with multiple agents moving in similar directions
    for direction, agents in movement_groups.items():
        if len(agents) >= 3:  # Minimum flock size
            # Check if agents are spatially close
            positions = [agent["position"] for agent in agents]
            if _are_positions_clustered(positions, max_distance=15.0):
                flocking_groups.append(
                    {
                        "pattern_type": "flocking",
                        "agents": [agent["agent_id"] for agent in agents],
                        "direction": direction,
                        "average_speed": sum(agent["speed"] for agent in agents)
                        / len(agents),
                        "cluster_center": _calculate_center(positions),
                        "confidence": min(
                            1.0, len(agents) / 5.0
                        ),  # Normalize by max expected flock size
                    }
                )

    return flocking_groups


def _detect_conversation_chains(
    agents_data: dict[str, Any], current_time: float, time_window: float
) -> list[dict[str, Any]]:
    """Detect conversation chain patterns."""
    conversation_chains = []

    # Collect recent messages with timing
    recent_messages = []
    for agent_id, data in agents_data.items():
        message_history = data.get("message_history", [])
        for message in message_history:
            if current_time - message["timestamp"] <= time_window:
                recent_messages.append(
                    {
                        "agent_id": agent_id,
                        "text": message["text"],
                        "timestamp": message["timestamp"],
                        "position": message.get("position"),
                    }
                )

    # Sort messages by timestamp
    recent_messages.sort(key=lambda x: x["timestamp"])

    # Find conversation chains (sequences of messages between nearby agents)
    if len(recent_messages) >= 3:
        chains = []
        current_chain = [recent_messages[0]]

        for i in range(1, len(recent_messages)):
            prev_msg = recent_messages[i - 1]
            curr_msg = recent_messages[i]

            # Check if messages are from different agents and temporally close
            time_gap = curr_msg["timestamp"] - prev_msg["timestamp"]
            if (
                curr_msg["agent_id"] != prev_msg["agent_id"]
                and time_gap <= 10.0  # Max 10 seconds between messages
                and _are_positions_close(
                    prev_msg.get("position"),
                    curr_msg.get("position"),
                    max_distance=20.0,
                )
            ):
                current_chain.append(curr_msg)
            else:
                if len(current_chain) >= 3:
                    chains.append(current_chain)
                current_chain = [curr_msg]

        # Add final chain if long enough
        if len(current_chain) >= 3:
            chains.append(current_chain)

        # Convert chains to patterns
        for chain in chains:
            conversation_chains.append(
                {
                    "pattern_type": "conversation_chain",
                    "agents": [msg["agent_id"] for msg in chain],
                    "message_count": len(chain),
                    "duration": chain[-1]["timestamp"] - chain[0]["timestamp"],
                    "confidence": min(
                        1.0, len(chain) / 8.0
                    ),  # Normalize by expected max chain length
                }
            )

    return conversation_chains


def _detect_helping_chains(
    agents_data: dict[str, Any], current_time: float, time_window: float
) -> list[dict[str, Any]]:
    """Detect helping chain patterns."""
    helping_chains = []

    # Look for sequences of help-related messages
    help_keywords = ["help", "assist", "support", "aid", "collaborate"]

    help_messages = []
    for agent_id, data in agents_data.items():
        message_history = data.get("message_history", [])
        for message in message_history:
            if current_time - message["timestamp"] <= time_window and any(
                keyword in message["text"].lower() for keyword in help_keywords
            ):
                help_messages.append(
                    {
                        "agent_id": agent_id,
                        "text": message["text"],
                        "timestamp": message["timestamp"],
                        "position": message.get("position"),
                    }
                )

    # Sort by timestamp and look for chains
    help_messages.sort(key=lambda x: x["timestamp"])

    if len(help_messages) >= 2:
        # Find sequences where agents help others who are also helping
        for i in range(len(help_messages) - 1):
            chain_agents = [help_messages[i]["agent_id"]]

            for j in range(i + 1, len(help_messages)):
                if (
                    help_messages[j]["timestamp"] - help_messages[i]["timestamp"]
                    <= 30.0
                    and help_messages[j]["agent_id"] not in chain_agents
                ):
                    chain_agents.append(help_messages[j]["agent_id"])

            if len(chain_agents) >= 3:
                helping_chains.append(
                    {
                        "pattern_type": "helping_chain",
                        "agents": chain_agents,
                        "chain_length": len(chain_agents),
                        "confidence": min(1.0, len(chain_agents) / 5.0),
                    }
                )

    return helping_chains


def _are_positions_clustered(
    positions: list[tuple], max_distance: float = 10.0
) -> bool:
    """Check if positions form a spatial cluster."""
    if len(positions) < 2:
        return True

    # Calculate pairwise distances
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = (
                sum(
                    (a - b) ** 2
                    for a, b in zip(positions[i], positions[j], strict=False)
                )
                ** 0.5
            )
            if distance > max_distance:
                return False

    return True


def _are_positions_close(
    pos1: tuple | None, pos2: tuple | None, max_distance: float = 10.0
) -> bool:
    """Check if two positions are close to each other."""
    if not pos1 or not pos2:
        return True  # Assume close if position data is missing

    distance = sum((a - b) ** 2 for a, b in zip(pos1, pos2, strict=False)) ** 0.5
    return distance <= max_distance


def _calculate_center(positions: list[tuple]) -> tuple:
    """Calculate the center point of a list of positions."""
    if not positions:
        return (0.0, 0.0, 0.0)

    center = [
        sum(pos[i] for pos in positions) / len(positions)
        for i in range(len(positions[0]))
    ]

    # Ensure 3D coordinates
    while len(center) < 3:
        center.append(0.0)

    return tuple(center[:3])


# Advanced coordination patterns


class CoordinationPatternAnalyzer:
    """Analyzes and tracks complex coordination patterns over time."""

    def __init__(self, pattern_memory_size: int = 100):
        """Initialize pattern analyzer.

        Args:
            pattern_memory_size: Maximum number of patterns to keep in memory
        """
        self.pattern_memory_size = pattern_memory_size
        self.pattern_history: list[dict[str, Any]] = []
        self.pattern_statistics: dict[str, Any] = {
            "total_patterns_detected": 0,
            "pattern_type_counts": {},
            "average_pattern_duration": 0.0,
            "most_collaborative_agents": [],
        }

    def analyze_coordination_evolution(
        self, current_patterns: list[CoordinationPattern], agents_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze how coordination patterns evolve over time.

        Args:
            current_patterns: Currently active coordination patterns
            agents_data: Current agent data

        Returns:
            Analysis results including pattern evolution metrics
        """
        current_time = time.time()

        # Track pattern evolution
        evolution_metrics = {
            "new_patterns": 0,
            "evolved_patterns": 0,
            "stable_patterns": 0,
            "dissolved_patterns": 0,
            "pattern_complexity_trend": "stable",
        }

        # Compare with previous patterns
        if self.pattern_history:
            prev_analysis = self.pattern_history[-1]
            prev_patterns = prev_analysis.get("active_patterns", [])

            # Analyze pattern changes
            evolution_metrics.update(
                self._compare_pattern_sets(prev_patterns, current_patterns)
            )

        # Calculate collaboration network metrics
        network_metrics = self._calculate_network_metrics(current_patterns, agents_data)

        # Store current analysis
        current_analysis = {
            "timestamp": current_time,
            "active_patterns": [
                {
                    "type": pattern.pattern_type,
                    "agents": pattern.involved_agents,
                    "strength": pattern.strength,
                    "duration": current_time - pattern.start_time,
                }
                for pattern in current_patterns
            ],
            "evolution_metrics": evolution_metrics,
            "network_metrics": network_metrics,
        }

        self.pattern_history.append(current_analysis)

        # Maintain memory size limit
        if len(self.pattern_history) > self.pattern_memory_size:
            self.pattern_history = self.pattern_history[-self.pattern_memory_size :]

        # Update statistics
        self._update_statistics(current_analysis)

        return current_analysis

    def _compare_pattern_sets(
        self,
        prev_patterns: list[dict[str, Any]],
        current_patterns: list[CoordinationPattern],
    ) -> dict[str, int]:
        """Compare two sets of patterns to detect evolution."""
        prev_pattern_keys = {
            (p["type"], tuple(sorted(p["agents"]))) for p in prev_patterns
        }

        current_pattern_keys = {
            (p.pattern_type, tuple(sorted(p.involved_agents))) for p in current_patterns
        }

        new_patterns = len(current_pattern_keys - prev_pattern_keys)
        dissolved_patterns = len(prev_pattern_keys - current_pattern_keys)
        stable_patterns = len(current_pattern_keys & prev_pattern_keys)

        return {
            "new_patterns": new_patterns,
            "dissolved_patterns": dissolved_patterns,
            "stable_patterns": stable_patterns,
            "evolved_patterns": 0,  # Would need more sophisticated tracking
        }

    def _calculate_network_metrics(
        self, patterns: list[CoordinationPattern], agents_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate collaboration network metrics."""
        # Build collaboration graph
        collaboration_graph = {}
        for pattern in patterns:
            for agent in pattern.involved_agents:
                if agent not in collaboration_graph:
                    collaboration_graph[agent] = set()

                # Add connections to other agents in the pattern
                for other_agent in pattern.involved_agents:
                    if other_agent != agent:
                        collaboration_graph[agent].add(other_agent)

        # Calculate network metrics
        total_agents = len(collaboration_graph)
        total_connections = (
            sum(len(connections) for connections in collaboration_graph.values()) // 2
        )

        # Calculate clustering coefficient (simplified)
        clustering_coefficient = 0.0
        if total_agents > 2:
            for agent, connections in collaboration_graph.items():
                if len(connections) > 1:
                    # Count triangles involving this agent
                    triangles = 0
                    possible_triangles = len(connections) * (len(connections) - 1) // 2

                    for conn1 in connections:
                        for conn2 in connections:
                            if conn1 != conn2 and conn2 in collaboration_graph.get(
                                conn1, set()
                            ):
                                triangles += 1

                    if possible_triangles > 0:
                        clustering_coefficient += triangles / (2 * possible_triangles)

            clustering_coefficient /= total_agents

        return {
            "total_collaborative_agents": total_agents,
            "total_collaboration_connections": total_connections,
            "network_density": total_connections
            / max(1, total_agents * (total_agents - 1) // 2),
            "clustering_coefficient": clustering_coefficient,
            "average_connections_per_agent": total_connections / max(1, total_agents),
        }

    def _update_statistics(self, analysis: dict[str, Any]) -> None:
        """Update overall pattern statistics."""
        self.pattern_statistics["total_patterns_detected"] += len(
            analysis["active_patterns"]
        )

        # Update pattern type counts
        for pattern in analysis["active_patterns"]:
            pattern_type = pattern["type"]
            if pattern_type not in self.pattern_statistics["pattern_type_counts"]:
                self.pattern_statistics["pattern_type_counts"][pattern_type] = 0
            self.pattern_statistics["pattern_type_counts"][pattern_type] += 1

        # Calculate average pattern duration
        if analysis["active_patterns"]:
            total_duration = sum(p["duration"] for p in analysis["active_patterns"])
            avg_duration = total_duration / len(analysis["active_patterns"])

            # Update running average
            if self.pattern_statistics["average_pattern_duration"] == 0.0:
                self.pattern_statistics["average_pattern_duration"] = avg_duration
            else:
                # Exponential moving average
                alpha = 0.1
                self.pattern_statistics["average_pattern_duration"] = (
                    alpha * avg_duration
                    + (1 - alpha) * self.pattern_statistics["average_pattern_duration"]
                )

    def get_collaboration_insights(self) -> dict[str, Any]:
        """Get insights about collaboration patterns.

        Returns:
            Dictionary with collaboration insights and recommendations
        """
        insights = {
            "summary": self.pattern_statistics.copy(),
            "trends": {},
            "recommendations": [],
        }

        # Analyze trends from recent history
        if len(self.pattern_history) >= 5:
            recent_analyses = self.pattern_history[-5:]

            # Pattern count trend
            pattern_counts = [
                len(analysis["active_patterns"]) for analysis in recent_analyses
            ]
            if len(set(pattern_counts)) > 1:
                if pattern_counts[-1] > pattern_counts[0]:
                    insights["trends"]["pattern_count"] = "increasing"
                elif pattern_counts[-1] < pattern_counts[0]:
                    insights["trends"]["pattern_count"] = "decreasing"
                else:
                    insights["trends"]["pattern_count"] = "stable"

            # Network density trend
            densities = [
                analysis["network_metrics"]["network_density"]
                for analysis in recent_analyses
                if "network_metrics" in analysis
            ]
            if len(densities) >= 2:
                if densities[-1] > densities[0]:
                    insights["trends"]["collaboration_density"] = "increasing"
                elif densities[-1] < densities[0]:
                    insights["trends"]["collaboration_density"] = "decreasing"
                else:
                    insights["trends"]["collaboration_density"] = "stable"

        # Generate recommendations
        if insights["trends"].get("pattern_count") == "decreasing":
            insights["recommendations"].append(
                "Consider introducing collaborative tasks or events to encourage more coordination"
            )

        if insights["trends"].get("collaboration_density") == "decreasing":
            insights["recommendations"].append(
                "Agents may be becoming more isolated - consider adjusting observation policies or interaction incentives"
            )

        # Check for dominant pattern types
        if self.pattern_statistics["pattern_type_counts"]:
            most_common_pattern = max(
                self.pattern_statistics["pattern_type_counts"].items(),
                key=lambda x: x[1],
            )
            if (
                most_common_pattern[1]
                > sum(self.pattern_statistics["pattern_type_counts"].values()) * 0.6
            ):
                insights["recommendations"].append(
                    f"Pattern diversity is low - most patterns are '{most_common_pattern[0]}'. "
                    f"Consider encouraging other types of collaboration"
                )

        return insights
