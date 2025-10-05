"""Collaborative behavior patterns for multi-agent coordination.

This module provides helper methods and patterns for detecting collaboration
opportunities, implementing coordination patterns, and enabling emergent
collaborative behavior without explicit synchronization.

Requirements addressed:
- 3.6: Multi-agent task coordination without explicit synchronization
- 4.6: Collaborative opportunities through observation
- 14.9: Coordination through observed actions and communication
"""

from dataclasses import dataclass
from typing import Any

from gunn.schemas.messages import View


@dataclass
class CollaborationOpportunity:
    """Represents a detected collaboration opportunity.

    Attributes:
        opportunity_type: Type of collaboration (e.g., "task", "resource", "conversation")
        description: Human-readable description of the opportunity
        involved_agents: List of agent IDs that could participate
        location: Optional spatial location of the opportunity
        priority: Priority level (0-10, higher is more important)
        metadata: Additional context-specific information
    """

    opportunity_type: str
    description: str
    involved_agents: list[str]
    location: tuple[float, float, float] | None = None
    priority: int = 5
    metadata: dict[str, Any] | None = None


@dataclass
class CoordinationPattern:
    """Represents a coordination pattern between agents.

    Attributes:
        pattern_type: Type of pattern (e.g., "following", "helping", "group_conversation")
        initiator: Agent ID that initiated the pattern
        participants: List of participating agent IDs
        status: Current status ("active", "completed", "failed")
        start_time: When the pattern started
        metadata: Additional pattern-specific information
    """

    pattern_type: str
    initiator: str
    participants: list[str]
    status: str = "active"
    start_time: float = 0.0
    metadata: dict[str, Any] | None = None


class CollaborationDetector:
    """Detects collaboration opportunities from agent observations.

    This class analyzes observations to identify situations where agents
    could benefit from coordinating their actions, such as:
    - Nearby agents working on similar tasks
    - Resource sharing opportunities
    - Group conversation possibilities
    - Helping behaviors (one agent assisting another)
    """

    def __init__(
        self,
        proximity_threshold: float = 15.0,
        task_similarity_threshold: float = 0.7,
    ):
        """Initialize collaboration detector.

        Args:
            proximity_threshold: Maximum distance for spatial collaboration
            task_similarity_threshold: Minimum similarity for task collaboration
        """
        self.proximity_threshold = proximity_threshold
        self.task_similarity_threshold = task_similarity_threshold

    def detect_opportunities(
        self, observation: View, agent_id: str
    ) -> list[CollaborationOpportunity]:
        """Detect collaboration opportunities from an observation.

        Args:
            observation: Current agent observation
            agent_id: ID of the observing agent

        Returns:
            List of detected collaboration opportunities
        """
        opportunities = []

        # Detect spatial clustering (agents nearby)
        spatial_opps = self._detect_spatial_clustering(observation, agent_id)
        opportunities.extend(spatial_opps)

        # Detect task-based collaboration
        task_opps = self._detect_task_collaboration(observation, agent_id)
        opportunities.extend(task_opps)

        # Detect conversation opportunities
        conversation_opps = self._detect_conversation_opportunities(
            observation, agent_id
        )
        opportunities.extend(conversation_opps)

        # Detect helping opportunities
        helping_opps = self._detect_helping_opportunities(observation, agent_id)
        opportunities.extend(helping_opps)

        # Detect resource sharing opportunities
        resource_opps = self._detect_resource_sharing(observation, agent_id)
        opportunities.extend(resource_opps)

        return opportunities

    def _detect_spatial_clustering(
        self, observation: View, agent_id: str
    ) -> list[CollaborationOpportunity]:
        """Detect opportunities based on spatial proximity.

        Args:
            observation: Current observation
            agent_id: Observing agent ID

        Returns:
            List of spatial collaboration opportunities
        """
        opportunities = []

        # Get agent's position
        agent_pos = None
        if agent_id in observation.visible_entities:
            entity_data = observation.visible_entities[agent_id]
            if "position" in entity_data:
                pos = entity_data["position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    agent_pos = (
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2]) if len(pos) > 2 else 0.0,
                    )

        if not agent_pos:
            return opportunities

        # Find nearby agents
        nearby_agents = []
        for entity_id, entity_data in observation.visible_entities.items():
            if not entity_id.startswith("agent_") or entity_id == agent_id:
                continue

            if "position" in entity_data:
                pos = entity_data["position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    other_pos = (
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2]) if len(pos) > 2 else 0.0,
                    )

                    distance = self._calculate_distance(agent_pos, other_pos)
                    if distance <= self.proximity_threshold:
                        nearby_agents.append((entity_id, distance, other_pos))

        # Create opportunity if multiple agents are nearby
        if len(nearby_agents) >= 1:
            # Calculate centroid of nearby agents
            all_positions = [agent_pos] + [pos for _, _, pos in nearby_agents]
            centroid = tuple(
                sum(p[i] for p in all_positions) / len(all_positions) for i in range(3)
            )

            opportunities.append(
                CollaborationOpportunity(
                    opportunity_type="spatial_clustering",
                    description=f"{len(nearby_agents) + 1} agents are in close proximity",
                    involved_agents=[agent_id] + [aid for aid, _, _ in nearby_agents],
                    location=centroid,
                    priority=min(10, 3 + len(nearby_agents)),
                    metadata={
                        "agent_distances": {
                            aid: dist for aid, dist, _ in nearby_agents
                        },
                        "cluster_size": len(nearby_agents) + 1,
                    },
                )
            )

        return opportunities

    def _detect_task_collaboration(
        self, observation: View, agent_id: str
    ) -> list[CollaborationOpportunity]:
        """Detect opportunities for task-based collaboration.

        Args:
            observation: Current observation
            agent_id: Observing agent ID

        Returns:
            List of task collaboration opportunities
        """
        opportunities = []

        # Look for task events or collaborative tasks in the observation
        for entity_id, entity_data in observation.visible_entities.items():
            if isinstance(entity_data, dict):
                # Check for task events
                if entity_data.get("type") == "task" or "task" in entity_id.lower():
                    collaboration_required = entity_data.get(
                        "collaboration_required", False
                    )
                    difficulty = entity_data.get("difficulty", "medium")

                    if collaboration_required:
                        # Find agents who could help with this task
                        potential_helpers = self._find_capable_agents(
                            observation, entity_data, agent_id
                        )

                        if potential_helpers:
                            opportunities.append(
                                CollaborationOpportunity(
                                    opportunity_type="task_collaboration",
                                    description=f"Task '{entity_data.get('description', entity_id)}' requires collaboration",
                                    involved_agents=[agent_id] + potential_helpers,
                                    location=self._extract_position(entity_data),
                                    priority=self._task_priority_from_difficulty(
                                        difficulty
                                    ),
                                    metadata={
                                        "task_id": entity_id,
                                        "difficulty": difficulty,
                                        "task_data": entity_data,
                                    },
                                )
                            )

        return opportunities

    def _detect_conversation_opportunities(
        self, observation: View, agent_id: str
    ) -> list[CollaborationOpportunity]:
        """Detect opportunities for group conversations.

        Args:
            observation: Current observation
            agent_id: Observing agent ID

        Returns:
            List of conversation opportunities
        """
        opportunities = []

        # Look for recent messages or active conversations
        recent_speakers = set()
        conversation_topics = []

        for entity_id, entity_data in observation.visible_entities.items():
            if not isinstance(entity_data, dict):
                continue

            # Check for recent messages
            if "recent_message" in entity_data:
                recent_speakers.add(entity_id)
                message_data = entity_data["recent_message"]
                if isinstance(message_data, dict):
                    text = message_data.get("text", "")
                    if text:
                        conversation_topics.append(text)

        # If multiple agents are speaking, there's a conversation opportunity
        if len(recent_speakers) >= 2:
            opportunities.append(
                CollaborationOpportunity(
                    opportunity_type="group_conversation",
                    description=f"Active conversation with {len(recent_speakers)} participants",
                    involved_agents=[agent_id] + list(recent_speakers),
                    priority=6,
                    metadata={
                        "speaker_count": len(recent_speakers),
                        "topics": conversation_topics[-5:],  # Last 5 topics
                    },
                )
            )

        return opportunities

    def _detect_helping_opportunities(
        self, observation: View, agent_id: str
    ) -> list[CollaborationOpportunity]:
        """Detect opportunities to help other agents.

        Args:
            observation: Current observation
            agent_id: Observing agent ID

        Returns:
            List of helping opportunities
        """
        opportunities = []

        # Look for agents that might need help
        for entity_id, entity_data in observation.visible_entities.items():
            if not entity_id.startswith("agent_") or entity_id == agent_id:
                continue

            if not isinstance(entity_data, dict):
                continue

            # Check for distress signals or help requests
            needs_help = False
            help_reason = ""

            if "status" in entity_data:
                status = entity_data["status"]
                if isinstance(status, str) and any(
                    word in status.lower()
                    for word in ["stuck", "blocked", "need help", "struggling"]
                ):
                    needs_help = True
                    help_reason = status

            if "recent_message" in entity_data:
                message_data = entity_data["recent_message"]
                if isinstance(message_data, dict):
                    text = message_data.get("text", "")
                    if any(
                        word in text.lower()
                        for word in ["help", "assist", "stuck", "can't", "unable"]
                    ):
                        needs_help = True
                        help_reason = text

            if needs_help:
                opportunities.append(
                    CollaborationOpportunity(
                        opportunity_type="helping",
                        description=f"Agent {entity_id} may need assistance: {help_reason}",
                        involved_agents=[agent_id, entity_id],
                        location=self._extract_position(entity_data),
                        priority=8,
                        metadata={
                            "target_agent": entity_id,
                            "help_reason": help_reason,
                        },
                    )
                )

        return opportunities

    def _detect_resource_sharing(
        self, observation: View, agent_id: str
    ) -> list[CollaborationOpportunity]:
        """Detect opportunities for resource sharing.

        Args:
            observation: Current observation
            agent_id: Observing agent ID

        Returns:
            List of resource sharing opportunities
        """
        opportunities = []

        # Look for resources that could be shared
        for entity_id, entity_data in observation.visible_entities.items():
            if not isinstance(entity_data, dict):
                continue

            # Check for resource entities
            entity_type = entity_data.get("type", "")
            if entity_type in ["resource", "shared_resource", "depot"]:
                # Find nearby agents who could share this resource
                resource_pos = self._extract_position(entity_data)
                if resource_pos:
                    nearby_agents = self._find_agents_near_location(
                        observation, resource_pos, agent_id
                    )

                    if len(nearby_agents) >= 1:
                        opportunities.append(
                            CollaborationOpportunity(
                                opportunity_type="resource_sharing",
                                description=f"Resource '{entity_data.get('name', entity_id)}' available for sharing",
                                involved_agents=[agent_id] + nearby_agents,
                                location=resource_pos,
                                priority=5,
                                metadata={
                                    "resource_id": entity_id,
                                    "resource_type": entity_type,
                                    "resource_data": entity_data,
                                },
                            )
                        )

        return opportunities

    def _find_capable_agents(
        self, observation: View, task_data: dict[str, Any], agent_id: str
    ) -> list[str]:
        """Find agents capable of helping with a task.

        Args:
            observation: Current observation
            task_data: Task information
            agent_id: Observing agent ID

        Returns:
            List of capable agent IDs
        """
        capable_agents = []

        # Simple heuristic: nearby agents are considered capable
        task_pos = self._extract_position(task_data)
        if task_pos:
            capable_agents = self._find_agents_near_location(
                observation, task_pos, agent_id
            )

        return capable_agents

    def _find_agents_near_location(
        self,
        observation: View,
        location: tuple[float, float, float],
        exclude_agent: str | None = None,
    ) -> list[str]:
        """Find agents near a specific location.

        Args:
            observation: Current observation
            location: Target location
            exclude_agent: Agent ID to exclude from results

        Returns:
            List of nearby agent IDs
        """
        nearby_agents = []

        for entity_id, entity_data in observation.visible_entities.items():
            if not entity_id.startswith("agent_"):
                continue

            if exclude_agent and entity_id == exclude_agent:
                continue

            entity_pos = self._extract_position(entity_data)
            if entity_pos:
                distance = self._calculate_distance(location, entity_pos)
                if distance <= self.proximity_threshold:
                    nearby_agents.append(entity_id)

        return nearby_agents

    def _extract_position(
        self, entity_data: dict[str, Any]
    ) -> tuple[float, float, float] | None:
        """Extract position from entity data.

        Args:
            entity_data: Entity data dictionary

        Returns:
            Position tuple or None if not available
        """
        if "position" in entity_data:
            pos = entity_data["position"]
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                return (
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]) if len(pos) > 2 else 0.0,
                )
        return None

    def _calculate_distance(
        self, pos1: tuple[float, float, float], pos2: tuple[float, float, float]
    ) -> float:
        """Calculate Euclidean distance between two positions.

        Args:
            pos1: First position
            pos2: Second position

        Returns:
            Distance between positions
        """
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2, strict=False)) ** 0.5

    def _task_priority_from_difficulty(self, difficulty: str) -> int:
        """Convert task difficulty to priority level.

        Args:
            difficulty: Difficulty level string

        Returns:
            Priority level (0-10)
        """
        difficulty_map = {
            "easy": 3,
            "medium": 5,
            "hard": 7,
            "very_hard": 9,
        }
        return difficulty_map.get(difficulty.lower(), 5)


class CoordinationPatternTracker:
    """Tracks active coordination patterns between agents.

    This class maintains state about ongoing coordination patterns,
    allowing agents to recognize and participate in emergent behaviors
    like following, helping, or group activities.
    """

    def __init__(self):
        """Initialize coordination pattern tracker."""
        self.active_patterns: dict[str, CoordinationPattern] = {}
        self.pattern_history: list[CoordinationPattern] = []
        self.max_history_size = 100

    def start_pattern(
        self,
        pattern_type: str,
        initiator: str,
        participants: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Start tracking a new coordination pattern.

        Args:
            pattern_type: Type of coordination pattern
            initiator: Agent that initiated the pattern
            participants: List of participating agents
            metadata: Additional pattern information

        Returns:
            Pattern ID for tracking
        """
        import time
        import uuid

        pattern_id = str(uuid.uuid4())
        pattern = CoordinationPattern(
            pattern_type=pattern_type,
            initiator=initiator,
            participants=participants,
            status="active",
            start_time=time.time(),
            metadata=metadata or {},
        )

        self.active_patterns[pattern_id] = pattern
        return pattern_id

    def update_pattern(
        self,
        pattern_id: str,
        status: str | None = None,
        participants: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing coordination pattern.

        Args:
            pattern_id: ID of pattern to update
            status: New status (if changing)
            participants: Updated participant list (if changing)
            metadata: Additional metadata to merge

        Returns:
            True if pattern was updated, False if not found
        """
        if pattern_id not in self.active_patterns:
            return False

        pattern = self.active_patterns[pattern_id]

        if status is not None:
            pattern.status = status

        if participants is not None:
            pattern.participants = participants

        if metadata is not None:
            if pattern.metadata is None:
                pattern.metadata = {}
            pattern.metadata.update(metadata)

        # Move to history if completed or failed
        if status in ["completed", "failed"]:
            self.pattern_history.append(pattern)
            del self.active_patterns[pattern_id]

            # Trim history if needed
            if len(self.pattern_history) > self.max_history_size:
                self.pattern_history = self.pattern_history[-self.max_history_size :]

        return True

    def get_active_patterns(
        self, agent_id: str | None = None, pattern_type: str | None = None
    ) -> list[tuple[str, CoordinationPattern]]:
        """Get active coordination patterns.

        Args:
            agent_id: Filter by agent participation (optional)
            pattern_type: Filter by pattern type (optional)

        Returns:
            List of (pattern_id, pattern) tuples
        """
        patterns = []

        for pattern_id, pattern in self.active_patterns.items():
            # Filter by agent if specified
            if agent_id is not None:
                if (
                    agent_id not in pattern.participants
                    and agent_id != pattern.initiator
                ):
                    continue

            # Filter by type if specified
            if pattern_type is not None:
                if pattern.pattern_type != pattern_type:
                    continue

            patterns.append((pattern_id, pattern))

        return patterns

    def is_agent_coordinating(self, agent_id: str) -> bool:
        """Check if an agent is currently involved in any coordination.

        Args:
            agent_id: Agent ID to check

        Returns:
            True if agent is coordinating, False otherwise
        """
        for pattern in self.active_patterns.values():
            if agent_id in pattern.participants or agent_id == pattern.initiator:
                return True
        return False

    def get_coordination_partners(self, agent_id: str) -> set[str]:
        """Get all agents currently coordinating with the given agent.

        Args:
            agent_id: Agent ID to check

        Returns:
            Set of agent IDs coordinating with this agent
        """
        partners = set()

        for pattern in self.active_patterns.values():
            if agent_id in pattern.participants or agent_id == pattern.initiator:
                # Add all other participants
                partners.update(pattern.participants)
                partners.add(pattern.initiator)

        # Remove the agent itself
        partners.discard(agent_id)
        return partners


def detect_following_pattern(
    observation: View, agent_id: str, target_agent: str, distance_threshold: float = 5.0
) -> bool:
    """Detect if an agent is following another agent.

    Args:
        observation: Current observation
        agent_id: ID of potential follower
        target_agent: ID of potential leader
        distance_threshold: Maximum distance to be considered following

    Returns:
        True if following pattern is detected
    """
    # Get positions
    agent_pos = None
    target_pos = None

    if agent_id in observation.visible_entities:
        entity_data = observation.visible_entities[agent_id]
        if "position" in entity_data:
            pos = entity_data["position"]
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                agent_pos = (
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]) if len(pos) > 2 else 0.0,
                )

    if target_agent in observation.visible_entities:
        entity_data = observation.visible_entities[target_agent]
        if "position" in entity_data:
            pos = entity_data["position"]
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                target_pos = (
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]) if len(pos) > 2 else 0.0,
                )

    if not agent_pos or not target_pos:
        return False

    # Calculate distance
    distance = (
        sum((a - b) ** 2 for a, b in zip(agent_pos, target_pos, strict=False)) ** 0.5
    )

    # Following if within threshold
    return distance <= distance_threshold


def suggest_collaborative_action(
    opportunity: CollaborationOpportunity,
    agent_id: str,
    agent_position: tuple[float, float, float] | None,
) -> dict[str, Any]:
    """Suggest an action to participate in a collaboration opportunity.

    Args:
        opportunity: Detected collaboration opportunity
        agent_id: ID of the agent
        agent_position: Current position of the agent

    Returns:
        Dictionary with suggested action details
    """
    if opportunity.opportunity_type == "spatial_clustering":
        # Suggest moving to the cluster center or initiating conversation
        if opportunity.location and agent_position:
            distance = (
                sum(
                    (a - b) ** 2
                    for a, b in zip(agent_position, opportunity.location, strict=False)
                )
                ** 0.5
            )

            if distance > 2.0:
                return {
                    "action_type": "move",
                    "target_position": list(opportunity.location),
                    "reasoning": "Moving to join nearby agents",
                }
            else:
                return {
                    "action_type": "speak",
                    "text": "Hello everyone! I see we're all nearby. Anyone want to collaborate?",
                    "reasoning": "Initiating group conversation",
                }

    elif opportunity.opportunity_type == "task_collaboration":
        # Suggest offering to help with the task
        task_desc = opportunity.metadata.get("task_data", {}).get("description", "task")
        return {
            "action_type": "speak",
            "text": f"I can help with {task_desc}. What needs to be done?",
            "reasoning": "Offering to collaborate on task",
        }

    elif opportunity.opportunity_type == "group_conversation":
        # Suggest joining the conversation
        return {
            "action_type": "speak",
            "text": "I'd like to join this conversation. What are we discussing?",
            "reasoning": "Joining group conversation",
        }

    elif opportunity.opportunity_type == "helping":
        # Suggest offering assistance
        target = opportunity.metadata.get("target_agent", "someone")
        return {
            "action_type": "speak",
            "text": f"Hey {target}, I noticed you might need help. How can I assist?",
            "reasoning": "Offering help to another agent",
        }

    elif opportunity.opportunity_type == "resource_sharing":
        # Suggest coordinating resource use
        resource_name = opportunity.metadata.get("resource_data", {}).get(
            "name", "resource"
        )
        return {
            "action_type": "speak",
            "text": f"I see the {resource_name} is available. Should we coordinate how to use it?",
            "reasoning": "Proposing resource coordination",
        }

    # Default: observe and wait
    return {
        "action_type": "wait",
        "reasoning": "Observing collaboration opportunity",
    }
