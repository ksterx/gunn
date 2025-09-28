"""Observation policies for partial observation and view filtering.

This module implements the ObservationPolicy interface that controls what agents
can observe in the simulation world, including distance-based filtering,
relationship constraints, and efficient observation delta generation using
RFC6902 JSON Patch operations.
"""

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any, Protocol

import jsonpatch
from pydantic import BaseModel, Field

from gunn.schemas.messages import View, WorldState
from gunn.schemas.types import Effect, ObservationDelta
from gunn.utils.hashing import canonical_json


class LatencyModel(Protocol):
    """Protocol for calculating observation delivery latency."""

    def calculate_delay(self, from_agent: str, to_agent: str, effect: Effect) -> float:
        """Calculate delivery delay in seconds for an observation."""
        ...


class PolicyConfig(BaseModel):
    """Configuration for observation policies."""

    distance_limit: float = Field(
        default=100.0,
        ge=0.0,
        description="Maximum distance for spatial observation",
    )
    relationship_filter: list[str] = Field(
        default_factory=list,
        description="List of relationship types to include in observations",
    )
    field_visibility: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-field visibility settings",
    )
    max_patch_ops: int = Field(
        default=50,
        ge=1,
        description="Maximum JSON patch operations before fallback to full snapshot",
    )
    include_spatial_index: bool = Field(
        default=True,
        description="Whether to include spatial coordinates in observations",
    )
    relationship_depth: int = Field(
        default=2,
        ge=0,
        description="Maximum depth for relationship traversal",
    )


class NoLatencyModel:
    """Default latency model with no delay."""

    def calculate_delay(self, from_agent: str, to_agent: str, effect: Effect) -> float:
        """Return zero delay for immediate delivery."""
        return 0.0


class DistanceLatencyModel:
    """Latency model based on spatial distance between agents."""

    def __init__(self, base_latency: float = 0.01, distance_factor: float = 0.001):
        """Initialize distance-based latency model.

        Args:
            base_latency: Base latency in seconds
            distance_factor: Additional latency per unit distance
        """
        self.base_latency = base_latency
        self.distance_factor = distance_factor

    def calculate_delay(self, from_agent: str, to_agent: str, effect: Effect) -> float:
        """Calculate latency based on distance between agents."""
        # In a real implementation, this would look up agent positions
        # For now, return base latency
        return self.base_latency


class ObservationPolicy(ABC):
    """Abstract base class for observation policies.

    Observation policies control what agents can see in the simulation world,
    including spatial filtering, relationship constraints, and field visibility.
    """

    def __init__(self, config: PolicyConfig):
        """Initialize observation policy with configuration.

        Args:
            config: Policy configuration parameters
        """
        self.config = config
        self.latency_model: LatencyModel = NoLatencyModel()

    def set_latency_model(self, latency_model: LatencyModel) -> None:
        """Set the latency model for observation delivery.

        Args:
            latency_model: Latency model to use for calculating delays
        """
        self.latency_model = latency_model

    @abstractmethod
    def filter_world_state(self, world_state: WorldState, agent_id: str) -> View:
        """Generate agent's filtered view of the world state.

        Args:
            world_state: Complete world state
            agent_id: Agent requesting the view

        Returns:
            Filtered view containing only what the agent should see
        """
        ...

    @abstractmethod
    def should_observe_event(
        self, effect: Effect, agent_id: str, world_state: WorldState
    ) -> bool:
        """Determine if an agent should observe a specific effect.

        Args:
            effect: Effect that occurred
            agent_id: Agent to check observation for
            world_state: Current world state

        Returns:
            True if agent should observe this effect
        """
        ...

    def calculate_observation_delta(
        self, old_view: View, new_view: View
    ) -> ObservationDelta:
        """Generate RFC6902 JSON Patch between two views.

        Args:
            old_view: Previous view state
            new_view: New view state

        Returns:
            ObservationDelta with JSON patch operations

        Raises:
            ValueError: If views are for different agents
        """
        if old_view.agent_id != new_view.agent_id:
            raise ValueError("Cannot generate delta between views for different agents")

        # Convert views to dictionaries for patch generation
        old_dict = old_view.model_dump()
        new_dict = new_view.model_dump()

        # Remove view_seq and context_digest from patch calculation
        # as these are metadata that shouldn't be patched
        old_dict.pop("view_seq", None)
        old_dict.pop("context_digest", None)
        new_dict.pop("view_seq", None)
        new_dict.pop("context_digest", None)

        # Generate JSON patch
        patches = jsonpatch.make_patch(old_dict, new_dict)
        patch_ops = list(patches)

        # Check if patch is too large and fallback to full snapshot
        if len(patch_ops) > self.config.max_patch_ops:
            # Return a "replace" operation for the entire visible state
            patch_ops = [
                {
                    "op": "replace",
                    "path": "/visible_entities",
                    "value": new_view.visible_entities,
                },
                {
                    "op": "replace",
                    "path": "/visible_relationships",
                    "value": new_view.visible_relationships,
                },
            ]

        return ObservationDelta(
            view_seq=new_view.view_seq,
            patches=patch_ops,
            context_digest=new_view.context_digest,
            schema_version="1.0.0",
        )

    def _calculate_distance(
        self, pos1: tuple[float, float, float], pos2: tuple[float, float, float]
    ) -> float:
        """Calculate Euclidean distance between two 3D positions.

        Args:
            pos1: First position (x, y, z)
            pos2: Second position (x, y, z)

        Returns:
            Euclidean distance between positions
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _generate_context_digest(self, view_data: dict[str, Any]) -> str:
        """Generate SHA-256 hash digest of view context.

        Args:
            view_data: View data to hash

        Returns:
            SHA-256 hex digest of the view context
        """
        # Create a copy without view_seq for stable hashing
        stable_data = view_data.copy()
        stable_data.pop("view_seq", None)
        stable_data.pop("context_digest", None)

        canonical_bytes = canonical_json(stable_data)
        return hashlib.sha256(canonical_bytes).hexdigest()


class DefaultObservationPolicy(ObservationPolicy):
    """Default observation policy with distance and relationship filtering.

    This policy implements:
    - Distance-based spatial filtering
    - Relationship-based entity filtering
    - Field-level visibility controls
    - Stable path generation for JSON patches
    """

    def filter_world_state(self, world_state: WorldState, agent_id: str) -> View:
        """Generate agent's filtered view with distance and relationship constraints.

        Args:
            world_state: Complete world state
            agent_id: Agent requesting the view

        Returns:
            Filtered view containing only observable entities and relationships
        """
        # Get agent's position for distance calculations
        agent_position = world_state.spatial_index.get(agent_id)
        if agent_position is None:
            # Agent has no position, can't do spatial filtering
            agent_position = (0.0, 0.0, 0.0)

        visible_entities: dict[str, Any] = {}
        visible_relationships: dict[str, list[str]] = {}

        # Filter entities based on distance and relationships
        for entity_id, entity_data in world_state.entities.items():
            if self._should_observe_entity(
                entity_id, entity_data, agent_id, agent_position, world_state
            ):
                # Apply field-level filtering
                filtered_entity = self._filter_entity_fields(entity_data)
                visible_entities[entity_id] = filtered_entity

                # Include relationships for visible entities
                if entity_id in world_state.relationships:
                    # Filter relationships to only include visible targets
                    filtered_relations = [
                        target_id
                        for target_id in world_state.relationships[entity_id]
                        if self._should_observe_relationship(
                            entity_id, target_id, agent_id, world_state
                        )
                    ]
                    if filtered_relations:
                        visible_relationships[entity_id] = filtered_relations

        # Calculate context digest
        view_data_for_digest = {
            "agent_id": agent_id,
            "view_seq": 0,
            "visible_entities": visible_entities,
            "visible_relationships": visible_relationships,
        }
        context_digest = self._generate_context_digest(view_data_for_digest)

        return View(
            agent_id=agent_id,
            view_seq=0,  # Will be set by caller
            visible_entities=visible_entities,
            visible_relationships=visible_relationships,
            context_digest=context_digest,
        )

    def should_observe_event(
        self, effect: Effect, agent_id: str, world_state: WorldState
    ) -> bool:
        """Determine if agent should observe an effect based on policy rules.

        Args:
            effect: Effect that occurred
            agent_id: Agent to check observation for
            world_state: Current world state

        Returns:
            True if agent should observe this effect
        """
        # Always observe effects that directly involve the agent
        if effect.get("source_id") == agent_id:
            return True

        source_id = effect.get("source_id")
        if not source_id:
            return False

        # Check if agent can observe the source entity of the effect
        agent_position = world_state.spatial_index.get(agent_id, (0.0, 0.0, 0.0))
        source_entity_data = world_state.entities.get(source_id, {})

        # Check if agent can observe the source entity
        can_observe_source = self._should_observe_entity(
            source_id,
            source_entity_data,
            agent_id,
            agent_position,
            world_state,
        )

        if can_observe_source:
            return True

        # Check if effect involves entities the agent can observe
        effect_payload = effect.get("payload", {})

        # For spatial effects, check distance
        if "position" in effect_payload:
            effect_position = effect_payload["position"]
            spatial_position: tuple[
                float, float, float
            ] | None = world_state.spatial_index.get(agent_id)

            if spatial_position is not None and isinstance(
                effect_position, list | tuple
            ):
                if len(effect_position) >= 3:
                    distance = self._calculate_distance(
                        spatial_position, tuple(effect_position[:3])
                    )
                    if distance <= self.config.distance_limit:
                        return True

        # For entity-specific effects, check if agent can observe the entity
        if "entity_id" in effect_payload:
            target_entity = effect_payload["entity_id"]
            if target_entity in world_state.entities:
                return self._should_observe_entity(
                    target_entity,
                    world_state.entities[target_entity],
                    agent_id,
                    world_state.spatial_index.get(agent_id, (0.0, 0.0, 0.0)),
                    world_state,
                )

        # For relationship effects, check if agent can observe the relationship
        if "from_entity" in effect_payload and "to_entity" in effect_payload:
            from_entity = effect_payload["from_entity"]
            to_entity = effect_payload["to_entity"]
            return self._should_observe_relationship(
                from_entity, to_entity, agent_id, world_state
            )

        # Default: don't observe effects that don't match any criteria
        return False

    def _should_observe_entity(
        self,
        entity_id: str,
        entity_data: Any,
        agent_id: str,
        agent_position: tuple[float, float, float],
        world_state: WorldState,
    ) -> bool:
        """Check if agent should observe a specific entity.

        Args:
            entity_id: ID of entity to check
            entity_data: Entity data
            agent_id: Observing agent ID
            agent_position: Agent's position
            world_state: Current world state

        Returns:
            True if agent should observe this entity
        """
        # Always observe self
        if entity_id == agent_id:
            return True

        # Unknown entities are not observable unless explicitly configured
        if entity_id not in world_state.entities:
            return False

        # Check distance constraint
        entity_position = world_state.spatial_index.get(entity_id)
        if entity_position:
            distance = self._calculate_distance(agent_position, entity_position)
            if distance > self.config.distance_limit:
                return False

        # Check relationship constraints
        if self.config.relationship_filter:
            # Agent must have a relationship of the specified type
            agent_relationships = world_state.relationships.get(agent_id, [])
            if entity_id not in agent_relationships:
                # Check if there's an indirect relationship within depth limit
                if not self._has_relationship_path(
                    agent_id, entity_id, world_state, self.config.relationship_depth
                ):
                    return False

        return True

    def _should_observe_relationship(
        self, from_entity: str, to_entity: str, agent_id: str, world_state: WorldState
    ) -> bool:
        """Check if agent should observe a relationship between entities.

        Args:
            from_entity: Source entity of relationship
            to_entity: Target entity of relationship
            agent_id: Observing agent ID
            world_state: Current world state

        Returns:
            True if agent should observe this relationship
        """
        # Agent must be able to observe both entities
        agent_position = world_state.spatial_index.get(agent_id, (0.0, 0.0, 0.0))

        can_observe_from = self._should_observe_entity(
            from_entity,
            world_state.entities.get(from_entity, {}),
            agent_id,
            agent_position,
            world_state,
        )

        can_observe_to = self._should_observe_entity(
            to_entity,
            world_state.entities.get(to_entity, {}),
            agent_id,
            agent_position,
            world_state,
        )

        return can_observe_from and can_observe_to

    def _has_relationship_path(
        self,
        from_entity: str,
        to_entity: str,
        world_state: WorldState,
        max_depth: int,
        visited: set[str] | None = None,
    ) -> bool:
        """Check if there's a relationship path between entities within max depth.

        Args:
            from_entity: Starting entity
            to_entity: Target entity
            world_state: Current world state
            max_depth: Maximum relationship depth to search
            visited: Set of already visited entities (for cycle detection)

        Returns:
            True if a relationship path exists within max_depth
        """
        if max_depth <= 0:
            return False

        if visited is None:
            visited = set()

        if from_entity in visited:
            return False  # Cycle detected

        if from_entity == to_entity:
            return True

        visited.add(from_entity)

        # Check direct relationships
        relationships = world_state.relationships.get(from_entity, [])
        if to_entity in relationships:
            return True

        # Check indirect relationships
        for related_entity in relationships:
            if self._has_relationship_path(
                related_entity, to_entity, world_state, max_depth - 1, visited.copy()
            ):
                return True

        return False

    def _filter_entity_fields(self, entity_data: Any) -> Any:
        """Apply field-level visibility filtering to entity data.

        Args:
            entity_data: Raw entity data

        Returns:
            Filtered entity data based on field visibility settings
        """
        if not isinstance(entity_data, dict):
            return entity_data

        if not self.config.field_visibility:
            return entity_data  # No field filtering configured

        filtered_data = {}
        for field_name, field_value in entity_data.items():
            # Check if field should be visible (default to True if not specified)
            if self.config.field_visibility.get(field_name, True):
                filtered_data[field_name] = field_value

        return filtered_data


class ConversationObservationPolicy(ObservationPolicy):
    """Observation policy optimized for conversation scenarios.

    This policy is designed for multi-agent conversations where agents
    should observe speaking events and participant changes, but may have
    limited spatial awareness.
    """

    def filter_world_state(self, world_state: WorldState, agent_id: str) -> View:
        """Generate conversation-focused view of the world state.

        Args:
            world_state: Complete world state
            agent_id: Agent requesting the view

        Returns:
            View focused on conversation participants and recent messages
        """
        visible_entities: dict[str, Any] = {}
        visible_relationships: dict[str, list[str]] = {}

        # Include self
        if agent_id in world_state.entities:
            visible_entities[agent_id] = world_state.entities[agent_id]

        # Include conversation participants (entities with "participant" relationship)
        agent_relationships = world_state.relationships.get(agent_id, [])
        for related_entity in agent_relationships:
            if related_entity in world_state.entities:
                entity_data = world_state.entities[related_entity]
                # Include if it's a conversation participant
                if isinstance(entity_data, dict) and entity_data.get("type") == "agent":
                    visible_entities[related_entity] = entity_data

        # Include relationships between visible entities
        for entity_id in visible_entities:
            if entity_id in world_state.relationships:
                filtered_relations = [
                    target_id
                    for target_id in world_state.relationships[entity_id]
                    if target_id in visible_entities
                ]
                if filtered_relations:
                    visible_relationships[entity_id] = filtered_relations

        # Calculate context digest
        view_data_for_digest = {
            "agent_id": agent_id,
            "view_seq": 0,
            "visible_entities": visible_entities,
            "visible_relationships": visible_relationships,
        }
        context_digest = self._generate_context_digest(view_data_for_digest)

        return View(
            agent_id=agent_id,
            view_seq=0,
            visible_entities=visible_entities,
            visible_relationships=visible_relationships,
            context_digest=context_digest,
        )

    def should_observe_event(
        self, effect: Effect, agent_id: str, world_state: WorldState
    ) -> bool:
        """Determine if agent should observe conversation-related effects.

        Args:
            effect: Effect that occurred
            agent_id: Agent to check observation for
            world_state: Current world state

        Returns:
            True if agent should observe this conversation effect
        """
        # Always observe own effects
        if effect.get("source_id") == agent_id:
            return True

        # Observe speaking effects from conversation participants
        if effect.get("kind") in ["Speak", "SpeakResponse", "MessageEmitted"]:
            source_id = effect.get("source_id")
            if source_id:
                # Check if source is a conversation participant
                agent_relationships = world_state.relationships.get(agent_id, [])
                return source_id in agent_relationships

        # Observe participant join/leave effects
        if effect.get("kind") in ["ParticipantJoined", "ParticipantLeft"]:
            return True

        return False


# Factory function for creating observation policies
def create_observation_policy(
    policy_type: str, config: PolicyConfig
) -> ObservationPolicy:
    """Create an observation policy of the specified type.

    Args:
        policy_type: Type of policy to create ("default" or "conversation")
        config: Policy configuration

    Returns:
        Configured observation policy instance

    Raises:
        ValueError: If policy_type is not recognized
    """
    if policy_type == "default":
        return DefaultObservationPolicy(config)
    elif policy_type == "conversation":
        return ConversationObservationPolicy(config)
    else:
        raise ValueError(f"Unknown observation policy type: {policy_type}")
