"""
Gunn orchestrator integration and observation policies.

This module contains the wrapper around Gunn's Orchestrator and
implements battle-specific observation policies with team-based filtering,
fog of war, and team communication visibility.
"""

import asyncio
import hashlib
import math
from typing import Any

from gunn.policies.observation import ObservationPolicy, PolicyConfig
from gunn.schemas.messages import View, WorldState
from gunn.schemas.types import Effect
from gunn.utils.hashing import canonical_json
from gunn.utils.telemetry import get_logger

from ..shared.models import BattleWorldState
from .effect_processor import EffectProcessor, GameStatusManager


class BattleObservationPolicy(ObservationPolicy):
    """Team-based observation policy with fog of war and team communication filtering.

    This policy implements:
    - Team-based visibility (always see teammates)
    - Fog of war for enemies (limited by vision range)
    - Team-only communication visibility
    - Map location visibility (always visible)
    - Partial enemy information (position and status only)
    """

    def __init__(
        self, team: str, vision_range: float = 30.0, communication_range: float = 50.0
    ):
        """Initialize battle observation policy for a specific team.

        Args:
            team: Team identifier ("team_a" or "team_b")
            vision_range: Maximum distance to see enemies
            communication_range: Maximum distance for team communication
        """
        # Create policy config with battle-specific settings
        config = PolicyConfig(
            distance_limit=vision_range,
            relationship_filter=[],  # No relationship filtering needed
            field_visibility={},  # No field filtering needed
            max_patch_ops=50,
            include_spatial_index=True,
            relationship_depth=1,
        )

        super().__init__(config)

        if team not in ["team_a", "team_b"]:
            raise ValueError(f"Invalid team: {team}. Must be 'team_a' or 'team_b'")

        self.team = team
        self.vision_range = vision_range
        self.communication_range = communication_range

    def filter_world_state(self, world_state: WorldState, agent_id: str) -> View:
        """Generate agent's filtered view based on team membership and vision range.

        Args:
            world_state: Complete world state from Gunn
            agent_id: Agent requesting the view

        Returns:
            Filtered view containing only what the agent should see
        """
        # Get agent's position for distance calculations
        agent_position = world_state.spatial_index.get(agent_id)
        if agent_position is None:
            # Agent has no position, use default
            agent_position = (0.0, 0.0, 0.0)

        visible_entities: dict[str, Any] = {}
        visible_relationships: dict[str, list[str]] = {}

        # Filter entities based on team membership and vision range
        for entity_id, entity_data in world_state.entities.items():
            if self._should_observe_entity(
                entity_id, entity_data, agent_id, agent_position, world_state
            ):
                # Apply entity filtering based on type
                filtered_entity = self._filter_entity_data(
                    entity_id, entity_data, agent_id
                )
                if filtered_entity is not None:
                    visible_entities[entity_id] = filtered_entity

        # Always include team communications in visible entities
        team_communications = self._get_team_communications(world_state)
        visible_entities["team_communications"] = {
            "type": "communications",
            "team": self.team,
            "messages": team_communications,
        }

        # Include relationships for visible entities (simplified for battle demo)
        for entity_id in visible_entities:
            if entity_id in world_state.relationships:
                # Only include relationships to other visible entities
                filtered_relations = [
                    target_id
                    for target_id in world_state.relationships[entity_id]
                    if target_id in visible_entities
                ]
                if filtered_relations:
                    visible_relationships[entity_id] = filtered_relations

        # Calculate context digest
        view_data_for_digest: dict[str, Any] = {
            "agent_id": agent_id,
            "view_seq": 0,
            "visible_entities": visible_entities,
            "visible_relationships": visible_relationships,
        }
        context_digest = self._generate_context_digest(view_data_for_digest)

        return View(
            agent_id=agent_id,
            view_seq=0,  # Will be set by Gunn orchestrator
            visible_entities=visible_entities,
            visible_relationships=visible_relationships,
            context_digest=context_digest,
        )

    def should_observe_event(
        self, effect: Effect, agent_id: str, world_state: WorldState
    ) -> bool:
        """Determine if agent should observe a specific effect.

        Args:
            effect: Effect that occurred
            agent_id: Agent to check observation for
            world_state: Current world state

        Returns:
            True if agent should observe this effect
        """
        # Always observe own effects
        if effect.get("source_id") == agent_id:
            return True

        # Check for team communication effects (special handling)
        effect_kind = effect.get("kind", "")
        if effect_kind in ["TeamMessage", "CommunicateAction", "Communication"]:
            return self.should_observe_communication(effect, agent_id)

        # Get agent's position for distance calculations
        agent_position = world_state.spatial_index.get(agent_id)
        if agent_position is None:
            agent_position = (0.0, 0.0, 0.0)

        source_id = effect.get("source_id")
        effect_payload = effect.get("payload", {})

        # Primary rule: Can observe effect if we can observe the source
        source_observable = False
        if source_id and source_id in world_state.entities:
            source_entity = world_state.entities[source_id]
            source_observable = self._should_observe_entity(
                source_id, source_entity, agent_id, agent_position, world_state
            )
        elif source_id and source_id in world_state.spatial_index:
            # Source has position but no entity data - check distance
            source_position = world_state.spatial_index[source_id]
            distance = self._calculate_distance_3d(agent_position, source_position)
            source_observable = distance <= self.vision_range

        if source_observable:
            return True

        # Secondary rule: Can observe effect if it happens at a position we can see
        if "position" in effect_payload:
            effect_position = effect_payload["position"]
            if isinstance(effect_position, (list, tuple)) and len(effect_position) >= 2:
                # Convert 2D position to 3D for distance calculation
                effect_pos_3d = (
                    float(effect_position[0]),
                    float(effect_position[1]),
                    0.0 if len(effect_position) < 3 else float(effect_position[2]),
                )
                distance = self._calculate_distance_3d(agent_position, effect_pos_3d)
                if distance <= self.vision_range:
                    return True

        # Tertiary rule: Can observe effect if it targets us (even if we can't see the source)
        # This handles cases like being attacked by an invisible enemy
        for key in ["target_id", "target_agent_id", "entity_id"]:
            if key in effect_payload:
                target_id = effect_payload[key]
                if target_id == agent_id:
                    return True

        return False

    def should_observe_communication(self, effect: Effect, agent_id: str) -> bool:
        """Determine if agent should see a communication effect.

        This method implements team-only message visibility - agents can only
        see communications from their own team members.

        Args:
            effect: Effect to check (should be a communication effect)
            agent_id: Agent to check observation for

        Returns:
            True if agent should see this communication effect
        """
        # Only filter communication effects
        if effect.get("kind") not in [
            "TeamMessage",
            "CommunicateAction",
            "Communication",
        ]:
            return True  # Non-communication effects use normal visibility rules

        # Get sender team from effect payload
        effect_payload = effect.get("payload", {})
        sender_team = effect_payload.get("sender_team")

        # If no sender team specified, check if sender is on our team
        if not sender_team:
            sender_id = effect.get("source_id") or effect_payload.get("sender_id")
            if sender_id:
                # Extract team from agent ID (assumes format like "team_a_agent_1")
                if sender_id.startswith("team_a_"):
                    sender_team = "team_a"
                elif sender_id.startswith("team_b_"):
                    sender_team = "team_b"

        # Only show messages from same team
        return sender_team == self.team

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
            agent_position: Agent's 3D position
            world_state: Current world state

        Returns:
            True if agent should observe this entity
        """
        # Always observe self
        if entity_id == agent_id:
            return True

        # Always observe map locations
        if isinstance(entity_data, dict) and entity_data.get("type") == "map_location":
            return True

        # Check if entity is an agent
        if isinstance(entity_data, dict) and "team" in entity_data:
            entity_team = entity_data.get("team")

            # Always observe teammates
            if entity_team == self.team:
                return True

            # For enemies, check vision range
            entity_position = world_state.spatial_index.get(entity_id)
            if entity_position:
                distance = self._calculate_distance_3d(agent_position, entity_position)
                return distance <= self.vision_range

        # For other entities, use distance-based visibility
        entity_position = world_state.spatial_index.get(entity_id)
        if entity_position:
            distance = self._calculate_distance_3d(agent_position, entity_position)
            return distance <= self.vision_range

        # If no position information, don't show by default
        return False

    def _filter_entity_data(
        self, entity_id: str, entity_data: Any, observer_id: str
    ) -> dict[str, Any] | None:
        """Filter entity data based on observer's permissions.

        Args:
            entity_id: ID of the entity
            entity_data: Raw entity data
            observer_id: ID of the observing agent

        Returns:
            Filtered entity data or None if entity should not be visible
        """
        if not isinstance(entity_data, dict):
            return entity_data

        # Always show full data for self
        if entity_id == observer_id:
            return entity_data

        # Always show full data for map locations
        if entity_data.get("type") == "map_location":
            return entity_data

        # For agents, filter based on team membership
        if "team" in entity_data:
            entity_team = entity_data.get("team")

            # Show full data for teammates
            if entity_team == self.team:
                return entity_data

            # Show limited data for enemies (fog of war)
            return {
                "agent_id": entity_id,
                "team": entity_team,
                "position": entity_data.get("position"),
                "status": entity_data.get("status"),
                "health": entity_data.get(
                    "health"
                ),  # Show health for tactical decisions
                # Hide detailed info like weapon condition, ranges, etc.
            }

        # For other entities, show full data
        return entity_data

    def _get_team_communications(self, world_state: WorldState) -> list[dict[str, Any]]:
        """Extract team communications from world state metadata with urgency-based prioritization.

        Args:
            world_state: Current world state

        Returns:
            List of recent team communications for this team, prioritized by urgency
        """
        metadata = world_state.metadata or {}
        team_communications = metadata.get("team_communications", {})

        # Get communications for this team
        team_messages = team_communications.get(self.team, [])

        # Convert to serializable format first
        serialized_messages = []
        for msg in team_messages:
            if isinstance(msg, dict):
                serialized_messages.append(msg)
            elif hasattr(msg, "model_dump"):  # Pydantic model
                serialized_messages.append(msg.model_dump())
            else:
                # Try to convert to dict
                try:
                    serialized_messages.append(
                        {
                            "sender_id": getattr(msg, "sender_id", "unknown"),
                            "message": getattr(msg, "message", str(msg)),
                            "urgency": getattr(msg, "urgency", "medium"),
                            "timestamp": getattr(msg, "timestamp", 0.0),
                            "team": getattr(msg, "team", self.team),
                        }
                    )
                except:
                    continue

        # Sort by urgency (high -> medium -> low) then by timestamp (newest first)
        urgency_priority = {"high": 3, "medium": 2, "low": 1}

        def sort_key(msg):
            urgency = msg.get("urgency", "medium")
            timestamp = msg.get("timestamp", 0.0)
            return (urgency_priority.get(urgency, 2), timestamp)

        # Sort messages by priority and recency
        sorted_messages = sorted(serialized_messages, key=sort_key, reverse=True)

        # Return last 10 messages (prioritized)
        return sorted_messages[-10:] if sorted_messages else []

    def _calculate_distance_3d(
        self, pos1: tuple[float, float, float], pos2: tuple[float, float, float]
    ) -> float:
        """Calculate 3D Euclidean distance between two positions.

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


class BattleOrchestrator:
    """Wrapper around Gunn orchestrator for battle simulation.

    This class provides the integration layer between the battle demo and Gunn's
    core orchestrator, handling concurrent agent decision processing and intent
    submission with deterministic ordering.
    """

    def __init__(self):
        """Initialize the battle orchestrator."""
        from gunn import Orchestrator, OrchestratorConfig
        from gunn.core.orchestrator import DefaultEffectValidator

        # Configure Gunn orchestrator for battle simulation
        config = OrchestratorConfig(
            max_agents=6,  # 3 per team
            staleness_threshold=1,
            debounce_ms=50.0,
            deadline_ms=3000.0,
            token_budget=500,
            backpressure_policy="defer",
            default_priority=0,
            max_queue_depth=100,
            use_in_memory_dedup=True,  # For demo simplicity
            dedup_ttl_minutes=5,
            max_dedup_entries=1000,
        )

        self.orchestrator = Orchestrator(
            config, world_id="battle_demo", effect_validator=DefaultEffectValidator()
        )

        # Battle-specific state
        self.world_state = BattleWorldState()
        self.ai_decision_maker = None  # Will be set externally
        self._initialized = False

        # Effect processing
        self.effect_processor = EffectProcessor(action_callback=None)
        self.game_status_manager = GameStatusManager()

        # Async agent tasks (independent agent loops)
        self._agent_tasks: list[asyncio.Task] = []

        # Logging
        self._logger = get_logger("battle_orchestrator")

    async def initialize(self, ai_decision_maker, force_reinit: bool = False) -> None:
        """Initialize the orchestrator and register agents with Gunn.

        Note: World state should be initialized separately using GameInitializer
        before calling this method.

        Args:
            ai_decision_maker: AIDecisionMaker instance for agent decisions
            force_reinit: If True, reinitialize even if already initialized
        """
        if self._initialized and not force_reinit:
            return

        self.ai_decision_maker = ai_decision_maker

        # Initialize or reinitialize the orchestrator
        if not self._initialized or force_reinit:
            await self.orchestrator.initialize()

        # Register existing agents with Gunn (world state should already be populated)
        await self._register_agents_with_gunn()

        # Sync with Gunn's world state
        await self._sync_world_state()

        self._initialized = True

    async def reset(self) -> None:
        """Reset the orchestrator to allow reinitialization."""
        self._logger.info("Resetting battle orchestrator")
        self._initialized = False

        # Cancel all agent tasks
        for task in self._agent_tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if self._agent_tasks:
            await asyncio.gather(*self._agent_tasks, return_exceptions=True)

        self._agent_tasks.clear()

        # Clear agent registry in the underlying orchestrator
        # This allows agents to be re-registered without conflicts
        if hasattr(self.orchestrator, "_agents"):
            self.orchestrator._agents.clear()

        self._logger.info("Battle orchestrator reset complete")

    def set_action_callback(self, callback) -> None:
        """Set the callback for action result notifications.

        Args:
            callback: Async function with signature:
                     async def callback(agent_id: str, action_type: str, success: bool, details: str)
        """
        self.effect_processor._action_callback = callback
        self._logger.info("Action callback set for effect processor")

    async def _setup_battle_world(self) -> None:
        """Create initial world state with teams and map locations."""
        from ..shared.enums import LocationType
        from ..shared.models import Agent, MapLocation

        # Create map locations
        forge_a = MapLocation(position=(20.0, 80.0), location_type=LocationType.FORGE)
        forge_b = MapLocation(position=(180.0, 20.0), location_type=LocationType.FORGE)

        self.world_state.map_locations = {"forge_a": forge_a, "forge_b": forge_b}

        # Create teams
        team_a_agents = []
        team_b_agents = []

        for i in range(3):
            # Team A agents
            agent_a = Agent(
                agent_id=f"team_a_agent_{i + 1}",
                team="team_a",
                position=(30.0 + i * 10, 90.0),
            )
            team_a_agents.append(agent_a)

            # Team B agents
            agent_b = Agent(
                agent_id=f"team_b_agent_{i + 1}",
                team="team_b",
                position=(170.0 - i * 10, 10.0),
            )
            team_b_agents.append(agent_b)

        # Register agents with Gunn
        for agent in team_a_agents + team_b_agents:
            policy = BattleObservationPolicy(agent.team, agent.vision_range)
            await self.orchestrator.register_agent(agent.agent_id, policy)

            # Add to world state
            self.world_state.agents[agent.agent_id] = agent

        # Initialize team scores
        self.world_state.team_scores = {"team_a": 0, "team_b": 0}

        # Sync with Gunn's world state
        await self._sync_world_state()

    async def _register_agents_with_gunn(self) -> None:
        """Register all agents in world state with Gunn orchestrator using async agent loops."""
        self._logger.info("Registering agents with Gunn orchestrator (async mode)")

        from .battle_agent import BattleAgent

        self._agent_tasks = []

        for agent_id, agent in self.world_state.agents.items():
            # Create observation policy for this agent
            policy = BattleObservationPolicy(agent.team, agent.vision_range)

            # Register agent with orchestrator
            handle = await self.orchestrator.register_agent(agent_id, policy)

            # Create BattleAgent instance
            battle_agent = BattleAgent(
                agent_id=agent_id,
                ai_decision_maker=self.ai_decision_maker,
                world_state=self.world_state,
            )

            # Start async agent loop for independent execution
            agent_task = asyncio.create_task(
                handle.run_async_loop(battle_agent), name=f"battle_agent_{agent_id}"
            )
            self._agent_tasks.append(agent_task)

            self._logger.debug(
                f"Registered agent {agent_id} with team {agent.team} (async loop started)"
            )

    async def _sync_world_state(self) -> None:
        """Synchronize our world state with Gunn's world state (bidirectional)."""
        # First, sync FROM Gunn TO BattleWorldState (e.g., Move effects)
        for agent_id in self.world_state.agents:
            if agent_id in self.orchestrator.world_state.spatial_index:
                new_pos = self.orchestrator.world_state.spatial_index[agent_id]
                old_pos = self.world_state.agents[agent_id].position

                # Only update if position changed
                if old_pos != (new_pos[0], new_pos[1]):
                    self.world_state.agents[agent_id].position = (
                        new_pos[0],
                        new_pos[1],
                    )
                    self._logger.info(
                        f"Agent {agent_id} position updated: {old_pos} -> {(new_pos[0], new_pos[1])}"
                    )

        # Then, sync FROM BattleWorldState TO Gunn (for other state updates)
        gunn_entities = {}
        spatial_index = {}

        # Add agents with their updated positions
        for agent_id, agent in self.world_state.agents.items():
            gunn_entities[agent_id] = agent.model_dump()
            spatial_index[agent_id] = (*agent.position, 0.0)

        # Add map locations
        for loc_id, location in self.world_state.map_locations.items():
            gunn_entities[loc_id] = {"type": "map_location", **location.model_dump()}
            spatial_index[loc_id] = (*location.position, 0.0)

        # Update Gunn's world state
        self.orchestrator.world_state.entities = gunn_entities
        self.orchestrator.world_state.spatial_index = spatial_index
        self.orchestrator.world_state.metadata = {
            "team_scores": self.world_state.team_scores,
            "game_time": self.world_state.game_time,
            "game_status": self.world_state.game_status,
            "team_communications": {
                team: [msg.model_dump() for msg in messages]
                for team, messages in self.world_state.team_communications.items()
            },
        }

    async def process_effects(self, effects: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Process effects and update world state.

        This method integrates with the effect processor to handle all
        effect types and maintain world state consistency.

        Args:
            effects: List of effects to process

        Returns:
            Processing results from effect processor
        """
        if not effects:
            return {
                "processed_count": 0,
                "failed_count": 0,
                "effect_types": {},
                "game_status": self.world_state.game_status,
                "status_changed": False,
            }

        # Process effects using the effect processor
        result = await self.effect_processor.process_effects(effects, self.world_state)

        # Sync updated world state back to Gunn
        await self._sync_world_state()

        # Log processing results
        self._logger.info(
            f"Processed {result['processed_count']} effects, "
            f"{result['failed_count']} failed. Game status: {result['game_status']}"
        )

        if result.get("status_changed"):
            self._logger.info(f"Game ended: {result['game_status']}")

        return result

    def get_game_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive game statistics.

        Returns:
            Dictionary containing current game statistics
        """
        return self.game_status_manager.get_game_statistics(self.world_state)
