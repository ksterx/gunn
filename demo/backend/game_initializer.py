"""
Game initialization and auto-start system for the battle demo.

This module provides comprehensive game initialization capabilities including
automatic team creation, strategic agent positioning, forge placement,
and deterministic game setup with reset and restart functionality.
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Literal

from ..shared.constants import GAME_CONFIG
from ..shared.enums import AgentStatus, LocationType, WeaponCondition
from ..shared.models import Agent, BattleWorldState, MapLocation

logger = logging.getLogger(__name__)


@dataclass
class InitializationConfig:
    """Configuration for game initialization."""

    # Team configuration
    agents_per_team: int = 3
    team_names: list[str] = None

    # Map configuration
    map_width: float = 200.0
    map_height: float = 100.0

    # Agent positioning
    positioning_strategy: Literal["corners", "lines", "random", "custom"] = "corners"
    team_spacing: float = 20.0
    agent_spacing: float = 10.0

    # Forge placement
    forge_placement: Literal["corners", "sides", "center", "custom"] = "corners"
    forge_radius: float = 5.0

    # Game settings
    initial_health: int = 100
    initial_weapon_condition: WeaponCondition = WeaponCondition.EXCELLENT

    # Randomization
    use_random_seed: bool = True
    random_seed: int | None = None
    position_jitter: float = 2.0  # Random offset for positions

    def __post_init__(self):
        """Initialize default values after creation."""
        if self.team_names is None:
            self.team_names = ["team_a", "team_b"]


class GameInitializer:
    """Handles comprehensive game initialization with deterministic setup."""

    def __init__(self, config: InitializationConfig | None = None):
        """
        Initialize the game initializer.

        Args:
            config: Initialization configuration, uses defaults if None
        """
        self.config = config or InitializationConfig()
        self._logger = logger
        self._initialization_count = 0

        # Store random state for deterministic initialization
        self._random_state = None
        if self.config.use_random_seed:
            seed = self.config.random_seed or 42
            self._random_state = random.Random(seed)
            self._logger.info(f"Using random seed: {seed}")
        else:
            self._random_state = random.Random()

    async def initialize_game(
        self, world_state: BattleWorldState, reset_existing: bool = True
    ) -> BattleWorldState:
        """
        Initialize a complete game with teams, agents, and map locations.

        Args:
            world_state: World state to initialize
            reset_existing: Whether to reset existing state

        Returns:
            Initialized world state
        """
        self._initialization_count += 1
        init_id = f"init_{self._initialization_count}"

        self._logger.info(f"Starting game initialization {init_id}")

        try:
            # Reset world state if requested
            if reset_existing:
                await self._reset_world_state(world_state)

            # Initialize map locations (forges, etc.)
            await self._initialize_map_locations(world_state)

            # Create and position teams
            await self._create_teams(world_state)

            # Initialize game metadata
            await self._initialize_game_metadata(world_state)

            # Validate initialization
            await self._validate_initialization(world_state)

            self._logger.info(
                f"Game initialization {init_id} completed successfully. "
                f"Teams: {len(self.config.team_names)}, "
                f"Agents: {len(world_state.agents)}, "
                f"Map locations: {len(world_state.map_locations)}"
            )

            return world_state

        except Exception as e:
            self._logger.error(f"Game initialization {init_id} failed: {e}")
            raise

    async def _reset_world_state(self, world_state: BattleWorldState) -> None:
        """Reset world state to clean initial conditions."""
        self._logger.debug("Resetting world state")

        # Clear existing data
        world_state.agents.clear()
        world_state.map_locations.clear()
        world_state.team_communications.clear()

        # Reset game state
        world_state.game_time = 0.0
        world_state.game_status = "active"
        world_state.team_scores = {}

    async def _initialize_map_locations(self, world_state: BattleWorldState) -> None:
        """Initialize strategic map locations including forges."""
        self._logger.debug("Initializing map locations")

        # Generate forge positions based on strategy
        forge_positions = self._generate_forge_positions()

        # Create forges for each team
        for i, team_name in enumerate(self.config.team_names):
            if i < len(forge_positions):
                forge_id = f"forge_{team_name.split('_')[1]}"  # team_a -> forge_a
                forge = MapLocation(
                    position=forge_positions[i],
                    location_type=LocationType.FORGE,
                    radius=self.config.forge_radius,
                    metadata={
                        "team": team_name,
                        "repair_capability": True,
                        "strategic_value": "high",
                    },
                )
                world_state.map_locations[forge_id] = forge
                self._logger.debug(f"Created {forge_id} at {forge_positions[i]}")

        # Add additional strategic locations if needed
        await self._add_additional_locations(world_state)

    def _generate_forge_positions(self) -> list[tuple[float, float]]:
        """Generate forge positions based on placement strategy."""
        positions = []

        if self.config.forge_placement == "corners":
            # Place forges in opposite corners
            positions = [
                (20.0, self.config.map_height - 20.0),  # Top-left for team_a
                (self.config.map_width - 20.0, 20.0),  # Bottom-right for team_b
            ]

        elif self.config.forge_placement == "sides":
            # Place forges on opposite sides
            mid_height = self.config.map_height / 2
            positions = [
                (20.0, mid_height),  # Left side for team_a
                (self.config.map_width - 20.0, mid_height),  # Right side for team_b
            ]

        elif self.config.forge_placement == "center":
            # Place forges near center but separated
            center_x = self.config.map_width / 2
            center_y = self.config.map_height / 2
            offset = 30.0
            positions = [
                (center_x - offset, center_y),  # Left of center for team_a
                (center_x + offset, center_y),  # Right of center for team_b
            ]

        else:  # custom or fallback
            positions = [
                (20.0, 80.0),  # Default team_a forge
                (180.0, 20.0),  # Default team_b forge
            ]

        # Apply position jitter if configured
        if self.config.position_jitter > 0:
            jittered_positions = []
            for x, y in positions:
                jitter_x = self._random_state.uniform(
                    -self.config.position_jitter, self.config.position_jitter
                )
                jitter_y = self._random_state.uniform(
                    -self.config.position_jitter, self.config.position_jitter
                )
                jittered_positions.append((x + jitter_x, y + jitter_y))
            positions = jittered_positions

        return positions

    async def _add_additional_locations(self, world_state: BattleWorldState) -> None:
        """Add additional strategic locations to the map."""
        # Add cover locations for tactical gameplay
        cover_positions = [
            (self.config.map_width / 2, self.config.map_height / 2),  # Center cover
            (
                self.config.map_width * 0.3,
                self.config.map_height * 0.7,
            ),  # Strategic points
            (self.config.map_width * 0.7, self.config.map_height * 0.3),
        ]

        for i, position in enumerate(cover_positions):
            cover_id = f"cover_{i + 1}"
            cover = MapLocation(
                position=position,
                location_type=LocationType.COVER,
                radius=8.0,
                metadata={
                    "cover_bonus": 0.2,  # 20% damage reduction
                    "strategic_value": "medium",
                },
            )
            world_state.map_locations[cover_id] = cover

    async def _create_teams(self, world_state: BattleWorldState) -> None:
        """Create teams with strategically positioned agents."""
        self._logger.debug("Creating teams and positioning agents")

        for team_name in self.config.team_names:
            # Generate positions for this team
            team_positions = self._generate_team_positions(team_name)

            # Create agents for this team
            for i in range(self.config.agents_per_team):
                agent_id = f"{team_name}_agent_{i + 1}"
                position = (
                    team_positions[i] if i < len(team_positions) else (50.0, 50.0)
                )

                agent = Agent(
                    agent_id=agent_id,
                    team=team_name,
                    position=position,
                    health=self.config.initial_health,
                    status=AgentStatus.ALIVE,
                    weapon_condition=self.config.initial_weapon_condition,
                    last_action_time=0.0,
                    communication_range=GAME_CONFIG.get("communication_range", 50.0),
                    vision_range=GAME_CONFIG.get("vision_range", 30.0),
                    attack_range=GAME_CONFIG.get("attack_range", 15.0),
                )

                world_state.agents[agent_id] = agent
                self._logger.debug(f"Created {agent_id} at {position}")

            # Initialize team score
            world_state.team_scores[team_name] = 0

    def _generate_team_positions(self, team_name: str) -> list[tuple[float, float]]:
        """Generate positions for a team based on positioning strategy."""
        positions = []

        if self.config.positioning_strategy == "corners":
            # Position teams in opposite corners
            if team_name == "team_a":
                base_x, base_y = 30.0, self.config.map_height - 10.0
            else:  # team_b
                base_x, base_y = self.config.map_width - 30.0, 10.0

            # Arrange agents in a line from the base position
            for i in range(self.config.agents_per_team):
                x = base_x + (i * self.config.agent_spacing)
                y = base_y
                positions.append((x, y))

        elif self.config.positioning_strategy == "lines":
            # Position teams in horizontal lines
            if team_name == "team_a":
                base_x, base_y = 30.0, self.config.map_height - 10.0
            else:  # team_b
                base_x, base_y = 30.0, 10.0

            for i in range(self.config.agents_per_team):
                x = base_x + (i * self.config.agent_spacing)
                y = base_y
                positions.append((x, y))

        elif self.config.positioning_strategy == "random":
            # Random positions within team area
            if team_name == "team_a":
                area_x = (10.0, self.config.map_width / 2 - 10.0)
                area_y = (self.config.map_height / 2, self.config.map_height - 10.0)
            else:  # team_b
                area_x = (
                    self.config.map_width / 2 + 10.0,
                    self.config.map_width - 10.0,
                )
                area_y = (10.0, self.config.map_height / 2)

            for _ in range(self.config.agents_per_team):
                x = self._random_state.uniform(area_x[0], area_x[1])
                y = self._random_state.uniform(area_y[0], area_y[1])
                positions.append((x, y))

        else:  # custom or fallback
            # Default corner positioning
            if team_name == "team_a":
                base_positions = [(30.0, 90.0), (40.0, 90.0), (50.0, 90.0)]
            else:  # team_b
                base_positions = [(170.0, 10.0), (160.0, 10.0), (150.0, 10.0)]

            positions = base_positions[: self.config.agents_per_team]

        # Apply position jitter if configured
        if self.config.position_jitter > 0:
            jittered_positions = []
            for x, y in positions:
                jitter_x = self._random_state.uniform(
                    -self.config.position_jitter, self.config.position_jitter
                )
                jitter_y = self._random_state.uniform(
                    -self.config.position_jitter, self.config.position_jitter
                )

                # Ensure positions stay within map bounds
                new_x = max(5.0, min(self.config.map_width - 5.0, x + jitter_x))
                new_y = max(5.0, min(self.config.map_height - 5.0, y + jitter_y))

                jittered_positions.append((new_x, new_y))
            positions = jittered_positions

        return positions

    async def _initialize_game_metadata(self, world_state: BattleWorldState) -> None:
        """Initialize game metadata and communication systems."""
        self._logger.debug("Initializing game metadata")

        # Initialize team communications
        for team_name in self.config.team_names:
            world_state.team_communications[team_name] = []

        # Set initial game state
        world_state.game_time = 0.0
        world_state.game_status = "active"

        # Add initialization metadata
        from ..shared.models import TeamCommunication

        world_state.team_communications["system"] = [
            TeamCommunication(
                sender_id="system",
                team="system",
                message=f"Battle initialized with {len(world_state.agents)} agents",
                urgency="low",
                timestamp=0.0,
            )
        ]

    async def _validate_initialization(self, world_state: BattleWorldState) -> None:
        """Validate that initialization was successful and consistent."""
        self._logger.debug("Validating initialization")

        # Validate configuration first
        if self.config.agents_per_team <= 0:
            raise ValueError(
                f"agents_per_team must be positive, got {self.config.agents_per_team}"
            )

        # Check team balance
        team_counts = {}
        for agent in world_state.agents.values():
            team_counts[agent.team] = team_counts.get(agent.team, 0) + 1

        expected_count = self.config.agents_per_team
        for team_name in self.config.team_names:
            actual_count = team_counts.get(team_name, 0)
            if actual_count != expected_count:
                raise ValueError(
                    f"Team {team_name} has {actual_count} agents, expected {expected_count}"
                )

        # Check forge availability
        expected_forges = len(self.config.team_names)
        actual_forges = sum(
            1
            for loc in world_state.map_locations.values()
            if loc.location_type == LocationType.FORGE
        )
        if actual_forges != expected_forges:
            raise ValueError(
                f"Found {actual_forges} forges, expected {expected_forges}"
            )

        # Check agent health and status
        for agent in world_state.agents.values():
            if agent.health != self.config.initial_health:
                raise ValueError(f"Agent {agent.agent_id} has incorrect initial health")
            if agent.status != AgentStatus.ALIVE:
                raise ValueError(
                    f"Agent {agent.agent_id} is not alive at initialization"
                )

        # Check position bounds
        for agent in world_state.agents.values():
            x, y = agent.position
            if not (
                0 <= x <= self.config.map_width and 0 <= y <= self.config.map_height
            ):
                raise ValueError(
                    f"Agent {agent.agent_id} position {agent.position} is out of bounds"
                )

        self._logger.info("Initialization validation passed")

    def get_initialization_summary(
        self, world_state: BattleWorldState
    ) -> dict[str, any]:
        """Get a summary of the current initialization state."""
        team_counts = {}
        for agent in world_state.agents.values():
            team_counts[agent.team] = team_counts.get(agent.team, 0) + 1

        forge_count = sum(
            1
            for loc in world_state.map_locations.values()
            if loc.location_type == LocationType.FORGE
        )

        return {
            "initialization_count": self._initialization_count,
            "total_agents": len(world_state.agents),
            "team_counts": team_counts,
            "total_forges": forge_count,
            "total_map_locations": len(world_state.map_locations),
            "game_status": world_state.game_status,
            "game_time": world_state.game_time,
            "config": {
                "positioning_strategy": self.config.positioning_strategy,
                "forge_placement": self.config.forge_placement,
                "agents_per_team": self.config.agents_per_team,
                "map_size": (self.config.map_width, self.config.map_height),
            },
        }


class AutoStartManager:
    """Manages automatic game startup and restart functionality."""

    def __init__(self, game_initializer: GameInitializer):
        """
        Initialize the auto-start manager.

        Args:
            game_initializer: Game initializer instance
        """
        self.game_initializer = game_initializer
        self._logger = logger
        self._auto_start_enabled = True
        self._restart_count = 0
        self._max_restart_attempts = 3

    async def auto_start_game(
        self, orchestrator, ai_decision_maker, max_attempts: int = 3
    ) -> bool:
        """
        Automatically start a game with retry logic.

        Args:
            orchestrator: Battle orchestrator instance
            ai_decision_maker: AI decision maker instance
            max_attempts: Maximum startup attempts

        Returns:
            True if startup was successful, False otherwise
        """
        if not self._auto_start_enabled:
            self._logger.info("Auto-start is disabled")
            return False

        for attempt in range(1, max_attempts + 1):
            try:
                self._logger.info(f"Auto-start attempt {attempt}/{max_attempts}")

                # Reset orchestrator on retry attempts
                if attempt > 1:
                    await orchestrator.reset()

                # Initialize the game
                await self.game_initializer.initialize_game(
                    orchestrator.world_state, reset_existing=True
                )

                # Initialize orchestrator with AI decision maker
                await orchestrator.initialize(
                    ai_decision_maker, force_reinit=(attempt > 1)
                )

                self._logger.info("Auto-start completed successfully")
                return True

            except Exception as e:
                self._logger.error(f"Auto-start attempt {attempt} failed: {e}")

                if attempt < max_attempts:
                    # Wait before retry with exponential backoff
                    wait_time = 2 ** (attempt - 1)
                    self._logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    self._logger.error("All auto-start attempts failed")

        return False

    async def restart_game(
        self, orchestrator, ai_decision_maker, reason: str = "manual_restart"
    ) -> bool:
        """
        Restart the game with proper cleanup.

        Args:
            orchestrator: Battle orchestrator instance
            ai_decision_maker: AI decision maker instance
            reason: Reason for restart

        Returns:
            True if restart was successful, False otherwise
        """
        self._restart_count += 1
        restart_id = f"restart_{self._restart_count}"

        self._logger.info(f"Starting game restart {restart_id}, reason: {reason}")

        try:
            # Stop current game if running
            await self._cleanup_current_game(orchestrator)

            # Reinitialize the game
            success = await self.auto_start_game(orchestrator, ai_decision_maker)

            if success:
                self._logger.info(f"Game restart {restart_id} completed successfully")
            else:
                self._logger.error(f"Game restart {restart_id} failed")

            return success

        except Exception as e:
            self._logger.error(f"Game restart {restart_id} failed with exception: {e}")
            return False

    async def _cleanup_current_game(self, orchestrator) -> None:
        """Clean up current game state before restart."""
        self._logger.debug("Cleaning up current game state")

        try:
            # Reset orchestrator state if needed
            if hasattr(orchestrator, "_initialized") and orchestrator._initialized:
                # Clear agent handles and policies
                if hasattr(orchestrator.orchestrator, "agent_handles"):
                    orchestrator.orchestrator.agent_handles.clear()
                if hasattr(orchestrator.orchestrator, "observation_policies"):
                    orchestrator.orchestrator.observation_policies.clear()

                # Reset initialization flag
                orchestrator._initialized = False

            # Clear world state
            orchestrator.world_state = type(orchestrator.world_state)()

        except Exception as e:
            self._logger.warning(f"Error during cleanup: {e}")

    def enable_auto_start(self) -> None:
        """Enable automatic game startup."""
        self._auto_start_enabled = True
        self._logger.info("Auto-start enabled")

    def disable_auto_start(self) -> None:
        """Disable automatic game startup."""
        self._auto_start_enabled = False
        self._logger.info("Auto-start disabled")

    def get_restart_statistics(self) -> dict[str, any]:
        """Get restart statistics."""
        return {
            "restart_count": self._restart_count,
            "auto_start_enabled": self._auto_start_enabled,
            "max_restart_attempts": self._max_restart_attempts,
        }
