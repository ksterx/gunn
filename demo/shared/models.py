"""
Pydantic models for the battle demo game state.

This module defines the core data structures used throughout the battle simulation,
including agents, world state, and map locations with comprehensive validation
and team management capabilities.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import AgentStatus, LocationType, WeaponCondition


class Agent(BaseModel):
    """Represents a battle agent with combat capabilities and team coordination."""

    agent_id: str = Field(
        ..., min_length=1, description="Unique identifier for the agent"
    )
    team: Literal["team_a", "team_b"] = Field(..., description="Team assignment")
    position: tuple[float, float] = Field(..., description="Current world coordinates")
    health: int = Field(default=100, ge=0, le=100, description="Current health points")
    status: AgentStatus = Field(
        default=AgentStatus.ALIVE, description="Current agent status"
    )
    weapon_condition: WeaponCondition = Field(
        default=WeaponCondition.EXCELLENT, description="Weapon condition"
    )
    last_action_time: float = Field(
        default=0.0, ge=0.0, description="Timestamp of last action"
    )
    communication_range: float = Field(
        default=50.0, gt=0.0, description="Range for team communication"
    )
    vision_range: float = Field(
        default=30.0, gt=0.0, description="Range for observing enemies"
    )
    attack_range: float = Field(
        default=15.0, gt=0.0, description="Range for attacking enemies"
    )

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate position coordinates are within reasonable bounds."""
        x, y = v
        if x < 0 or y < 0:
            raise ValueError("Position coordinates must be non-negative")
        if x > 1000 or y > 1000:  # Reasonable upper bound
            raise ValueError("Position coordinates exceed maximum map size")
        return v

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        """Validate agent ID format."""
        if not v.strip():
            raise ValueError("Agent ID cannot be empty")
        # Should follow pattern like "team_a_agent_1"
        if not (v.startswith("team_a_") or v.startswith("team_b_")):
            raise ValueError("Agent ID must start with 'team_a_' or 'team_b_'")
        return v

    @model_validator(mode="after")
    def validate_team_consistency(self) -> "Agent":
        """Ensure agent_id is consistent with team assignment."""
        expected_prefix = f"{self.team}_"
        if not self.agent_id.startswith(expected_prefix):
            raise ValueError(
                f"Agent ID '{self.agent_id}' must start with '{expected_prefix}'"
            )
        return self

    def is_alive(self) -> bool:
        """Check if agent is alive and can take actions."""
        return self.status == AgentStatus.ALIVE and self.health > 0

    def can_attack(self) -> bool:
        """Check if agent can perform attacks."""
        return self.is_alive() and self.weapon_condition != WeaponCondition.BROKEN

    def get_teammates(self, all_agents: dict[str, "Agent"]) -> list[str]:
        """Get list of teammate agent IDs."""
        return [
            agent_id
            for agent_id, agent in all_agents.items()
            if agent.team == self.team and agent_id != self.agent_id
        ]

    def get_enemies(self, all_agents: dict[str, "Agent"]) -> list[str]:
        """Get list of enemy agent IDs."""
        return [
            agent_id
            for agent_id, agent in all_agents.items()
            if agent.team != self.team
        ]


class MapLocation(BaseModel):
    """Represents a strategic location on the battle map."""

    position: tuple[float, float] = Field(
        ..., description="World coordinates of the location"
    )
    location_type: LocationType = Field(..., description="Type of location")
    radius: float = Field(default=5.0, gt=0.0, description="Interaction radius")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional location data"
    )

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate position coordinates are within reasonable bounds."""
        x, y = v
        if x < 0 or y < 0:
            raise ValueError("Position coordinates must be non-negative")
        if x > 1000 or y > 1000:  # Reasonable upper bound
            raise ValueError("Position coordinates exceed maximum map size")
        return v

    def is_agent_in_range(self, agent_position: tuple[float, float]) -> bool:
        """Check if an agent is within interaction range of this location."""
        from .utils import calculate_distance

        distance = calculate_distance(self.position, agent_position)
        return distance <= self.radius


class TeamCommunication(BaseModel):
    """Represents team communication messages."""

    sender_id: str = Field(..., description="ID of the sending agent")
    team: Literal["team_a", "team_b", "system"] = Field(
        ..., description="Team the message belongs to"
    )
    message: str = Field(..., max_length=200, description="Message content")
    urgency: Literal["low", "medium", "high"] = Field(
        default="medium", description="Message urgency"
    )
    timestamp: float = Field(..., ge=0.0, description="When the message was sent")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate message content."""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class BattleWorldState(BaseModel):
    """Complete state of the battle simulation with team management."""

    agents: dict[str, Agent] = Field(
        default_factory=dict, description="All agents in the simulation"
    )
    map_locations: dict[str, MapLocation] = Field(
        default_factory=dict, description="Strategic map locations"
    )
    team_scores: dict[str, int] = Field(
        default_factory=dict, description="Current team scores"
    )
    game_time: float = Field(
        default=0.0, ge=0.0, description="Elapsed game time in seconds"
    )
    game_status: Literal["active", "team_a_wins", "team_b_wins", "draw"] = Field(
        default="active", description="Current game status"
    )
    team_communications: dict[str, list[TeamCommunication]] = Field(
        default_factory=dict, description="Team communication history"
    )

    @model_validator(mode="after")
    def validate_world_state(self) -> "BattleWorldState":
        """Validate the overall world state consistency."""
        # Ensure team scores exist for both teams
        if "team_a" not in self.team_scores:
            self.team_scores["team_a"] = 0
        if "team_b" not in self.team_scores:
            self.team_scores["team_b"] = 0

        # Ensure team communications exist for both teams
        if "team_a" not in self.team_communications:
            self.team_communications["team_a"] = []
        if "team_b" not in self.team_communications:
            self.team_communications["team_b"] = []

        # Validate agent team consistency
        for agent_id, agent in self.agents.items():
            if agent_id != agent.agent_id:
                raise ValueError(
                    f"Agent key '{agent_id}' doesn't match agent.agent_id '{agent.agent_id}'"
                )

        return self

    def get_team_agents(self, team: str) -> dict[str, Agent]:
        """Get all agents for a specific team."""
        return {
            agent_id: agent
            for agent_id, agent in self.agents.items()
            if agent.team == team
        }

    def get_alive_agents(self, team: str | None = None) -> dict[str, Agent]:
        """Get all alive agents, optionally filtered by team."""
        agents = self.agents if team is None else self.get_team_agents(team)
        return {
            agent_id: agent for agent_id, agent in agents.items() if agent.is_alive()
        }

    def add_team_message(
        self, sender_id: str, message: str, urgency: str = "medium"
    ) -> None:
        """Add a message to team communications."""
        if sender_id not in self.agents:
            raise ValueError(f"Unknown sender: {sender_id}")

        sender = self.agents[sender_id]
        team_message = TeamCommunication(
            sender_id=sender_id,
            team=sender.team,
            message=message,
            urgency=urgency,
            timestamp=self.game_time,
        )

        self.team_communications[sender.team].append(team_message)

        # Keep only last 50 messages per team to prevent memory bloat
        if len(self.team_communications[sender.team]) > 50:
            self.team_communications[sender.team] = self.team_communications[
                sender.team
            ][-50:]

    def get_recent_team_messages(
        self, team: str, count: int = 10
    ) -> list[TeamCommunication]:
        """Get recent team messages."""
        messages = self.team_communications.get(team, [])
        return messages[-count:] if messages else []

    def get_prioritized_team_messages(
        self, team: str, count: int = 10
    ) -> list[TeamCommunication]:
        """Get team messages prioritized by urgency and recency.

        Args:
            team: Team to get messages for
            count: Maximum number of messages to return

        Returns:
            List of team messages prioritized by urgency (high -> medium -> low)
            then by timestamp (newest first)
        """
        messages = self.team_communications.get(team, [])
        if not messages:
            return []

        # Sort by urgency (high -> medium -> low) then by timestamp (newest first)
        urgency_priority = {"high": 3, "medium": 2, "low": 1}

        def sort_key(msg):
            urgency = msg.urgency
            timestamp = msg.timestamp
            return (urgency_priority.get(urgency, 2), timestamp)

        # Sort messages by priority and recency
        sorted_messages = sorted(messages, key=sort_key, reverse=True)

        # Return top messages up to count
        return sorted_messages[:count]

    def check_win_condition(
        self,
    ) -> Literal["active", "team_a_wins", "team_b_wins", "draw"]:
        """Check if any team has won the battle."""
        team_a_alive = len(self.get_alive_agents("team_a"))
        team_b_alive = len(self.get_alive_agents("team_b"))

        if team_a_alive == 0 and team_b_alive == 0:
            return "draw"
        elif team_a_alive == 0:
            return "team_b_wins"
        elif team_b_alive == 0:
            return "team_a_wins"
        else:
            return "active"

    def update_game_status(self) -> None:
        """Update game status based on current conditions."""
        self.game_status = self.check_win_condition()
