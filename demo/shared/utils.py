"""
Common utility functions for the battle demo.

This module provides helper functions used across both backend and frontend
components, including coordinate calculations, distance measurements, and
data conversion utilities with enhanced team management support.
"""

import math
from typing import Any

from .models import Agent, BattleWorldState, MapLocation


def calculate_distance(pos1: tuple[float, float], pos2: tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def calculate_manhattan_distance(
    pos1: tuple[float, float], pos2: tuple[float, float]
) -> float:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def calculate_angle(
    from_pos: tuple[float, float], to_pos: tuple[float, float]
) -> float:
    """Calculate angle in radians from one position to another."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    return math.atan2(dy, dx)


def calculate_direction_vector(
    from_pos: tuple[float, float], to_pos: tuple[float, float]
) -> tuple[float, float]:
    """Calculate normalized direction vector from one position to another."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    distance = math.sqrt(dx * dx + dy * dy)

    if distance == 0:
        return (0.0, 0.0)

    return (dx / distance, dy / distance)


def normalize_position(
    position: tuple[float, float], map_width: float, map_height: float
) -> tuple[float, float]:
    """Normalize position to stay within map boundaries."""
    x, y = position
    x = max(0, min(x, map_width))
    y = max(0, min(y, map_height))
    return (x, y)


def is_within_range(
    agent_pos: tuple[float, float], target_pos: tuple[float, float], max_range: float
) -> bool:
    """Check if target is within range of agent."""
    distance = calculate_distance(agent_pos, target_pos)
    return distance <= max_range


def get_agents_in_range(
    agent: Agent,
    all_agents: dict[str, Agent],
    max_range: float,
    include_teammates: bool = True,
    include_enemies: bool = True,
) -> list[str]:
    """Get list of agent IDs within range of the given agent."""
    agents_in_range = []

    for other_id, other_agent in all_agents.items():
        if other_id == agent.agent_id:
            continue

        # Filter by team if specified
        is_teammate = other_agent.team == agent.team
        if is_teammate and not include_teammates:
            continue
        if not is_teammate and not include_enemies:
            continue

        if is_within_range(agent.position, other_agent.position, max_range):
            agents_in_range.append(other_id)

    return agents_in_range


def get_nearest_agent(
    agent: Agent,
    all_agents: dict[str, Agent],
    include_teammates: bool = True,
    include_enemies: bool = True,
) -> str | None:
    """Get the nearest agent to the given agent."""
    nearest_id = None
    nearest_distance = float("inf")

    for other_id, other_agent in all_agents.items():
        if other_id == agent.agent_id:
            continue

        # Filter by team if specified
        is_teammate = other_agent.team == agent.team
        if is_teammate and not include_teammates:
            continue
        if not is_teammate and not include_enemies:
            continue

        distance = calculate_distance(agent.position, other_agent.position)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_id = other_id

    return nearest_id


def get_nearest_location(
    position: tuple[float, float],
    locations: dict[str, MapLocation],
    location_type: str | None = None,
) -> str | None:
    """Get the nearest map location to a position, optionally filtered by type."""
    nearest_id = None
    nearest_distance = float("inf")

    for loc_id, location in locations.items():
        if location_type and location.location_type.value != location_type:
            continue

        distance = calculate_distance(position, location.position)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_id = loc_id

    return nearest_id


def convert_to_screen_coords(
    world_pos: tuple[float, float],
    world_width: float,
    world_height: float,
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    """Convert world coordinates to screen coordinates for rendering."""
    x_ratio = screen_width / world_width
    y_ratio = screen_height / world_height

    screen_x = int(world_pos[0] * x_ratio)
    screen_y = int(world_pos[1] * y_ratio)

    return (screen_x, screen_y)


def convert_to_world_coords(
    screen_pos: tuple[int, int],
    world_width: float,
    world_height: float,
    screen_width: int,
    screen_height: int,
) -> tuple[float, float]:
    """Convert screen coordinates to world coordinates."""
    x_ratio = world_width / screen_width
    y_ratio = world_height / screen_height

    world_x = screen_pos[0] * x_ratio
    world_y = screen_pos[1] * y_ratio

    return (world_x, world_y)


def interpolate_position(
    start_pos: tuple[float, float], end_pos: tuple[float, float], progress: float
) -> tuple[float, float]:
    """Interpolate between two positions based on progress (0.0 to 1.0)."""
    progress = max(0.0, min(1.0, progress))  # Clamp to valid range

    x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
    y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress

    return (x, y)


def calculate_movement_time(
    start_pos: tuple[float, float], end_pos: tuple[float, float], speed: float
) -> float:
    """Calculate time needed to move between two positions at given speed."""
    if speed <= 0:
        return float("inf")

    distance = calculate_distance(start_pos, end_pos)
    return distance / speed


def format_game_time(seconds: float) -> str:
    """Format game time in MM:SS format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def format_health_percentage(health: int, max_health: int = 100) -> str:
    """Format health as percentage string."""
    percentage = (health / max_health) * 100
    return f"{percentage:.0f}%"


def serialize_agent_for_api(agent: Agent) -> dict[str, Any]:
    """Serialize agent for API responses."""
    return {
        "agent_id": agent.agent_id,
        "team": agent.team,
        "position": agent.position,
        "health": agent.health,
        "status": agent.status.value,
        "weapon_condition": agent.weapon_condition.value,
        "last_action_time": agent.last_action_time,
        "communication_range": agent.communication_range,
        "vision_range": agent.vision_range,
        "attack_range": agent.attack_range,
        "is_alive": agent.is_alive(),
        "can_attack": agent.can_attack(),
    }


def serialize_world_state_for_api(world_state: BattleWorldState) -> dict[str, Any]:
    """Serialize complete world state for API responses."""
    return {
        "agents": {
            agent_id: serialize_agent_for_api(agent)
            for agent_id, agent in world_state.agents.items()
        },
        "map_locations": {
            loc_id: {
                "position": location.position,
                "location_type": location.location_type.value,
                "radius": location.radius,
                "metadata": location.metadata,
            }
            for loc_id, location in world_state.map_locations.items()
        },
        "team_scores": world_state.team_scores,
        "game_time": world_state.game_time,
        "game_status": world_state.game_status,
        "team_communications": {
            team: [
                {
                    "sender_id": msg.sender_id,
                    "message": msg.message,
                    "urgency": msg.urgency,
                    "timestamp": msg.timestamp,
                }
                for msg in messages
            ]
            for team, messages in world_state.team_communications.items()
        },
    }


def validate_team_name(team: str) -> bool:
    """Validate if team name is valid."""
    return team in ["team_a", "team_b"]


def get_enemy_team(team: str) -> str:
    """Get the enemy team name."""
    if team == "team_a":
        return "team_b"
    elif team == "team_b":
        return "team_a"
    else:
        raise ValueError(f"Invalid team name: {team}")


def calculate_team_stats(world_state: BattleWorldState, team: str) -> dict[str, Any]:
    """Calculate comprehensive statistics for a team."""
    team_agents = world_state.get_team_agents(team)
    alive_agents = world_state.get_alive_agents(team)

    total_health = sum(agent.health for agent in team_agents.values())
    avg_health = total_health / len(team_agents) if team_agents else 0

    weapon_conditions = [agent.weapon_condition.value for agent in team_agents.values()]

    return {
        "total_agents": len(team_agents),
        "alive_agents": len(alive_agents),
        "dead_agents": len(team_agents) - len(alive_agents),
        "total_health": total_health,
        "average_health": avg_health,
        "weapon_conditions": weapon_conditions,
        "team_score": world_state.team_scores.get(team, 0),
        "recent_messages": len(world_state.get_recent_team_messages(team, 5)),
    }
