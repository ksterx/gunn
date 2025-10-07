"""
Game configuration constants and settings.

This module contains all the configurable parameters for the battle simulation,
including combat mechanics, timing, and display settings.
"""

from typing import Any

# Game mechanics constants
GAME_CONFIG: dict[str, Any] = {
    # Combat settings
    "attack_damage": 70,
    "heal_amount": 30,
    "weapon_degradation_rate": 0.1,
    "movement_speed": 5.0,
    "attack_cooldown": 2.0,
    "heal_cooldown": 3.0,
    # Agent capabilities
    "default_health": 100,
    "communication_range": 50.0,
    "vision_range": 180.0,
    "attack_range": 15.0,
    # Map settings
    "map_width": 200.0,
    "map_height": 100.0,
    "forge_radius": 5.0,
    # Team configuration
    "agents_per_team": 3,
    "team_names": ["team_a", "team_b"],
    # Timing settings
    "game_loop_interval": 0.1,  # seconds
    "decision_timeout": 3.0,  # seconds
    "max_game_duration": 300.0,  # seconds (5 minutes)
    # Display settings
    "window_width": 800,
    "window_height": 600,
    "fps": 60,
    # AI settings
    "openai_model": "gpt-4.1-mini",
    "decision_temperature": 0.7,
    "max_tokens": 1000,
}

# Team colors for visualization
TEAM_COLORS = {
    "team_a": (0, 100, 255),  # Blue
    "team_b": (255, 100, 0),  # Red
    "neutral": (128, 128, 128),  # Gray
}

# Map locations configuration
MAP_LOCATIONS = {
    "forge_a": {"position": (20.0, 80.0), "location_type": "forge", "team": "team_a"},
    "forge_b": {"position": (180.0, 20.0), "location_type": "forge", "team": "team_b"},
}

# Agent spawn positions
SPAWN_POSITIONS = {
    "team_a": [(30.0, 90.0), (40.0, 90.0), (50.0, 90.0)],
    "team_b": [(170.0, 10.0), (160.0, 10.0), (150.0, 10.0)],
}
