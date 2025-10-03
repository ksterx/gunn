"""
Enumerations used throughout the battle demo.

This module defines the various enums for agent status, weapon conditions,
location types, and other game state indicators.
"""

from enum import Enum


class AgentStatus(str, Enum):
    """Status of an agent in the battle."""

    ALIVE = "alive"
    DEAD = "dead"
    HEALING = "healing"


class WeaponCondition(str, Enum):
    """Condition of an agent's weapon."""

    EXCELLENT = "excellent"
    GOOD = "good"
    DAMAGED = "damaged"
    BROKEN = "broken"


class LocationType(str, Enum):
    """Type of location on the battle map."""

    OPEN_FIELD = "open_field"
    FORGE = "forge"
    COVER = "cover"
    SPAWN_POINT = "spawn_point"
