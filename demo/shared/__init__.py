"""
Shared module for the Gunn Battle Demo

This module contains common data models, schemas, enums, and utilities
used by both the backend and frontend components.

Components:
- models.py: Pydantic models for game state (Agent, BattleWorldState, etc.)
- schemas.py: OpenAI structured output schemas for AI decisions
- enums.py: Game-related enumerations (AgentStatus, WeaponCondition, etc.)
- utils.py: Common utility functions and helpers
- constants.py: Game configuration constants and settings
"""

from .constants import GAME_CONFIG
from .enums import AgentStatus, LocationType, WeaponCondition
from .models import Agent, BattleWorldState, MapLocation, TeamCommunication
from .schemas import (
    AgentDecision,
    AttackAction,
    CommunicateAction,
    HealAction,
    MoveAction,
    RepairAction,
)

__all__ = [
    "GAME_CONFIG",
    "Agent",
    "AgentDecision",
    "AgentStatus",
    "AttackAction",
    "BattleWorldState",
    "CommunicateAction",
    "HealAction",
    "LocationType",
    "MapLocation",
    "MoveAction",
    "RepairAction",
    "TeamCommunication",
    "WeaponCondition",
]
