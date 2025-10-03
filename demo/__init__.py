"""
Gunn Multi-Agent Battle Demo

This package contains a complete demonstration of Gunn's multi-agent simulation
capabilities through a 2D real-time strategy battle simulation.

The demo showcases:
- Two teams of 3 CPU agents each engaging in tactical combat
- OpenAI structured outputs for AI decision-making
- Team-based observation policies and communication
- Real-time visualization with Pygame
- FastAPI backend with WebSocket support
- Concurrent agent processing and deterministic behavior

Package Structure:
- backend/: FastAPI server, game logic, and Gunn integration
- frontend/: Pygame visualization and user interface
- shared/: Common models, schemas, and utilities
"""

__version__ = "1.0.0"
__author__ = "Gunn Development Team"

# Public exports for demo package
from .shared.models import Agent, BattleWorldState, MapLocation
from .shared.schemas import AgentDecision, AttackAction, MoveAction

__all__ = [
    "Agent",
    "AgentDecision",
    "AttackAction",
    "BattleWorldState",
    "MapLocation",
    "MoveAction",
]
