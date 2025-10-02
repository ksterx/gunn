"""
Backend module for the Gunn Battle Demo

This module contains the FastAPI server, game logic, battle mechanics,
and Gunn orchestrator integration for the multi-agent battle simulation.

Components:
- server.py: FastAPI application with REST API and WebSocket endpoints
- game_manager.py: Game state management and battle orchestration
- battle_mechanics.py: Combat calculations and game rules
- ai_decision.py: OpenAI structured output integration
- gunn_integration.py: Wrapper around Gunn's Orchestrator
"""

from .battle_mechanics import BattleMechanics, CombatManager
from .game_manager import GameManager
from .server import BattleAPIServer

__all__ = [
    "BattleAPIServer",
    "BattleMechanics",
    "CombatManager",
    "GameManager",
]
