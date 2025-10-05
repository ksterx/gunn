"""
FastAPI server for the battle demo backend.

This module contains the main FastAPI application with REST endpoints
and WebSocket handlers for real-time game state updates, auto-startup
game initialization, and concurrent agent processing.
"""

import asyncio
import json
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..shared.enums import WeaponCondition
from ..shared.errors import NetworkError, SystemError, WebSocketError
from ..shared.schemas import AgentDecision
from .ai_decision import AIDecisionMaker
from .battle_mechanics import BattleMechanics, CombatManager
from .error_handler import BattleErrorHandler
from .game_initializer import AutoStartManager, GameInitializer, InitializationConfig
from .gunn_integration import BattleOrchestrator
from .performance_monitor import performance_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# API Request/Response Models
class GameStateResponse(BaseModel):
    """Response model for game state."""

    agents: dict[str, dict[str, Any]]
    map_locations: dict[str, dict[str, Any]]
    team_scores: dict[str, int]
    game_time: float
    game_status: str
    team_communications: dict[str, list[dict[str, Any]]]


class StartGameRequest(BaseModel):
    """Request model for starting a game."""

    game_mode: str = Field(default="demo", description="Game mode")
    ai_difficulty: str = Field(default="normal", description="AI difficulty level")
    auto_start: bool = Field(
        default=True, description="Whether to auto-start the game loop"
    )
    positioning_strategy: str = Field(
        default="corners", description="Agent positioning strategy"
    )
    forge_placement: str = Field(
        default="corners", description="Forge placement strategy"
    )
    agents_per_team: int = Field(
        default=3, ge=1, le=5, description="Number of agents per team"
    )
    use_random_seed: bool = Field(
        default=True, description="Use deterministic random seed"
    )
    random_seed: int | None = Field(
        default=None, description="Random seed for deterministic setup"
    )


class GameControlRequest(BaseModel):
    """Request model for game control actions."""

    action: str = Field(
        description="Control action: 'pause', 'resume', 'reset', 'stop'"
    )


class WebSocketMessage(BaseModel):
    """WebSocket message format."""

    type: str = Field(description="Message type")
    data: dict[str, Any] = Field(description="Message data")
    timestamp: float = Field(description="Message timestamp")


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: set[WebSocket] = set()
        self.connection_count = 0

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_count += 1
        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def send_personal_message(
        self, message: dict[str, Any], websocket: WebSocket
    ) -> None:
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send message to WebSocket: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected WebSockets."""
        if not self.active_connections:
            return

        message_text = json.dumps(message)
        disconnected = set()

        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.warning(f"Failed to broadcast to WebSocket: {e}")
                disconnected.add(connection)

        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_action_result(
        self,
        agent_id: str,
        action_type: str,
        success: bool,
        details: str = "",
    ) -> None:
        """Broadcast an action result to all connected clients."""
        message = {
            "type": "action_result",
            "data": {
                "agent_id": agent_id,
                "action_type": action_type,
                "success": success,
                "details": details,
            },
            "timestamp": time.time(),
        }
        await self.broadcast(message)


class BattleAPIServer:
    """FastAPI server for the battle demo with auto-startup and real-time updates."""

    def __init__(self, openai_api_key: str | None = None):
        """
        Initialize the battle API server.

        Args:
            openai_api_key: OpenAI API key for AI decision making
        """
        # Core components
        self.orchestrator: BattleOrchestrator | None = None
        self.ai_decision_maker: AIDecisionMaker | None = None
        self.combat_manager: CombatManager | None = None
        self.connection_manager = ConnectionManager()
        self.error_handler = BattleErrorHandler()

        # Game initialization components
        self.game_initializer: GameInitializer | None = None
        self.auto_start_manager: AutoStartManager | None = None

        # Game state
        self.game_running = False
        self.game_paused = False
        self.game_loop_task: asyncio.Task | None = None
        self.shutdown_event = asyncio.Event()

        # Configuration
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.game_tick_rate = 2.0  # Seconds between game ticks
        self.max_game_time = 300.0  # 5 minutes max game time

        # Performance monitoring
        self.performance_monitor = performance_monitor

        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title="Gunn Battle Demo API",
            version="1.0.0",
            description="Multi-agent battle simulation showcasing Gunn capabilities",
            lifespan=self._lifespan,
        )

        self._setup_middleware()
        self._setup_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Handle application startup and shutdown."""
        # Startup
        logger.info("Starting Battle API Server...")
        await self._initialize_components()

        # Start performance monitoring
        await self.performance_monitor.start_monitoring()

        # Auto-start game if configured
        if not self.openai_api_key:
            logger.warning(
                "No OpenAI API key provided. Game will use fallback decisions."
            )

        # Start the game automatically (no user intervention required)
        await self._auto_start_game()

        yield

        # Shutdown
        logger.info("Shutting down Battle API Server...")
        await self.performance_monitor.stop_monitoring()
        await self._graceful_shutdown()

    def _setup_middleware(self) -> None:
        """Configure CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        """Set up API routes and WebSocket endpoints."""

        @self.app.get("/", response_class=JSONResponse)
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "Gunn Battle Demo API",
                "version": "1.0.0",
                "status": "running",
                "game_running": self.game_running,
                "connections": len(self.connection_manager.active_connections),
                "endpoints": {
                    "game_state": "/api/game/state",
                    "start_game": "/api/game/start",
                    "control_game": "/api/game/control",
                    "websocket": "/ws",
                },
            }

        @self.app.get("/api/game/state", response_model=GameStateResponse)
        async def get_game_state():
            """Get current game state."""
            if not self.orchestrator:
                raise HTTPException(status_code=503, detail="Game not initialized")

            return self._serialize_game_state()

        @self.app.post("/api/game/start", response_model=GameStateResponse)
        async def start_game(request: StartGameRequest):
            """Start a new battle simulation with configurable initialization."""
            try:
                if self.game_running:
                    # Reset current game
                    await self._stop_game()

                # Initialize game with request parameters
                await self._initialize_game(request)

                if request.auto_start:
                    await self._start_game_loop()

                logger.info(
                    f"Game started - Mode: {request.game_mode}, "
                    f"Difficulty: {request.ai_difficulty}, "
                    f"Agents per team: {request.agents_per_team}, "
                    f"Positioning: {request.positioning_strategy}"
                )
                return self._serialize_game_state()

            except Exception as e:
                logger.error(f"Failed to start game: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to start game: {e!s}"
                )

        @self.app.post("/api/game/control")
        async def control_game(request: GameControlRequest):
            """Control game execution (pause, resume, reset, stop)."""
            try:
                if request.action == "pause":
                    self.game_paused = True
                    logger.info("Game paused")

                elif request.action == "resume":
                    self.game_paused = False
                    logger.info("Game resumed")

                elif request.action == "reset":
                    if self.auto_start_manager:
                        success = await self.auto_start_manager.restart_game(
                            self.orchestrator,
                            self.ai_decision_maker,
                            reason="manual_reset",
                        )
                        if success:
                            await self._start_game_loop()
                            logger.info("Game reset successfully")
                        else:
                            raise HTTPException(
                                status_code=500, detail="Game reset failed"
                            )
                    else:
                        # Fallback to old method
                        await self._stop_game()
                        await self._initialize_game()
                        await self._start_game_loop()
                        logger.info("Game reset")

                elif request.action == "stop":
                    await self._stop_game()
                    logger.info("Game stopped")

                else:
                    raise HTTPException(
                        status_code=400, detail=f"Unknown action: {request.action}"
                    )

                return {
                    "action": request.action,
                    "game_running": self.game_running,
                    "game_paused": self.game_paused,
                    "timestamp": time.time(),
                }

            except Exception as e:
                logger.error(f"Failed to control game: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to control game: {e!s}"
                )

        @self.app.get("/api/game/stats")
        async def get_game_stats():
            """Get game statistics and performance metrics."""
            try:
                if not self.orchestrator:
                    return {"error": "Game not initialized"}

                world_state = self.orchestrator.world_state

                # Calculate team statistics
                team_stats = {}
                for team in ["team_a", "team_b"]:
                    team_agents = world_state.get_alive_agents(team)
                    total_health = sum(agent.health for agent in team_agents.values())
                    avg_weapon_condition = self._calculate_avg_weapon_condition(
                        team_agents.values()
                    )

                    team_stats[team] = {
                        "alive_agents": len(team_agents),
                        "total_health": total_health,
                        "avg_weapon_condition": avg_weapon_condition,
                        "score": world_state.team_scores.get(team, 0),
                    }

                # Include error handling statistics
                error_stats = self.error_handler.get_error_statistics()
                if self.ai_decision_maker:
                    error_stats.update(self.ai_decision_maker.get_error_statistics())

                return {
                    "game_time": world_state.game_time,
                    "game_status": world_state.game_status,
                    "team_stats": team_stats,
                    "total_agents": len(world_state.agents),
                    "connections": len(self.connection_manager.active_connections),
                    "game_running": self.game_running,
                    "game_paused": self.game_paused,
                    "error_statistics": error_stats,
                }

            except Exception as e:
                logger.error(f"Error getting game stats: {e}")
                return {"error": f"Failed to get stats: {e!s}"}

        @self.app.get("/api/system/health")
        async def health_check():
            """Health check endpoint with error recovery status."""
            try:
                health_status = {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "components": {
                        "orchestrator": self.orchestrator is not None,
                        "ai_decision_maker": self.ai_decision_maker is not None,
                        "combat_manager": self.combat_manager is not None,
                        "error_handler": True,
                    },
                    "game_state": {
                        "running": self.game_running,
                        "paused": self.game_paused,
                    },
                    "connections": len(self.connection_manager.active_connections),
                }

                # Add error statistics
                if self.error_handler:
                    health_status["error_statistics"] = (
                        self.error_handler.get_error_statistics()
                    )

                # Check if any critical errors occurred
                error_stats = health_status.get("error_statistics", {})
                critical_errors = sum(
                    count
                    for error_type, count in error_stats.get(
                        "error_counts_by_type", {}
                    ).items()
                    if "CRITICAL" in error_type.upper()
                )

                if critical_errors > 0:
                    health_status["status"] = "degraded"
                    health_status["warnings"] = [
                        f"{critical_errors} critical errors detected"
                    ]

                return health_status

            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": time.time(),
                }

        @self.app.post("/api/system/reset-errors")
        async def reset_error_tracking():
            """Reset error tracking statistics."""
            try:
                self.error_handler.reset_error_tracking()
                if self.ai_decision_maker:
                    self.ai_decision_maker.reset_error_tracking()

                return {
                    "status": "success",
                    "message": "Error tracking reset",
                    "timestamp": time.time(),
                }

            except Exception as e:
                logger.error(f"Failed to reset error tracking: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to reset error tracking: {e!s}"
                )

        @self.app.get("/api/game/initialization")
        async def get_initialization_info():
            """Get current game initialization information."""
            try:
                if not self.game_initializer or not self.orchestrator:
                    return {"error": "Game not initialized"}

                summary = self.game_initializer.get_initialization_summary(
                    self.orchestrator.world_state
                )

                # Add auto-start manager statistics
                if self.auto_start_manager:
                    summary["auto_start"] = (
                        self.auto_start_manager.get_restart_statistics()
                    )

                return summary

            except Exception as e:
                logger.error(f"Error getting initialization info: {e}")
                return {"error": f"Failed to get initialization info: {e!s}"}

        @self.app.post("/api/game/reinitialize")
        async def reinitialize_game(request: StartGameRequest):
            """Reinitialize the game with new parameters without starting the game loop."""
            try:
                # Stop current game if running
                if self.game_running:
                    await self._stop_game()

                # Reinitialize with new parameters
                await self._initialize_game(request)

                logger.info("Game reinitialized successfully")
                return {
                    "status": "success",
                    "message": "Game reinitialized",
                    "game_state": self._serialize_game_state().model_dump(),
                    "timestamp": time.time(),
                }

            except Exception as e:
                logger.error(f"Failed to reinitialize game: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to reinitialize game: {e!s}"
                )

        @self.app.post("/api/system/auto-start/enable")
        async def enable_auto_start():
            """Enable automatic game startup."""
            if not self.auto_start_manager:
                raise HTTPException(
                    status_code=503, detail="Auto-start manager not available"
                )

            try:
                self.auto_start_manager.enable_auto_start()
                return {
                    "status": "success",
                    "message": "Auto-start enabled",
                    "timestamp": time.time(),
                }

            except Exception as e:
                logger.error(f"Failed to enable auto-start: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to enable auto-start: {e!s}"
                )

        @self.app.post("/api/system/auto-start/disable")
        async def disable_auto_start():
            """Disable automatic game startup."""
            if not self.auto_start_manager:
                raise HTTPException(
                    status_code=503, detail="Auto-start manager not available"
                )

            try:
                self.auto_start_manager.disable_auto_start()
                return {
                    "status": "success",
                    "message": "Auto-start disabled",
                    "timestamp": time.time(),
                }

            except Exception as e:
                logger.error(f"Failed to disable auto-start: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to disable auto-start: {e!s}"
                )

        @self.app.get("/api/performance/summary")
        async def get_performance_summary():
            """Get comprehensive performance summary."""
            try:
                async with self.performance_monitor.monitor_api_request(
                    "/api/performance/summary", "GET"
                ):
                    summary = self.performance_monitor.get_performance_summary()
                    return summary
            except Exception as e:
                logger.error(f"Failed to get performance summary: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get performance summary: {e!s}",
                )

        @self.app.post("/api/performance/frontend")
        async def receive_frontend_metrics(metrics: dict):
            """Receive performance metrics from frontend."""
            try:
                # Record frontend frame rate
                if "fps" in metrics:
                    self.performance_monitor.record_frame_rate(metrics["fps"])

                # Record queue sizes if provided
                if "queue_sizes" in metrics:
                    for queue_type, size in metrics["queue_sizes"].items():
                        self.performance_monitor.record_queue_size(queue_type, size)

                return {"status": "success", "timestamp": time.time()}

            except Exception as e:
                logger.error(f"Failed to process frontend metrics: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process frontend metrics: {e!s}",
                )

        @self.app.post("/api/performance/test")
        async def run_performance_test(duration: float = 60.0):
            """Run a comprehensive performance test."""
            try:
                if duration > 300.0:  # Max 5 minutes
                    raise HTTPException(
                        status_code=400,
                        detail="Test duration too long (max 300 seconds)",
                    )

                test_results = await self.performance_monitor.run_performance_test(
                    duration
                )
                return test_results

            except Exception as e:
                logger.error(f"Failed to run performance test: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to run performance test: {e!s}"
                )

        @self.app.post("/api/performance/start-monitoring")
        async def start_performance_monitoring():
            """Start continuous performance monitoring."""
            try:
                await self.performance_monitor.start_monitoring()
                return {
                    "status": "success",
                    "message": "Performance monitoring started",
                    "timestamp": time.time(),
                }
            except Exception as e:
                logger.error(f"Failed to start performance monitoring: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to start performance monitoring: {e!s}",
                )

        @self.app.post("/api/performance/stop-monitoring")
        async def stop_performance_monitoring():
            """Stop continuous performance monitoring."""
            try:
                await self.performance_monitor.stop_monitoring()
                return {
                    "status": "success",
                    "message": "Performance monitoring stopped",
                    "timestamp": time.time(),
                }
            except Exception as e:
                logger.error(f"Failed to stop performance monitoring: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to stop performance monitoring: {e!s}",
                )

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time game updates with comprehensive error handling."""
            connection_id = f"ws_{id(websocket)}"

            try:
                await self.connection_manager.connect(websocket)
                logger.info(f"WebSocket connected: {connection_id}")

                # Send initial game state with error handling
                if self.orchestrator:
                    try:
                        initial_state = {
                            "type": "game_state",
                            "data": self._serialize_game_state().model_dump(),
                            "timestamp": time.time(),
                        }
                        await self.connection_manager.send_personal_message(
                            initial_state, websocket
                        )
                    except Exception as e:
                        await self.error_handler.handle_websocket_error(
                            e, connection_id, auto_reconnect=False
                        )

                # Keep connection alive and handle incoming messages
                while True:
                    try:
                        # Wait for messages with timeout
                        message = await asyncio.wait_for(
                            websocket.receive_text(), timeout=30.0
                        )

                        # Handle client messages (e.g., ping/pong)
                        try:
                            data = json.loads(message)
                            if data.get("type") == "ping":
                                await self.connection_manager.send_personal_message(
                                    {
                                        "type": "pong",
                                        "data": {},
                                        "timestamp": time.time(),
                                    },
                                    websocket,
                                )
                            elif data.get("type") == "error_report":
                                # Client reporting an error
                                client_error = data.get("data", {})
                                logger.warning(
                                    f"Client error reported by {connection_id}: {client_error}"
                                )

                        except json.JSONDecodeError:
                            json_error = WebSocketError(
                                message=f"Invalid JSON received: {message[:100]}...",
                                connection_id=connection_id,
                            )
                            await self.error_handler.handle_websocket_error(
                                json_error, connection_id
                            )

                    except TimeoutError:
                        # Send keepalive ping with error handling
                        try:
                            await self.connection_manager.send_personal_message(
                                {
                                    "type": "keepalive",
                                    "data": {"game_running": self.game_running},
                                    "timestamp": time.time(),
                                },
                                websocket,
                            )
                        except Exception as e:
                            timeout_error = WebSocketError(
                                message=f"Keepalive failed: {e!s}",
                                connection_id=connection_id,
                            )
                            await self.error_handler.handle_websocket_error(
                                timeout_error, connection_id
                            )
                            break  # Exit loop if keepalive fails

                    except (WebSocketDisconnect, RuntimeError) as e:
                        # Client disconnected or connection already closed
                        if "disconnect" in str(e).lower():
                            logger.info(
                                f"WebSocket client disconnected: {connection_id}"
                            )
                        else:
                            logger.debug(
                                f"WebSocket connection closed: {connection_id}"
                            )
                        break  # Exit loop cleanly

                    except Exception as e:
                        ws_error = WebSocketError(
                            message=f"Unexpected WebSocket error: {e!s}",
                            connection_id=connection_id,
                        )
                        success = await self.error_handler.handle_websocket_error(
                            ws_error, connection_id
                        )
                        if not success:
                            break  # Exit loop if error handling fails

            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected: {connection_id}")
            except Exception as e:
                disconnect_error = WebSocketError(
                    message=f"WebSocket connection error: {e!s}",
                    connection_id=connection_id,
                )
                await self.error_handler.handle_websocket_error(
                    disconnect_error, connection_id
                )
            finally:
                self.connection_manager.disconnect(websocket)
                logger.info(f"WebSocket cleanup completed: {connection_id}")

    async def _initialize_components(self) -> None:
        """Initialize core components."""
        try:
            # Initialize AI decision maker
            self.ai_decision_maker = AIDecisionMaker(api_key=self.openai_api_key)

            # Initialize battle mechanics and combat manager
            battle_mechanics = BattleMechanics()
            self.combat_manager = CombatManager(battle_mechanics)

            # Initialize orchestrator
            self.orchestrator = BattleOrchestrator()

            # Initialize game initialization components
            init_config = InitializationConfig()
            self.game_initializer = GameInitializer(init_config)
            self.auto_start_manager = AutoStartManager(self.game_initializer)

            logger.info("Core components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def _auto_start_game(self) -> None:
        """Auto-start game on server startup (no user intervention required)."""
        try:
            logger.info("Auto-starting game...")

            if not self.auto_start_manager:
                raise RuntimeError("Auto-start manager not initialized")

            # Use the auto-start manager for robust initialization
            success = await self.auto_start_manager.auto_start_game(
                self.orchestrator, self.ai_decision_maker, max_attempts=3
            )

            if success:
                # Connect action result callback to broadcast to WebSocket clients
                self.orchestrator.set_action_callback(
                    lambda agent_id, action_type, success, details: asyncio.create_task(
                        self.connection_manager.broadcast_action_result(
                            agent_id, action_type, success, details
                        )
                    )
                )
                logger.info("Action result callback connected")

                # Connect broadcast callback for Speak effects to WebSocket clients
                self.orchestrator.effect_processor._broadcast_callback = (
                    lambda team, payload: asyncio.create_task(
                        self._broadcast_team_communication(team, payload)
                    )
                )
                logger.info("Broadcast callback connected for Speak effects")

                await self._start_game_loop()
                logger.info("Game auto-started successfully")
            else:
                logger.warning(
                    "Auto-start failed, server will continue without active game"
                )

        except Exception as e:
            logger.error(f"Failed to auto-start game: {e}")
            # Don't raise - server should still start even if game fails

    async def _initialize_game(self, request: StartGameRequest | None = None) -> None:
        """Initialize a new game session with configurable parameters."""
        if (
            not self.orchestrator
            or not self.ai_decision_maker
            or not self.game_initializer
        ):
            raise RuntimeError("Components not initialized")

        # Create initialization config from request
        if request:
            init_config = InitializationConfig(
                agents_per_team=request.agents_per_team,
                positioning_strategy=request.positioning_strategy,
                forge_placement=request.forge_placement,
                use_random_seed=request.use_random_seed,
                random_seed=request.random_seed,
            )
            # Update game initializer config
            self.game_initializer.config = init_config

        # Initialize the game world
        await self.game_initializer.initialize_game(
            self.orchestrator.world_state, reset_existing=True
        )

        # Initialize orchestrator with AI decision maker
        await self.orchestrator.initialize(self.ai_decision_maker)

        # Reset game state
        self.game_running = False
        self.game_paused = False

        # Get initialization summary
        summary = self.game_initializer.get_initialization_summary(
            self.orchestrator.world_state
        )
        logger.info(
            f"Game initialized: {summary['total_agents']} agents, {summary['total_forges']} forges"
        )

    async def _start_game_loop(self) -> None:
        """Start the main game loop with concurrent agent processing."""
        if self.game_running:
            return

        self.game_running = True
        self.game_paused = False
        self.shutdown_event.clear()

        # Start game loop task
        self.game_loop_task = asyncio.create_task(self._game_loop())

        logger.info("Game loop started")

    async def _stop_game(self) -> None:
        """Stop the current game."""
        self.game_running = False
        self.shutdown_event.set()

        if self.game_loop_task and not self.game_loop_task.done():
            self.game_loop_task.cancel()
            try:
                await self.game_loop_task
            except asyncio.CancelledError:
                pass

        logger.info("Game stopped")

    async def _game_loop(self) -> None:
        """Main game loop with concurrent agent processing and comprehensive error handling."""
        consecutive_errors = 0
        max_consecutive_errors = 5

        try:
            logger.info("Starting main game loop")

            while self.game_running and not self.shutdown_event.is_set():
                try:
                    # Skip processing if paused
                    if self.game_paused:
                        await asyncio.sleep(0.1)
                        continue

                    # Check win condition
                    if self.orchestrator.world_state.game_status != "active":
                        logger.info(
                            f"Game ended: {self.orchestrator.world_state.game_status}"
                        )
                        await self._broadcast_game_end()
                        break

                    # Check max game time
                    if self.orchestrator.world_state.game_time >= self.max_game_time:
                        logger.info("Game ended: Maximum time reached")
                        self.orchestrator.world_state.game_status = "draw"
                        await self._broadcast_game_end()
                        break

                    # Process one game tick with error handling
                    async with self.error_handler.error_boundary(
                        "game_tick_processing", fallback_result=None
                    ):
                        await self._process_game_tick()

                    # Broadcast updated state with error handling
                    async with self.error_handler.error_boundary(
                        "game_state_broadcast", fallback_result=None
                    ):
                        await self._broadcast_game_state()

                    # Reset consecutive error counter on success
                    consecutive_errors = 0

                    # Wait for next tick
                    await asyncio.sleep(self.game_tick_rate)

                except Exception as e:
                    consecutive_errors += 1

                    game_loop_error = SystemError(
                        message=f"Game loop error (consecutive: {consecutive_errors}): {e!s}",
                        system_component="game_loop",
                        context={
                            "consecutive_errors": consecutive_errors,
                            "game_time": getattr(
                                self.orchestrator.world_state, "game_time", 0
                            )
                            if self.orchestrator
                            else 0,
                        },
                    )

                    success, _ = await self.error_handler.handle_error(game_loop_error)

                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical(
                            f"Too many consecutive game loop errors ({consecutive_errors}), stopping game"
                        )
                        self.game_running = False
                        break

                    # Exponential backoff for errors
                    error_delay = min(5.0, 0.5 * (2**consecutive_errors))
                    logger.warning(
                        f"Game loop error, retrying in {error_delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(error_delay)

        except asyncio.CancelledError:
            logger.info("Game loop cancelled")
        except Exception as e:
            fatal_error = SystemError(
                message=f"Fatal game loop error: {e!s}",
                system_component="game_loop",
                severity="critical",
                recoverable=False,
            )
            await self.error_handler.handle_error(fatal_error)
            logger.critical(f"Fatal error in game loop: {e}")
        finally:
            self.game_running = False
            logger.info("Game loop ended")

    async def _process_game_tick(self) -> None:
        """Process one game tick with state updates.

        Note: Agent decisions and effect processing are now handled independently
        by async agent loops and the orchestrator. This method only updates
        game time and checks win conditions.
        """
        try:
            # Update game time
            self.orchestrator.world_state.game_time += self.game_tick_rate

            # Sync our world state with Gunn's world state
            # (Effects are already being applied by the orchestrator)
            await self.orchestrator._sync_world_state()

            # Check and update win condition
            self.orchestrator.world_state.update_game_status()

        except Exception as e:
            logger.error(f"Error processing game tick: {e}")
            raise

    async def _process_agent_effects(self, agent_decisions: dict[str, Any]) -> None:
        """Process effects from agent decisions using combat manager."""
        try:
            all_effects = []

            for agent_id, decision in agent_decisions.items():
                if isinstance(decision, Exception):
                    logger.warning(
                        f"Skipping effects for {agent_id} due to decision error: {decision}"
                    )
                    continue

                # Process primary action effects
                effects = await self._process_action_effects(agent_id, decision)
                all_effects.extend(effects)

            # Apply all effects to world state
            await self._apply_effects_to_world_state(all_effects)

        except Exception as e:
            logger.error(f"Error processing agent effects: {e}")
            raise

    async def _process_action_effects(
        self, agent_id: str, decision: AgentDecision
    ) -> list[dict[str, Any]]:
        """Process effects for a single agent's decision."""
        effects = []
        action = decision.primary_action
        world_state = self.orchestrator.world_state

        try:
            # Process primary action
            if action.action_type == "attack":
                effects.extend(
                    await self.combat_manager.process_attack(
                        agent_id, action.target_agent_id, world_state
                    )
                )
            elif action.action_type == "heal":
                target_id = action.target_agent_id or agent_id
                effects.extend(
                    await self.combat_manager.process_heal(
                        agent_id, target_id, world_state
                    )
                )
            elif action.action_type == "repair":
                effects.extend(
                    await self.combat_manager.process_repair(agent_id, world_state)
                )
            elif action.action_type == "move":
                # Update agent position
                agent = world_state.agents.get(agent_id)
                if agent:
                    old_position = agent.position
                    agent.position = action.target_position
                    agent.last_action_time = world_state.game_time

                    effects.append(
                        {
                            "kind": "AgentMoved",
                            "payload": {
                                "agent_id": agent_id,
                                "old_position": old_position,
                                "new_position": action.target_position,
                                "reason": action.reason,
                                "timestamp": world_state.game_time,
                            },
                            "source_id": "game_manager",
                            "schema_version": "1.0.0",
                        }
                    )

            # Process communication if present
            if decision.communication:
                comm_effects = await self.combat_manager.process_communication(
                    agent_id,
                    decision.communication.message,
                    decision.communication.urgency,
                    world_state,
                )
                effects.extend(comm_effects)

        except Exception as e:
            logger.error(f"Error processing action for {agent_id}: {e}")
            # Add error effect
            effects.append(
                {
                    "kind": "ActionFailed",
                    "payload": {
                        "agent_id": agent_id,
                        "action_type": action.action_type,
                        "error": str(e),
                        "timestamp": world_state.game_time,
                    },
                    "source_id": "game_manager",
                    "schema_version": "1.0.0",
                }
            )

        return effects

    async def _apply_effects_to_world_state(
        self, effects: list[dict[str, Any]]
    ) -> None:
        """Apply effects to the world state, update team scores, and broadcast team communications."""
        for effect in effects:
            try:
                effect_kind = effect.get("kind", "")
                payload = effect.get("payload", {})

                # Update team scores based on effects
                # Update team scores for kills
                if effect_kind == "AgentDied":
                    killer_id = payload.get("killer_id")
                    if killer_id and killer_id in self.orchestrator.world_state.agents:
                        killer_team = self.orchestrator.world_state.agents[
                            killer_id
                        ].team
                        self.orchestrator.world_state.team_scores[killer_team] += 10

                # Note: Speak and TeamMessage effects are now handled by
                # EffectProcessor via the effect polling loop

            except Exception as e:
                logger.error(
                    f"Error applying effect {effect.get('kind', 'unknown')}: {e}"
                )

    async def _broadcast_game_state(self) -> None:
        """Broadcast current game state to all WebSocket connections with error recovery."""
        try:
            if not self.connection_manager.active_connections:
                logger.debug("[BROADCAST] No active WebSocket connections")
                return

            # Use error handler for network operations
            async def broadcast_operation():
                game_state = self._serialize_game_state()

                # Log agent positions being broadcast
                logger.info("[BROADCAST] Broadcasting agent positions:")
                for agent_id, agent in game_state.agents.items():
                    logger.info(f"  {agent_id}: {agent['position']}")

                message = {
                    "type": "game_state_update",
                    "data": game_state.model_dump(),
                    "timestamp": time.time(),
                }
                await self.connection_manager.broadcast(message)
                logger.info(
                    f"[BROADCAST] Sent to {len(self.connection_manager.active_connections)} clients"
                )
                return True

            success, _ = await self.error_handler.handle_network_error(
                Exception("Broadcast operation"),  # Placeholder
                broadcast_operation,
                max_retries=2,
            )

            if not success:
                logger.warning("Failed to broadcast game state after retries")

        except Exception as e:
            broadcast_error = NetworkError(
                message=f"Game state broadcast failed: {e!s}",
                endpoint="websocket_broadcast",
            )
            await self.error_handler.handle_error(broadcast_error)

    async def _broadcast_team_communication(
        self, team: str, communication: dict[str, Any]
    ) -> None:
        """Broadcast team communication effect with team visibility filtering.

        This method implements team-only visibility by only broadcasting
        communication effects to connections that should see them.

        Args:
            team: Team that sent the communication
            communication: Communication data to broadcast
        """
        try:
            if not self.connection_manager.active_connections:
                logger.warning("[COMM] No active connections for team communication")
                return

            # Create team communication message
            message = {
                "type": "team_communication",
                "data": {
                    "team": team,
                    "sender_id": communication.get("sender_id"),
                    "message": communication.get("message"),
                    "urgency": communication.get("urgency", "medium"),
                    "timestamp": communication.get("timestamp", time.time()),
                    "team_only": True,  # Indicates this is team-only visibility
                },
                "timestamp": time.time(),
            }

            logger.info(
                f"[COMM] Broadcasting team communication: sender={communication.get('sender_id')}, message={communication.get('message')[:50]}"
            )

            # For demo purposes, broadcast to all connections
            # In a real implementation, you would filter by team membership
            await self.connection_manager.broadcast(message)

            logger.debug(
                f"Broadcasted team communication from {team}: {communication.get('message', '')[:50]}..."
            )

        except Exception as e:
            logger.error(f"Error broadcasting team communication: {e}")

    async def _broadcast_game_end(self) -> None:
        """Broadcast game end message."""
        try:
            message = {
                "type": "game_ended",
                "data": {
                    "final_status": self.orchestrator.world_state.game_status,
                    "final_scores": self.orchestrator.world_state.team_scores,
                    "game_time": self.orchestrator.world_state.game_time,
                },
                "timestamp": time.time(),
            }

            await self.connection_manager.broadcast(message)
            self.game_running = False

        except Exception as e:
            logger.error(f"Error broadcasting game end: {e}")

    def _serialize_game_state(self) -> GameStateResponse:
        """Serialize current game state for API responses."""
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized")

        world_state = self.orchestrator.world_state

        # Serialize agents
        agents_data = {
            agent_id: agent.model_dump()
            for agent_id, agent in world_state.agents.items()
        }

        # Serialize map locations
        locations_data = {
            loc_id: location.model_dump()
            for loc_id, location in world_state.map_locations.items()
        }

        # Serialize team communications
        comms_data = {
            team: [
                msg.model_dump() if hasattr(msg, "model_dump") else msg
                for msg in messages
            ]
            for team, messages in world_state.team_communications.items()
        }

        return GameStateResponse(
            agents=agents_data,
            map_locations=locations_data,
            team_scores=world_state.team_scores,
            game_time=world_state.game_time,
            game_status=world_state.game_status,
            team_communications=comms_data,
        )

    def _calculate_avg_weapon_condition(self, agents) -> float:
        """Calculate average weapon condition for a set of agents."""
        if not agents:
            return 0.0

        condition_values = {
            WeaponCondition.EXCELLENT: 4,
            WeaponCondition.GOOD: 3,
            WeaponCondition.DAMAGED: 2,
            WeaponCondition.BROKEN: 1,
        }

        total_value = sum(
            condition_values.get(agent.weapon_condition, 1) for agent in agents
        )
        return total_value / len(agents)

    async def _graceful_shutdown(self) -> None:
        """Perform graceful shutdown of the server."""
        try:
            # Stop game loop
            await self._stop_game()

            # Close all WebSocket connections
            if self.connection_manager.active_connections:
                disconnect_message = {
                    "type": "server_shutdown",
                    "data": {"reason": "Server is shutting down"},
                    "timestamp": time.time(),
                }
                await self.connection_manager.broadcast(disconnect_message)

                # Give connections time to close gracefully
                await asyncio.sleep(1.0)

            logger.info("Graceful shutdown completed")

        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
        """
        Run the FastAPI server.

        Args:
            host: Host to bind to
            port: Port to bind to
            **kwargs: Additional uvicorn configuration
        """

        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self._graceful_shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run server
        config = {
            "host": host,
            "port": port,
            "log_level": "info",
            "access_log": True,
            **kwargs,
        }

        logger.info(f"Starting Battle API Server on {host}:{port}")
        uvicorn.run(self.app, **config)


def main():
    """Main entry point for the server."""
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    server = BattleAPIServer(openai_api_key=api_key)
    server.run()


if __name__ == "__main__":
    main()
