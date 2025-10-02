"""
Pygame-based battle renderer for the multi-agent demo.

This module provides the BattleRenderer class that visualizes the battle simulation
with agent rendering, team colors, health bars, weapon condition indicators,
map locations, and real-time game state updates from the backend API.
"""

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp
import pygame
import pygame.font
import pygame.gfxdraw
import websockets

from ..shared.constants import GAME_CONFIG, TEAM_COLORS
from ..shared.enums import AgentStatus, LocationType, WeaponCondition
from ..shared.models import Agent, BattleWorldState, MapLocation, TeamCommunication
from .performance_monitor import FrontendPerformanceMonitor, RenderingOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BattleRenderer:
    """
    Pygame-based renderer for the battle simulation with real-time updates.

    Features:
    - Agent rendering with team colors, health bars, and weapon condition indicators
    - Map location rendering for forges and strategic points
    - Real-time game state fetching from backend API
    - Optional manual controls (SPACE/ESC) while maintaining auto-start functionality
    - Team communication display with message history
    """

    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        window_size: tuple[int, int] = (800, 600),
    ):
        """
        Initialize the battle renderer.

        Args:
            backend_url: URL of the backend API server
            window_size: Window dimensions (width, height)
        """
        # Configuration
        self.backend_url = backend_url.rstrip("/")
        self.window_size = window_size
        self.window_width, self.window_height = window_size

        # Game world to screen scaling
        self.world_width = GAME_CONFIG["map_width"]
        self.world_height = GAME_CONFIG["map_height"]
        self.scale_x = (
            self.window_width - 200
        ) / self.world_width  # Leave space for UI
        self.scale_y = (
            self.window_height - 100
        ) / self.world_height  # Leave space for UI
        self.offset_x = 20
        self.offset_y = 20

        # Pygame components
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font_small: pygame.font.Font | None = None
        self.font_medium: pygame.font.Font | None = None
        self.font_large: pygame.font.Font | None = None

        # Game state
        self.game_state: BattleWorldState | None = None
        self.last_update_time = 0.0
        self.connection_status = "disconnected"
        self.error_message: str | None = None

        # UI state
        self.show_debug_info = True
        self.show_communication = True
        self.paused = False
        self.running = True

        # Network components
        self.session: aiohttp.ClientSession | None = None
        self.websocket: websockets.WebSocketServerProtocol | None = None
        self.websocket_task: asyncio.Task | None = None

        # Performance monitoring
        self.performance_monitor = FrontendPerformanceMonitor(
            target_fps=60.0, backend_url=backend_url
        )
        self.rendering_optimizer = RenderingOptimizer(self.performance_monitor)

        # Colors
        self.colors = {
            "background": (20, 25, 30),
            "ui_background": (40, 45, 50),
            "text": (255, 255, 255),
            "text_dim": (180, 180, 180),
            "health_bg": (60, 60, 60),
            "health_good": (0, 200, 0),
            "health_medium": (255, 255, 0),
            "health_low": (255, 100, 0),
            "health_critical": (255, 0, 0),
            "weapon_excellent": (0, 255, 0),
            "weapon_good": (255, 255, 0),
            "weapon_damaged": (255, 150, 0),
            "weapon_broken": (255, 0, 0),
            "forge_team_a": (100, 150, 255),
            "forge_team_b": (255, 150, 100),
            "communication_bg": (30, 30, 40),
            "border": (100, 100, 100),
            "error": (255, 100, 100),
            "success": (100, 255, 100),
        }

    async def initialize(self) -> None:
        """Initialize Pygame and network components."""
        try:
            # Initialize Pygame
            pygame.init()
            pygame.font.init()

            # Create display
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Gunn Battle Demo - Multi-Agent Simulation")

            # Initialize clock
            self.clock = pygame.time.Clock()

            # Initialize fonts
            try:
                self.font_small = pygame.font.Font(None, 16)
                self.font_medium = pygame.font.Font(None, 20)
                self.font_large = pygame.font.Font(None, 24)
            except Exception:
                # Fallback to default font
                self.font_small = pygame.font.SysFont("arial", 14)
                self.font_medium = pygame.font.SysFont("arial", 18)
                self.font_large = pygame.font.SysFont("arial", 22)

            # Initialize network session
            self.session = aiohttp.ClientSession()

            # Start WebSocket connection
            await self._connect_websocket()

            logger.info("Battle renderer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize renderer: {e}")
            self.error_message = f"Initialization failed: {e!s}"
            raise

    async def _connect_websocket(self) -> None:
        """Connect to the backend WebSocket for real-time updates."""
        try:
            ws_url = (
                self.backend_url.replace("http://", "ws://").replace(
                    "https://", "wss://"
                )
                + "/ws"
            )

            self.websocket = await websockets.connect(ws_url)
            self.connection_status = "connected"

            # Start WebSocket listener task
            self.websocket_task = asyncio.create_task(self._websocket_listener())

            logger.info(f"Connected to WebSocket: {ws_url}")

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self.connection_status = "error"
            self.error_message = f"WebSocket connection failed: {e!s}"

    async def _websocket_listener(self) -> None:
        """Listen for WebSocket messages and update game state."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type", "")

                    if message_type == "game_state_update":
                        await self._update_game_state(data.get("data", {}))
                    elif message_type == "game_ended":
                        logger.info(f"Game ended: {data.get('data', {})}")
                    elif message_type == "keepalive":
                        # Send pong response
                        await self.websocket.send(
                            json.dumps({"type": "pong", "timestamp": time.time()})
                        )

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from WebSocket: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.connection_status = "disconnected"
        except Exception as e:
            logger.error(f"WebSocket listener error: {e}")
            self.connection_status = "error"
            self.error_message = f"WebSocket error: {e!s}"

    async def _update_game_state(self, state_data: dict[str, Any]) -> None:
        """Update the game state from received data."""
        try:
            # Convert the received data back to our models
            agents = {}
            for agent_id, agent_data in state_data.get("agents", {}).items():
                agents[agent_id] = Agent(**agent_data)

            map_locations = {}
            for loc_id, loc_data in state_data.get("map_locations", {}).items():
                map_locations[loc_id] = MapLocation(**loc_data)

            team_communications = {}
            for team, messages_data in state_data.get(
                "team_communications", {}
            ).items():
                team_communications[team] = [
                    TeamCommunication(**msg_data) for msg_data in messages_data
                ]

            # Create updated world state
            self.game_state = BattleWorldState(
                agents=agents,
                map_locations=map_locations,
                team_scores=state_data.get("team_scores", {}),
                game_time=state_data.get("game_time", 0.0),
                game_status=state_data.get("game_status", "active"),
                team_communications=team_communications,
            )

            self.last_update_time = time.time()

        except Exception as e:
            logger.error(f"Error updating game state: {e}")
            self.error_message = f"State update error: {e!s}"

    async def fetch_initial_game_state(self) -> None:
        """Fetch initial game state from the backend API."""
        try:
            if not self.session:
                raise RuntimeError("HTTP session not initialized")

            async with self.session.get(
                f"{self.backend_url}/api/game/state"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    await self._update_game_state(data)
                    logger.info("Initial game state fetched successfully")
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {error_text}")

        except Exception as e:
            logger.error(f"Failed to fetch initial game state: {e}")
            self.error_message = f"Failed to fetch game state: {e!s}"

    def world_to_screen(self, world_pos: tuple[float, float]) -> tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        x, y = world_pos
        screen_x = int(self.offset_x + x * self.scale_x)
        screen_y = int(self.offset_y + y * self.scale_y)
        return screen_x, screen_y

    def screen_to_world(self, screen_pos: tuple[int, int]) -> tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        x, y = screen_pos
        world_x = (x - self.offset_x) / self.scale_x
        world_y = (y - self.offset_y) / self.scale_y
        return world_x, world_y

    def render_agent(self, agent: Agent) -> None:
        """Render an agent with team colors, health bar, and weapon condition."""
        if not self.screen:
            return

        # Get screen position
        screen_x, screen_y = self.world_to_screen(agent.position)

        # Agent color based on team and status
        if agent.status == AgentStatus.DEAD:
            agent_color = (100, 100, 100)  # Gray for dead agents
        else:
            agent_color = TEAM_COLORS.get(agent.team, TEAM_COLORS["neutral"])

        # Agent size based on status
        agent_radius = 12 if agent.status == AgentStatus.ALIVE else 8

        # Draw agent circle
        pygame.draw.circle(self.screen, agent_color, (screen_x, screen_y), agent_radius)
        pygame.draw.circle(
            self.screen, self.colors["border"], (screen_x, screen_y), agent_radius, 2
        )

        # Draw agent ID
        agent_text = self.font_small.render(
            agent.agent_id.split("_")[-1], True, self.colors["text"]
        )
        text_rect = agent_text.get_rect(center=(screen_x, screen_y))
        self.screen.blit(agent_text, text_rect)

        if agent.status == AgentStatus.ALIVE:
            # Draw health bar
            self._draw_health_bar(screen_x, screen_y - agent_radius - 15, agent.health)

            # Draw weapon condition indicator
            self._draw_weapon_condition(
                screen_x, screen_y + agent_radius + 5, agent.weapon_condition
            )

            # Draw vision range (if debug mode)
            if self.show_debug_info:
                vision_radius = int(agent.vision_range * self.scale_x)
                pygame.draw.circle(
                    self.screen,
                    (*agent_color, 30),
                    (screen_x, screen_y),
                    vision_radius,
                    1,
                )

    def _draw_health_bar(self, x: int, y: int, health: int) -> None:
        """Draw a health bar above an agent."""
        if not self.screen:
            return

        bar_width = 24
        bar_height = 4

        # Background
        bg_rect = pygame.Rect(x - bar_width // 2, y, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.colors["health_bg"], bg_rect)

        # Health fill
        health_ratio = max(0, min(1, health / 100))
        fill_width = int(bar_width * health_ratio)

        if health_ratio > 0:
            # Color based on health level
            if health_ratio > 0.7:
                health_color = self.colors["health_good"]
            elif health_ratio > 0.4:
                health_color = self.colors["health_medium"]
            elif health_ratio > 0.2:
                health_color = self.colors["health_low"]
            else:
                health_color = self.colors["health_critical"]

            fill_rect = pygame.Rect(x - bar_width // 2, y, fill_width, bar_height)
            pygame.draw.rect(self.screen, health_color, fill_rect)

        # Border
        pygame.draw.rect(self.screen, self.colors["border"], bg_rect, 1)

    def _draw_weapon_condition(
        self, x: int, y: int, condition: WeaponCondition
    ) -> None:
        """Draw weapon condition indicator below an agent."""
        if not self.screen:
            return

        # Color based on weapon condition
        condition_colors = {
            WeaponCondition.EXCELLENT: self.colors["weapon_excellent"],
            WeaponCondition.GOOD: self.colors["weapon_good"],
            WeaponCondition.DAMAGED: self.colors["weapon_damaged"],
            WeaponCondition.BROKEN: self.colors["weapon_broken"],
        }

        color = condition_colors.get(condition, self.colors["weapon_broken"])

        # Draw small indicator circle
        pygame.draw.circle(self.screen, color, (x, y), 3)
        pygame.draw.circle(self.screen, self.colors["border"], (x, y), 3, 1)

    def render_map_location(self, location_id: str, location: MapLocation) -> None:
        """Render map locations like forges and strategic points."""
        if not self.screen:
            return

        screen_x, screen_y = self.world_to_screen(location.position)

        # Color and size based on location type
        if location.location_type == LocationType.FORGE:
            # Determine team based on location ID
            if "forge_a" in location_id:
                color = self.colors["forge_team_a"]
                team_text = "A"
            elif "forge_b" in location_id:
                color = self.colors["forge_team_b"]
                team_text = "B"
            else:
                color = self.colors["ui_background"]
                team_text = "?"

            # Draw forge as a square
            size = int(location.radius * self.scale_x)
            rect = pygame.Rect(screen_x - size, screen_y - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.colors["border"], rect, 2)

            # Draw team indicator
            team_surface = self.font_medium.render(team_text, True, self.colors["text"])
            team_rect = team_surface.get_rect(center=(screen_x, screen_y))
            self.screen.blit(team_surface, team_rect)

        elif location.location_type == LocationType.COVER:
            # Draw cover as a filled circle
            radius = int(location.radius * self.scale_x)
            pygame.draw.circle(
                self.screen, self.colors["ui_background"], (screen_x, screen_y), radius
            )
            pygame.draw.circle(
                self.screen, self.colors["border"], (screen_x, screen_y), radius, 2
            )

        # Draw location label
        if self.show_debug_info:
            label = location_id.replace("_", " ").title()
            label_surface = self.font_small.render(label, True, self.colors["text_dim"])
            label_rect = label_surface.get_rect(center=(screen_x, screen_y + 25))
            self.screen.blit(label_surface, label_rect)

    def render_ui(self) -> None:
        """Render the user interface with game information and controls."""
        if not self.screen or not self.game_state:
            return

        # UI panel background
        ui_rect = pygame.Rect(self.window_width - 180, 0, 180, self.window_height)
        pygame.draw.rect(self.screen, self.colors["ui_background"], ui_rect)
        pygame.draw.line(
            self.screen,
            self.colors["border"],
            (self.window_width - 180, 0),
            (self.window_width - 180, self.window_height),
            2,
        )

        y_offset = 10

        # Game status
        status_text = f"Status: {self.game_state.game_status.replace('_', ' ').title()}"
        status_surface = self.font_medium.render(status_text, True, self.colors["text"])
        self.screen.blit(status_surface, (self.window_width - 170, y_offset))
        y_offset += 25

        # Game time
        time_text = f"Time: {self.game_state.game_time:.1f}s"
        time_surface = self.font_small.render(time_text, True, self.colors["text_dim"])
        self.screen.blit(time_surface, (self.window_width - 170, y_offset))
        y_offset += 20

        # Connection status
        conn_color = (
            self.colors["success"]
            if self.connection_status == "connected"
            else self.colors["error"]
        )
        conn_text = f"Connection: {self.connection_status}"
        conn_surface = self.font_small.render(conn_text, True, conn_color)
        self.screen.blit(conn_surface, (self.window_width - 170, y_offset))
        y_offset += 30

        # Team scores
        for team, score in self.game_state.team_scores.items():
            team_color = TEAM_COLORS.get(team, self.colors["text"])
            team_name = team.replace("_", " ").title()
            score_text = f"{team_name}: {score}"
            score_surface = self.font_medium.render(score_text, True, team_color)
            self.screen.blit(score_surface, (self.window_width - 170, y_offset))
            y_offset += 25

        y_offset += 10

        # Team statistics
        for team in ["team_a", "team_b"]:
            team_agents = self.game_state.get_alive_agents(team)
            team_color = TEAM_COLORS.get(team, self.colors["text"])
            team_name = team.replace("_", " ").title()

            # Team header
            header_text = f"{team_name} ({len(team_agents)} alive)"
            header_surface = self.font_small.render(header_text, True, team_color)
            self.screen.blit(header_surface, (self.window_width - 170, y_offset))
            y_offset += 18

            # Agent details
            for agent_id, agent in team_agents.items():
                agent_name = agent_id.split("_")[-1]
                health_text = f"  {agent_name}: {agent.health}HP"
                health_surface = self.font_small.render(
                    health_text, True, self.colors["text_dim"]
                )
                self.screen.blit(health_surface, (self.window_width - 170, y_offset))
                y_offset += 15

            y_offset += 10

        # Controls
        y_offset = self.window_height - 120
        controls_title = self.font_small.render("Controls:", True, self.colors["text"])
        self.screen.blit(controls_title, (self.window_width - 170, y_offset))
        y_offset += 18

        controls = [
            "SPACE: Pause/Resume",
            "ESC: Quit",
            "D: Toggle Debug",
            "C: Toggle Chat",
        ]

        for control in controls:
            control_surface = self.font_small.render(
                control, True, self.colors["text_dim"]
            )
            self.screen.blit(control_surface, (self.window_width - 170, y_offset))
            y_offset += 15

    def render_team_communications(self) -> None:
        """Render team communication messages with urgency-based prioritization."""
        if not self.screen or not self.game_state or not self.show_communication:
            return

        # Communication panel
        comm_rect = pygame.Rect(10, self.window_height - 180, 450, 170)
        pygame.draw.rect(self.screen, self.colors["communication_bg"], comm_rect)
        pygame.draw.rect(self.screen, self.colors["border"], comm_rect, 2)

        # Title
        title_surface = self.font_medium.render(
            "Team Communications", True, self.colors["text"]
        )
        self.screen.blit(title_surface, (20, self.window_height - 175))

        y_offset = self.window_height - 155

        # Collect and prioritize messages from both teams
        all_messages = []
        for team in ["team_a", "team_b"]:
            # Get prioritized messages for each team
            prioritized_messages = self.game_state.get_prioritized_team_messages(
                team, 10
            )
            for msg in prioritized_messages:
                all_messages.append((team, msg))

        # Sort all messages by urgency (high -> medium -> low) then by timestamp (newest first)
        urgency_priority = {"high": 3, "medium": 2, "low": 1}

        def sort_key(item):
            team, msg = item
            urgency = getattr(msg, "urgency", "medium")
            timestamp = getattr(msg, "timestamp", 0.0)
            return (urgency_priority.get(urgency, 2), timestamp)

        all_messages.sort(key=sort_key, reverse=True)

        # Display messages (show up to 8 messages)
        displayed_count = 0
        for team, msg in all_messages:
            if displayed_count >= 8:
                break

            team_color = TEAM_COLORS.get(team, self.colors["text"])

            # Format message with enhanced urgency indicators
            sender_name = (
                msg.sender_id.split("_")[-1] if hasattr(msg, "sender_id") else "Unknown"
            )
            urgency = getattr(msg, "urgency", "medium")
            message_content = getattr(msg, "message", str(msg))

            # Urgency indicators and colors
            if urgency == "high":
                urgency_indicator = "!!!"
                urgency_color = self.colors["error"]  # Red for high urgency
            elif urgency == "medium":
                urgency_indicator = "!"
                urgency_color = (255, 255, 0)  # Yellow for medium urgency
            else:
                urgency_indicator = ""
                urgency_color = team_color

            # Truncate message to fit
            max_message_length = 45
            if len(message_content) > max_message_length:
                message_content = message_content[: max_message_length - 3] + "..."

            # Format final message text
            if urgency_indicator:
                message_text = f"{sender_name} {urgency_indicator}: {message_content}"
            else:
                message_text = f"{sender_name}: {message_content}"

            # Render message with urgency color for high priority messages
            display_color = urgency_color if urgency == "high" else team_color
            msg_surface = self.font_small.render(message_text, True, display_color)
            self.screen.blit(msg_surface, (20, y_offset))

            # Add timestamp for recent messages (last 30 seconds)
            current_time = getattr(self.game_state, "game_time", 0.0)
            msg_time = getattr(msg, "timestamp", 0.0)
            if current_time - msg_time < 30.0:  # Show timestamp for recent messages
                time_text = f"({msg_time:.1f}s)"
                time_surface = self.font_small.render(
                    time_text, True, self.colors["text_dim"]
                )
                self.screen.blit(time_surface, (400, y_offset))

            y_offset += 16
            displayed_count += 1

        # Show communication stats
        if all_messages:
            stats_y = self.window_height - 25
            team_a_count = len([m for t, m in all_messages if t == "team_a"])
            team_b_count = len([m for t, m in all_messages if t == "team_b"])

            stats_text = f"Messages: Team A({team_a_count}) Team B({team_b_count})"
            stats_surface = self.font_small.render(
                stats_text, True, self.colors["text_dim"]
            )
            self.screen.blit(stats_surface, (20, stats_y))

    def render_error_message(self) -> None:
        """Render error message if present."""
        if not self.screen or not self.error_message:
            return

        # Error panel
        error_rect = pygame.Rect(50, 50, self.window_width - 100, 100)
        pygame.draw.rect(self.screen, self.colors["error"], error_rect)
        pygame.draw.rect(self.screen, self.colors["border"], error_rect, 3)

        # Error text
        error_title = self.font_large.render("Error", True, self.colors["text"])
        self.screen.blit(error_title, (60, 60))

        # Wrap error message
        words = self.error_message.split(" ")
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            if self.font_small.size(test_line)[0] < self.window_width - 120:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        for i, line in enumerate(lines[:3]):  # Max 3 lines
            line_surface = self.font_small.render(line, True, self.colors["text"])
            self.screen.blit(line_surface, (60, 85 + i * 15))

    def render_frame(self) -> None:
        """Render a complete frame with performance monitoring."""
        if not self.screen:
            return

        # Start frame timing
        frame_start_time = self.performance_monitor.start_frame()
        render_start_time = time.perf_counter()

        # Clear screen
        self.screen.fill(self.colors["background"])

        # Render error message if present
        if self.error_message:
            self.render_error_message()
            return

        # Render game elements if we have game state
        if self.game_state:
            # Render map locations first (background)
            for location_id, location in self.game_state.map_locations.items():
                self.render_map_location(location_id, location)

            # Optimize agent rendering based on performance
            screen_bounds = (0, 0, self.window_width, self.window_height)
            agents_to_render = self.rendering_optimizer.optimize_agent_rendering(
                list(self.game_state.agents.values()), screen_bounds
            )

            # Render optimized agent list
            for agent in agents_to_render:
                self.render_agent(agent)

            # Render UI
            self.render_ui()

            # Render team communications
            self.render_team_communications()
        else:
            # Show loading message
            loading_text = self.font_large.render(
                "Loading game state...", True, self.colors["text"]
            )
            loading_rect = loading_text.get_rect(
                center=(self.window_width // 2, self.window_height // 2)
            )
            self.screen.blit(loading_text, loading_rect)

        # Update display
        pygame.display.flip()

        # End frame timing
        render_end_time = time.perf_counter()
        render_time = render_end_time - render_start_time

        # Count agents and UI elements for performance tracking
        total_agents = len(self.game_state.agents) if self.game_state else 0
        visible_agents = (
            len(
                [
                    a
                    for a in self.game_state.agents.values()
                    if a.status == AgentStatus.ALIVE
                ]
            )
            if self.game_state
            else 0
        )
        ui_elements = 10  # Approximate count of UI elements

        # Record frame performance
        self.performance_monitor.end_frame(
            frame_start_time=frame_start_time,
            render_time=render_time,
            event_time=0.0,  # Will be updated by handle_events
            network_time=0.0,  # Will be updated by network operations
            total_agents=total_agents,
            visible_agents=visible_agents,
            ui_elements=ui_elements,
        )

    def handle_events(self) -> None:
        """Handle Pygame events and manual controls."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    # Toggle pause (send control request to backend)
                    asyncio.create_task(self._toggle_pause())

                elif event.key == pygame.K_d:
                    # Toggle debug info
                    self.show_debug_info = not self.show_debug_info
                    logger.info(
                        f"Debug info: {'ON' if self.show_debug_info else 'OFF'}"
                    )

                elif event.key == pygame.K_c:
                    # Toggle communication display
                    self.show_communication = not self.show_communication
                    logger.info(
                        f"Communication display: {'ON' if self.show_communication else 'OFF'}"
                    )

                elif event.key == pygame.K_r:
                    # Reset game
                    asyncio.create_task(self._reset_game())

                elif event.key == pygame.K_F5:
                    # Refresh game state
                    asyncio.create_task(self.fetch_initial_game_state())

    async def _toggle_pause(self) -> None:
        """Toggle game pause state via backend API."""
        try:
            if not self.session:
                return

            action = "resume" if self.paused else "pause"

            async with self.session.post(
                f"{self.backend_url}/api/game/control", json={"action": action}
            ) as response:
                if response.status == 200:
                    self.paused = not self.paused
                    logger.info(f"Game {'paused' if self.paused else 'resumed'}")
                else:
                    logger.error(f"Failed to {action} game: {response.status}")

        except Exception as e:
            logger.error(f"Error toggling pause: {e}")

    async def _reset_game(self) -> None:
        """Reset the game via backend API."""
        try:
            if not self.session:
                return

            async with self.session.post(
                f"{self.backend_url}/api/game/control", json={"action": "reset"}
            ) as response:
                if response.status == 200:
                    logger.info("Game reset successfully")
                    self.error_message = None
                else:
                    logger.error(f"Failed to reset game: {response.status}")

        except Exception as e:
            logger.error(f"Error resetting game: {e}")

    async def run(self) -> None:
        """Main render loop with real-time updates."""
        try:
            # Initialize components
            await self.initialize()

            # Fetch initial game state
            await self.fetch_initial_game_state()

            logger.info("Starting render loop...")

            # Main loop
            while self.running:
                # Handle events
                self.handle_events()

                # Render frame
                self.render_frame()

                # Control frame rate
                if self.clock:
                    self.clock.tick(GAME_CONFIG["fps"])

                # Small async yield to allow other tasks to run
                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in render loop: {e}")
            self.error_message = f"Render loop error: {e!s}"
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Cancel WebSocket task
            if self.websocket_task and not self.websocket_task.done():
                self.websocket_task.cancel()
                try:
                    await self.websocket_task
                except asyncio.CancelledError:
                    pass

            # Close WebSocket
            if self.websocket:
                await self.websocket.close()

            # Close HTTP session
            if self.session:
                await self.session.close()

            # Quit Pygame
            pygame.quit()

            logger.info("Renderer cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main function to run the battle renderer."""
    import argparse

    parser = argparse.ArgumentParser(description="Gunn Battle Demo Frontend")
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Backend API URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--window-size",
        default="800x600",
        help="Window size in format WIDTHxHEIGHT (default: 800x600)",
    )

    args = parser.parse_args()

    # Parse window size
    try:
        width, height = map(int, args.window_size.split("x"))
        window_size = (width, height)
    except ValueError:
        logger.error("Invalid window size format. Use WIDTHxHEIGHT (e.g., 800x600)")
        return

    # Create and run renderer
    renderer = BattleRenderer(backend_url=args.backend_url, window_size=window_size)

    try:
        await renderer.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await renderer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
