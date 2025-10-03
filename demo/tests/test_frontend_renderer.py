"""
Tests for the battle demo frontend renderer.

This module tests the BattleRenderer class functionality including
coordinate conversion, rendering components, and network communication.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Mock pygame modules before importing
pygame_mock = MagicMock()
pygame_mock.QUIT = 256
pygame_mock.KEYDOWN = 768
pygame_mock.K_ESCAPE = 27
pygame_mock.K_SPACE = 32
pygame_mock.K_d = 100
pygame_mock.K_c = 99
pygame_mock.K_r = 114
pygame_mock.K_F5 = 286

with patch.dict(
    "sys.modules",
    {
        "pygame": pygame_mock,
        "pygame.font": MagicMock(),
        "pygame.gfxdraw": MagicMock(),
        "pygame.display": MagicMock(),
        "pygame.time": MagicMock(),
        "pygame.draw": MagicMock(),
        "aiohttp": MagicMock(),
        "websockets": MagicMock(),
    },
):
    from ..frontend.renderer import BattleRenderer

from ..shared.enums import AgentStatus, LocationType, WeaponCondition
from ..shared.models import Agent, BattleWorldState, MapLocation, TeamCommunication


class TestBattleRenderer:
    """Test cases for the BattleRenderer class."""

    @pytest.fixture
    def mock_pygame(self):
        """Mock pygame components."""
        with patch("demo.frontend.renderer.pygame") as mock_pg:
            # Mock pygame constants
            mock_pg.QUIT = 256
            mock_pg.KEYDOWN = 768
            mock_pg.K_ESCAPE = 27
            mock_pg.K_SPACE = 32
            mock_pg.K_d = 100
            mock_pg.K_c = 99
            mock_pg.K_r = 114
            mock_pg.K_F5 = 286

            # Mock pygame components
            mock_pg.init.return_value = None
            mock_pg.font.init.return_value = None
            mock_pg.display.set_mode.return_value = Mock()
            mock_pg.display.set_caption.return_value = None
            mock_pg.time.Clock.return_value = Mock()
            mock_pg.font.Font.return_value = Mock()
            mock_pg.font.SysFont.return_value = Mock()
            mock_pg.quit.return_value = None

            # Mock drawing functions
            mock_pg.draw.circle = Mock()
            mock_pg.draw.rect = Mock()
            mock_pg.draw.line = Mock()
            mock_pg.display.flip = Mock()
            mock_pg.event.get = Mock(return_value=[])

            # Mock font rendering
            mock_font = Mock()
            mock_surface = Mock()
            mock_surface.get_rect.return_value = Mock()
            mock_font.render.return_value = mock_surface
            mock_font.size.return_value = (100, 20)
            mock_pg.font.Font.return_value = mock_font
            mock_pg.font.SysFont.return_value = mock_font

            yield mock_pg

    @pytest.fixture
    def sample_game_state(self) -> BattleWorldState:
        """Create a sample game state for testing."""
        agents = {
            "team_a_agent_1": Agent(
                agent_id="team_a_agent_1",
                team="team_a",
                position=(30.0, 90.0),
                health=100,
                status=AgentStatus.ALIVE,
                weapon_condition=WeaponCondition.EXCELLENT,
            ),
            "team_b_agent_1": Agent(
                agent_id="team_b_agent_1",
                team="team_b",
                position=(170.0, 10.0),
                health=75,
                status=AgentStatus.ALIVE,
                weapon_condition=WeaponCondition.GOOD,
            ),
            "team_a_agent_2": Agent(
                agent_id="team_a_agent_2",
                team="team_a",
                position=(40.0, 85.0),
                health=0,
                status=AgentStatus.DEAD,
                weapon_condition=WeaponCondition.BROKEN,
            ),
        }

        map_locations = {
            "forge_a": MapLocation(
                position=(20.0, 80.0), location_type=LocationType.FORGE, radius=5.0
            ),
            "forge_b": MapLocation(
                position=(180.0, 20.0), location_type=LocationType.FORGE, radius=5.0
            ),
        }

        team_communications = {
            "team_a": [
                TeamCommunication(
                    sender_id="team_a_agent_1",
                    team="team_a",
                    message="Enemy spotted at forge B!",
                    urgency="high",
                    timestamp=10.5,
                )
            ],
            "team_b": [
                TeamCommunication(
                    sender_id="team_b_agent_1",
                    team="team_b",
                    message="Moving to attack position",
                    urgency="medium",
                    timestamp=12.0,
                )
            ],
        }

        return BattleWorldState(
            agents=agents,
            map_locations=map_locations,
            team_scores={"team_a": 10, "team_b": 5},
            game_time=15.0,
            game_status="active",
            team_communications=team_communications,
        )

    @pytest.fixture
    def renderer(self, mock_pygame) -> BattleRenderer:
        """Create a BattleRenderer instance for testing."""
        return BattleRenderer(
            backend_url="http://localhost:8000", window_size=(800, 600)
        )

    def test_initialization(self, renderer: BattleRenderer):
        """Test renderer initialization."""
        assert renderer.backend_url == "http://localhost:8000"
        assert renderer.window_size == (800, 600)
        assert renderer.window_width == 800
        assert renderer.window_height == 600
        assert renderer.world_width == 200.0
        assert renderer.world_height == 100.0
        assert renderer.game_state is None
        assert renderer.connection_status == "disconnected"
        assert renderer.running is True

    def test_coordinate_conversion(self, renderer: BattleRenderer):
        """Test world to screen coordinate conversion."""
        # Test world to screen conversion
        world_pos = (100.0, 50.0)  # Center of world
        screen_pos = renderer.world_to_screen(world_pos)

        # Should be roughly in the center of the game area
        expected_x = int(renderer.offset_x + 100.0 * renderer.scale_x)
        expected_y = int(renderer.offset_y + 50.0 * renderer.scale_y)
        assert screen_pos == (expected_x, expected_y)

        # Test screen to world conversion (inverse)
        converted_world = renderer.screen_to_world(screen_pos)
        assert abs(converted_world[0] - world_pos[0]) < 0.1
        assert abs(converted_world[1] - world_pos[1]) < 0.1

    def test_coordinate_conversion_edges(self, renderer: BattleRenderer):
        """Test coordinate conversion at world edges."""
        # Test origin
        origin_screen = renderer.world_to_screen((0.0, 0.0))
        assert origin_screen == (renderer.offset_x, renderer.offset_y)

        # Test max coordinates
        max_world = (renderer.world_width, renderer.world_height)
        max_screen = renderer.world_to_screen(max_world)

        expected_max_x = int(
            renderer.offset_x + renderer.world_width * renderer.scale_x
        )
        expected_max_y = int(
            renderer.offset_y + renderer.world_height * renderer.scale_y
        )
        assert max_screen == (expected_max_x, expected_max_y)

    @pytest.mark.asyncio
    async def test_update_game_state(
        self, renderer: BattleRenderer, sample_game_state: BattleWorldState
    ):
        """Test game state update from API data."""
        # Convert game state to API format
        state_data = {
            "agents": {
                agent_id: agent.model_dump()
                for agent_id, agent in sample_game_state.agents.items()
            },
            "map_locations": {
                loc_id: location.model_dump()
                for loc_id, location in sample_game_state.map_locations.items()
            },
            "team_scores": sample_game_state.team_scores,
            "game_time": sample_game_state.game_time,
            "game_status": sample_game_state.game_status,
            "team_communications": {
                team: [msg.model_dump() for msg in messages]
                for team, messages in sample_game_state.team_communications.items()
            },
        }

        # Update game state
        await renderer._update_game_state(state_data)

        # Verify state was updated correctly
        assert renderer.game_state is not None
        assert len(renderer.game_state.agents) == 3
        assert len(renderer.game_state.map_locations) == 2
        assert renderer.game_state.team_scores == {"team_a": 10, "team_b": 5}
        assert renderer.game_state.game_time == 15.0
        assert renderer.game_state.game_status == "active"

        # Check specific agent
        agent = renderer.game_state.agents["team_a_agent_1"]
        assert agent.team == "team_a"
        assert agent.position == (30.0, 90.0)
        assert agent.health == 100
        assert agent.status == AgentStatus.ALIVE

    @pytest.mark.asyncio
    async def test_update_game_state_invalid_data(self, renderer: BattleRenderer):
        """Test game state update with invalid data."""
        # Test with invalid agent data
        invalid_state_data = {
            "agents": {
                "invalid_agent": {
                    "agent_id": "invalid_agent",
                    "team": "invalid_team",  # Invalid team
                    "position": (30.0, 90.0),
                    "health": 100,
                }
            },
            "map_locations": {},
            "team_scores": {},
            "game_time": 0.0,
            "game_status": "active",
            "team_communications": {},
        }

        # Should handle the error gracefully
        await renderer._update_game_state(invalid_state_data)

        # Error should be recorded
        assert renderer.error_message is not None
        assert "State update error" in renderer.error_message

    def test_render_agent_alive(self, renderer: BattleRenderer, mock_pygame):
        """Test rendering of alive agents."""
        # Create mock screen
        renderer.screen = Mock()
        renderer.font_small = Mock()
        renderer.font_small.render.return_value = Mock()

        # Create test agent
        agent = Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(50.0, 50.0),
            health=75,
            status=AgentStatus.ALIVE,
            weapon_condition=WeaponCondition.GOOD,
        )

        # Render agent
        renderer.render_agent(agent)

        # Verify drawing calls were made
        assert mock_pygame.draw.circle.called
        assert renderer.screen.blit.called

    def test_render_agent_dead(self, renderer: BattleRenderer, mock_pygame):
        """Test rendering of dead agents."""
        # Create mock screen
        renderer.screen = Mock()
        renderer.font_small = Mock()
        renderer.font_small.render.return_value = Mock()

        # Create dead agent
        agent = Agent(
            agent_id="team_b_agent_1",
            team="team_b",
            position=(150.0, 25.0),
            health=0,
            status=AgentStatus.DEAD,
            weapon_condition=WeaponCondition.BROKEN,
        )

        # Render agent
        renderer.render_agent(agent)

        # Verify drawing calls were made (dead agents still rendered)
        assert mock_pygame.draw.circle.called

    def test_render_map_location_forge(self, renderer: BattleRenderer, mock_pygame):
        """Test rendering of forge locations."""
        # Create mock screen
        renderer.screen = Mock()
        renderer.font_medium = Mock()
        renderer.font_medium.render.return_value = Mock()
        renderer.font_small = Mock()
        renderer.font_small.render.return_value = Mock()

        # Create forge location
        forge = MapLocation(
            position=(20.0, 80.0), location_type=LocationType.FORGE, radius=5.0
        )

        # Render forge
        renderer.render_map_location("forge_a", forge)

        # Verify drawing calls were made
        assert mock_pygame.draw.rect.called
        assert renderer.screen.blit.called

    def test_render_ui_with_game_state(
        self, renderer: BattleRenderer, sample_game_state: BattleWorldState, mock_pygame
    ):
        """Test UI rendering with game state."""
        # Setup mocks
        renderer.screen = Mock()
        renderer.font_small = Mock()
        renderer.font_medium = Mock()
        renderer.font_large = Mock()

        mock_surface = Mock()
        mock_surface.get_rect.return_value = Mock()
        renderer.font_small.render.return_value = mock_surface
        renderer.font_medium.render.return_value = mock_surface
        renderer.font_large.render.return_value = mock_surface

        # Set game state
        renderer.game_state = sample_game_state

        # Render UI
        renderer.render_ui()

        # Verify drawing calls were made
        assert mock_pygame.draw.rect.called
        assert mock_pygame.draw.line.called
        assert renderer.screen.blit.called

    def test_render_team_communications(
        self, renderer: BattleRenderer, sample_game_state: BattleWorldState, mock_pygame
    ):
        """Test team communication rendering."""
        # Setup mocks
        renderer.screen = Mock()
        renderer.font_small = Mock()
        renderer.font_medium = Mock()

        mock_surface = Mock()
        renderer.font_small.render.return_value = mock_surface
        renderer.font_medium.render.return_value = mock_surface

        # Set game state and enable communication display
        renderer.game_state = sample_game_state
        renderer.show_communication = True

        # Render communications
        renderer.render_team_communications()

        # Verify drawing calls were made
        assert mock_pygame.draw.rect.called
        assert renderer.screen.blit.called

    def test_render_error_message(self, renderer: BattleRenderer, mock_pygame):
        """Test error message rendering."""
        # Setup mocks
        renderer.screen = Mock()
        renderer.font_small = Mock()
        renderer.font_large = Mock()

        mock_surface = Mock()
        renderer.font_small.render.return_value = mock_surface
        renderer.font_large.render.return_value = mock_surface
        renderer.font_small.size.return_value = (100, 15)

        # Set error message
        renderer.error_message = "Test error message that is quite long and should wrap"

        # Render error
        renderer.render_error_message()

        # Verify drawing calls were made
        assert mock_pygame.draw.rect.called
        assert renderer.screen.blit.called

    def test_render_frame_no_game_state(self, renderer: BattleRenderer, mock_pygame):
        """Test frame rendering without game state."""
        # Setup mocks
        renderer.screen = Mock()
        renderer.font_large = Mock()
        mock_surface = Mock()
        mock_surface.get_rect.return_value = Mock()
        renderer.font_large.render.return_value = mock_surface

        # Render frame
        renderer.render_frame()

        # Should show loading message
        assert renderer.font_large.render.called
        assert renderer.screen.blit.called
        assert mock_pygame.display.flip.called

    def test_render_frame_with_error(self, renderer: BattleRenderer, mock_pygame):
        """Test frame rendering with error message."""
        # Setup mocks
        renderer.screen = Mock()
        renderer.font_small = Mock()
        renderer.font_large = Mock()

        mock_surface = Mock()
        renderer.font_small.render.return_value = mock_surface
        renderer.font_large.render.return_value = mock_surface
        renderer.font_small.size.return_value = (100, 15)

        # Set error message
        renderer.error_message = "Test error"

        # Render frame
        renderer.render_frame()

        # Should show error message
        assert renderer.font_large.render.called
        assert mock_pygame.display.flip.called

    @pytest.mark.asyncio
    async def test_fetch_initial_game_state_success(self, renderer: BattleRenderer):
        """Test successful initial game state fetch."""
        # Mock HTTP session with proper async context manager
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "agents": {},
                "map_locations": {},
                "team_scores": {"team_a": 0, "team_b": 0},
                "game_time": 0.0,
                "game_status": "active",
                "team_communications": {"team_a": [], "team_b": []},
            }
        )

        mock_session = Mock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None
        mock_session.get.return_value = mock_context_manager
        renderer.session = mock_session

        # Fetch initial state
        await renderer.fetch_initial_game_state()

        # Verify state was updated
        assert renderer.game_state is not None
        assert renderer.game_state.game_status == "active"

    @pytest.mark.asyncio
    async def test_fetch_initial_game_state_error(self, renderer: BattleRenderer):
        """Test initial game state fetch with HTTP error."""
        # Mock HTTP session with error
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session = Mock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None
        mock_session.get.return_value = mock_context_manager
        renderer.session = mock_session

        # Fetch initial state
        await renderer.fetch_initial_game_state()

        # Should have error message
        assert renderer.error_message is not None
        assert "Failed to fetch game state" in renderer.error_message

    @pytest.mark.asyncio
    async def test_toggle_pause(self, renderer: BattleRenderer):
        """Test pause toggle functionality."""
        # Mock HTTP session
        mock_response = Mock()
        mock_response.status = 200

        mock_session = Mock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None
        mock_session.post.return_value = mock_context_manager
        renderer.session = mock_session

        # Test pause
        renderer.paused = False
        await renderer._toggle_pause()

        # Verify pause state changed
        assert renderer.paused is True

        # Verify API call was made
        mock_session.post.assert_called_with(
            "http://localhost:8000/api/game/control", json={"action": "pause"}
        )

    @pytest.mark.asyncio
    async def test_reset_game(self, renderer: BattleRenderer):
        """Test game reset functionality."""
        # Mock HTTP session
        mock_response = Mock()
        mock_response.status = 200

        mock_session = Mock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_response
        mock_context_manager.__aexit__.return_value = None
        mock_session.post.return_value = mock_context_manager
        renderer.session = mock_session

        # Set error message to test clearing
        renderer.error_message = "Previous error"

        # Reset game
        await renderer._reset_game()

        # Verify API call was made
        mock_session.post.assert_called_with(
            "http://localhost:8000/api/game/control", json={"action": "reset"}
        )

        # Error message should be cleared
        assert renderer.error_message is None

    def test_handle_events_quit(self, renderer: BattleRenderer, mock_pygame):
        """Test quit event handling."""
        # Mock quit event
        quit_event = Mock()
        quit_event.type = mock_pygame.QUIT

        mock_pygame.event.get.return_value = [quit_event]

        # Handle events
        renderer.handle_events()

        # Should set running to False
        assert renderer.running is False

    def test_handle_events_escape(self, renderer: BattleRenderer, mock_pygame):
        """Test escape key event handling."""
        # Mock escape key event
        key_event = Mock()
        key_event.type = mock_pygame.KEYDOWN
        key_event.key = mock_pygame.K_ESCAPE

        mock_pygame.event.get.return_value = [key_event]

        # Handle events
        renderer.handle_events()

        # Should set running to False
        assert renderer.running is False

    def test_handle_events_debug_toggle(self, renderer: BattleRenderer, mock_pygame):
        """Test debug info toggle."""
        # Mock 'D' key event
        key_event = Mock()
        key_event.type = mock_pygame.KEYDOWN
        key_event.key = mock_pygame.K_d

        mock_pygame.event.get.return_value = [key_event]

        # Initial state
        initial_debug = renderer.show_debug_info

        # Handle events
        renderer.handle_events()

        # Should toggle debug info
        assert renderer.show_debug_info != initial_debug

    def test_handle_events_communication_toggle(
        self, renderer: BattleRenderer, mock_pygame
    ):
        """Test communication display toggle."""
        # Mock 'C' key event
        key_event = Mock()
        key_event.type = mock_pygame.KEYDOWN
        key_event.key = mock_pygame.K_c

        mock_pygame.event.get.return_value = [key_event]

        # Initial state
        initial_comm = renderer.show_communication

        # Handle events
        renderer.handle_events()

        # Should toggle communication display
        assert renderer.show_communication != initial_comm

    @pytest.mark.asyncio
    async def test_cleanup(self, renderer: BattleRenderer, mock_pygame):
        """Test cleanup functionality."""
        # Setup mocks
        renderer.websocket_task = Mock()
        renderer.websocket_task.done.return_value = False
        renderer.websocket_task.cancel = Mock()
        renderer.websocket = Mock()
        renderer.websocket.close = AsyncMock()
        renderer.session = Mock()
        renderer.session.close = AsyncMock()

        # Cleanup
        await renderer.cleanup()

        # Verify cleanup calls
        renderer.websocket_task.cancel.assert_called_once()
        renderer.websocket.close.assert_called_once()
        renderer.session.close.assert_called_once()
        mock_pygame.quit.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
