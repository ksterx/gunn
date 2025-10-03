"""
Integration tests for server initialization and auto-start functionality.

This module tests the integration between the FastAPI server and the
game initialization system, including API endpoints and auto-start behavior.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from demo.backend.server import BattleAPIServer
from demo.shared.models import BattleWorldState


class TestServerInitialization:
    """Test server initialization functionality."""

    @pytest.fixture
    def mock_openai_key(self):
        """Mock OpenAI API key."""
        return "test_openai_key"

    @pytest.fixture
    def server(self, mock_openai_key):
        """Create test server instance."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": mock_openai_key}):
            return BattleAPIServer(openai_api_key=mock_openai_key)

    @pytest.fixture
    def client(self, server):
        """Create test client."""
        return TestClient(server.app)

    def test_server_components_initialization(self, server):
        """Test that server components are properly initialized."""
        # Components should be None initially (initialized in lifespan)
        assert server.orchestrator is None
        assert server.ai_decision_maker is None
        assert server.game_initializer is None
        assert server.auto_start_manager is None

        # Other components should be initialized
        assert server.connection_manager is not None
        assert server.error_handler is not None

    @pytest.mark.asyncio
    async def test_initialize_components(self, server):
        """Test component initialization."""
        await server._initialize_components()

        # All components should be initialized
        assert server.orchestrator is not None
        assert server.ai_decision_maker is not None
        assert server.game_initializer is not None
        assert server.auto_start_manager is not None
        assert server.combat_manager is not None

    def test_start_game_request_validation(self, client):
        """Test start game request validation."""
        # Test valid request
        valid_request = {
            "game_mode": "demo",
            "ai_difficulty": "normal",
            "auto_start": True,
            "positioning_strategy": "corners",
            "forge_placement": "sides",
            "agents_per_team": 3,
            "use_random_seed": True,
            "random_seed": 42,
        }

        # This will fail because components aren't initialized in test,
        # but we can check that the request structure is accepted
        response = client.post("/api/game/start", json=valid_request)
        # Should get 500 (internal error) not 422 (validation error)
        assert response.status_code == 500

        # Test invalid request (agents_per_team out of range)
        invalid_request = valid_request.copy()
        invalid_request["agents_per_team"] = 10  # Too many

        response = client.post("/api/game/start", json=invalid_request)
        assert response.status_code == 422  # Validation error


class TestServerAutoStart:
    """Test server auto-start functionality."""

    @pytest.fixture
    def mock_server(self):
        """Create server with mocked components."""
        server = BattleAPIServer()

        # Mock components
        server.orchestrator = Mock()
        server.orchestrator.world_state = BattleWorldState()
        server.ai_decision_maker = Mock()
        server.game_initializer = Mock()
        server.auto_start_manager = Mock()
        server.combat_manager = Mock()

        return server

    @pytest.mark.asyncio
    async def test_auto_start_success(self, mock_server):
        """Test successful auto-start."""
        # Mock successful auto-start
        mock_server.auto_start_manager.auto_start_game = AsyncMock(return_value=True)

        await mock_server._auto_start_game()

        # Verify auto-start was called
        mock_server.auto_start_manager.auto_start_game.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_start_failure(self, mock_server):
        """Test auto-start failure handling."""
        # Mock failed auto-start
        mock_server.auto_start_manager.auto_start_game = AsyncMock(return_value=False)

        # Should not raise exception even on failure
        await mock_server._auto_start_game()

        mock_server.auto_start_manager.auto_start_game.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_start_exception(self, mock_server):
        """Test auto-start exception handling."""
        # Mock exception during auto-start
        mock_server.auto_start_manager.auto_start_game = AsyncMock(
            side_effect=Exception("Auto-start failed")
        )

        # Should not raise exception
        await mock_server._auto_start_game()

        mock_server.auto_start_manager.auto_start_game.assert_called_once()


class TestInitializationEndpoints:
    """Test initialization-related API endpoints."""

    @pytest.fixture
    def mock_server_with_client(self):
        """Create server with mocked components and test client."""
        server = BattleAPIServer()

        # Mock components
        server.orchestrator = Mock()
        server.orchestrator.world_state = BattleWorldState()
        server.ai_decision_maker = Mock()
        server.combat_manager = Mock()

        # Mock game initializer with summary
        server.game_initializer = Mock()
        server.game_initializer.get_initialization_summary.return_value = {
            "initialization_count": 1,
            "total_agents": 6,
            "team_counts": {"team_a": 3, "team_b": 3},
            "total_forges": 2,
            "total_map_locations": 5,
            "game_status": "active",
            "game_time": 0.0,
            "config": {
                "positioning_strategy": "corners",
                "forge_placement": "corners",
                "agents_per_team": 3,
                "map_size": [200.0, 100.0],
            },
        }

        # Mock auto-start manager
        server.auto_start_manager = Mock()
        server.auto_start_manager.get_restart_statistics.return_value = {
            "restart_count": 0,
            "auto_start_enabled": True,
            "max_restart_attempts": 3,
        }

        client = TestClient(server.app)
        return server, client

    def test_get_initialization_info(self, mock_server_with_client):
        """Test initialization info endpoint."""
        server, client = mock_server_with_client

        response = client.get("/api/game/initialization")

        assert response.status_code == 200
        data = response.json()

        assert "initialization_count" in data
        assert "total_agents" in data
        assert "team_counts" in data
        assert "auto_start" in data
        assert data["total_agents"] == 6
        assert data["team_counts"]["team_a"] == 3

    def test_enable_auto_start(self, mock_server_with_client):
        """Test enable auto-start endpoint."""
        server, client = mock_server_with_client

        response = client.post("/api/system/auto-start/enable")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "Auto-start enabled" in data["message"]
        server.auto_start_manager.enable_auto_start.assert_called_once()

    def test_disable_auto_start(self, mock_server_with_client):
        """Test disable auto-start endpoint."""
        server, client = mock_server_with_client

        response = client.post("/api/system/auto-start/disable")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "Auto-start disabled" in data["message"]
        server.auto_start_manager.disable_auto_start.assert_called_once()

    def test_reinitialize_game(self, mock_server_with_client):
        """Test game reinitialization endpoint."""
        server, client = mock_server_with_client

        # Mock _initialize_game method
        server._initialize_game = AsyncMock()
        server._serialize_game_state = Mock(
            return_value=Mock(model_dump=Mock(return_value={}))
        )
        server._stop_game = AsyncMock()
        server.game_running = False

        request_data = {
            "game_mode": "demo",
            "positioning_strategy": "lines",
            "agents_per_team": 4,
        }

        response = client.post("/api/game/reinitialize", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "reinitialized" in data["message"]


class TestGameResetFunctionality:
    """Test game reset and restart functionality."""

    @pytest.fixture
    def mock_server(self):
        """Create server with mocked components."""
        server = BattleAPIServer()

        # Mock components
        server.orchestrator = Mock()
        server.orchestrator.world_state = BattleWorldState()
        server.ai_decision_maker = Mock()
        server.game_initializer = Mock()
        server.auto_start_manager = Mock()
        server.combat_manager = Mock()

        # Mock game state methods
        server._serialize_game_state = Mock(
            return_value=Mock(model_dump=Mock(return_value={}))
        )
        server._stop_game = AsyncMock()
        server._start_game_loop = AsyncMock()

        return server

    @pytest.mark.asyncio
    async def test_game_reset_with_auto_start_manager(self, mock_server):
        """Test game reset using auto-start manager."""
        # Mock successful restart
        mock_server.auto_start_manager.restart_game = AsyncMock(return_value=True)

        client = TestClient(mock_server.app)

        request_data = {"action": "reset"}
        response = client.post("/api/game/control", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["action"] == "reset"
        mock_server.auto_start_manager.restart_game.assert_called_once()
        mock_server._start_game_loop.assert_called_once()

    @pytest.mark.asyncio
    async def test_game_reset_failure(self, mock_server):
        """Test game reset failure handling."""
        # Mock failed restart
        mock_server.auto_start_manager.restart_game = AsyncMock(return_value=False)

        client = TestClient(mock_server.app)

        request_data = {"action": "reset"}
        response = client.post("/api/game/control", json=request_data)

        assert response.status_code == 500
        mock_server.auto_start_manager.restart_game.assert_called_once()
        mock_server._start_game_loop.assert_not_called()

    @pytest.mark.asyncio
    async def test_game_reset_fallback(self, mock_server):
        """Test game reset fallback when auto-start manager unavailable."""
        # Remove auto-start manager
        mock_server.auto_start_manager = None
        mock_server._initialize_game = AsyncMock()

        client = TestClient(mock_server.app)

        request_data = {"action": "reset"}
        response = client.post("/api/game/control", json=request_data)

        assert response.status_code == 200
        mock_server._stop_game.assert_called_once()
        mock_server._initialize_game.assert_called_once()
        mock_server._start_game_loop.assert_called_once()


class TestInitializationConsistency:
    """Test initialization consistency and determinism."""

    @pytest.mark.asyncio
    async def test_deterministic_initialization(self):
        """Test that initialization produces consistent results."""
        # Create two servers with same configuration
        server1 = BattleAPIServer()
        server2 = BattleAPIServer()

        await server1._initialize_components()
        await server2._initialize_components()

        # Both should have same configuration
        config1 = server1.game_initializer.config
        config2 = server2.game_initializer.config

        assert config1.agents_per_team == config2.agents_per_team
        assert config1.positioning_strategy == config2.positioning_strategy
        assert config1.forge_placement == config2.forge_placement

    @pytest.mark.asyncio
    async def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        server = BattleAPIServer()
        await server._initialize_components()

        # Create custom request
        from demo.backend.server import StartGameRequest

        request = StartGameRequest(
            positioning_strategy="lines",
            forge_placement="sides",
            agents_per_team=4,
            use_random_seed=True,
            random_seed=123,
        )

        # Mock the initialization process
        server.orchestrator.world_state = BattleWorldState()
        server.orchestrator.initialize = AsyncMock()

        await server._initialize_game(request)

        # Check that config was updated
        config = server.game_initializer.config
        assert config.positioning_strategy == "lines"
        assert config.forge_placement == "sides"
        assert config.agents_per_team == 4
        assert config.random_seed == 123

    def test_initialization_config_validation(self):
        """Test that initialization config is properly validated."""
        from demo.backend.server import StartGameRequest

        # Test valid config
        valid_request = StartGameRequest(
            agents_per_team=3, positioning_strategy="corners", forge_placement="corners"
        )

        assert valid_request.agents_per_team == 3
        assert valid_request.positioning_strategy == "corners"

        # Test invalid config (should be caught by Pydantic)
        with pytest.raises(ValueError):
            StartGameRequest(agents_per_team=0)  # Below minimum

        with pytest.raises(ValueError):
            StartGameRequest(agents_per_team=10)  # Above maximum


class TestInitializationErrorHandling:
    """Test error handling in initialization system."""

    @pytest.fixture
    def server_with_failing_components(self):
        """Create server with components that fail initialization."""
        server = BattleAPIServer()
        return server

    @pytest.mark.asyncio
    async def test_component_initialization_failure(
        self, server_with_failing_components
    ):
        """Test handling of component initialization failures."""
        server = server_with_failing_components

        # Mock AI decision maker to fail
        with patch(
            "demo.backend.server.AIDecisionMaker",
            side_effect=Exception("AI init failed"),
        ):
            with pytest.raises(Exception, match="AI init failed"):
                await server._initialize_components()

    @pytest.mark.asyncio
    async def test_game_initialization_failure(self):
        """Test handling of game initialization failures."""
        server = BattleAPIServer()
        await server._initialize_components()

        # Mock game initializer to fail
        server.game_initializer.initialize_game = AsyncMock(
            side_effect=Exception("Game init failed")
        )

        from demo.backend.server import StartGameRequest

        request = StartGameRequest()

        with pytest.raises(Exception, match="Game init failed"):
            await server._initialize_game(request)

    def test_missing_components_error(self):
        """Test error when components are missing."""
        server = BattleAPIServer()

        # Try to initialize game without components
        with pytest.raises(RuntimeError, match="Components not initialized"):
            asyncio.run(server._initialize_game())

    def test_auto_start_manager_unavailable(self):
        """Test handling when auto-start manager is unavailable."""
        server = BattleAPIServer()
        server.auto_start_manager = None

        client = TestClient(server.app)

        # Try to enable auto-start without manager
        response = client.post("/api/system/auto-start/enable")
        assert response.status_code == 503

        # Try to disable auto-start without manager
        response = client.post("/api/system/auto-start/disable")
        assert response.status_code == 503
