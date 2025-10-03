"""
Integration tests for the frontend renderer.

This module tests the frontend integration with the backend and
verifies that the renderer can be properly instantiated and configured.
"""

from unittest.mock import MagicMock, patch

import pytest

# Mock pygame before importing
pygame_mock = MagicMock()
pygame_mock.QUIT = 256
pygame_mock.KEYDOWN = 768

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


class TestFrontendIntegration:
    """Integration tests for the frontend renderer."""

    def test_renderer_import(self):
        """Test that the renderer can be imported successfully."""
        assert BattleRenderer is not None

    def test_renderer_instantiation(self):
        """Test that the renderer can be instantiated with default parameters."""
        renderer = BattleRenderer()

        assert renderer.backend_url == "http://localhost:8000"
        assert renderer.window_size == (800, 600)
        assert renderer.running is True
        assert renderer.game_state is None

    def test_renderer_custom_config(self):
        """Test renderer with custom configuration."""
        custom_url = "http://192.168.1.100:8080"
        custom_size = (1024, 768)

        renderer = BattleRenderer(backend_url=custom_url, window_size=custom_size)

        assert renderer.backend_url == custom_url
        assert renderer.window_size == custom_size
        assert renderer.window_width == 1024
        assert renderer.window_height == 768

    def test_coordinate_system_setup(self):
        """Test that the coordinate system is properly configured."""
        renderer = BattleRenderer(window_size=(800, 600))

        # Check scaling calculations
        assert renderer.world_width == 200.0
        assert renderer.world_height == 100.0
        assert renderer.scale_x > 0
        assert renderer.scale_y > 0
        assert renderer.offset_x == 20
        assert renderer.offset_y == 20

    def test_color_configuration(self):
        """Test that colors are properly configured."""
        renderer = BattleRenderer()

        # Check that essential colors are defined
        assert "background" in renderer.colors
        assert "text" in renderer.colors
        assert "health_good" in renderer.colors
        assert "weapon_excellent" in renderer.colors
        assert "forge_team_a" in renderer.colors
        assert "forge_team_b" in renderer.colors

    def test_ui_state_initialization(self):
        """Test that UI state is properly initialized."""
        renderer = BattleRenderer()

        assert renderer.show_debug_info is True
        assert renderer.show_communication is True
        assert renderer.paused is False
        assert renderer.running is True
        assert renderer.connection_status == "disconnected"
        assert renderer.error_message is None


if __name__ == "__main__":
    pytest.main([__file__])
