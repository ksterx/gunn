"""
Tests for performance monitoring and optimization functionality.

This module tests the performance monitoring system including:
- Decision making latency tracking
- Concurrent processing metrics
- API response time monitoring
- Frame rate monitoring
- Memory usage tracking
- Performance optimization triggers
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from demo.backend.performance_monitor import (
    BattlePerformanceMonitor,
    PerformanceMetrics,
    PerformanceOptimizer,
)
from demo.frontend.performance_monitor import (
    FrontendPerformanceMonitor,
    RenderingOptimizer,
)
from demo.shared.enums import AgentStatus, WeaponCondition
from demo.shared.models import Agent, BattleWorldState


class TestBattlePerformanceMonitor:
    """Test backend performance monitoring."""

    @pytest.fixture
    def performance_monitor(self):
        """Create a performance monitor for testing."""
        return BattlePerformanceMonitor(monitoring_interval=0.1)

    @pytest.fixture
    def sample_world_state(self):
        """Create a sample world state for testing."""
        agents = {
            "agent_1": Agent(
                agent_id="agent_1",
                team="team_a",
                position=(10.0, 10.0),
                health=100,
                status=AgentStatus.ALIVE,
                weapon_condition=WeaponCondition.EXCELLENT,
            ),
            "agent_2": Agent(
                agent_id="agent_2",
                team="team_b",
                position=(90.0, 90.0),
                health=80,
                status=AgentStatus.ALIVE,
                weapon_condition=WeaponCondition.GOOD,
            ),
        }

        return BattleWorldState(
            agents=agents,
            team_scores={"team_a": 0, "team_b": 0},
            game_time=10.0,
            game_status="active",
        )

    @pytest.mark.asyncio
    async def test_decision_making_monitoring(self, performance_monitor):
        """Test decision making latency monitoring."""
        agent_id = "test_agent"

        # Test decision making monitoring context
        async with performance_monitor.monitor_decision_making(agent_id):
            await asyncio.sleep(0.01)  # Simulate decision making time

        # Check that latency was recorded
        assert agent_id in performance_monitor.decision_latencies
        assert len(performance_monitor.decision_latencies[agent_id]) > 0

        # Check that the latency is reasonable
        recorded_latency = performance_monitor.decision_latencies[agent_id][-1]
        assert 0.005 < recorded_latency < 0.1  # Should be around 10ms

    @pytest.mark.asyncio
    async def test_concurrent_processing_monitoring(self, performance_monitor):
        """Test concurrent processing performance monitoring."""
        agent_count = 3

        # Test concurrent processing monitoring
        async with performance_monitor.monitor_concurrent_processing(agent_count):
            # Simulate concurrent processing
            tasks = [asyncio.sleep(0.01) for _ in range(agent_count)]
            await asyncio.gather(*tasks)

        # Verify that processing was monitored
        assert performance_monitor.total_decisions_processed >= 0

    @pytest.mark.asyncio
    async def test_api_request_monitoring(self, performance_monitor):
        """Test API request performance monitoring."""
        endpoint = "/api/test"
        method = "GET"

        # Test successful API request monitoring
        async with performance_monitor.monitor_api_request(endpoint, method):
            await asyncio.sleep(0.005)  # Simulate API processing

        # Check that response time was recorded
        endpoint_key = f"{method}_{endpoint}"
        assert endpoint_key in performance_monitor.api_response_times
        assert len(performance_monitor.api_response_times[endpoint_key]) > 0

        recorded_time = performance_monitor.api_response_times[endpoint_key][-1]
        assert 0.003 < recorded_time < 0.1

    @pytest.mark.asyncio
    async def test_websocket_update_monitoring(self, performance_monitor):
        """Test WebSocket update performance monitoring."""
        update_type = "game_state"

        # Test WebSocket update monitoring
        async with performance_monitor.monitor_websocket_update(update_type):
            await asyncio.sleep(0.002)  # Simulate WebSocket update

        # Verify counter was incremented
        assert performance_monitor.total_websocket_updates > 0

    def test_frame_rate_recording(self, performance_monitor):
        """Test frame rate recording and threshold checking."""
        # Test normal frame rate
        performance_monitor.record_frame_rate(60.0)
        assert "low_frame_rate" not in performance_monitor.active_alerts

        # Test low frame rate
        performance_monitor.record_frame_rate(15.0)
        assert "low_frame_rate" in performance_monitor.active_alerts

    def test_queue_size_recording(self, performance_monitor):
        """Test queue size recording and threshold checking."""
        queue_type = "agent_processing"

        # Test normal queue size
        performance_monitor.record_queue_size(queue_type, 5)
        assert f"large_queue_{queue_type}" not in performance_monitor.active_alerts

        # Test large queue size
        performance_monitor.record_queue_size(queue_type, 150)
        assert f"large_queue_{queue_type}" in performance_monitor.active_alerts

    @pytest.mark.asyncio
    async def test_performance_thresholds(self, performance_monitor):
        """Test performance threshold checking and alerting."""
        # Create metrics that exceed thresholds
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            decision_latencies={"agent_1": 10.0},  # Exceeds 5.0s threshold
            memory_usage=2 * 1024 * 1024 * 1024,  # 2GB, exceeds 1GB threshold
            cpu_usage=90.0,  # Exceeds 80% threshold
        )

        # Check thresholds
        await performance_monitor._check_performance_thresholds(metrics)

        # Verify alerts were triggered
        assert "high_decision_latency_agent_1" in performance_monitor.active_alerts
        assert "high_memory_usage" in performance_monitor.active_alerts
        assert "high_cpu_usage" in performance_monitor.active_alerts

    def test_performance_summary(self, performance_monitor):
        """Test performance summary generation."""
        # Add some test data
        performance_monitor.decision_latencies["agent_1"].extend([0.1, 0.2, 0.15])
        performance_monitor.total_decisions_processed = 10
        performance_monitor.total_api_requests = 5

        summary = performance_monitor.get_performance_summary()

        assert "system" in summary
        assert "decision_making" in summary
        assert "counters" in summary
        assert "thresholds" in summary

        # Check decision making stats
        decision_stats = summary["decision_making"]
        assert "avg_latency" in decision_stats
        assert "max_latency" in decision_stats
        assert "count" in decision_stats

    @pytest.mark.asyncio
    async def test_performance_test_execution(self, performance_monitor):
        """Test performance test execution."""
        # Run a short performance test
        test_duration = 0.1  # 100ms test

        with patch.object(performance_monitor, "_collect_metrics") as mock_collect:
            mock_collect.return_value = PerformanceMetrics(
                timestamp=time.time(),
                memory_usage=100 * 1024 * 1024,  # 100MB
                cpu_usage=50.0,
            )

            results = await performance_monitor.run_performance_test(test_duration)

        assert "test_duration" in results
        assert "initial_memory" in results
        assert "final_memory" in results
        assert "memory_delta" in results
        assert results["test_duration"] >= test_duration


class TestPerformanceOptimizer:
    """Test performance optimization functionality."""

    @pytest.fixture
    def optimizer(self):
        """Create a performance optimizer for testing."""
        return PerformanceOptimizer()

    @pytest.mark.asyncio
    async def test_decision_making_optimization(self, optimizer):
        """Test decision making performance optimization."""
        # Test optimization trigger for high latency
        high_latency = 4.0
        result = await optimizer.optimize_decision_making(high_latency)

        assert result is True
        assert len(optimizer.optimization_history) > 0

        optimization = optimizer.optimization_history[-1]
        assert optimization["action"] == "reduce_model_complexity"
        assert optimization["trigger"] == "high_decision_latency"

    @pytest.mark.asyncio
    async def test_memory_optimization(self, optimizer):
        """Test memory usage optimization."""
        # Test optimization trigger for high memory usage
        high_memory = 600 * 1024 * 1024  # 600MB

        with patch("gc.collect", return_value=100) as mock_gc:
            result = await optimizer.optimize_memory_usage(high_memory)

        assert result is True
        assert mock_gc.called
        assert len(optimizer.optimization_history) > 0

        optimization = optimizer.optimization_history[-1]
        assert optimization["action"] == "garbage_collection"
        assert optimization["trigger"] == "high_memory_usage"

    @pytest.mark.asyncio
    async def test_concurrent_processing_optimization(self, optimizer):
        """Test concurrent processing optimization."""
        # Test optimization trigger for slow processing
        slow_processing_time = 6.0
        agent_count = 5

        result = await optimizer.optimize_concurrent_processing(
            slow_processing_time, agent_count
        )

        assert result is True
        assert len(optimizer.optimization_history) > 0

        optimization = optimizer.optimization_history[-1]
        assert optimization["action"] == "batch_size_reduction"
        assert optimization["trigger"] == "slow_concurrent_processing"


class TestFrontendPerformanceMonitor:
    """Test frontend performance monitoring."""

    @pytest.fixture
    def frontend_monitor(self):
        """Create a frontend performance monitor for testing."""
        return FrontendPerformanceMonitor(target_fps=60.0)

    def test_frame_timing(self, frontend_monitor):
        """Test frame timing measurement."""
        # Start and end a frame
        frame_start = frontend_monitor.start_frame()
        time.sleep(0.01)  # Simulate frame processing

        frontend_monitor.end_frame(
            frame_start_time=frame_start,
            render_time=0.008,
            event_time=0.001,
            network_time=0.001,
            total_agents=6,
            visible_agents=6,
            ui_elements=10,
        )

        # Check that frame time was recorded
        assert len(frontend_monitor.frame_times) > 0
        assert len(frontend_monitor.render_times) > 0

        # Check frame time is reasonable
        frame_time = frontend_monitor.frame_times[-1]
        assert 0.005 < frame_time < 0.1

    def test_fps_calculation(self, frontend_monitor):
        """Test FPS calculation."""
        # Simulate multiple frames
        for _ in range(10):
            frame_start = frontend_monitor.start_frame()
            time.sleep(0.016)  # ~60 FPS
            frontend_monitor.end_frame(
                frame_start_time=frame_start,
                render_time=0.01,
                total_agents=2,
                visible_agents=2,
            )

        # Force FPS calculation
        frontend_monitor._calculate_fps()

        current_fps = frontend_monitor.get_current_fps()
        assert 30 < current_fps < 120  # Should be around 60 FPS

    def test_adaptive_quality_adjustment(self, frontend_monitor):
        """Test adaptive quality adjustment based on performance."""
        # Enable adaptive quality
        frontend_monitor.set_adaptive_quality(True)

        # Simulate low FPS for extended period
        for _ in range(10):
            frontend_monitor._check_performance_optimization(20.0)  # Low FPS

        # Quality should be reduced
        assert frontend_monitor.current_quality_level < 1.0
        assert not frontend_monitor.enable_debug_rendering

    def test_quality_settings_application(self, frontend_monitor):
        """Test quality settings application."""
        # Test low quality settings
        frontend_monitor.current_quality_level = 0.4
        frontend_monitor._apply_quality_settings()

        assert not frontend_monitor.enable_debug_rendering
        assert not frontend_monitor.enable_particle_effects
        assert not frontend_monitor.enable_smooth_animations
        assert frontend_monitor.max_visible_agents == 10

        # Test high quality settings
        frontend_monitor.current_quality_level = 0.9
        frontend_monitor._apply_quality_settings()

        assert frontend_monitor.enable_debug_rendering
        assert frontend_monitor.enable_particle_effects
        assert frontend_monitor.enable_smooth_animations
        assert frontend_monitor.max_visible_agents == 20

    def test_performance_summary(self, frontend_monitor):
        """Test frontend performance summary generation."""
        # Add some test data
        frontend_monitor.frame_times.extend([0.016, 0.017, 0.015])
        frontend_monitor.render_times.extend([0.010, 0.011, 0.009])
        frontend_monitor.fps_history.extend([60.0, 58.0, 62.0])

        summary = frontend_monitor.get_performance_summary()

        assert "fps" in summary
        assert "frame_times_ms" in summary
        assert "render_times_ms" in summary
        assert "quality" in summary
        assert "counters" in summary

        # Check FPS stats
        fps_stats = summary["fps"]
        assert "current" in fps_stats
        assert "average" in fps_stats
        assert "target" in fps_stats


class TestRenderingOptimizer:
    """Test rendering optimization functionality."""

    @pytest.fixture
    def rendering_optimizer(self):
        """Create a rendering optimizer for testing."""
        monitor = FrontendPerformanceMonitor()
        return RenderingOptimizer(monitor)

    def test_frustum_culling(self, rendering_optimizer):
        """Test frustum culling for agents."""
        # Test agent within screen bounds
        agent_position = (50.0, 50.0)
        screen_bounds = (0, 0, 800, 600)

        should_render = rendering_optimizer.should_render_agent(
            agent_position, screen_bounds
        )
        assert should_render is True

        # Test agent outside screen bounds
        agent_position = (1000.0, 1000.0)
        should_render = rendering_optimizer.should_render_agent(
            agent_position, screen_bounds
        )
        assert should_render is False

    def test_agent_list_optimization(self, rendering_optimizer):
        """Test agent list optimization for rendering."""
        # Create test agents
        agents = []
        for i in range(25):  # More than max_visible_agents (20)
            agent = Mock()
            agent.position = (i * 10, i * 10)
            agents.append(agent)

        screen_bounds = (0, 0, 800, 600)

        # Optimize agent list
        optimized_agents = rendering_optimizer.optimize_agent_rendering(
            agents, screen_bounds
        )

        # Should limit to max visible agents
        max_agents = rendering_optimizer.performance_monitor.get_max_visible_agents()
        assert len(optimized_agents) <= max_agents

    def test_cache_functionality(self, rendering_optimizer):
        """Test surface and font caching."""

        # Test surface caching
        def create_surface():
            return Mock()

        surface1 = rendering_optimizer.get_cached_surface("test_key", create_surface)
        surface2 = rendering_optimizer.get_cached_surface("test_key", create_surface)

        # Should return the same cached surface
        assert surface1 is surface2

        # Test cache clearing
        rendering_optimizer.clear_caches()
        assert len(rendering_optimizer.surface_cache) == 0


@pytest.mark.asyncio
async def test_integration_performance_monitoring():
    """Test integration between backend and frontend performance monitoring."""
    backend_monitor = BattlePerformanceMonitor(monitoring_interval=0.1)
    frontend_monitor = FrontendPerformanceMonitor(target_fps=60.0)

    # Start backend monitoring
    await backend_monitor.start_monitoring()

    try:
        # Simulate some performance data
        async with backend_monitor.monitor_decision_making("test_agent"):
            await asyncio.sleep(0.01)

        # Simulate frontend frame
        frame_start = frontend_monitor.start_frame()
        time.sleep(0.016)
        frontend_monitor.end_frame(
            frame_start_time=frame_start,
            render_time=0.01,
            total_agents=2,
            visible_agents=2,
        )

        # Get summaries
        backend_summary = backend_monitor.get_performance_summary()
        frontend_summary = frontend_monitor.get_performance_summary()

        # Verify both have data
        assert backend_summary["status"] != "no_data"
        assert "fps" in frontend_summary

    finally:
        await backend_monitor.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__])
