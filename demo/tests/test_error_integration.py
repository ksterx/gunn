"""
Integration tests for error handling across the entire battle demo system.

This module tests error handling and recovery in realistic scenarios
involving the full system stack from AI decisions to WebSocket communication.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from ..backend.error_handler import BattleErrorHandler
from ..backend.server import BattleAPIServer
from ..shared.enums import WeaponCondition
from ..shared.errors import NetworkError, OpenAIAPIError, WebSocketError


class TestSystemErrorIntegration:
    """Test error handling integration across system components."""

    @pytest.fixture
    async def battle_server(self):
        """Create a battle server for testing."""
        server = BattleAPIServer(openai_api_key="test_key")
        await server._initialize_components()
        yield server
        await server._graceful_shutdown()

    @pytest.mark.asyncio
    async def test_ai_decision_failure_recovery(self, battle_server):
        """Test recovery from AI decision failures during game loop."""
        # Initialize game
        await battle_server._initialize_game()

        # Mock AI decision maker to fail
        original_make_decision = battle_server.ai_decision_maker.make_decision

        async def failing_decision(*args, **kwargs):
            raise OpenAIAPIError(
                message="API rate limit exceeded",
                agent_id=args[0] if args else "unknown",
                status_code=429,
            )

        battle_server.ai_decision_maker.make_decision = failing_decision

        # Process a game tick - should handle errors gracefully
        try:
            await battle_server._process_game_tick()

            # Game should continue despite AI failures
            assert battle_server.game_running is True

            # Check error statistics
            stats = battle_server.error_handler.get_error_statistics()
            assert stats["total_errors_handled"] > 0

        finally:
            # Restore original method
            battle_server.ai_decision_maker.make_decision = original_make_decision

    @pytest.mark.asyncio
    async def test_websocket_error_recovery(self, battle_server):
        """Test WebSocket error handling and recovery."""
        # Create mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.receive_text = AsyncMock(
            side_effect=Exception("Connection lost")
        )

        # Test WebSocket error handling
        connection_id = "test_connection"

        success = await battle_server.error_handler.handle_websocket_error(
            Exception("Connection lost"),
            connection_id=connection_id,
            auto_reconnect=True,
        )

        assert success is True  # Should handle gracefully

        # Check error statistics
        stats = battle_server.error_handler.get_error_statistics()
        assert "network:WebSocketError" in stats["error_counts_by_type"]

    @pytest.mark.asyncio
    async def test_concurrent_agent_processing_errors(self, battle_server):
        """Test error handling in concurrent agent processing."""
        # Initialize game with agents
        await battle_server._initialize_game()

        # Mock some agents to fail decision making
        original_batch_decisions = (
            battle_server.orchestrator.process_concurrent_decisions
        )

        async def mixed_results():
            # Simulate mixed success/failure results
            return {
                "team_a_agent_1": Exception("AI timeout"),
                "team_a_agent_2": Mock(primary_action=Mock(action_type="move")),
                "team_b_agent_1": Exception("Network error"),
                "team_b_agent_2": Mock(primary_action=Mock(action_type="attack")),
            }

        battle_server.orchestrator.process_concurrent_decisions = mixed_results

        try:
            # Process game tick with mixed results
            await battle_server._process_game_tick()

            # Should handle mixed results gracefully
            assert battle_server.game_running is True

        finally:
            # Restore original method
            battle_server.orchestrator.process_concurrent_decisions = (
                original_batch_decisions
            )

    @pytest.mark.asyncio
    async def test_game_loop_error_recovery(self, battle_server):
        """Test game loop error recovery and circuit breaker."""
        # Initialize game
        await battle_server._initialize_game()
        battle_server.game_running = True

        # Mock process_game_tick to fail multiple times
        original_process_tick = battle_server._process_game_tick
        call_count = 0

        async def failing_tick():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception(f"Game tick error {call_count}")
            # Succeed after 3 failures
            return

        battle_server._process_game_tick = failing_tick

        # Reduce game tick rate for faster testing
        original_tick_rate = battle_server.game_tick_rate
        battle_server.game_tick_rate = 0.1

        try:
            # Start game loop
            game_task = asyncio.create_task(battle_server._game_loop())

            # Let it run for a short time
            await asyncio.sleep(0.5)

            # Stop the game
            battle_server.game_running = False
            await game_task

            # Should have handled errors and continued
            assert call_count >= 3  # Multiple attempts made

            # Check error statistics
            stats = battle_server.error_handler.get_error_statistics()
            assert stats["total_errors_handled"] > 0

        finally:
            # Restore original methods
            battle_server._process_game_tick = original_process_tick
            battle_server.game_tick_rate = original_tick_rate

    @pytest.mark.asyncio
    async def test_network_broadcast_error_recovery(self, battle_server):
        """Test network error recovery in game state broadcasting."""
        # Initialize game
        await battle_server._initialize_game()

        # Mock connection manager to fail broadcasts
        original_broadcast = battle_server.connection_manager.broadcast

        async def failing_broadcast(message):
            raise Exception("Network broadcast failed")

        battle_server.connection_manager.broadcast = failing_broadcast

        try:
            # Attempt to broadcast game state
            await battle_server._broadcast_game_state()

            # Should handle error gracefully without crashing
            # Check error statistics
            stats = battle_server.error_handler.get_error_statistics()
            assert stats["total_errors_handled"] > 0

        finally:
            # Restore original method
            battle_server.connection_manager.broadcast = original_broadcast

    @pytest.mark.asyncio
    async def test_cascading_error_scenarios(self, battle_server):
        """Test handling of cascading error scenarios."""
        # Initialize game
        await battle_server._initialize_game()

        # Create a scenario with multiple failing components
        errors_to_inject = [
            ("ai_decision", OpenAIAPIError("API down", "team_a_agent_1")),
            ("network", NetworkError("Connection timeout")),
            ("websocket", WebSocketError("WebSocket disconnected")),
        ]

        # Inject errors into error handler
        for error_type, error in errors_to_inject:
            success, result = await battle_server.error_handler.handle_error(error)

            # Each error should be handled appropriately
            assert isinstance(success, bool)

        # System should remain stable
        assert battle_server.orchestrator is not None
        assert battle_server.ai_decision_maker is not None

        # Check comprehensive error statistics
        stats = battle_server.error_handler.get_error_statistics()
        assert stats["total_errors_handled"] >= len(errors_to_inject)
        assert len(stats["error_counts_by_type"]) > 0

    @pytest.mark.asyncio
    async def test_error_recovery_with_real_world_state(self, battle_server):
        """Test error recovery with realistic world state changes."""
        # Initialize game with full world state
        await battle_server._initialize_game()

        world_state = battle_server.orchestrator.world_state

        # Simulate agent taking damage and weapon degradation
        agent = list(world_state.agents.values())[0]
        agent.health = 25  # Low health
        agent.weapon_condition = WeaponCondition.BROKEN

        # Mock AI decision to fail
        original_make_decision = battle_server.ai_decision_maker.make_decision

        async def context_aware_failure(*args, **kwargs):
            agent_id = args[0] if args else "unknown"
            raise OpenAIAPIError(
                message="Context-aware API failure", agent_id=agent_id, status_code=503
            )

        battle_server.ai_decision_maker.make_decision = context_aware_failure

        try:
            # Process decision for low-health agent with broken weapon
            fallback_decision = await battle_server.ai_decision_maker.make_decision(
                agent.agent_id, {}, world_state
            )

            # Fallback should be intelligent based on agent state
            assert fallback_decision.confidence < 0.5

            # Should consider agent's critical state in fallback
            reason = fallback_decision.primary_action.reason.lower()
            assert any(
                keyword in reason
                for keyword in ["health", "weapon", "forge", "retreat"]
            )

        finally:
            # Restore original method
            battle_server.ai_decision_maker.make_decision = original_make_decision

    @pytest.mark.asyncio
    async def test_error_statistics_and_monitoring(self, battle_server):
        """Test error statistics collection and monitoring endpoints."""
        # Initialize components
        await battle_server._initialize_components()

        # Generate various types of errors
        test_errors = [
            OpenAIAPIError("API error 1", "agent_1"),
            NetworkError("Network error 1"),
            OpenAIAPIError("API error 2", "agent_2"),
            WebSocketError("WebSocket error 1"),
        ]

        # Process errors through error handler
        for error in test_errors:
            await battle_server.error_handler.handle_error(error)

        # Check statistics collection
        stats = battle_server.error_handler.get_error_statistics()

        assert stats["total_errors_handled"] == len(test_errors)
        assert len(stats["error_counts_by_type"]) > 0

        # Verify specific error type counts
        ai_errors = sum(
            count
            for error_type, count in stats["error_counts_by_type"].items()
            if "ai_decision" in error_type.lower()
        )
        assert ai_errors == 2  # Two AI errors

        network_errors = sum(
            count
            for error_type, count in stats["error_counts_by_type"].items()
            if "network" in error_type.lower()
        )
        assert network_errors == 2  # Network + WebSocket errors

    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, battle_server):
        """Test error recovery performance under load."""
        # Initialize game
        await battle_server._initialize_game()

        # Create many concurrent errors
        num_errors = 50
        error_tasks = []

        for i in range(num_errors):
            error = OpenAIAPIError(
                f"Load test error {i}",
                f"agent_{i % 6}",  # Distribute across 6 agents
                status_code=429,
            )
            task = battle_server.error_handler.handle_error(error)
            error_tasks.append(task)

        # Measure recovery time
        import time

        start_time = time.time()

        results = await asyncio.gather(*error_tasks, return_exceptions=True)

        end_time = time.time()
        recovery_time = end_time - start_time

        # All errors should be handled
        assert len(results) == num_errors

        # No exceptions should propagate
        for result in results:
            assert not isinstance(result, Exception)

        # Recovery should be reasonably fast (less than 5 seconds for 50 errors)
        assert recovery_time < 5.0

        # Check final statistics
        stats = battle_server.error_handler.get_error_statistics()
        assert stats["total_errors_handled"] == num_errors

    @pytest.mark.asyncio
    async def test_graceful_degradation_scenario(self, battle_server):
        """Test graceful degradation when multiple systems fail."""
        # Initialize game
        await battle_server._initialize_game()

        # Simulate multiple system failures
        # 1. AI decision maker fails
        battle_server.ai_decision_maker = None

        # 2. Network issues
        original_broadcast = battle_server.connection_manager.broadcast

        async def unreliable_broadcast(message):
            import random

            if random.random() < 0.7:  # 70% failure rate
                raise Exception("Network unreliable")
            return True

        battle_server.connection_manager.broadcast = unreliable_broadcast

        try:
            # System should continue operating with degraded functionality
            # Process a game tick
            await battle_server._process_game_tick()

            # Try to broadcast state
            await battle_server._broadcast_game_state()

            # Game should still be running despite failures
            assert battle_server.orchestrator is not None

            # Error handler should have recorded the issues
            stats = battle_server.error_handler.get_error_statistics()
            assert stats["total_errors_handled"] > 0

        finally:
            # Restore network function
            battle_server.connection_manager.broadcast = original_broadcast


class TestErrorRecoveryEdgeCases:
    """Test edge cases in error recovery."""

    @pytest.mark.asyncio
    async def test_error_during_error_handling(self):
        """Test handling of errors that occur during error recovery."""
        error_handler = BattleErrorHandler()

        # Mock a method to fail during error handling
        original_log_error = error_handler._log_error

        def failing_log_error(error):
            raise Exception("Logging system failed")

        error_handler._log_error = failing_log_error

        try:
            # This should not crash despite logging failure
            test_error = OpenAIAPIError("Test error", "agent_1")
            success, result = await error_handler.handle_error(test_error)

            # Should still attempt to handle the error
            assert isinstance(success, bool)

        finally:
            # Restore original method
            error_handler._log_error = original_log_error

    @pytest.mark.asyncio
    async def test_memory_pressure_during_errors(self):
        """Test error handling under memory pressure."""
        error_handler = BattleErrorHandler()

        # Create many large error objects to simulate memory pressure
        large_errors = []
        for i in range(100):
            error = OpenAIAPIError(
                f"Large error {i} with lots of context data " * 100,
                f"agent_{i}",
                context={"large_data": "x" * 10000},  # 10KB of data per error
            )
            large_errors.append(error)

        # Process all errors
        results = []
        for error in large_errors:
            success, result = await error_handler.handle_error(error)
            results.append((success, result))

        # All should be handled despite memory pressure
        assert len(results) == 100

        # Memory should be manageable (error handler should not accumulate unbounded data)
        stats = error_handler.get_error_statistics()
        assert isinstance(stats, dict)  # Should still be able to generate stats

    @pytest.mark.asyncio
    async def test_rapid_error_succession(self):
        """Test handling of rapid succession of errors."""
        error_handler = BattleErrorHandler()

        # Create rapid succession of errors (no delay)
        rapid_tasks = []
        for i in range(20):
            error = OpenAIAPIError(f"Rapid error {i}", "agent_1")
            task = asyncio.create_task(error_handler.handle_error(error))
            rapid_tasks.append(task)

        # All should complete successfully
        results = await asyncio.gather(*rapid_tasks)

        assert len(results) == 20
        for success, result in results:
            assert isinstance(success, bool)

        # Circuit breaker should activate due to rapid failures
        stats = error_handler.get_error_statistics()
        circuit_status = stats["circuit_breaker_status"]
        assert circuit_status.get("openai_api", False) is True


if __name__ == "__main__":
    pytest.main([__file__])
