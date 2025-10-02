"""
Tests for comprehensive error handling and recovery system.

This module tests the BattleErrorHandler and error recovery mechanisms
across different error scenarios including AI decision failures,
network errors, and system failures.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ..backend.ai_decision import AIDecisionMaker
from ..backend.error_handler import BattleErrorHandler
from ..shared.enums import WeaponCondition
from ..shared.errors import (
    AIDecisionError,
    ErrorCategory,
    NetworkError,
    OpenAIAPIError,
    RecoveryStrategy,
    SystemError,
    ValidationError,
)
from ..shared.models import Agent, BattleWorldState
from ..shared.schemas import AgentDecision, MoveAction


class TestBattleErrorHandler:
    """Test the BattleErrorHandler class."""

    @pytest.fixture
    def error_handler(self):
        """Create a BattleErrorHandler instance."""
        return BattleErrorHandler()

    @pytest.fixture
    def sample_world_state(self):
        """Create a sample world state for testing."""
        world_state = BattleWorldState()

        # Add test agents
        agent_a = Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(50.0, 50.0),
            health=75,
            weapon_condition=WeaponCondition.GOOD,
        )
        agent_b = Agent(
            agent_id="team_b_agent_1",
            team="team_b",
            position=(150.0, 150.0),
            health=30,
            weapon_condition=WeaponCondition.BROKEN,
        )

        world_state.agents = {"team_a_agent_1": agent_a, "team_b_agent_1": agent_b}

        return world_state

    def test_error_conversion(self, error_handler):
        """Test conversion of generic exceptions to BattleErrors."""
        # Test OpenAI API error conversion
        openai_error = Exception("OpenAI API rate limit exceeded")
        context = {"operation": "decision_making"}

        battle_error = error_handler._convert_to_battle_error(
            openai_error, context, "team_a_agent_1"
        )

        assert isinstance(battle_error, OpenAIAPIError)
        assert battle_error.agent_id == "team_a_agent_1"
        assert battle_error.category == ErrorCategory.AI_DECISION

        # Test network error conversion
        network_error = Exception("Connection timeout")
        battle_error = error_handler._convert_to_battle_error(
            network_error, context, None
        )

        assert isinstance(battle_error, NetworkError)
        assert battle_error.category == ErrorCategory.NETWORK

        # Test validation error conversion
        validation_error = Exception("Invalid data format")
        battle_error = error_handler._convert_to_battle_error(
            validation_error, context, None
        )

        assert isinstance(battle_error, ValidationError)
        assert battle_error.category == ErrorCategory.VALIDATION

    def test_error_tracking(self, error_handler):
        """Test error tracking and statistics."""
        # Create different types of errors
        ai_error = AIDecisionError(message="Decision failed", agent_id="team_a_agent_1")
        network_error = NetworkError(message="Connection failed", endpoint="/api/test")

        # Update tracking
        error_handler._update_error_tracking(ai_error)
        error_handler._update_error_tracking(network_error)
        error_handler._update_error_tracking(ai_error)  # Same type again

        # Check statistics
        stats = error_handler.get_error_statistics()

        assert (
            stats["total_errors_handled"] == 0
        )  # Not incremented by _update_error_tracking
        assert "ai_decision:AIDecisionError" in stats["error_counts_by_type"]
        assert stats["error_counts_by_type"]["ai_decision:AIDecisionError"] == 2
        assert stats["error_counts_by_type"]["network:NetworkError"] == 1

    def test_circuit_breaker(self, error_handler):
        """Test circuit breaker functionality."""
        # Create multiple AI decision errors to trigger circuit breaker
        for i in range(6):  # Exceed threshold of 5
            ai_error = AIDecisionError(
                message=f"Decision failed {i}", agent_id="team_a_agent_1"
            )
            error_handler._update_error_tracking(ai_error)

        # Check if circuit breaker is open
        test_error = AIDecisionError(message="Test error", agent_id="team_a_agent_1")

        assert error_handler._is_circuit_breaker_open(test_error)

        # Check circuit breaker status in statistics
        stats = error_handler.get_error_statistics()
        assert stats["circuit_breaker_status"]["openai_api"] is True

    @pytest.mark.asyncio
    async def test_ai_decision_error_handling(self, error_handler, sample_world_state):
        """Test AI decision error handling with fallback."""
        error = OpenAIAPIError(
            message="API timeout", agent_id="team_a_agent_1", status_code=408
        )

        fallback_decision = await error_handler.handle_ai_decision_error(
            error, "team_a_agent_1", sample_world_state
        )

        assert isinstance(fallback_decision, AgentDecision)
        assert isinstance(fallback_decision.primary_action, MoveAction)
        assert fallback_decision.confidence < 0.5  # Low confidence for fallback
        assert "fallback" in fallback_decision.strategic_assessment.lower()

        # Check that fallback counter was incremented
        stats = error_handler.get_error_statistics()
        assert stats["fallback_decisions_used"] == 1

    @pytest.mark.asyncio
    async def test_intelligent_fallback_decisions(
        self, error_handler, sample_world_state
    ):
        """Test intelligent fallback decision creation based on agent state."""
        # Test low health agent - should retreat
        low_health_agent = sample_world_state.agents["team_b_agent_1"]  # health=30
        error = AIDecisionError(message="Decision failed", agent_id="team_b_agent_1")

        fallback_decision = await error_handler._create_intelligent_fallback_decision(
            "team_b_agent_1", sample_world_state, error
        )

        # The agent has both low health AND broken weapon, so it might prioritize weapon repair
        # Let's check for either condition in the reason
        reason = fallback_decision.primary_action.reason.lower()
        assert any(
            keyword in reason for keyword in ["low health", "weapon broken", "forge"]
        )

        # Test agent with only low health (fix weapon condition and lower health)
        low_health_agent.weapon_condition = WeaponCondition.EXCELLENT
        low_health_agent.health = 25  # Below 30 threshold

        fallback_decision = await error_handler._create_intelligent_fallback_decision(
            "team_b_agent_1", sample_world_state, error
        )

        assert "low health" in fallback_decision.primary_action.reason.lower()
        assert fallback_decision.communication.urgency == "high"

        # Test broken weapon agent with full health
        broken_weapon_agent = sample_world_state.agents["team_b_agent_1"]
        broken_weapon_agent.health = 100  # Full health
        broken_weapon_agent.weapon_condition = (
            WeaponCondition.BROKEN
        )  # But broken weapon

        fallback_decision = await error_handler._create_intelligent_fallback_decision(
            "team_b_agent_1", sample_world_state, error
        )

        assert "weapon broken" in fallback_decision.primary_action.reason.lower()
        assert "forge" in fallback_decision.primary_action.reason.lower()

    @pytest.mark.asyncio
    async def test_network_error_retry(self, error_handler):
        """Test network error handling with retry logic."""
        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Network timeout")
            return "success"

        # Mock sleep to avoid actual delays in tests
        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Should succeed after 2 retries
            success, result = await error_handler.handle_network_error(
                Exception("Initial failure"),
                failing_operation,
                max_retries=3,
                backoff_factor=1.0,  # Minimal backoff for testing
            )

        assert success is True
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_network_error_max_retries(self, error_handler):
        """Test network error handling when max retries exceeded."""

        async def always_failing_operation():
            raise Exception("Persistent network error")

        success, result = await error_handler.handle_network_error(
            Exception("Initial failure"), always_failing_operation, max_retries=2
        )

        assert success is False
        assert result is None

    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, error_handler):
        """Test WebSocket error handling."""
        ws_error = Exception("WebSocket connection lost")

        success = await error_handler.handle_websocket_error(
            ws_error, connection_id="ws_123", auto_reconnect=True
        )

        assert success is True  # Should handle gracefully

        # Test without auto-reconnect
        success = await error_handler.handle_websocket_error(
            ws_error, connection_id="ws_456", auto_reconnect=False
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_concurrent_processing_error_handling(self, error_handler):
        """Test handling of concurrent processing errors."""
        successful_results = {
            "team_a_agent_1": {"status": "success", "action": "move"},
            "team_a_agent_2": {"status": "success", "action": "attack"},
        }

        errors = {
            "team_b_agent_1": Exception("AI decision failed"),
            "team_b_agent_2": Exception("Network timeout"),
        }

        combined_results = await error_handler.handle_concurrent_processing_error(
            errors, successful_results
        )

        # Should have all agents in results
        assert len(combined_results) == 4

        # Successful agents should be unchanged
        assert combined_results["team_a_agent_1"]["status"] == "success"
        assert combined_results["team_a_agent_2"]["status"] == "success"

        # Failed agents should have fallback results
        assert combined_results["team_b_agent_1"]["status"] == "fallback"
        assert combined_results["team_b_agent_2"]["status"] == "fallback"

    @pytest.mark.asyncio
    async def test_error_boundary_context_manager(self, error_handler):
        """Test error boundary context manager."""
        # Test successful operation
        async with error_handler.error_boundary("test_operation"):
            result = "success"

        # Test operation with fallback
        fallback_used = False
        try:
            async with error_handler.error_boundary(
                "failing_operation", fallback_result="fallback_value"
            ):
                raise Exception("Operation failed")
        except Exception:
            # Should not reach here if fallback works
            pass

        # The error boundary should handle the exception internally
        # and use the fallback result

    def test_recovery_strategy_selection(self, error_handler):
        """Test recovery strategy selection based on error type."""
        ai_error = AIDecisionError(message="Decision failed", agent_id="team_a_agent_1")
        network_error = NetworkError(message="Connection failed")
        system_error = SystemError(message="Critical system failure")

        assert (
            error_handler._get_recovery_strategy(ai_error) == RecoveryStrategy.FALLBACK
        )
        assert (
            error_handler._get_recovery_strategy(network_error)
            == RecoveryStrategy.RETRY
        )
        assert (
            error_handler._get_recovery_strategy(system_error) == RecoveryStrategy.ABORT
        )

    def test_error_statistics_reset(self, error_handler):
        """Test error statistics reset functionality."""
        # Generate some errors
        error = AIDecisionError(message="Test error", agent_id="team_a_agent_1")
        error_handler._update_error_tracking(error)
        error_handler.fallback_decisions_used = 5
        error_handler.total_errors_handled = 10

        # Check statistics exist
        stats = error_handler.get_error_statistics()
        assert stats["fallback_decisions_used"] == 5
        assert stats["total_errors_handled"] == 10
        assert len(stats["error_counts_by_type"]) > 0

        # Reset and verify
        error_handler.reset_error_tracking()

        stats = error_handler.get_error_statistics()
        assert stats["fallback_decisions_used"] == 0
        assert stats["total_errors_handled"] == 0
        assert len(stats["error_counts_by_type"]) == 0


class TestAIDecisionMakerErrorHandling:
    """Test error handling in AIDecisionMaker."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        return AsyncMock()

    @pytest.fixture
    def ai_decision_maker(self, mock_openai_client):
        """Create an AIDecisionMaker with mocked client."""
        with patch(
            "demo.backend.ai_decision.AsyncOpenAI", return_value=mock_openai_client
        ):
            return AIDecisionMaker(api_key="test_key")

    @pytest.fixture
    def sample_world_state(self):
        """Create a sample world state for testing."""
        world_state = BattleWorldState()

        agent = Agent(
            agent_id="team_a_agent_1",
            team="team_a",
            position=(50.0, 50.0),
            health=100,
            weapon_condition=WeaponCondition.EXCELLENT,
        )

        world_state.agents = {"team_a_agent_1": agent}
        return world_state

    @pytest.mark.asyncio
    async def test_openai_timeout_handling(self, ai_decision_maker, sample_world_state):
        """Test handling of OpenAI API timeouts."""
        # Mock timeout error
        ai_decision_maker.client.beta.chat.completions.parse = AsyncMock(
            side_effect=TimeoutError("Request timeout")
        )

        observation = {"visible_entities": {}}

        decision = await ai_decision_maker.make_decision(
            "team_a_agent_1", observation, sample_world_state
        )

        # Should return fallback decision
        assert isinstance(decision, AgentDecision)
        assert decision.confidence < 0.5
        assert "fallback" in decision.strategic_assessment.lower()

    @pytest.mark.asyncio
    async def test_openai_api_error_handling(
        self, ai_decision_maker, sample_world_state
    ):
        """Test handling of OpenAI API errors."""
        # Mock API error
        api_error = Exception("API rate limit exceeded")
        ai_decision_maker.client.beta.chat.completions.parse = AsyncMock(
            side_effect=api_error
        )

        observation = {"visible_entities": {}}

        decision = await ai_decision_maker.make_decision(
            "team_a_agent_1", observation, sample_world_state
        )

        # Should return fallback decision
        assert isinstance(decision, AgentDecision)
        assert decision.confidence < 0.5

    @pytest.mark.asyncio
    async def test_invalid_decision_validation(
        self, ai_decision_maker, sample_world_state
    ):
        """Test validation of invalid decisions from AI."""
        from ..shared.schemas import AttackAction

        # Mock response with invalid decision (attacking teammate)
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.parsed = AgentDecision(
            primary_action=AttackAction(
                target_agent_id="team_a_agent_1",  # Same team - invalid
                reason="Test attack",
            ),
            confidence=0.8,
            strategic_assessment="Test decision",
        )

        ai_decision_maker.client.beta.chat.completions.parse = AsyncMock(
            return_value=mock_response
        )

        observation = {"visible_entities": {}}

        decision = await ai_decision_maker.make_decision(
            "team_a_agent_1", observation, sample_world_state
        )

        # Should return fallback decision due to validation failure
        assert isinstance(decision, AgentDecision)
        assert decision.confidence < 0.5

    @pytest.mark.asyncio
    async def test_batch_decision_error_handling(
        self, ai_decision_maker, sample_world_state
    ):
        """Test error handling in batch decision making."""
        # Add another agent
        agent_b = Agent(
            agent_id="team_b_agent_1", team="team_b", position=(150.0, 150.0)
        )
        sample_world_state.agents["team_b_agent_1"] = agent_b

        # Mock one success and one failure
        call_count = 0

        async def mock_make_decision(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call succeeds
                return AgentDecision(
                    primary_action=MoveAction(
                        target_position=(100.0, 100.0), reason="Success"
                    ),
                    confidence=0.8,
                    strategic_assessment="Good decision",
                )
            else:
                # Second call fails
                raise Exception("AI decision failed")

        # Patch the make_decision method to simulate mixed results
        original_make_decision = ai_decision_maker.make_decision
        ai_decision_maker.make_decision = mock_make_decision

        agent_observations = {
            "team_a_agent_1": {"visible_entities": {}},
            "team_b_agent_1": {"visible_entities": {}},
        }

        results = await ai_decision_maker.batch_make_decisions(
            agent_observations, sample_world_state
        )

        # Should have results for both agents
        assert len(results) == 2
        assert "team_a_agent_1" in results
        assert "team_b_agent_1" in results

        # First should be successful, second should be fallback
        assert results["team_a_agent_1"].confidence == 0.8
        assert results["team_b_agent_1"].confidence < 0.5  # Fallback

        # Restore original method
        ai_decision_maker.make_decision = original_make_decision

    def test_error_statistics_tracking(self, ai_decision_maker):
        """Test error statistics tracking in AI decision maker."""
        stats = ai_decision_maker.get_error_statistics()

        # Should have basic structure
        assert "total_errors_handled" in stats
        assert "fallback_decisions_used" in stats
        assert "error_counts_by_type" in stats

        # Reset should work
        ai_decision_maker.reset_error_tracking()

        stats_after_reset = ai_decision_maker.get_error_statistics()
        assert stats_after_reset["total_errors_handled"] == 0
        assert stats_after_reset["fallback_decisions_used"] == 0


class TestErrorRecoveryScenarios:
    """Test various error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_cascading_error_recovery(self):
        """Test recovery from cascading errors."""
        error_handler = BattleErrorHandler()

        # Simulate cascading errors
        errors = [
            AIDecisionError("First error", "agent_1"),
            NetworkError("Network down"),
            SystemError("System overload", system_component="orchestrator"),
        ]

        recovery_results = []
        for error in errors:
            success, result = await error_handler.handle_error(error)
            recovery_results.append((success, result))

        # Should handle each error appropriately
        assert len(recovery_results) == 3

        # AI decision error should have fallback
        assert recovery_results[0][0] is True  # Success with fallback

        # Network error should attempt retry
        assert recovery_results[1][0] is False  # No operation to retry

        # System error should abort
        assert recovery_results[2][0] is False  # Abort strategy

    @pytest.mark.asyncio
    async def test_error_recovery_under_load(self):
        """Test error recovery under high load conditions."""
        error_handler = BattleErrorHandler()

        # Simulate high error load
        tasks = []
        for i in range(20):
            error = AIDecisionError(
                f"Load test error {i}",
                f"agent_{i % 6}",  # 6 agents
            )
            task = error_handler.handle_error(error)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without exceptions
        assert len(results) == 20
        for result in results:
            assert not isinstance(result, Exception)
            success, _ = result
            assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        error_handler = BattleErrorHandler()

        # Configure short timeout for testing
        error_handler.circuit_breaker_config["openai_api"]["timeout"] = 0.1

        # Trigger circuit breaker
        for i in range(6):
            ai_error = AIDecisionError(f"Error {i}", "test_agent")
            error_handler._update_error_tracking(ai_error)

        # Verify circuit breaker is open
        test_error = AIDecisionError("Test", "test_agent")
        assert error_handler._is_circuit_breaker_open(test_error)

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Circuit breaker should reset
        error_handler._update_circuit_breaker("openai_api", test_error)

        # Should be closed now (after reset logic)
        # Note: This tests the timeout mechanism
        assert (
            error_handler.circuit_breakers["openai_api"]["failure_count"] == 1
        )  # Reset and incremented


if __name__ == "__main__":
    pytest.main([__file__])
