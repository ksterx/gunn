"""
Battle error handler with comprehensive recovery strategies.

This module provides centralized error handling with AI decision fallbacks,
retry logic, and graceful degradation for various error scenarios.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from ..shared.errors import (
    AIDecisionError,
    BattleError,
    ConcurrentProcessingError,
    ErrorCategory,
    ErrorRecoveryInfo,
    ErrorSeverity,
    NetworkError,
    OpenAIAPIError,
    RecoveryStrategy,
    SystemError,
    ValidationError,
    WebSocketError,
)
from ..shared.models import Agent, BattleWorldState
from ..shared.schemas import AgentDecision, CommunicateAction, MoveAction

logger = logging.getLogger(__name__)


class BattleErrorHandler:
    """Centralized error handler with recovery strategies and fallback mechanisms."""

    def __init__(self):
        self.error_counts: dict[str, int] = {}
        self.recovery_info: dict[str, ErrorRecoveryInfo] = {}
        self.circuit_breakers: dict[str, dict[str, Any]] = {}
        self.fallback_decisions_used = 0
        self.total_errors_handled = 0

        # Configure recovery strategies for different error types
        self.recovery_strategies = {
            ErrorCategory.AI_DECISION: RecoveryStrategy.FALLBACK,
            ErrorCategory.NETWORK: RecoveryStrategy.FALLBACK,  # Changed from RETRY to FALLBACK
            ErrorCategory.GAME_STATE: RecoveryStrategy.RESET,
            ErrorCategory.VALIDATION: RecoveryStrategy.SKIP,
            ErrorCategory.SYSTEM: RecoveryStrategy.ABORT,
        }

        # Configure circuit breaker thresholds
        self.circuit_breaker_config = {
            "openai_api": {"failure_threshold": 5, "timeout": 60.0},
            "websocket": {"failure_threshold": 3, "timeout": 30.0},
            "game_state": {"failure_threshold": 10, "timeout": 120.0},
        }

    async def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> tuple[bool, Any]:
        """
        Handle an error with appropriate recovery strategy.

        Args:
            error: The exception that occurred
            context: Additional context about the error
            agent_id: ID of the agent involved (if applicable)

        Returns:
            Tuple of (recovered_successfully, result_or_fallback)
        """
        self.total_errors_handled += 1
        context = context or {}

        # Convert to BattleError if needed
        battle_error = self._convert_to_battle_error(error, context, agent_id)

        # Log the error
        self._log_error(battle_error)

        # Update error tracking
        self._update_error_tracking(battle_error)

        # Check circuit breakers
        if self._is_circuit_breaker_open(battle_error):
            logger.warning(
                f"Circuit breaker open for {battle_error.category.value}, using fallback"
            )
            return await self._handle_circuit_breaker_fallback(battle_error)

        # Determine recovery strategy
        strategy = self._get_recovery_strategy(battle_error)

        # Execute recovery strategy
        return await self._execute_recovery_strategy(battle_error, strategy)

    async def handle_ai_decision_error(
        self,
        error: Exception,
        agent_id: str,
        world_state: BattleWorldState,
        retry_count: int = 0,
    ) -> AgentDecision:
        """
        Handle AI decision errors with intelligent fallbacks.

        Args:
            error: The error that occurred during decision making
            agent_id: ID of the agent that failed to make a decision
            world_state: Current world state for context
            retry_count: Number of retries already attempted

        Returns:
            A fallback AgentDecision
        """
        context = {
            "agent_id": agent_id,
            "retry_count": retry_count,
            "world_state_time": world_state.game_time,
        }

        # Convert to AI decision error
        if isinstance(error, AIDecisionError):
            ai_error = error
        else:
            ai_error = AIDecisionError(
                message=f"Decision making failed: {error!s}",
                agent_id=agent_id,
                api_error=error,
                context=context,
            )

        # Log the error
        self._log_error(ai_error)

        # Create intelligent fallback decision
        fallback_decision = await self._create_intelligent_fallback_decision(
            agent_id, world_state, ai_error
        )

        self.fallback_decisions_used += 1

        logger.info(
            f"Created fallback decision for {agent_id}: {fallback_decision.primary_action.action_type}"
        )

        return fallback_decision

    async def handle_network_error(
        self,
        error: Exception,
        operation: Callable[[], Awaitable[Any]],
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> tuple[bool, Any]:
        """
        Handle network errors with exponential backoff retry.

        Args:
            error: The network error that occurred
            operation: The operation to retry
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier

        Returns:
            Tuple of (success, result)
        """
        network_error = NetworkError(
            message=f"Network operation failed: {error!s}",
            severity=ErrorSeverity.MEDIUM,
            context={"original_error": str(error)},
        )

        recovery_key = f"network_{id(operation)}"

        if recovery_key not in self.recovery_info:
            self.recovery_info[recovery_key] = ErrorRecoveryInfo(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=max_retries,
                backoff_factor=backoff_factor,
            )

        recovery = self.recovery_info[recovery_key]
        current_time = time.time()

        if not recovery.should_retry(current_time):
            logger.error(
                f"Max retries exceeded for network operation (attempts: {recovery.attempts}/{recovery.max_attempts}): {error}"
            )
            return False, None

        try:
            # Wait for backoff delay if this isn't the first attempt
            if recovery.attempts > 0:
                delay = recovery.get_next_delay()
                if delay > 0:
                    logger.info(
                        f"Retrying network operation in {delay:.2f}s (attempt {recovery.attempts + 1})"
                    )
                    await asyncio.sleep(delay)

            result = await operation()
            recovery.record_attempt(current_time, success=True)
            logger.info(
                f"Network operation succeeded after {recovery.attempts} attempts"
            )
            return True, result

        except Exception as retry_error:
            recovery.record_attempt(current_time, success=False)
            logger.warning(
                f"Network retry attempt {recovery.attempts} failed: {retry_error}"
            )

            # Check if we should retry (after recording the failed attempt)
            if recovery.should_retry(time.time()):
                return await self.handle_network_error(
                    retry_error, operation, max_retries, backoff_factor
                )
            else:
                return False, None

    async def handle_websocket_error(
        self,
        error: Exception,
        connection_id: str | None = None,
        auto_reconnect: bool = True,
    ) -> bool:
        """
        Handle WebSocket errors with connection recovery.

        Args:
            error: The WebSocket error
            connection_id: ID of the failed connection
            auto_reconnect: Whether to attempt automatic reconnection

        Returns:
            True if recovery was successful
        """
        ws_error = WebSocketError(
            message=str(error),
            connection_id=connection_id,
            severity=ErrorSeverity.MEDIUM,
            context={"auto_reconnect": auto_reconnect},
        )

        # Use the main error handling pipeline
        success, _ = await self.handle_error(ws_error)

        if not auto_reconnect:
            return False

        # Implement connection recovery logic here
        # For now, just log and return success to indicate graceful handling
        logger.info(
            f"WebSocket connection {connection_id} will be handled by connection manager"
        )
        return success

    async def handle_concurrent_processing_error(
        self, errors: dict[str, Exception], successful_results: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle errors in concurrent agent processing.

        Args:
            errors: Map of agent_id to error for failed agents
            successful_results: Map of agent_id to result for successful agents

        Returns:
            Combined results with fallbacks for failed agents
        """
        failed_agents = list(errors.keys())

        concurrent_error = ConcurrentProcessingError(
            message=f"Concurrent processing failed for {len(failed_agents)} agents",
            failed_agents=failed_agents,
            severity=ErrorSeverity.MEDIUM,
            context={
                "successful_count": len(successful_results),
                "failed_count": len(failed_agents),
            },
        )

        self._log_error(concurrent_error)

        # Start with successful results
        combined_results = successful_results.copy()

        # Create fallbacks for failed agents
        for agent_id, error in errors.items():
            logger.warning(f"Creating fallback for failed agent {agent_id}: {error}")

            # For now, create a simple fallback result
            # In a real implementation, this would depend on the operation type
            combined_results[agent_id] = {
                "status": "fallback",
                "error": str(error),
                "agent_id": agent_id,
            }

        return combined_results

    @asynccontextmanager
    async def error_boundary(
        self,
        operation_name: str,
        agent_id: str | None = None,
        fallback_result: Any = None,
    ):
        """
        Context manager that provides error boundary with automatic handling.

        Args:
            operation_name: Name of the operation for logging
            agent_id: Agent ID if operation is agent-specific
            fallback_result: Result to return if operation fails
        """
        try:
            yield
        except Exception as e:
            context = {"operation_name": operation_name, "agent_id": agent_id}

            success, result = await self.handle_error(e, context, agent_id)

            if not success and fallback_result is not None:
                logger.info(f"Using fallback result for {operation_name}")
                yield fallback_result
            else:
                # Re-raise if no fallback available
                raise

    def get_error_statistics(self) -> dict[str, Any]:
        """Get error handling statistics."""
        return {
            "total_errors_handled": self.total_errors_handled,
            "fallback_decisions_used": self.fallback_decisions_used,
            "error_counts_by_type": self.error_counts.copy(),
            "active_recovery_operations": len(self.recovery_info),
            "circuit_breaker_status": {
                name: breaker.get("is_open", False)
                for name, breaker in self.circuit_breakers.items()
            },
        }

    def reset_error_tracking(self) -> None:
        """Reset error tracking counters."""
        self.error_counts.clear()
        self.recovery_info.clear()
        self.circuit_breakers.clear()
        self.fallback_decisions_used = 0
        self.total_errors_handled = 0
        logger.info("Error tracking reset")

    def _convert_to_battle_error(
        self, error: Exception, context: dict[str, Any], agent_id: str | None
    ) -> BattleError:
        """Convert a generic exception to a BattleError."""
        if isinstance(error, BattleError):
            return error

        error_message = str(error)
        error_type = type(error).__name__

        # Determine category based on error type and context
        if "openai" in error_message.lower() or "api" in error_message.lower():
            return OpenAIAPIError(
                message=error_message, agent_id=agent_id or "unknown", context=context
            )
        elif (
            "network" in error_message.lower() or "connection" in error_message.lower()
        ):
            return NetworkError(message=error_message, context=context)
        elif (
            "validation" in error_message.lower() or "invalid" in error_message.lower()
        ):
            return ValidationError(
                message=error_message, validation_type=error_type, context=context
            )
        else:
            return SystemError(
                message=f"Unexpected error: {error_message}",
                system_component=error_type,
                context=context,
            )

    def _log_error(self, error: BattleError) -> None:
        """Log error with appropriate level based on severity."""
        log_message = f"{error} | Context: {error.context}"

        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def _update_error_tracking(self, error: BattleError) -> None:
        """Update error tracking counters and circuit breakers."""
        error_key = f"{error.category.value}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Update circuit breakers
        if error.category == ErrorCategory.AI_DECISION:
            self._update_circuit_breaker("openai_api", error)
        elif error.category == ErrorCategory.NETWORK:
            self._update_circuit_breaker("websocket", error)
        elif error.category == ErrorCategory.GAME_STATE:
            self._update_circuit_breaker("game_state", error)

    def _update_circuit_breaker(self, breaker_name: str, error: BattleError) -> None:
        """Update circuit breaker state based on error."""
        if breaker_name not in self.circuit_breakers:
            config = self.circuit_breaker_config.get(breaker_name, {})
            self.circuit_breakers[breaker_name] = {
                "failure_count": 0,
                "last_failure_time": 0.0,
                "is_open": False,
                "failure_threshold": config.get("failure_threshold", 5),
                "timeout": config.get("timeout", 60.0),
            }

        breaker = self.circuit_breakers[breaker_name]
        current_time = time.time()

        # Check if circuit breaker should be reset
        if (
            breaker["is_open"]
            and (current_time - breaker["last_failure_time"]) > breaker["timeout"]
        ):
            breaker["is_open"] = False
            breaker["failure_count"] = 0
            logger.info(f"Circuit breaker {breaker_name} reset")

        # Update failure count
        if not breaker["is_open"]:
            breaker["failure_count"] += 1
            breaker["last_failure_time"] = current_time

            # Open circuit breaker if threshold exceeded
            if breaker["failure_count"] >= breaker["failure_threshold"]:
                breaker["is_open"] = True
                logger.warning(
                    f"Circuit breaker {breaker_name} opened after {breaker['failure_count']} failures"
                )

    def _is_circuit_breaker_open(self, error: BattleError) -> bool:
        """Check if circuit breaker is open for this error type."""
        breaker_name = None

        if error.category == ErrorCategory.AI_DECISION:
            breaker_name = "openai_api"
        elif error.category == ErrorCategory.NETWORK:
            breaker_name = "websocket"
        elif error.category == ErrorCategory.GAME_STATE:
            breaker_name = "game_state"

        if breaker_name and breaker_name in self.circuit_breakers:
            return self.circuit_breakers[breaker_name]["is_open"]

        return False

    def _get_recovery_strategy(self, error: BattleError) -> RecoveryStrategy:
        """Determine the appropriate recovery strategy for an error."""
        return self.recovery_strategies.get(error.category, RecoveryStrategy.FALLBACK)

    async def _execute_recovery_strategy(
        self, error: BattleError, strategy: RecoveryStrategy
    ) -> tuple[bool, Any]:
        """Execute the specified recovery strategy."""
        if strategy == RecoveryStrategy.FALLBACK:
            return await self._handle_fallback_recovery(error)
        elif strategy == RecoveryStrategy.RETRY:
            return await self._handle_retry_recovery(error)
        elif strategy == RecoveryStrategy.SKIP:
            return await self._handle_skip_recovery(error)
        elif strategy == RecoveryStrategy.RESET:
            return await self._handle_reset_recovery(error)
        elif strategy == RecoveryStrategy.ABORT:
            return await self._handle_abort_recovery(error)
        else:
            logger.error(f"Unknown recovery strategy: {strategy}")
            return False, None

    async def _handle_fallback_recovery(self, error: BattleError) -> tuple[bool, Any]:
        """Handle recovery using fallback mechanisms."""
        logger.info(f"Using fallback recovery for {error}")

        if isinstance(error, AIDecisionError) and error.agent_id:
            # Create a basic fallback decision
            fallback_decision = AgentDecision(
                primary_action=MoveAction(
                    target_position=(100.0, 100.0),
                    reason=f"Fallback due to: {error.message}",
                ),
                communication=CommunicateAction(
                    message="Experiencing technical difficulties", urgency="medium"
                ),
                confidence=0.1,
                strategic_assessment=f"Fallback mode: {error.message}",
            )
            return True, fallback_decision

        return True, None

    async def _handle_retry_recovery(self, error: BattleError) -> tuple[bool, Any]:
        """Handle recovery using retry logic."""
        logger.info(f"Using retry recovery for {error}")
        # Retry logic would be implemented here
        return False, None

    async def _handle_skip_recovery(self, error: BattleError) -> tuple[bool, Any]:
        """Handle recovery by skipping the failed operation."""
        logger.info(f"Skipping operation due to {error}")
        return True, None

    async def _handle_reset_recovery(self, error: BattleError) -> tuple[bool, Any]:
        """Handle recovery by resetting relevant state."""
        logger.warning(f"Resetting state due to {error}")
        return True, None

    async def _handle_abort_recovery(self, error: BattleError) -> tuple[bool, Any]:
        """Handle recovery by aborting the operation."""
        logger.error(f"Aborting operation due to {error}")
        return False, None

    async def _handle_circuit_breaker_fallback(
        self, error: BattleError
    ) -> tuple[bool, Any]:
        """Handle fallback when circuit breaker is open."""
        if isinstance(error, AIDecisionError) and error.agent_id:
            fallback_decision = AgentDecision(
                primary_action=MoveAction(
                    target_position=(100.0, 100.0),
                    reason="Circuit breaker open - using safe fallback",
                ),
                confidence=0.05,
                strategic_assessment="Circuit breaker protection active",
            )
            return True, fallback_decision

        return True, None

    async def _create_intelligent_fallback_decision(
        self, agent_id: str, world_state: BattleWorldState, error: AIDecisionError
    ) -> AgentDecision:
        """Create an intelligent fallback decision based on game state."""
        agent = world_state.agents.get(agent_id)

        if not agent:
            # Agent not found - create basic fallback
            return AgentDecision(
                primary_action=MoveAction(
                    target_position=(100.0, 100.0),
                    reason="Agent not found in world state",
                ),
                confidence=0.1,
                strategic_assessment="Emergency fallback - agent missing",
            )

        # Analyze agent's situation for intelligent fallback
        if agent.health < 30:
            # Low health - prioritize healing or retreat
            action = MoveAction(
                target_position=self._find_safe_position(agent, world_state),
                reason="Low health - retreating to safety",
            )
            communication = CommunicateAction(
                message="Taking cover - low health", urgency="high"
            )
        elif agent.weapon_condition.value == "broken":
            # Broken weapon - head to forge
            forge_position = self._find_team_forge(agent.team, world_state)
            action = MoveAction(
                target_position=forge_position,
                reason="Weapon broken - heading to forge",
            )
            communication = CommunicateAction(
                message="Weapon broken, going to forge", urgency="medium"
            )
        else:
            # Default - move toward team center
            team_center = self._calculate_team_center(agent.team, world_state)
            action = MoveAction(
                target_position=team_center,
                reason="AI decision failed - regrouping with team",
            )
            communication = CommunicateAction(
                message="AI malfunction - regrouping", urgency="medium"
            )

        return AgentDecision(
            primary_action=action,
            communication=communication,
            confidence=0.3,  # Low but not zero confidence
            strategic_assessment=f"Intelligent fallback: {action.reason}",
        )

    def _find_safe_position(
        self, agent: Agent, world_state: BattleWorldState
    ) -> tuple[float, float]:
        """Find a safe position for an agent to retreat to."""
        # Simple implementation - move away from enemies toward team forge
        forge_position = self._find_team_forge(agent.team, world_state)

        # Add some randomness to avoid clustering
        import random

        offset_x = random.uniform(-20.0, 20.0)
        offset_y = random.uniform(-20.0, 20.0)

        safe_x = max(0, min(1000, forge_position[0] + offset_x))
        safe_y = max(0, min(1000, forge_position[1] + offset_y))

        return (safe_x, safe_y)

    def _find_team_forge(
        self, team: str, world_state: BattleWorldState
    ) -> tuple[float, float]:
        """Find the position of the team's forge."""
        team_suffix = team.split("_")[1]  # "team_a" -> "a"
        forge_id = f"forge_{team_suffix}"

        forge = world_state.map_locations.get(forge_id)
        if forge:
            return forge.position

        # Fallback positions if forge not found
        if team == "team_a":
            return (50.0, 50.0)
        else:
            return (950.0, 950.0)

    def _calculate_team_center(
        self, team: str, world_state: BattleWorldState
    ) -> tuple[float, float]:
        """Calculate the center position of team members."""
        team_agents = world_state.get_alive_agents(team)

        if not team_agents:
            # No team members - use default position
            return (500.0, 500.0)

        total_x = sum(agent.position[0] for agent in team_agents.values())
        total_y = sum(agent.position[1] for agent in team_agents.values())
        count = len(team_agents)

        return (total_x / count, total_y / count)
