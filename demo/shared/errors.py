"""
Comprehensive error handling system for the battle demo.

This module defines a hierarchy of battle-specific errors and provides
error handling utilities with fallback mechanisms and recovery strategies.
"""

from enum import Enum
from typing import Any


class ErrorSeverity(str, Enum):
    """Severity levels for battle errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Categories of battle errors."""

    AI_DECISION = "ai_decision"
    NETWORK = "network"
    GAME_STATE = "game_state"
    VALIDATION = "validation"
    SYSTEM = "system"


class BattleError(Exception):
    """Base class for all battle-specific errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        agent_id: str | None = None,
        recoverable: bool = True,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.agent_id = agent_id
        self.recoverable = recoverable
        self.context = context or {}

    def __str__(self) -> str:
        agent_info = f" (Agent: {self.agent_id})" if self.agent_id else ""
        return f"[{self.category.value.upper()}:{self.severity.value.upper()}] {self.message}{agent_info}"

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "agent_id": self.agent_id,
            "recoverable": self.recoverable,
            "context": self.context,
        }


class AIDecisionError(BattleError):
    """Errors related to AI decision making."""

    def __init__(
        self,
        message: str,
        agent_id: str,
        decision_type: str | None = None,
        api_error: Exception | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update(
            {
                "decision_type": decision_type,
                "api_error": str(api_error) if api_error else None,
            }
        )
        kwargs["context"] = context

        super().__init__(
            message=message,
            category=ErrorCategory.AI_DECISION,
            agent_id=agent_id,
            **kwargs,
        )
        self.decision_type = decision_type
        self.api_error = api_error


class OpenAIAPIError(AIDecisionError):
    """Specific errors from OpenAI API calls."""

    def __init__(
        self,
        message: str,
        agent_id: str,
        status_code: int | None = None,
        retry_count: int = 0,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update({"status_code": status_code, "retry_count": retry_count})
        kwargs["context"] = context

        super().__init__(
            message=f"OpenAI API Error: {message}", agent_id=agent_id, **kwargs
        )
        self.status_code = status_code
        self.retry_count = retry_count


class NetworkError(BattleError):
    """Network-related errors."""

    def __init__(
        self,
        message: str,
        endpoint: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update({"endpoint": endpoint, "timeout": timeout})
        kwargs["context"] = context

        super().__init__(message=message, category=ErrorCategory.NETWORK, **kwargs)
        self.endpoint = endpoint
        self.timeout = timeout


class GameStateError(BattleError):
    """Errors related to game state consistency."""

    def __init__(self, message: str, state_component: str | None = None, **kwargs):
        context = kwargs.get("context", {})
        context.update({"state_component": state_component})
        kwargs["context"] = context

        super().__init__(message=message, category=ErrorCategory.GAME_STATE, **kwargs)
        self.state_component = state_component


class ValidationError(BattleError):
    """Errors related to data validation."""

    def __init__(
        self,
        message: str,
        validation_type: str | None = None,
        invalid_data: Any = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update(
            {
                "validation_type": validation_type,
                "invalid_data": str(invalid_data) if invalid_data is not None else None,
            }
        )
        kwargs["context"] = context

        super().__init__(message=message, category=ErrorCategory.VALIDATION, **kwargs)
        self.validation_type = validation_type
        self.invalid_data = invalid_data


class SystemError(BattleError):
    """System-level errors."""

    def __init__(self, message: str, system_component: str | None = None, **kwargs):
        context = kwargs.get("context", {})
        context.update({"system_component": system_component})
        kwargs["context"] = context

        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs,
        )
        self.system_component = system_component


class PerformanceError(BattleError):
    """Performance-related errors."""

    def __init__(
        self,
        message: str,
        metric_type: str | None = None,
        threshold_value: float | None = None,
        actual_value: float | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update(
            {
                "metric_type": metric_type,
                "threshold_value": threshold_value,
                "actual_value": actual_value,
            }
        )
        kwargs["context"] = context

        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )
        self.metric_type = metric_type
        self.threshold_value = threshold_value
        self.actual_value = actual_value


class ConcurrentProcessingError(BattleError):
    """Errors in concurrent agent processing."""

    def __init__(self, message: str, failed_agents: list[str] | None = None, **kwargs):
        context = kwargs.get("context", {})
        context.update({"failed_agents": failed_agents or []})
        kwargs["context"] = context

        super().__init__(message=message, category=ErrorCategory.SYSTEM, **kwargs)
        self.failed_agents = failed_agents or []


class WebSocketError(NetworkError):
    """WebSocket-specific errors."""

    def __init__(self, message: str, connection_id: str | None = None, **kwargs):
        context = kwargs.get("context", {})
        context.update({"connection_id": connection_id})
        kwargs["context"] = context

        super().__init__(message=f"WebSocket Error: {message}", **kwargs)
        self.connection_id = connection_id


class EffectProcessingError(BattleError):
    """Errors in effect processing."""

    def __init__(
        self,
        message: str,
        effect_type: str | None = None,
        effect_data: dict[str, Any] | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update({"effect_type": effect_type, "effect_data": effect_data})
        kwargs["context"] = context

        super().__init__(message=message, category=ErrorCategory.GAME_STATE, **kwargs)
        self.effect_type = effect_type
        self.effect_data = effect_data


class RecoveryStrategy(str, Enum):
    """Available recovery strategies for errors."""

    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    RESET = "reset"


class ErrorRecoveryInfo:
    """Information about error recovery attempts."""

    def __init__(
        self,
        strategy: RecoveryStrategy,
        max_attempts: int = 3,
        backoff_factor: float = 1.5,
        timeout: float = 30.0,
    ):
        self.strategy = strategy
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.attempts = 0
        self.last_attempt_time = 0.0
        self.success = False

    def should_retry(self, current_time: float) -> bool:
        """Check if another retry attempt should be made."""
        if self.attempts >= self.max_attempts:
            return False

        if self.strategy != RecoveryStrategy.RETRY:
            return False

        # Always allow retry if we haven't exceeded max attempts
        return True

    def record_attempt(self, current_time: float, success: bool = False) -> None:
        """Record an attempt and its outcome."""
        self.attempts += 1
        self.last_attempt_time = current_time
        self.success = success

    def get_next_delay(self) -> float:
        """Get the delay before the next retry attempt."""
        if self.attempts == 0:
            return 0.0  # No delay for first attempt
        return self.backoff_factor ** (self.attempts - 1)
