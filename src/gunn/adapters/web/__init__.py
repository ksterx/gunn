"""Web adapter for REST and WebSocket endpoints.

This module provides FastAPI-based REST and WebSocket endpoints for external
system integration with authentication, authorization, and rate limiting.
"""

from gunn.adapters.web.server import (
    AuthToken,
    ErrorResponse,
    IntentRequest,
    IntentResponse,
    ObservationResponse,
    WebAdapter,
    create_web_adapter,
)

__all__ = [
    "AuthToken",
    "WebAdapter",
    "create_web_adapter",
    "ErrorResponse",
    "IntentRequest",
    "IntentResponse",
    "ObservationResponse",
]
