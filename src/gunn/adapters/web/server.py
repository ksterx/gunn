"""FastAPI-based Web adapter for external system integration.

This module provides REST and WebSocket endpoints for external systems to
interact with the multi-agent simulation core, including authentication,
authorization, and rate limiting.

Requirements addressed:
- 8.3: Expose REST/WebSocket endpoints for external system integration
- 12.1: Authenticate using token-based auth and authorize per agent_id/world_id
- 12.4: Enforce rate limiting per tenant and agent
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from gunn.core.orchestrator import Orchestrator
from gunn.schemas.types import Intent
from gunn.utils.errors import (
    BackpressureError,
    QuotaExceededError,
    StaleContextError,
    ValidationError,
)
from gunn.utils.telemetry import get_logger


class AuthToken(BaseModel):
    """Authentication token with permissions."""

    token: str
    world_id: str
    agent_id: str
    permissions: list[str] = Field(default_factory=list)
    expires_at: float | None = None


class IntentRequest(BaseModel):
    """Request model for intent submission."""

    kind: str = Field(..., description="Type of intent")
    payload: dict[str, Any] = Field(default_factory=dict, description="Intent payload")
    priority: int = Field(default=0, description="Intent priority")
    context_seq: int = Field(default=0, description="Context sequence number")
    schema_version: str = Field(default="1.0.0", description="Schema version")


class IntentResponse(BaseModel):
    """Response model for intent submission."""

    req_id: str = Field(..., description="Request ID for tracking")
    status: str = Field(..., description="Submission status")
    message: str | None = Field(None, description="Status message")


class ObservationResponse(BaseModel):
    """Response model for observation requests."""

    view_seq: int = Field(..., description="View sequence number")
    patches: list[dict[str, Any]] = Field(..., description="JSON Patch operations")
    context_digest: str = Field(..., description="Context digest")
    schema_version: str = Field(..., description="Schema version")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional details"
    )


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = {}

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for the given key.

        Args:
            key: Rate limiting key (e.g., "world_id:agent_id")

        Returns:
            True if request is allowed
        """
        now = time.time()
        cutoff = now - self.window_seconds

        # Initialize or clean up old requests
        if key not in self.requests:
            self.requests[key] = []
        else:
            self.requests[key] = [t for t in self.requests[key] if t > cutoff]

        # Check if under limit
        if len(self.requests[key]) >= self.max_requests:
            return False

        # Record this request
        self.requests[key].append(now)
        return True


class WebAdapter:
    """FastAPI-based Web adapter for external integration."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        auth_tokens: dict[str, AuthToken] | None = None,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60,
    ):
        """Initialize Web adapter.

        Args:
            orchestrator: Orchestrator instance
            auth_tokens: Authentication tokens mapping
            rate_limit_requests: Rate limit requests per window
            rate_limit_window: Rate limit window in seconds
        """
        self.orchestrator = orchestrator
        self.auth_tokens = auth_tokens or {}
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)
        self.websocket_connections: dict[str, list[WebSocket]] = {}
        self.logger = get_logger("gunn.web_adapter")

        # Create FastAPI app with lifespan
        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            # Startup
            await self.orchestrator.initialize()
            self.logger.info("Web adapter started")
            yield
            # Shutdown
            await self.shutdown()
            self.logger.info("Web adapter stopped")

        self.app = FastAPI(
            title="Gunn Multi-Agent Simulation API",
            description="REST and WebSocket API for multi-agent simulation core",
            version="1.0.0",
            lifespan=lifespan,
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up API routes."""
        security = HTTPBearer()

        async def get_current_auth(
            credentials: HTTPAuthorizationCredentials = Depends(security),
        ) -> AuthToken:
            """Get current authentication token."""
            token = credentials.credentials
            if token not in self.auth_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication token",
                )

            auth_token = self.auth_tokens[token]

            # Check expiration
            if auth_token.expires_at and time.time() > auth_token.expires_at:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication token expired",
                )

            return auth_token

        async def check_rate_limit(
            request: Request, auth: AuthToken = Depends(get_current_auth)
        ) -> None:
            """Check rate limiting."""
            rate_key = f"{auth.world_id}:{auth.agent_id}"
            if not self.rate_limiter.is_allowed(rate_key):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                )

        @self.app.get("/health")
        async def health_check() -> dict[str, Any]:
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": time.time()}

        @self.app.post(
            "/worlds/{world_id}/agents/{agent_id}/intents",
            response_model=IntentResponse,
            responses={
                400: {"model": ErrorResponse},
                401: {"model": ErrorResponse},
                429: {"model": ErrorResponse},
                409: {"model": ErrorResponse},
                422: {"model": ErrorResponse},
            },
        )
        async def submit_intent(
            world_id: str,
            agent_id: str,
            intent_request: IntentRequest,
            auth: AuthToken = Depends(get_current_auth),
            _: None = Depends(check_rate_limit),
        ) -> IntentResponse:
            """Submit an intent for processing."""
            # Verify authorization
            if auth.world_id != world_id or auth.agent_id != agent_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized for this world/agent",
                )

            # Check permissions
            if "submit_intent" not in auth.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Missing submit_intent permission",
                )

            try:
                # Generate unique request ID
                req_id = f"web_{int(time.time() * 1000000)}"

                # Create intent
                intent: Intent = {
                    "kind": intent_request.kind,  # type: ignore
                    "payload": intent_request.payload,
                    "context_seq": intent_request.context_seq,
                    "req_id": req_id,
                    "agent_id": agent_id,
                    "priority": intent_request.priority,
                    "schema_version": intent_request.schema_version,
                }

                # Submit intent
                await self.orchestrator.submit_intent(intent)

                return IntentResponse(
                    req_id=req_id,
                    status="accepted",
                    message="Intent submitted successfully",
                )

            except StaleContextError as e:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=ErrorResponse(
                        error="STALE_CONTEXT",
                        message=str(e),
                        details={
                            "expected_seq": e.expected_seq,
                            "actual_seq": e.actual_seq,
                            "threshold": e.threshold,
                        },
                    ).model_dump(),
                ) from e
            except QuotaExceededError as e:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=ErrorResponse(
                        error="QUOTA_EXCEEDED",
                        message=str(e),
                        details={
                            "quota_type": e.quota_type,
                            "limit": e.limit,
                            "current": e.current,
                        },
                    ).model_dump(),
                ) from e
            except BackpressureError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=ErrorResponse(
                        error="BACKPRESSURE",
                        message=str(e),
                        details={
                            "queue_type": e.queue_type,
                            "current_depth": e.current_depth,
                            "max_depth": e.threshold,
                            "policy": e.policy,
                        },
                    ).model_dump(),
                ) from e
            except ValidationError as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=ErrorResponse(
                        error="VALIDATION_ERROR",
                        message=str(e),
                    ).model_dump(),
                ) from e
            except Exception as e:
                self.logger.error(
                    "Intent submission failed",
                    agent_id=agent_id,
                    world_id=world_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=ErrorResponse(
                        error="INTERNAL_ERROR",
                        message="Internal server error",
                    ).model_dump(),
                ) from e

        @self.app.get(
            "/worlds/{world_id}/agents/{agent_id}/observations",
            response_model=ObservationResponse,
            responses={
                401: {"model": ErrorResponse},
                403: {"model": ErrorResponse},
                429: {"model": ErrorResponse},
                408: {"model": ErrorResponse},
            },
        )
        async def get_observation(
            world_id: str,
            agent_id: str,
            timeout: float = 30.0,
            auth: AuthToken = Depends(get_current_auth),
            _: None = Depends(check_rate_limit),
        ) -> ObservationResponse:
            """Get next observation for an agent."""
            # Verify authorization
            if auth.world_id != world_id or auth.agent_id != agent_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized for this world/agent",
                )

            # Check permissions
            if "get_observation" not in auth.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Missing get_observation permission",
                )

            try:
                # Get agent handle
                if agent_id not in self.orchestrator.agent_handles:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Agent {agent_id} not found",
                    )

                agent_handle = self.orchestrator.agent_handles[agent_id]

                # Get next observation with timeout
                try:
                    observation = await asyncio.wait_for(
                        agent_handle.next_observation(), timeout=timeout
                    )
                except TimeoutError as e:
                    raise HTTPException(
                        status_code=status.HTTP_408_REQUEST_TIMEOUT,
                        detail="Observation timeout",
                    ) from e

                # Convert to response format
                if isinstance(observation, dict):
                    return ObservationResponse(
                        view_seq=observation.get("view_seq", 0),
                        patches=observation.get("patches", []),
                        context_digest=observation.get("context_digest", ""),
                        schema_version=observation.get("schema_version", "1.0.0"),
                    )
                else:
                    # Handle other observation formats
                    return ObservationResponse(
                        view_seq=getattr(observation, "view_seq", 0),
                        patches=getattr(observation, "patches", []),
                        context_digest=getattr(observation, "context_digest", ""),
                        schema_version=getattr(observation, "schema_version", "1.0.0"),
                    )

            except HTTPException:
                # Re-raise HTTP exceptions (like timeout)
                raise
            except Exception as e:
                self.logger.error(
                    "Observation retrieval failed",
                    agent_id=agent_id,
                    world_id=world_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=ErrorResponse(
                        error="INTERNAL_ERROR",
                        message="Internal server error",
                    ).model_dump(),
                ) from e

        @self.app.websocket("/worlds/{world_id}/agents/{agent_id}/observations/stream")
        async def websocket_observations(
            websocket: WebSocket, world_id: str, agent_id: str, token: str
        ) -> None:
            """WebSocket endpoint for real-time observation streaming."""
            # Authenticate
            if token not in self.auth_tokens:
                await websocket.close(code=4001, reason="Invalid token")
                return

            auth_token = self.auth_tokens[token]

            # Check authorization
            if auth_token.world_id != world_id or auth_token.agent_id != agent_id:
                await websocket.close(code=4003, reason="Not authorized")
                return

            # Check permissions
            if "stream_observations" not in auth_token.permissions:
                await websocket.close(code=4003, reason="Missing permission")
                return

            # Check expiration
            if auth_token.expires_at and time.time() > auth_token.expires_at:
                await websocket.close(code=4001, reason="Token expired")
                return

            await websocket.accept()

            # Add to connections
            connection_key = f"{world_id}:{agent_id}"
            if connection_key not in self.websocket_connections:
                self.websocket_connections[connection_key] = []
            self.websocket_connections[connection_key].append(websocket)

            try:
                # Get agent handle
                if agent_id not in self.orchestrator.agent_handles:
                    await websocket.send_json(
                        {
                            "error": "AGENT_NOT_FOUND",
                            "message": f"Agent {agent_id} not found",
                        }
                    )
                    return

                agent_handle = self.orchestrator.agent_handles[agent_id]

                # Stream observations
                while True:
                    try:
                        observation = await agent_handle.next_observation()

                        # Convert to JSON-serializable format
                        if isinstance(observation, dict):
                            await websocket.send_json(observation)
                        else:
                            # Handle other observation formats
                            obs_dict = {
                                "view_seq": getattr(observation, "view_seq", 0),
                                "patches": getattr(observation, "patches", []),
                                "context_digest": getattr(
                                    observation, "context_digest", ""
                                ),
                                "schema_version": getattr(
                                    observation, "schema_version", "1.0.0"
                                ),
                            }
                            await websocket.send_json(obs_dict)

                    except Exception as e:
                        self.logger.error(
                            "WebSocket observation streaming error",
                            agent_id=agent_id,
                            world_id=world_id,
                            error=str(e),
                            error_type=type(e).__name__,
                        )
                        await websocket.send_json(
                            {"error": "STREAM_ERROR", "message": str(e)}
                        )
                        break

            except WebSocketDisconnect:
                self.logger.info(
                    "WebSocket disconnected",
                    agent_id=agent_id,
                    world_id=world_id,
                )
            finally:
                # Remove from connections
                if connection_key in self.websocket_connections:
                    try:
                        self.websocket_connections[connection_key].remove(websocket)
                        if not self.websocket_connections[connection_key]:
                            del self.websocket_connections[connection_key]
                    except ValueError:
                        pass  # Already removed

    def add_auth_token(
        self,
        token: str,
        world_id: str,
        agent_id: str,
        permissions: list[str],
        expires_at: float | None = None,
    ) -> None:
        """Add authentication token.

        Args:
            token: Authentication token
            world_id: World ID this token is valid for
            agent_id: Agent ID this token is valid for
            permissions: List of permissions
            expires_at: Optional expiration timestamp
        """
        self.auth_tokens[token] = AuthToken(
            token=token,
            world_id=world_id,
            agent_id=agent_id,
            permissions=permissions,
            expires_at=expires_at,
        )

        self.logger.info(
            "Authentication token added",
            world_id=world_id,
            agent_id=agent_id,
            permissions=permissions,
            expires_at=expires_at,
        )

    def remove_auth_token(self, token: str) -> None:
        """Remove authentication token.

        Args:
            token: Authentication token to remove
        """
        if token in self.auth_tokens:
            auth_token = self.auth_tokens.pop(token)
            self.logger.info(
                "Authentication token removed",
                world_id=auth_token.world_id,
                agent_id=auth_token.agent_id,
            )

    async def shutdown(self) -> None:
        """Shutdown the web adapter."""
        # Close all WebSocket connections
        for connections in self.websocket_connections.values():
            for websocket in connections:
                try:
                    await websocket.close()
                except Exception:
                    pass  # Ignore errors during shutdown

        self.websocket_connections.clear()
        self.logger.info("Web adapter shutdown complete")


def create_web_adapter(
    orchestrator: Orchestrator,
    auth_tokens: dict[str, AuthToken] | None = None,
    rate_limit_requests: int = 100,
    rate_limit_window: int = 60,
) -> WebAdapter:
    """Create a Web adapter instance.

    Args:
        orchestrator: Orchestrator instance
        auth_tokens: Authentication tokens mapping
        rate_limit_requests: Rate limit requests per window
        rate_limit_window: Rate limit window in seconds

    Returns:
        WebAdapter instance
    """
    return WebAdapter(
        orchestrator=orchestrator,
        auth_tokens=auth_tokens,
        rate_limit_requests=rate_limit_requests,
        rate_limit_window=rate_limit_window,
    )
