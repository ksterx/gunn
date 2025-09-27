"""Core type definitions using TypedDict for performance and schema versioning."""

import asyncio
from typing import Any, Literal, TypedDict


class Intent(TypedDict):
    """Intent submitted by an agent to perform an action."""

    kind: Literal["Speak", "Move", "Interact", "Custom"]
    payload: dict[str, Any]
    context_seq: int
    req_id: str
    agent_id: str
    priority: int  # Higher numbers = higher priority
    schema_version: str


class EffectDraft(TypedDict):
    """External input for effects - Orchestrator fills in uuid, global_seq, sim_time."""

    kind: str
    payload: dict[str, Any]
    source_id: str
    schema_version: str


class Effect(TypedDict):
    """Complete effect with all fields filled by Orchestrator."""

    uuid: str  # Unique ID for ordering tie-breaker
    kind: str
    payload: dict[str, Any]
    global_seq: int
    sim_time: float
    source_id: str
    schema_version: str  # Semantic versioning (e.g., "1.0.0")


class ObservationDelta(TypedDict):
    """RFC6902 JSON Patch operations for incremental view updates."""

    view_seq: int
    patches: list[dict[str, Any]]  # RFC6902 JSON Patch operations with stable paths
    context_digest: str
    schema_version: str


class CancelToken:
    """Token for tracking and cancelling agent generation operations.

    This class provides a thread-safe way to signal cancellation of long-running
    operations like LLM generation, with reason tracking for debugging.
    """

    def __init__(self, req_id: str, agent_id: str) -> None:
        """Initialize a new cancel token.

        Args:
            req_id: Unique request identifier
            agent_id: Agent identifier this token belongs to

        Raises:
            ValueError: If req_id or agent_id is empty
        """
        if not req_id.strip():
            raise ValueError("req_id cannot be empty")
        if not agent_id.strip():
            raise ValueError("agent_id cannot be empty")

        self.req_id = req_id
        self.agent_id = agent_id
        self._cancelled = asyncio.Event()
        self.reason: str | None = None

    @property
    def cancelled(self) -> bool:
        """Check if the token has been cancelled."""
        return self._cancelled.is_set()

    def cancel(self, reason: str) -> None:
        """Cancel the token with a specific reason.

        Args:
            reason: Human-readable reason for cancellation

        Raises:
            ValueError: If reason is empty
        """
        if not reason.strip():
            raise ValueError("Cancellation reason cannot be empty")

        self.reason = reason
        self._cancelled.set()

    async def wait_cancelled(self) -> None:
        """Wait until token is cancelled.

        This method will block until cancel() is called on this token.
        """
        await self._cancelled.wait()

    def __repr__(self) -> str:
        status = "cancelled" if self.cancelled else "active"
        reason_info = f", reason={self.reason}" if self.reason else ""
        return f"CancelToken(req_id={self.req_id}, agent_id={self.agent_id}, status={status}{reason_info})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on req_id and agent_id."""
        if not isinstance(other, CancelToken):
            return NotImplemented
        return self.req_id == other.req_id and self.agent_id == other.agent_id

    def __hash__(self) -> int:
        """Hash based on req_id and agent_id for use in sets/dicts."""
        return hash((self.req_id, self.agent_id))
