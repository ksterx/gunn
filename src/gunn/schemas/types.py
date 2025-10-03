"""Core type definitions using TypedDict for performance and schema versioning."""

import asyncio
from typing import Annotated, Any, Literal, TypedDict


class Intent(TypedDict):
    """Intent submitted by an agent to perform an action.

    This represents a request from an agent to perform a specific action in the world.
    Intents go through validation before being converted to Effects.
    """

    kind: Annotated[
        Literal["Speak", "Move", "Interact", "Custom"], "Type of action to perform"
    ]
    payload: Annotated[
        dict[str, Any], "Action-specific data (e.g., {'to': [x,y,z]} for Move)"
    ]
    context_seq: Annotated[
        int,
        "Agent's view sequence when this intent was created (for staleness detection)",
    ]
    req_id: Annotated[str, "Unique request identifier for tracking and deduplication"]
    agent_id: Annotated[str, "ID of the agent submitting this intent"]
    priority: Annotated[int, "Processing priority (higher numbers = higher priority)"]
    schema_version: Annotated[str, "Schema version for compatibility (e.g., '1.0.0')"]


class EffectDraft(TypedDict):
    """External input for effects - Orchestrator fills in uuid, global_seq, sim_time.

    This is the incomplete form of an Effect, used when external systems (like game engines)
    want to broadcast events. The Orchestrator completes it with timing and sequence info.
    """

    kind: Annotated[str, "Type of effect (e.g., 'Move', 'Speak', 'EnvironmentChanged')"]
    payload: Annotated[dict[str, Any], "Effect-specific data"]
    source_id: Annotated[str, "ID of the system/agent that created this effect"]
    schema_version: Annotated[str, "Schema version for compatibility"]


class Effect(TypedDict):
    """Complete effect with all fields filled by Orchestrator."""

    uuid: Annotated[str, "Unique ID for ordering tie-breaker"]
    kind: Annotated[str, "Type of effect (e.g., 'Move', 'Speak', 'EnvironmentChanged')"]
    payload: Annotated[dict[str, Any], "Effect-specific data"]
    global_seq: Annotated[int, "Global sequence number for deterministic ordering"]
    sim_time: Annotated[float, "Simulation time when this effect occurred"]
    source_id: Annotated[str, "ID of the system/agent that created this effect"]
    schema_version: Annotated[str, "Semantic versioning (e.g., '1.0.0')"]


class ObservationDelta(TypedDict):
    """RFC6902 JSON Patch operations for incremental view updates."""

    view_seq: Annotated[int, "Sequence number of the view this delta updates to"]
    patches: Annotated[
        list[dict[str, Any]], "RFC6902 JSON Patch operations with stable paths"
    ]
    context_digest: Annotated[
        str, "Hash digest of the resulting view state for integrity checking"
    ]
    schema_version: Annotated[str, "Schema version for compatibility"]


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
