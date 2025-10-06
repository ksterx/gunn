"""Pydantic models for type-safe Intent and Effect validation.

These models provide optional type-safe wrappers around Gunn's TypedDict-based
types. Use at API boundaries for better developer experience while maintaining
TypedDict performance internally.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class IntentModel(BaseModel):
    """Pydantic model for Intent with validation.

    Example:
        >>> intent = IntentModel(
        ...     kind="Move",
        ...     payload={"to": [10.0, 20.0]},
        ...     context_seq=0,
        ...     req_id="move_1",
        ...     agent_id="agent_a",
        ...     priority=1,
        ...     schema_version="1.0.0"
        ... )
        >>> intent_dict = intent.to_dict()
    """

    model_config = ConfigDict(
        strict=True,
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    kind: Literal["Speak", "Move", "Interact", "Custom"] = Field(
        ..., description="Type of action to perform"
    )
    payload: dict[str, Any] = Field(
        ..., description="Action-specific data (e.g., {'to': [x,y,z]} for Move)"
    )
    context_seq: int = Field(
        ...,
        description="Agent's view sequence when this intent was created (for staleness detection)",
        ge=0,
    )
    req_id: str = Field(
        ..., description="Unique request identifier for tracking and deduplication"
    )
    agent_id: str = Field(..., description="ID of the agent submitting this intent")
    priority: int = Field(
        ..., description="Processing priority (higher numbers = higher priority)"
    )
    schema_version: str = Field(
        ..., description="Schema version for compatibility (e.g., '1.0.0')"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to TypedDict-compatible dictionary for internal use."""
        return self.model_dump(mode="python")


class EffectDraftModel(BaseModel):
    """Pydantic model for EffectDraft with validation.

    External input for effects - Orchestrator fills in uuid, global_seq, sim_time.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    kind: str = Field(
        ..., description="Type of effect (e.g., 'Move', 'Speak', 'EnvironmentChanged')"
    )
    payload: dict[str, Any] = Field(..., description="Effect-specific data")
    source_id: str = Field(
        ..., description="ID of the system/agent that created this effect"
    )
    schema_version: str = Field(..., description="Schema version for compatibility")

    def to_dict(self) -> dict[str, Any]:
        """Convert to TypedDict-compatible dictionary for internal use."""
        return self.model_dump(mode="python")


class EffectModel(BaseModel):
    """Pydantic model for Effect with validation.

    Complete effect with all fields filled by Orchestrator.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    uuid: str = Field(..., description="Unique ID for ordering tie-breaker")
    kind: str = Field(
        ..., description="Type of effect (e.g., 'Move', 'Speak', 'EnvironmentChanged')"
    )
    payload: dict[str, Any] = Field(..., description="Effect-specific data")
    global_seq: int = Field(
        ..., description="Global sequence number for deterministic ordering", ge=0
    )
    sim_time: float = Field(
        ..., description="Simulation time when this effect occurred", ge=0.0
    )
    source_id: str = Field(
        ..., description="ID of the system/agent that created this effect"
    )
    schema_version: str = Field(
        ..., description="Semantic versioning (e.g., '1.0.0')"
    )
    req_id: str | None = Field(
        None, description="Request ID for tracking action completion"
    )
    duration_ms: float | None = Field(
        None, description="Optional duration for interval effects", ge=0.0
    )
    apply_at: float | None = Field(
        None, description="Optional delayed application timestamp", ge=0.0
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to TypedDict-compatible dictionary for internal use."""
        return self.model_dump(mode="python")


class ObservationDeltaModel(BaseModel):
    """Pydantic model for ObservationDelta with validation.

    RFC6902 JSON Patch operations for incremental view updates.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    view_seq: int = Field(
        ...,
        description="Sequence number of the view this delta updates to",
        ge=0,
    )
    patches: list[dict[str, Any]] = Field(
        ..., description="RFC6902 JSON Patch operations with stable paths"
    )
    context_digest: str = Field(
        ..., description="Hash digest of the resulting view state for integrity checking"
    )
    schema_version: str = Field(..., description="Schema version for compatibility")
    delivery_id: str = Field(
        ..., description="Unique identifier for this delivery attempt"
    )
    redelivery: bool = Field(
        ..., description="Flag indicating this is a redelivery attempt"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to TypedDict-compatible dictionary for internal use."""
        return self.model_dump(mode="python")
