"""Pydantic models for data validation and JSON schema generation."""

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorldState(BaseModel):
    """Complete world state containing all entities and relationships."""

    entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Map of entity_id to entity data",
        json_schema_extra={"example": {"agent1": {"name": "Alice", "health": 100}}},
    )
    relationships: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map of entity_id to list of related entity_ids",
        json_schema_extra={"example": {"agent1": ["agent2", "agent3"]}},
    )
    spatial_index: dict[str, tuple[float, float, float]] = Field(
        default_factory=dict,
        description="Map of entity_id to (x, y, z) coordinates",
        json_schema_extra={"example": {"agent1": [10.0, 20.0, 0.0]}},
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional world metadata",
        json_schema_extra={
            "example": {"world_name": "simulation_1", "version": "1.0.0"}
        },
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "entities": {"agent1": {"name": "Alice", "health": 100}},
                    "relationships": {"agent1": ["agent2"]},
                    "spatial_index": {"agent1": [10.0, 20.0, 0.0]},
                    "metadata": {"world_name": "test_world"},
                }
            ]
        },
    )

    @field_validator("spatial_index")
    @classmethod
    def validate_spatial_coordinates(
        cls, v: dict[str, tuple[float, float, float]]
    ) -> dict[str, tuple[float, float, float]]:
        """Validate that spatial coordinates are 3D tuples of floats."""
        # Pydantic v2 handles most validation automatically via type annotations
        # This validator is mainly for custom business logic if needed
        return v


class View(BaseModel):
    """Agent's filtered view of the world state."""

    agent_id: str = Field(
        description="ID of the agent this view belongs to",
        min_length=1,
        json_schema_extra={"example": "agent_001"},
    )
    view_seq: int = Field(
        description="Sequence number of this view",
        ge=0,
        json_schema_extra={"example": 42},
    )
    visible_entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Entities visible to this agent",
        json_schema_extra={"example": {"entity1": {"type": "player", "name": "Alice"}}},
    )
    visible_relationships: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Relationships visible to this agent",
        json_schema_extra={"example": {"entity1": ["entity2"]}},
    )
    context_digest: str = Field(
        description="Hash digest of the view context",
        min_length=1,
        json_schema_extra={"example": "sha256:abc123def456..."},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "agent_id": "agent_001",
                    "view_seq": 42,
                    "visible_entities": {
                        "entity1": {"type": "player", "name": "Alice"}
                    },
                    "visible_relationships": {"entity1": ["entity2"]},
                    "context_digest": "sha256:abc123def456",
                }
            ]
        },
    )

    # view_seq validation is handled by Field(ge=0) constraint


class EventLogEntry(BaseModel):
    """Single entry in the event log with integrity checking."""

    global_seq: int = Field(
        description="Global sequence number", ge=0, json_schema_extra={"example": 1001}
    )
    sim_time: float = Field(
        description="Simulation time when event occurred",
        ge=0.0,
        json_schema_extra={"example": 123.456},
    )
    wall_time: float = Field(
        default_factory=time.time,
        description="Wall clock time when event was recorded",
        ge=0.0,
        json_schema_extra={"example": 1640995200.0},
    )
    effect: dict[str, Any] = Field(
        description="The effect that occurred",
        json_schema_extra={
            "example": {
                "uuid": "550e8400-e29b-41d4-a716-446655440000",
                "kind": "Move",
                "payload": {"x": 10, "y": 20},
                "global_seq": 1001,
                "sim_time": 123.456,
                "source_id": "agent_001",
                "schema_version": "1.0.0",
            }
        },
    )
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the event source",
        json_schema_extra={"example": {"adapter": "unity", "world_id": "world_001"}},
    )
    checksum: str = Field(
        description="Hash chain checksum for integrity",
        min_length=1,
        json_schema_extra={"example": "sha256:def789abc123..."},
    )
    req_id: str = Field(
        default="",
        description="Request ID for idempotency checking",
        json_schema_extra={"example": "req_550e8400-e29b-41d4"},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "global_seq": 1001,
                    "sim_time": 123.456,
                    "wall_time": 1640995200.0,
                    "effect": {
                        "uuid": "550e8400-e29b-41d4-a716-446655440000",
                        "kind": "Move",
                        "payload": {"x": 10, "y": 20},
                        "global_seq": 1001,
                        "sim_time": 123.456,
                        "source_id": "agent_001",
                        "schema_version": "1.0.0",
                    },
                    "source_metadata": {"adapter": "unity", "world_id": "world_001"},
                    "checksum": "sha256:def789abc123",
                    "req_id": "req_550e8400-e29b-41d4",
                }
            ]
        },
    )

    # global_seq, sim_time, wall_time validation is handled by Field(ge=0) constraints

    @field_validator("effect")
    @classmethod
    def validate_effect_structure(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate that effect has required fields."""
        required_fields = ["kind", "payload", "source_id", "schema_version"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Effect must contain '{field}' field")
        return v

    def __repr__(self) -> str:
        return (
            f"EventLogEntry(global_seq={self.global_seq}, "
            f"sim_time={self.sim_time}, effect_kind={self.effect.get('kind', 'unknown')})"
        )
