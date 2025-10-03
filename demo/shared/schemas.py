"""
OpenAI structured output schemas for AI decision making.

This module defines the Pydantic models used for OpenAI's structured outputs,
ensuring that AI agents make valid decisions in the battle simulation.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class MoveAction(BaseModel):
    """Action to move to a target position."""

    action_type: Literal["move"] = "move"
    target_position: Annotated[
        list[float],
        Field(description="Target coordinates as [x, y]", min_length=2, max_length=2),
    ]
    reason: str = Field(description="Why this move is strategic", min_length=1)


class AttackAction(BaseModel):
    """Action to attack another agent."""

    action_type: Literal["attack"] = "attack"
    target_agent_id: str = Field(description="ID of the agent to attack", min_length=1)
    reason: str = Field(description="Why this target was chosen", min_length=1)


class HealAction(BaseModel):
    """Action to heal self or teammate."""

    action_type: Literal["heal"] = "heal"
    target_agent_id: str | None = Field(
        default=None, description="Agent to heal (None for self)"
    )
    reason: str = Field(description="Why healing is needed now", min_length=1)


class RepairAction(BaseModel):
    """Action to repair weapon at forge."""

    action_type: Literal["repair"] = "repair"
    reason: str = Field(description="Why weapon repair is needed", min_length=1)


class CommunicateAction(BaseModel):
    """Action to communicate with team."""

    action_type: Literal["communicate"] = "communicate"
    message: str = Field(
        description="Message to send to team", max_length=200, min_length=1
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="Message urgency level"
    )

    @field_validator("message")
    @classmethod
    def validate_message_not_empty(cls, v: str) -> str:
        """Ensure message is not just whitespace."""
        if not v.strip():
            raise ValueError("Message cannot be empty or just whitespace")
        return v.strip()


class AgentDecision(BaseModel):
    """Complete decision made by an agent."""

    primary_action: MoveAction | AttackAction | HealAction | RepairAction
    communication: CommunicateAction | None = Field(
        default=None, description="Optional team communication"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this decision")
    strategic_assessment: str = Field(
        description="Current situation assessment", min_length=1
    )
