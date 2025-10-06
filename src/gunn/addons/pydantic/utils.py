"""Conversion utilities between TypedDict and Pydantic models.

These utilities provide zero-copy conversions between Gunn's TypedDict-based
types and the optional Pydantic models for validation at API boundaries.
"""

from typing import Any

from gunn.addons.pydantic.models import (
    EffectDraftModel,
    EffectModel,
    IntentModel,
    ObservationDeltaModel,
)
from gunn.schemas.types import Effect, EffectDraft, Intent, ObservationDelta


def intent_to_dict(intent: IntentModel) -> Intent:
    """Convert IntentModel to TypedDict Intent.

    Args:
        intent: Pydantic intent model

    Returns:
        TypedDict Intent for internal use

    Example:
        >>> model = IntentModel(kind="Move", ...)
        >>> intent_dict = intent_to_dict(model)
    """
    return intent.to_dict()  # type: ignore[return-value]


def intent_from_dict(data: Intent | dict[str, Any]) -> IntentModel:
    """Convert TypedDict Intent to IntentModel.

    Args:
        data: TypedDict Intent or dict

    Returns:
        Validated Pydantic intent model

    Raises:
        ValidationError: If data doesn't match Intent schema

    Example:
        >>> intent_dict = {"kind": "Move", ...}
        >>> model = intent_from_dict(intent_dict)
    """
    return IntentModel.model_validate(data)


def effect_draft_to_dict(effect_draft: EffectDraftModel) -> EffectDraft:
    """Convert EffectDraftModel to TypedDict EffectDraft.

    Args:
        effect_draft: Pydantic effect draft model

    Returns:
        TypedDict EffectDraft for internal use
    """
    return effect_draft.to_dict()  # type: ignore[return-value]


def effect_draft_from_dict(data: EffectDraft | dict[str, Any]) -> EffectDraftModel:
    """Convert TypedDict EffectDraft to EffectDraftModel.

    Args:
        data: TypedDict EffectDraft or dict

    Returns:
        Validated Pydantic effect draft model

    Raises:
        ValidationError: If data doesn't match EffectDraft schema
    """
    return EffectDraftModel.model_validate(data)


def effect_to_dict(effect: EffectModel) -> Effect:
    """Convert EffectModel to TypedDict Effect.

    Args:
        effect: Pydantic effect model

    Returns:
        TypedDict Effect for internal use
    """
    return effect.to_dict()  # type: ignore[return-value]


def effect_from_dict(data: Effect | dict[str, Any]) -> EffectModel:
    """Convert TypedDict Effect to EffectModel.

    Args:
        data: TypedDict Effect or dict

    Returns:
        Validated Pydantic effect model

    Raises:
        ValidationError: If data doesn't match Effect schema
    """
    return EffectModel.model_validate(data)


def observation_delta_to_dict(delta: ObservationDeltaModel) -> ObservationDelta:
    """Convert ObservationDeltaModel to TypedDict ObservationDelta.

    Args:
        delta: Pydantic observation delta model

    Returns:
        TypedDict ObservationDelta for internal use
    """
    return delta.to_dict()  # type: ignore[return-value]


def observation_delta_from_dict(
    data: ObservationDelta | dict[str, Any],
) -> ObservationDeltaModel:
    """Convert TypedDict ObservationDelta to ObservationDeltaModel.

    Args:
        data: TypedDict ObservationDelta or dict

    Returns:
        Validated Pydantic observation delta model

    Raises:
        ValidationError: If data doesn't match ObservationDelta schema
    """
    return ObservationDeltaModel.model_validate(data)
