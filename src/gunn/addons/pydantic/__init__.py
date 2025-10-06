"""Pydantic addon for type-safe intent/effect validation.

This addon provides optional Pydantic wrappers around Gunn's TypedDict-based
Intent and Effect types. Use this at API boundaries for better developer
experience while maintaining TypedDict performance internally.

Example:
    >>> from gunn.addons.pydantic import IntentModel
    >>> intent = IntentModel(
    ...     kind="Move",
    ...     payload={"to": [10.0, 20.0]},
    ...     context_seq=0,
    ...     req_id="move_1",
    ...     agent_id="agent_a",
    ...     priority=1,
    ...     schema_version="1.0.0"
    ... )
    >>> # Convert to TypedDict for internal use
    >>> intent_dict = intent.to_dict()
"""

from gunn.addons.pydantic.models import (
    EffectDraftModel,
    EffectModel,
    IntentModel,
    ObservationDeltaModel,
)
from gunn.addons.pydantic.utils import (
    effect_draft_from_dict,
    effect_draft_to_dict,
    effect_from_dict,
    effect_to_dict,
    intent_from_dict,
    intent_to_dict,
    observation_delta_from_dict,
    observation_delta_to_dict,
)

__all__ = [
    "EffectDraftModel",
    "EffectModel",
    # Models
    "IntentModel",
    "ObservationDeltaModel",
    "effect_draft_from_dict",
    "effect_draft_to_dict",
    "effect_from_dict",
    "effect_to_dict",
    "intent_from_dict",
    # Conversion utilities
    "intent_to_dict",
    "observation_delta_from_dict",
    "observation_delta_to_dict",
]
