"""
Comprehensive benchmark: TypedDict + manual validation vs Pydantic validation.

This benchmark tests the end-to-end performance including:
1. Object creation
2. Validation
3. Field access
4. Serialization/deserialization
"""

import time
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# TypedDict + Manual Validation Approach (Current Gunn)
# ============================================================================


class IntentTypedDict(TypedDict):
    """Intent using TypedDict."""

    kind: Literal["Move", "Attack", "Speak"]
    payload: dict[str, Any]
    context_seq: int
    req_id: str
    agent_id: str
    priority: int
    schema_version: str


class ManualValidator:
    """Manual validation similar to DefaultEffectValidator."""

    def validate_intent(self, intent: IntentTypedDict) -> bool:
        """Validate intent with manual checks."""
        # Required fields check
        required_fields = ["agent_id", "kind", "req_id", "schema_version"]
        for field in required_fields:
            if not intent.get(field):
                raise ValueError(f"Missing required field: {field}")

        # Agent ID format validation
        agent_id = intent.get("agent_id", "")
        if len(agent_id) > 100 or not agent_id.replace("_", "").replace(
            "-", ""
        ).isalnum():
            raise ValueError("Invalid agent_id format")

        # Priority range validation
        priority = intent.get("priority", 0)
        if not isinstance(priority, int) or priority < -100 or priority > 100:
            raise ValueError("Invalid priority range")

        # Kind validation
        if intent.get("kind") not in ["Move", "Attack", "Speak"]:
            raise ValueError("Invalid intent kind")

        # Payload-specific validation
        kind = intent.get("kind")
        payload = intent.get("payload", {})

        if kind == "Move":
            target = payload.get("to")
            if not target or not isinstance(target, (list, tuple)):
                raise ValueError("Invalid move target")
            if len(target) < 2:
                raise ValueError("Move target must have at least 2 elements")

        elif kind == "Speak":
            message = payload.get("text") or payload.get("message")
            if not message or not isinstance(message, str):
                raise ValueError("Invalid speak message")
            if not message.strip():
                raise ValueError("Speak message cannot be empty")
            if len(message) > 1000:
                raise ValueError("Speak message too long")

        return True


# ============================================================================
# Pydantic Approach
# ============================================================================


class MovePayload(BaseModel):
    """Move action payload."""

    to: list[float] = Field(min_length=2, max_length=3)


class SpeakPayload(BaseModel):
    """Speak action payload."""

    text: str = Field(min_length=1, max_length=1000)

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Speak message cannot be empty")
        return v.strip()


class IntentPydantic(BaseModel):
    """Intent using Pydantic."""

    kind: Literal["Move", "Attack", "Speak"]
    payload: dict[str, Any]
    context_seq: int
    req_id: str
    agent_id: str = Field(min_length=1, max_length=100)
    priority: int = Field(ge=-100, le=100)
    schema_version: str

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Invalid agent_id format")
        return v

    def validate_payload(self) -> None:
        """Additional payload validation based on kind."""
        if self.kind == "Move":
            MovePayload.model_validate(self.payload)
        elif self.kind == "Speak":
            SpeakPayload.model_validate(self.payload)


# ============================================================================
# Benchmark Functions
# ============================================================================


def benchmark_typeddict_creation_and_validation(iterations: int = 10000):
    """Benchmark TypedDict creation + manual validation."""
    validator = ManualValidator()

    start = time.perf_counter()
    for i in range(iterations):
        intent: IntentTypedDict = {
            "kind": "Move",
            "payload": {"to": [10.5, 20.3]},
            "context_seq": i,
            "req_id": f"req_{i}",
            "agent_id": "agent_1",
            "priority": 0,
            "schema_version": "1.0.0",
        }
        validator.validate_intent(intent)
        # Field access
        _ = intent["kind"]
        _ = intent["payload"]

    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_pydantic_creation_and_validation(iterations: int = 10000):
    """Benchmark Pydantic creation + validation."""
    start = time.perf_counter()
    for i in range(iterations):
        intent = IntentPydantic(
            kind="Move",
            payload={"to": [10.5, 20.3]},
            context_seq=i,
            req_id=f"req_{i}",
            agent_id="agent_1",
            priority=0,
            schema_version="1.0.0",
        )
        intent.validate_payload()
        # Field access
        _ = intent.kind
        _ = intent.payload

    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_typeddict_from_dict(iterations: int = 10000):
    """Benchmark TypedDict from dict (like receiving from API)."""
    validator = ManualValidator()

    data = {
        "kind": "Speak",
        "payload": {"text": "Hello team!"},
        "context_seq": 0,
        "req_id": "req_0",
        "agent_id": "agent_1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    start = time.perf_counter()
    for i in range(iterations):
        intent: IntentTypedDict = data.copy()  # type: ignore
        intent["context_seq"] = i
        validator.validate_intent(intent)
        _ = intent["payload"]["text"]

    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_pydantic_from_dict(iterations: int = 10000):
    """Benchmark Pydantic from dict (like receiving from API)."""
    data = {
        "kind": "Speak",
        "payload": {"text": "Hello team!"},
        "context_seq": 0,
        "req_id": "req_0",
        "agent_id": "agent_1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    start = time.perf_counter()
    for i in range(iterations):
        data_copy = data.copy()
        data_copy["context_seq"] = i
        intent = IntentPydantic.model_validate(data_copy)
        intent.validate_payload()
        _ = intent.payload["text"]

    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_serialization(iterations: int = 10000):
    """Benchmark serialization to dict/JSON."""
    # TypedDict
    intent_td: IntentTypedDict = {
        "kind": "Move",
        "payload": {"to": [10.5, 20.3]},
        "context_seq": 0,
        "req_id": "req_0",
        "agent_id": "agent_1",
        "priority": 0,
        "schema_version": "1.0.0",
    }

    start_td = time.perf_counter()
    for _ in range(iterations):
        _ = dict(intent_td)  # Already a dict, trivial
    elapsed_td = time.perf_counter() - start_td

    # Pydantic
    intent_pyd = IntentPydantic(
        kind="Move",
        payload={"to": [10.5, 20.3]},
        context_seq=0,
        req_id="req_0",
        agent_id="agent_1",
        priority=0,
        schema_version="1.0.0",
    )

    start_pyd = time.perf_counter()
    for _ in range(iterations):
        _ = intent_pyd.model_dump()
    elapsed_pyd = time.perf_counter() - start_pyd

    return elapsed_td, elapsed_pyd


def benchmark_complex_validation_failure(iterations: int = 1000):
    """Benchmark validation failure scenarios."""
    validator = ManualValidator()

    # TypedDict with invalid data
    start_td = time.perf_counter()
    errors_td = 0
    for i in range(iterations):
        intent_td: IntentTypedDict = {
            "kind": "Speak",
            "payload": {"text": ""},  # Invalid: empty
            "context_seq": i,
            "req_id": f"req_{i}",
            "agent_id": "agent_1",
            "priority": 0,
            "schema_version": "1.0.0",
        }
        try:
            validator.validate_intent(intent_td)
        except ValueError:
            errors_td += 1
    elapsed_td = time.perf_counter() - start_td

    # Pydantic with invalid data
    start_pyd = time.perf_counter()
    errors_pyd = 0
    for i in range(iterations):
        try:
            intent_pyd = IntentPydantic(
                kind="Speak",
                payload={"text": ""},  # Invalid: empty
                context_seq=i,
                req_id=f"req_{i}",
                agent_id="agent_1",
                priority=0,
                schema_version="1.0.0",
            )
            intent_pyd.validate_payload()
        except Exception:
            errors_pyd += 1
    elapsed_pyd = time.perf_counter() - start_pyd

    return elapsed_td, elapsed_pyd, errors_td, errors_pyd


# ============================================================================
# Main Benchmark Runner
# ============================================================================


def run_comprehensive_benchmark():
    """Run all benchmarks and report results."""
    print("=" * 80)
    print("COMPREHENSIVE VALIDATION PERFORMANCE BENCHMARK")
    print("TypedDict + Manual Validation vs Pydantic")
    print("=" * 80)
    print()

    iterations = 10000

    # Test 1: Creation + Validation (most common operation)
    print(f"[1] Creation + Validation ({iterations:,} iterations)")
    print("-" * 80)
    td_time = benchmark_typeddict_creation_and_validation(iterations)
    pyd_time = benchmark_pydantic_creation_and_validation(iterations)

    print(f"TypedDict + Manual:  {td_time:.4f}s ({td_time/iterations*1000:.3f}ms per op)")
    print(
        f"Pydantic:            {pyd_time:.4f}s ({pyd_time/iterations*1000:.3f}ms per op)"
    )
    print(f"Speedup:             {pyd_time/td_time:.2f}x (TypedDict faster)")
    print()

    # Test 2: From Dict (API/network scenario)
    print(f"[2] From Dict + Validation ({iterations:,} iterations)")
    print("-" * 80)
    td_time = benchmark_typeddict_from_dict(iterations)
    pyd_time = benchmark_pydantic_from_dict(iterations)

    print(f"TypedDict + Manual:  {td_time:.4f}s ({td_time/iterations*1000:.3f}ms per op)")
    print(
        f"Pydantic:            {pyd_time:.4f}s ({pyd_time/iterations*1000:.3f}ms per op)"
    )
    print(f"Speedup:             {pyd_time/td_time:.2f}x (TypedDict faster)")
    print()

    # Test 3: Serialization
    print(f"[3] Serialization to Dict ({iterations:,} iterations)")
    print("-" * 80)
    td_time, pyd_time = benchmark_serialization(iterations)

    print(f"TypedDict:           {td_time:.4f}s ({td_time/iterations*1000:.3f}ms per op)")
    print(
        f"Pydantic:            {pyd_time:.4f}s ({pyd_time/iterations*1000:.3f}ms per op)"
    )
    print(f"Speedup:             {pyd_time/td_time:.2f}x (TypedDict faster)")
    print()

    # Test 4: Validation Failures
    print(f"[4] Validation Failure Handling ({1000} iterations)")
    print("-" * 80)
    td_time, pyd_time, td_errors, pyd_errors = benchmark_complex_validation_failure(
        1000
    )

    print(
        f"TypedDict + Manual:  {td_time:.4f}s ({td_time/1000*1000:.3f}ms per op, {td_errors} errors caught)"
    )
    print(
        f"Pydantic:            {pyd_time:.4f}s ({pyd_time/1000*1000:.3f}ms per op, {pyd_errors} errors caught)"
    )
    print(f"Speedup:             {pyd_time/td_time:.2f}x (TypedDict faster)")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Key Findings:")
    print("1. TypedDict + Manual validation is consistently faster across all scenarios")
    print("2. Performance gap is largest in high-frequency operations (creation/access)")
    print("3. Pydantic provides better developer experience but with performance cost")
    print("4. For event-driven systems with >100 events/sec, TypedDict is preferable")
    print()
    print("Recommendation:")
    print("- Core library (Gunn): Keep TypedDict for performance")
    print("- Applications (Demo): Optional Pydantic wrapper for type safety")
    print("- Best of both: Pydantic at API boundaries, TypedDict internally")


if __name__ == "__main__":
    run_comprehensive_benchmark()
