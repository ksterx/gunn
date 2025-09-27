#!/usr/bin/env python3
"""Demonstration of hash chain utilities for log integrity."""

from gunn.utils.hashing import (
    canonical_json,
    chain_checksum,
    detect_corruption,
    validate_hash_chain,
    verify_sequence_integrity,
)


def main() -> None:
    """Demonstrate hash chain utilities."""
    print("=== Hash Chain Utilities Demo ===\n")

    # 1. Demonstrate canonical JSON
    print("1. Canonical JSON Serialization:")
    data = {"z": 1, "a": 2, "m": 3}
    canonical = canonical_json(data)
    print(f"   Input: {data}")
    print(f"   Canonical JSON: {canonical.decode()}")
    print()

    # 2. Build a hash chain
    print("2. Building Hash Chain:")
    events = [
        {"kind": "AgentRegistered", "agent_id": "alice", "timestamp": 1000},
        {
            "kind": "IntentSubmitted",
            "agent_id": "alice",
            "intent": "move_north",
            "timestamp": 1001,
        },
        {
            "kind": "EffectCreated",
            "effect_id": "eff_1",
            "effect_type": "Move",
            "timestamp": 1002,
        },
        {
            "kind": "ObservationSent",
            "target": "bob",
            "data": {"alice_position": "north"},
            "timestamp": 1003,
        },
    ]

    log_entries = []
    prev_checksum = None

    for i, event in enumerate(events):
        checksum = chain_checksum(event, prev_checksum)
        log_entries.append({"global_seq": i + 1, "effect": event, "checksum": checksum})
        print(f"   Event {i + 1}: {event['kind']}")
        print(f"   Checksum: {checksum[:16]}...")
        prev_checksum = checksum
    print()

    # 3. Validate chain integrity
    print("3. Chain Validation:")
    is_valid = validate_hash_chain(log_entries)
    print(f"   Chain is valid: {is_valid}")

    integrity_report = verify_sequence_integrity(log_entries)
    print(f"   Integrity report: {integrity_report}")
    print()

    # 4. Demonstrate corruption detection
    print("4. Corruption Detection:")
    # Create a corrupted copy
    corrupted_entries = log_entries.copy()
    corrupted_entries[2]["effect"]["kind"] = "HackedMove"  # Tamper with data

    print("   Tampering with event 3...")
    corrupted_indices = detect_corruption(corrupted_entries)
    print(f"   Corrupted entries detected at indices: {corrupted_indices}")

    is_valid_after = validate_hash_chain(corrupted_entries)
    print(f"   Chain is valid after tampering: {is_valid_after}")
    print()

    # 5. Demonstrate sequence gap detection
    print("5. Sequence Gap Detection:")
    gap_entries = [
        log_entries[0],  # seq 1
        log_entries[2],  # seq 3 (missing seq 2)
        log_entries[3],  # seq 4
    ]

    integrity_report = verify_sequence_integrity(gap_entries)
    print(f"   Integrity report with gaps: {integrity_report}")
    print()

    # 6. Show deterministic behavior
    print("6. Deterministic Behavior:")
    event = {"kind": "Test", "data": {"b": 2, "a": 1}}

    # Generate checksum multiple times
    checksums = [chain_checksum(event) for _ in range(5)]
    all_same = len(set(checksums)) == 1
    print(f"   Same event produces identical checksums: {all_same}")
    print(f"   Checksum: {checksums[0][:16]}...")
    print()

    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
