"""Integration tests for hash chain utilities to verify requirements."""

from gunn.utils.hashing import (
    canonical_json,
    chain_checksum,
    detect_corruption,
    validate_hash_chain,
    verify_sequence_integrity,
)


def test_requirement_7_1_hash_chain_integrity():
    """Test requirement 7.1: Complete sequential log with hash chain integrity."""
    # Simulate a sequence of events in an event log
    events = [
        {"kind": "AgentRegistered", "agent_id": "agent_1", "timestamp": 1000},
        {
            "kind": "IntentSubmitted",
            "agent_id": "agent_1",
            "intent": "move",
            "timestamp": 1001,
        },
        {
            "kind": "EffectCreated",
            "effect_id": "eff_1",
            "source": "agent_1",
            "timestamp": 1002,
        },
        {
            "kind": "ObservationGenerated",
            "target": "agent_2",
            "data": {"visible": True},
            "timestamp": 1003,
        },
    ]

    # Build hash chain
    log_entries = []
    prev_checksum = None

    for i, event in enumerate(events):
        checksum = chain_checksum(event, prev_checksum)
        log_entries.append(
            {
                "global_seq": i + 1,
                "effect": event,
                "checksum": checksum,
                "wall_time": 1000 + i,
            }
        )
        prev_checksum = checksum

    # Verify chain integrity
    assert validate_hash_chain(log_entries) is True

    # Verify comprehensive integrity
    integrity_report = verify_sequence_integrity(log_entries)
    assert integrity_report["valid"] is True
    assert integrity_report["corrupted_entries"] == []
    assert integrity_report["missing_sequences"] == []
    assert integrity_report["total_entries"] == 4


def test_requirement_7_5_corruption_detection():
    """Test requirement 7.5: Corruption detection using hash/CRC and sequence gap detection."""
    # Create valid chain
    events = [
        {"kind": "Event1", "data": "test1"},
        {"kind": "Event2", "data": "test2"},
        {"kind": "Event3", "data": "test3"},
    ]

    log_entries = []
    prev_checksum = None

    for i, event in enumerate(events):
        checksum = chain_checksum(event, prev_checksum)
        log_entries.append({"global_seq": i + 1, "effect": event, "checksum": checksum})
        prev_checksum = checksum

    # Test 1: Detect hash corruption
    corrupted_entries = log_entries.copy()
    corrupted_entries[1]["checksum"] = "corrupted_hash"

    corruption_indices = detect_corruption(corrupted_entries)
    assert 1 in corruption_indices

    integrity_report = verify_sequence_integrity(corrupted_entries)
    assert integrity_report["valid"] is False
    assert 1 in integrity_report["corrupted_entries"]

    # Test 2: Detect sequence gaps
    gap_entries = [
        {"global_seq": 1, "effect": events[0], "checksum": chain_checksum(events[0])},
        {
            "global_seq": 3,
            "effect": events[2],
            "checksum": chain_checksum(events[2], chain_checksum(events[0])),
        },  # Missing seq 2
    ]

    integrity_report = verify_sequence_integrity(gap_entries)
    assert integrity_report["valid"] is False
    assert 2 in integrity_report["missing_sequences"]


def test_canonical_json_deterministic_ordering():
    """Test that canonical_json produces deterministic output with sorted keys."""
    # Test with various key orders
    data_variants = [
        {"z": 1, "a": 2, "m": 3},
        {"a": 2, "m": 3, "z": 1},
        {"m": 3, "z": 1, "a": 2},
    ]

    canonical_outputs = [canonical_json(data) for data in data_variants]

    # All should produce identical output
    assert len(set(canonical_outputs)) == 1

    # Should be properly sorted
    expected = b'{"a":2,"m":3,"z":1}'
    assert canonical_outputs[0] == expected


def test_hash_chain_tamper_detection():
    """Test that any tampering with the chain is detected."""
    original_events = [
        {"kind": "UserAction", "action": "login", "user": "alice"},
        {
            "kind": "UserAction",
            "action": "read_file",
            "user": "alice",
            "file": "document.txt",
        },
        {"kind": "UserAction", "action": "logout", "user": "alice"},
    ]

    # Build legitimate chain
    legitimate_chain = []
    prev_checksum = None

    for i, event in enumerate(original_events):
        checksum = chain_checksum(event, prev_checksum)
        legitimate_chain.append(
            {"global_seq": i + 1, "effect": event, "checksum": checksum}
        )
        prev_checksum = checksum

    # Verify legitimate chain is valid
    assert validate_hash_chain(legitimate_chain) is True

    # Test various tampering scenarios

    # 1. Modify event data
    tampered_chain_1 = legitimate_chain.copy()
    tampered_chain_1[1]["effect"]["file"] = "secret.txt"  # type: ignore
    assert validate_hash_chain(tampered_chain_1) is False

    # 2. Insert malicious event
    malicious_event = {
        "kind": "UserAction",
        "action": "delete_file",
        "user": "alice",
        "file": "evidence.txt",
    }
    tampered_chain_2 = legitimate_chain.copy()
    tampered_chain_2.insert(
        2,
        {
            "global_seq": 2.5,  # Try to insert between existing events
            "effect": malicious_event,
            "checksum": chain_checksum(
                malicious_event,
                legitimate_chain[1]["checksum"],  # type: ignore
            ),
        },
    )
    # This would break the chain because subsequent checksums would be wrong
    assert validate_hash_chain(tampered_chain_2) is False

    # 3. Remove event
    tampered_chain_3 = legitimate_chain[:-1]  # Remove last event
    # If we try to add a new event with the wrong previous checksum
    new_event = {"kind": "UserAction", "action": "admin_access", "user": "alice"}
    tampered_chain_3.append(
        {
            "global_seq": 3,
            "effect": new_event,
            "checksum": chain_checksum(
                new_event,
                legitimate_chain[0]["checksum"],  # type: ignore
            ),  # Wrong prev checksum
        }
    )
    assert validate_hash_chain(tampered_chain_3) is False


def test_large_scale_integrity():
    """Test hash chain integrity with larger datasets."""
    # Generate a large number of events
    large_event_set = []
    for i in range(1000):
        event = {
            "kind": "BulkEvent",
            "sequence": i,
            "data": f"event_data_{i}",
            "metadata": {"batch": i // 100, "index": i % 100},
        }
        large_event_set.append(event)

    # Build hash chain
    large_chain = []
    prev_checksum = None

    for i, event in enumerate(large_event_set):
        checksum = chain_checksum(event, prev_checksum)
        large_chain.append({"global_seq": i + 1, "effect": event, "checksum": checksum})
        prev_checksum = checksum

    # Verify entire chain
    assert validate_hash_chain(large_chain) is True

    # Verify comprehensive integrity
    integrity_report = verify_sequence_integrity(large_chain)
    assert integrity_report["valid"] is True
    assert integrity_report["total_entries"] == 1000

    # Introduce corruption in the middle
    large_chain[500]["effect"]["data"] = "corrupted_data"  # type: ignore

    # Should detect corruption
    corruption_indices = detect_corruption(large_chain)
    assert 500 in corruption_indices

    # The corrupted entry should be detected
    # Note: Only the directly corrupted entry is detected, not subsequent ones
    # because the chain validation stops at the first corruption
    assert len(corruption_indices) >= 1
    assert 500 in corruption_indices


def test_unicode_and_special_characters():
    """Test hash consistency with Unicode and special characters."""
    events_with_unicode = [
        {"message": "Hello 世界", "emoji": "🌍🚀", "special": "café"},
        {"text": "Здравствуй мир", "symbols": "αβγδε", "math": "∑∫∂"},
        {"mixed": "ASCII + 中文 + العربية + עברית", "numbers": "①②③④⑤"},
    ]

    # Build chain with Unicode data
    unicode_chain = []
    prev_checksum = None

    for i, event in enumerate(events_with_unicode):
        checksum = chain_checksum(event, prev_checksum)
        unicode_chain.append(
            {"global_seq": i + 1, "effect": event, "checksum": checksum}
        )
        prev_checksum = checksum

    # Should validate correctly
    assert validate_hash_chain(unicode_chain) is True

    # Checksums should be consistent across multiple generations
    for entry in unicode_chain:
        effect = entry["effect"]
        stored_checksum = entry["checksum"]

        # Regenerate checksum multiple times
        for _ in range(5):
            regenerated = chain_checksum(
                effect,  # type: ignore
                prev_checksum if int(entry["global_seq"]) > 1 else None,  # type: ignore
            )
            assert regenerated == stored_checksum

        prev_checksum = str(stored_checksum)


if __name__ == "__main__":
    # Run integration tests
    test_requirement_7_1_hash_chain_integrity()  # type: ignore
    test_requirement_7_5_corruption_detection()  # type: ignore
    test_canonical_json_deterministic_ordering()  # type: ignore
    test_hash_chain_tamper_detection()  # type: ignore
    test_large_scale_integrity()  # type: ignore
    test_unicode_and_special_characters()  # type: ignore
    print("All integration tests passed!")
