"""Unit tests for hash chain utilities."""

from gunn.utils.hashing import (
    canonical_json,
    chain_checksum,
    detect_corruption,
    validate_hash_chain,
    verify_sequence_integrity,
)


class TestCanonicalJson:
    """Test canonical JSON serialization."""

    def test_sorted_keys(self):
        """Test that keys are sorted in output."""
        data = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(data)
        expected = b'{"a":2,"m":3,"z":1}'
        assert result == expected

    def test_nested_objects(self):
        """Test canonical serialization of nested objects."""
        data = {"outer": {"z": 1, "a": 2}, "array": [{"b": 2, "a": 1}]}
        result = canonical_json(data)
        # Should have sorted keys at all levels
        assert b'"a":1' in result
        assert b'"a":2' in result
        assert b'"b":2' in result
        assert b'"outer"' in result
        assert b'"array"' in result

    def test_deterministic_output(self):
        """Test that same input always produces same output."""
        data = {"random": 42, "order": "test", "keys": [1, 2, 3]}
        result1 = canonical_json(data)
        result2 = canonical_json(data)
        assert result1 == result2

    def test_empty_dict(self):
        """Test canonical JSON of empty dictionary."""
        result = canonical_json({})
        assert result == b"{}"

    def test_complex_types(self):
        """Test canonical JSON with various data types."""
        data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "nested": {"inner": "value"},
        }
        result = canonical_json(data)
        # Should be valid JSON bytes
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestChainChecksum:
    """Test hash chain checksum generation."""

    def test_first_entry_no_prev(self):
        """Test checksum generation for first entry (no previous)."""
        effect = {"kind": "test", "payload": {"data": "value"}}
        checksum = chain_checksum(effect)

        # Should be valid hex string
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_chained_entry(self):
        """Test checksum generation with previous checksum."""
        effect = {"kind": "test", "payload": {"data": "value"}}
        prev_checksum = "a" * 64  # Valid hex string

        checksum = chain_checksum(effect, prev_checksum)

        # Should be different from first entry
        checksum_no_prev = chain_checksum(effect)
        assert checksum != checksum_no_prev
        assert len(checksum) == 64

    def test_deterministic_hashing(self):
        """Test that same inputs produce same checksums."""
        effect = {"kind": "test", "payload": {"data": "value"}}
        prev_checksum = "b" * 64

        checksum1 = chain_checksum(effect, prev_checksum)
        checksum2 = chain_checksum(effect, prev_checksum)

        assert checksum1 == checksum2

    def test_different_effects_different_checksums(self):
        """Test that different effects produce different checksums."""
        effect1 = {"kind": "test1", "payload": {}}
        effect2 = {"kind": "test2", "payload": {}}

        checksum1 = chain_checksum(effect1)
        checksum2 = chain_checksum(effect2)

        assert checksum1 != checksum2

    def test_key_order_independence(self):
        """Test that key order doesn't affect checksum."""
        effect1 = {"kind": "test", "payload": {"a": 1, "b": 2}}
        effect2 = {"payload": {"b": 2, "a": 1}, "kind": "test"}

        checksum1 = chain_checksum(effect1)
        checksum2 = chain_checksum(effect2)

        assert checksum1 == checksum2


class TestValidateHashChain:
    """Test hash chain validation."""

    def test_empty_chain(self):
        """Test validation of empty chain."""
        assert validate_hash_chain([]) is True

    def test_single_entry_valid(self):
        """Test validation of single valid entry."""
        effect = {"kind": "test", "payload": {}}
        checksum = chain_checksum(effect)
        entries = [{"effect": effect, "checksum": checksum}]

        assert validate_hash_chain(entries) is True

    def test_multiple_entries_valid(self):
        """Test validation of valid multi-entry chain."""
        effect1 = {"kind": "test1", "payload": {}}
        effect2 = {"kind": "test2", "payload": {}}

        checksum1 = chain_checksum(effect1)
        checksum2 = chain_checksum(effect2, checksum1)

        entries = [
            {"effect": effect1, "checksum": checksum1},
            {"effect": effect2, "checksum": checksum2},
        ]

        assert validate_hash_chain(entries) is True

    def test_corrupted_checksum(self):
        """Test detection of corrupted checksum."""
        effect = {"kind": "test", "payload": {}}
        entries = [{"effect": effect, "checksum": "invalid_checksum"}]

        assert validate_hash_chain(entries) is False

    def test_missing_effect(self):
        """Test handling of missing effect field."""
        entries = [{"checksum": "some_checksum"}]
        assert validate_hash_chain(entries) is False

    def test_missing_checksum(self):
        """Test handling of missing checksum field."""
        entries = [{"effect": {"kind": "test"}}]
        assert validate_hash_chain(entries) is False

    def test_chain_break(self):
        """Test detection of broken chain."""
        effect1 = {"kind": "test1", "payload": {}}
        effect2 = {"kind": "test2", "payload": {}}

        checksum1 = chain_checksum(effect1)
        # Intentionally use wrong previous checksum
        checksum2 = chain_checksum(effect2, "wrong_prev_checksum")

        entries = [
            {"effect": effect1, "checksum": checksum1},
            {"effect": effect2, "checksum": checksum2},
        ]

        assert validate_hash_chain(entries) is False


class TestDetectCorruption:
    """Test corruption detection."""

    def test_no_corruption(self):
        """Test detection with no corruption."""
        effect1 = {"kind": "test1", "payload": {}}
        effect2 = {"kind": "test2", "payload": {}}

        checksum1 = chain_checksum(effect1)
        checksum2 = chain_checksum(effect2, checksum1)

        entries = [
            {"effect": effect1, "checksum": checksum1},
            {"effect": effect2, "checksum": checksum2},
        ]

        corrupted = detect_corruption(entries)
        assert corrupted == []

    def test_single_corruption(self):
        """Test detection of single corrupted entry."""
        effect1 = {"kind": "test1", "payload": {}}
        effect2 = {"kind": "test2", "payload": {}}

        checksum1 = chain_checksum(effect1)

        entries = [
            {"effect": effect1, "checksum": checksum1},
            {"effect": effect2, "checksum": "corrupted_checksum"},
        ]

        corrupted = detect_corruption(entries)
        assert corrupted == [1]

    def test_multiple_corruption(self):
        """Test detection of multiple corrupted entries."""
        entries = [
            {"effect": {"kind": "test1"}, "checksum": "bad1"},
            {"effect": {"kind": "test2"}, "checksum": "bad2"},
            {"effect": {"kind": "test3"}, "checksum": "bad3"},
        ]

        corrupted = detect_corruption(entries)
        assert corrupted == [0, 1, 2]

    def test_missing_fields(self):
        """Test handling of entries with missing fields."""
        entries = [
            {"checksum": "missing_effect"},
            {"effect": {"kind": "test"}},  # missing checksum
        ]

        corrupted = detect_corruption(entries)  # type: ignore
        assert corrupted == [0, 1]

    def test_empty_list(self):
        """Test corruption detection on empty list."""
        corrupted = detect_corruption([])
        assert corrupted == []


class TestVerifySequenceIntegrity:
    """Test comprehensive sequence integrity verification."""

    def test_valid_sequence(self):
        """Test verification of valid sequence."""
        effect1 = {"kind": "test1", "payload": {}}
        effect2 = {"kind": "test2", "payload": {}}

        checksum1 = chain_checksum(effect1)
        checksum2 = chain_checksum(effect2, checksum1)

        entries = [
            {"effect": effect1, "checksum": checksum1, "global_seq": 1},
            {"effect": effect2, "checksum": checksum2, "global_seq": 2},
        ]

        result = verify_sequence_integrity(entries)

        assert result["valid"] is True
        assert result["corrupted_entries"] == []
        assert result["missing_sequences"] == []
        assert result["total_entries"] == 2

    def test_corrupted_sequence(self):
        """Test verification with corrupted entries."""
        entries = [
            {"effect": {"kind": "test1"}, "checksum": "bad_checksum", "global_seq": 1},
            {"effect": {"kind": "test2"}, "checksum": "another_bad", "global_seq": 2},
        ]

        result = verify_sequence_integrity(entries)

        assert result["valid"] is False
        assert result["corrupted_entries"] == [0, 1]
        assert result["total_entries"] == 2

    def test_sequence_gaps(self):
        """Test detection of sequence gaps."""
        effect1 = {"kind": "test1", "payload": {}}
        effect2 = {"kind": "test2", "payload": {}}

        checksum1 = chain_checksum(effect1)
        checksum2 = chain_checksum(effect2, checksum1)

        entries = [
            {"effect": effect1, "checksum": checksum1, "global_seq": 1},
            {
                "effect": effect2,
                "checksum": checksum2,
                "global_seq": 3,
            },  # Gap: missing seq 2
        ]

        result = verify_sequence_integrity(entries)

        assert result["valid"] is False
        assert result["missing_sequences"] == [2]
        assert result["corrupted_entries"] == []

    def test_no_sequence_numbers(self):
        """Test verification without sequence numbers."""
        effect1 = {"kind": "test1", "payload": {}}
        checksum1 = chain_checksum(effect1)

        entries = [{"effect": effect1, "checksum": checksum1}]

        result = verify_sequence_integrity(entries)

        assert result["valid"] is True
        assert result["missing_sequences"] == []
        assert result["total_entries"] == 1

    def test_empty_sequence(self):
        """Test verification of empty sequence."""
        result = verify_sequence_integrity([])

        assert result["valid"] is True
        assert result["corrupted_entries"] == []
        assert result["missing_sequences"] == []
        assert result["total_entries"] == 0


class TestHashConsistency:
    """Test hash consistency across different scenarios."""

    def test_unicode_handling(self):
        """Test consistent hashing with Unicode characters."""
        effect = {"kind": "test", "message": "Hello 世界 🌍"}

        checksum1 = chain_checksum(effect)
        checksum2 = chain_checksum(effect)

        assert checksum1 == checksum2
        assert len(checksum1) == 64

    def test_large_payload_consistency(self):
        """Test consistency with large payloads."""
        large_data = {"data": "x" * 10000, "numbers": list(range(1000))}
        effect = {"kind": "large_test", "payload": large_data}

        checksum1 = chain_checksum(effect)
        checksum2 = chain_checksum(effect)

        assert checksum1 == checksum2

    def test_nested_structure_consistency(self):
        """Test consistency with deeply nested structures."""
        nested = {
            "level1": {"level2": {"level3": {"data": [{"item": i} for i in range(10)]}}}
        }
        effect = {"kind": "nested_test", "payload": nested}

        checksum1 = chain_checksum(effect)
        checksum2 = chain_checksum(effect)

        assert checksum1 == checksum2

    def test_chain_progression(self):
        """Test that chain progresses correctly through multiple entries."""
        effects = [
            {"kind": "step1", "data": 1},
            {"kind": "step2", "data": 2},
            {"kind": "step3", "data": 3},
        ]

        checksums = []
        prev_checksum = None

        for effect in effects:
            checksum = chain_checksum(effect, prev_checksum)
            checksums.append(checksum)
            prev_checksum = checksum

        # All checksums should be different
        assert len(set(checksums)) == len(checksums)

        # Chain should validate
        entries = [
            {"effect": effects[i], "checksum": checksums[i]}
            for i in range(len(effects))
        ]

        assert validate_hash_chain(entries) is True
