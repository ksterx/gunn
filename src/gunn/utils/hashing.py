"""Hash chain utilities for log integrity.

This module provides utilities for maintaining integrity of event logs through
hash chaining and canonical JSON serialization.
"""

import hashlib
from typing import Any

import orjson


def canonical_json(obj: dict[str, Any]) -> bytes:
    """Generate canonical JSON serialization for consistent hashing.

    Uses orjson with sorted keys to ensure deterministic output regardless
    of dictionary key ordering.

    Args:
        obj: Dictionary to serialize

    Returns:
        Canonical JSON bytes representation

    Example:
        >>> data = {"b": 2, "a": 1}
        >>> canonical_json(data)
        b'{"a":1,"b":2}'
    """
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)


def chain_checksum(effect: dict[str, Any], prev_checksum: str | None = None) -> str:
    """Generate hash chain checksum using SHA-256.

    Creates a hash chain by combining the previous checksum with the canonical
    JSON representation of the current effect.

    Args:
        effect: Effect dictionary to hash
        prev_checksum: Previous checksum in the chain (None for first entry)

    Returns:
        Hexadecimal SHA-256 hash string

    Example:
        >>> effect = {"kind": "test", "payload": {}}
        >>> chain_checksum(effect)
        'a1b2c3...'
        >>> chain_checksum(effect, "prev_hash")
        'd4e5f6...'
    """
    hasher = hashlib.sha256()

    # Include previous checksum if provided
    if prev_checksum:
        hasher.update(prev_checksum.encode("utf-8"))

    # Add canonical JSON of current effect
    hasher.update(canonical_json(effect))

    return hasher.hexdigest()


def validate_hash_chain(entries: list[dict[str, Any]]) -> bool:
    """Validate integrity of a hash chain sequence.

    Verifies that each entry's checksum correctly follows from the previous
    entry's checksum and the entry's effect data.

    Args:
        entries: List of log entries, each containing 'effect' and 'checksum' fields

    Returns:
        True if chain is valid, False if corruption detected

    Example:
        >>> entries = [
        ...     {"effect": {"kind": "test1"}, "checksum": "abc123"},
        ...     {"effect": {"kind": "test2"}, "checksum": "def456"}
        ... ]
        >>> validate_hash_chain(entries)
        True
    """
    if not entries:
        return True

    prev_checksum = None

    for entry in entries:
        effect = entry.get("effect")
        stored_checksum = entry.get("checksum")

        if not effect or not stored_checksum:
            return False

        # Calculate expected checksum
        expected_checksum = chain_checksum(effect, prev_checksum)

        # Compare with stored checksum
        if expected_checksum != stored_checksum:
            return False

        prev_checksum = stored_checksum

    return True


def detect_corruption(entries: list[dict[str, Any]]) -> list[int]:
    """Detect corrupted entries in a hash chain.

    Returns indices of entries that have invalid checksums, indicating
    potential corruption or tampering.

    Args:
        entries: List of log entries with 'effect' and 'checksum' fields

    Returns:
        List of indices where corruption was detected

    Example:
        >>> entries = [
        ...     {"effect": {"kind": "test1"}, "checksum": "valid_hash"},
        ...     {"effect": {"kind": "test2"}, "checksum": "invalid_hash"}
        ... ]
        >>> detect_corruption(entries)
        [1]
    """
    corrupted_indices: list[int] = []

    if not entries:
        return corrupted_indices

    prev_checksum = None

    for i, entry in enumerate(entries):
        effect = entry.get("effect")
        stored_checksum = entry.get("checksum")

        if not effect or not stored_checksum:
            corrupted_indices.append(i)
            continue

        # Calculate expected checksum
        expected_checksum = chain_checksum(effect, prev_checksum)

        # Check if checksum matches
        if expected_checksum != stored_checksum:
            corrupted_indices.append(i)

        prev_checksum = stored_checksum

    return corrupted_indices


def verify_sequence_integrity(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Comprehensive integrity verification of a log sequence.

    Performs multiple integrity checks including hash chain validation,
    sequence gap detection, and corruption analysis.

    Args:
        entries: List of log entries with required fields

    Returns:
        Dictionary containing integrity report with:
        - valid: Overall validity boolean
        - corrupted_entries: List of corrupted entry indices
        - missing_sequences: List of detected sequence gaps
        - total_entries: Total number of entries checked

    Example:
        >>> entries = [{"effect": {...}, "checksum": "...", "global_seq": 1}]
        >>> verify_sequence_integrity(entries)
        {"valid": True, "corrupted_entries": [], "missing_sequences": [], "total_entries": 1}
    """
    corrupted_entries: list[int] = []
    missing_sequences: list[int] = []

    result: dict[str, Any] = {
        "valid": True,
        "corrupted_entries": corrupted_entries,
        "missing_sequences": missing_sequences,
        "total_entries": len(entries),
    }

    if not entries:
        return result

    # Check hash chain integrity
    corrupted_entries.extend(detect_corruption(entries))

    # Check for sequence gaps (if global_seq is present)
    sequences: list[int] = []
    for entry in entries:
        if "global_seq" in entry:
            sequences.append(entry["global_seq"])

    if sequences:
        sequences.sort()
        expected_seq = sequences[0]

        for seq in sequences:
            if seq != expected_seq:
                missing_sequences.append(expected_seq)
                expected_seq = seq
            expected_seq += 1

    # Overall validity
    result["valid"] = len(corrupted_entries) == 0 and len(missing_sequences) == 0

    return result
