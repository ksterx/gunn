"""Persistent storage layer.

This package provides storage components for the multi-agent simulation,
including deduplication stores and snapshot management.
"""

from gunn.storage.dedup_store import DedupStore, InMemoryDedupStore

__all__ = ["DedupStore", "InMemoryDedupStore"]
