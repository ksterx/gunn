"""Unit tests for backpressure policies and queue management."""

import asyncio
from collections import deque
from typing import Any
from unittest.mock import patch

import pytest

from gunn.utils.backpressure import (
    BackpressureManager,
    BackpressurePolicy,
    BackpressureQueue,
    DeferPolicy,
    DropNewestPolicy,
    ShedOldestPolicy,
    backpressure_manager,
)
from gunn.utils.errors import BackpressureError


class TestDeferPolicy:
    """Test DeferPolicy class."""

    def test_initialization(self) -> None:
        """Test defer policy initialization."""
        policy: DeferPolicy[Any] = DeferPolicy(threshold=10, agent_id="agent_1")

        assert policy.threshold == 10
        assert policy.agent_id == "agent_1"
        assert policy.policy_name == "defer"

    @pytest.mark.asyncio
    async def test_handle_overflow_raises_backpressure_error(self) -> None:
        """Test defer policy raises BackpressureError on overflow."""
        policy: DeferPolicy[Any] = DeferPolicy(threshold=5, agent_id="agent_1")
        queue = deque([1, 2, 3, 4, 5, 6])  # 6 items, threshold is 5

        with pytest.raises(BackpressureError) as exc_info:
            await policy.handle_overflow(queue, 7)

        error = exc_info.value
        assert error.agent_id == "agent_1"
        assert error.queue_type == "queue"
        assert error.current_depth == 6
        assert error.threshold == 5
        assert error.policy == "defer"

    @patch("gunn.utils.backpressure.record_queue_high_watermark")
    @patch("gunn.utils.backpressure.record_backpressure_event")
    @pytest.mark.asyncio
    async def test_handle_overflow_records_metrics(
        self, mock_record_event, mock_record_watermark
    ) -> None:
        """Test defer policy records metrics on overflow."""
        policy: DeferPolicy[Any] = DeferPolicy(threshold=5, agent_id="agent_1")
        queue = deque([1, 2, 3, 4, 5, 6])

        with pytest.raises(BackpressureError):
            await policy.handle_overflow(queue, 7)

        mock_record_watermark.assert_called_once_with("agent_1", "queue", 6)
        mock_record_event.assert_called_once_with("agent_1", "queue", "defer")


class TestShedOldestPolicy:
    """Test ShedOldestPolicy class."""

    def test_initialization(self) -> None:
        """Test shed oldest policy initialization."""
        policy: ShedOldestPolicy[Any] = ShedOldestPolicy(
            threshold=10, agent_id="agent_1"
        )

        assert policy.threshold == 10
        assert policy.agent_id == "agent_1"
        assert policy.policy_name == "shed_oldest"

    @pytest.mark.asyncio
    async def test_handle_overflow_sheds_oldest_items(self) -> None:
        """Test shed oldest policy removes oldest items to make room."""
        policy: ShedOldestPolicy[Any] = ShedOldestPolicy(
            threshold=5, agent_id="agent_1"
        )
        queue = deque([1, 2, 3, 4, 5, 6])  # 6 items, threshold is 5

        result = await policy.handle_overflow(queue, 7)

        assert result is True
        assert len(queue) == 5  # Should be at threshold
        assert list(queue) == [
            3,
            4,
            5,
            6,
            7,
        ]  # Oldest items (1, 2) removed, new item added

    @pytest.mark.asyncio
    async def test_handle_overflow_multiple_items_shed(self) -> None:
        """Test shedding multiple items when far over threshold."""
        policy: ShedOldestPolicy[Any] = ShedOldestPolicy(
            threshold=3, agent_id="agent_1"
        )
        queue = deque([1, 2, 3, 4, 5, 6, 7])  # 7 items, threshold is 3

        result = await policy.handle_overflow(queue, 8)

        assert result is True
        assert len(queue) == 3  # Should be at threshold
        assert list(queue) == [6, 7, 8]  # Items 1-5 removed, new item added

    @pytest.mark.asyncio
    async def test_handle_no_overflow_adds_normally(self) -> None:
        """Test normal addition when under threshold."""
        policy: ShedOldestPolicy[Any] = ShedOldestPolicy(
            threshold=5, agent_id="agent_1"
        )
        queue = deque([1, 2, 3])  # 3 items, under threshold

        result = await policy.handle_overflow(queue, 4)

        assert result is True
        assert len(queue) == 4
        assert list(queue) == [1, 2, 3, 4]

    @patch("gunn.utils.backpressure.record_queue_high_watermark")
    @patch("gunn.utils.backpressure.record_backpressure_event")
    @pytest.mark.asyncio
    async def test_handle_overflow_records_metrics(
        self, mock_record_event, mock_record_watermark
    ) -> None:
        """Test shed oldest policy records metrics on overflow."""
        policy: ShedOldestPolicy[Any] = ShedOldestPolicy(
            threshold=3, agent_id="agent_1"
        )
        queue = deque([1, 2, 3, 4])

        await policy.handle_overflow(queue, 5)

        mock_record_watermark.assert_called_once_with("agent_1", "queue", 4)
        mock_record_event.assert_called_once_with("agent_1", "queue", "shed_oldest")


class TestDropNewestPolicy:
    """Test DropNewestPolicy class."""

    def test_initialization(self) -> None:
        """Test drop newest policy initialization."""
        policy: DropNewestPolicy[Any] = DropNewestPolicy(
            threshold=10, agent_id="agent_1"
        )

        assert policy.threshold == 10
        assert policy.agent_id == "agent_1"
        assert policy.policy_name == "drop_newest"

    @pytest.mark.asyncio
    async def test_handle_overflow_drops_new_item(self) -> None:
        """Test drop newest policy drops the new item on overflow."""
        policy: DropNewestPolicy[Any] = DropNewestPolicy(
            threshold=5, agent_id="agent_1"
        )
        queue = deque([1, 2, 3, 4, 5])  # 5 items, at threshold

        result = await policy.handle_overflow(queue, 6)

        assert result is True
        assert len(queue) == 5  # Should remain at threshold
        assert list(queue) == [
            1,
            2,
            3,
            4,
            5,
        ]  # Original items preserved, new item dropped

    @pytest.mark.asyncio
    async def test_handle_no_overflow_adds_normally(self) -> None:
        """Test normal addition when under threshold."""
        policy: DropNewestPolicy[Any] = DropNewestPolicy(
            threshold=5, agent_id="agent_1"
        )
        queue = deque([1, 2, 3])  # 3 items, under threshold

        result = await policy.handle_overflow(queue, 4)

        assert result is True
        assert len(queue) == 4
        assert list(queue) == [1, 2, 3, 4]

    @patch("gunn.utils.backpressure.record_queue_high_watermark")
    @patch("gunn.utils.backpressure.record_backpressure_event")
    @pytest.mark.asyncio
    async def test_handle_overflow_records_metrics(
        self, mock_record_event, mock_record_watermark
    ) -> None:
        """Test drop newest policy records metrics on overflow."""
        policy: DropNewestPolicy[Any] = DropNewestPolicy(
            threshold=3, agent_id="agent_1"
        )
        queue = deque([1, 2, 3, 4])

        await policy.handle_overflow(queue, 5)

        mock_record_watermark.assert_called_once_with("agent_1", "queue", 4)
        mock_record_event.assert_called_once_with("agent_1", "queue", "drop_newest")


class TestBackpressureManager:
    """Test BackpressureManager class."""

    def test_initialization(self) -> None:
        """Test backpressure manager initialization."""
        manager = BackpressureManager()

        expected_policies = ["defer", "shed_oldest", "drop_newest"]
        assert set(manager.available_policies) == set(expected_policies)

    def test_create_defer_policy(self) -> None:
        """Test creating defer policy."""
        manager = BackpressureManager()

        policy: BackpressurePolicy[Any] = manager.create_policy(
            "defer", threshold=10, agent_id="agent_1"
        )

        assert isinstance(policy, DeferPolicy)
        assert policy.threshold == 10
        assert policy.agent_id == "agent_1"

    def test_create_shed_oldest_policy(self) -> None:
        """Test creating shed oldest policy."""
        manager = BackpressureManager()

        policy: BackpressurePolicy[Any] = manager.create_policy(
            "shed_oldest", threshold=20, agent_id="agent_2"
        )

        assert isinstance(policy, ShedOldestPolicy)
        assert policy.threshold == 20
        assert policy.agent_id == "agent_2"

    def test_create_drop_newest_policy(self) -> None:
        """Test creating drop newest policy."""
        manager = BackpressureManager()

        policy: BackpressurePolicy[Any] = manager.create_policy(
            "drop_newest", threshold=15, agent_id="agent_3"
        )

        assert isinstance(policy, DropNewestPolicy)
        assert policy.threshold == 15
        assert policy.agent_id == "agent_3"

    def test_create_invalid_policy_raises_error(self) -> None:
        """Test creating invalid policy raises ValueError."""
        manager = BackpressureManager()

        with pytest.raises(ValueError) as exc_info:
            manager.create_policy("invalid_policy", threshold=10)

        assert "Unknown backpressure policy 'invalid_policy'" in str(exc_info.value)
        assert "defer, shed_oldest, drop_newest" in str(exc_info.value)

    def test_register_custom_policy(self) -> None:
        """Test registering custom backpressure policy."""
        manager = BackpressureManager()

        class CustomPolicy(DeferPolicy[Any]):
            @property
            def policy_name(self) -> str:
                return "custom"

        manager.register_policy("custom", CustomPolicy)

        assert "custom" in manager.available_policies

        policy: BackpressurePolicy[Any] = manager.create_policy(
            "custom", threshold=5, agent_id="test"
        )
        assert isinstance(policy, CustomPolicy)
        assert policy.policy_name == "custom"

    def test_global_manager_instance(self) -> None:
        """Test global backpressure manager instance."""
        assert backpressure_manager is not None
        assert isinstance(backpressure_manager, BackpressureManager)
        assert "defer" in backpressure_manager.available_policies


class TestBackpressureQueue:
    """Test BackpressureQueue class."""

    @pytest.mark.asyncio
    async def test_initialization(self) -> None:
        """Test backpressure queue initialization."""
        policy: DeferPolicy[Any] = DeferPolicy(threshold=5, agent_id="agent_1")
        queue = BackpressureQueue(policy, maxsize=10, queue_type="test_queue")

        assert queue.policy == policy
        assert queue.maxsize == 10
        assert queue.queue_type == "test_queue"
        assert queue.qsize() == 0
        assert queue.empty()
        assert not queue.full()

    @pytest.mark.asyncio
    async def test_put_under_threshold(self) -> None:
        """Test putting items when under threshold."""
        policy: DeferPolicy[Any] = DeferPolicy(threshold=5, agent_id="agent_1")
        queue = BackpressureQueue(policy)

        await queue.put("item1")
        await queue.put("item2")

        assert queue.qsize() == 2
        assert not queue.empty()

    @pytest.mark.asyncio
    async def test_put_over_threshold_with_defer_policy(self) -> None:
        """Test putting items over threshold with defer policy raises error."""
        policy: DeferPolicy[Any] = DeferPolicy(threshold=2, agent_id="agent_1")
        queue = BackpressureQueue(policy)

        # Fill to threshold
        await queue.put("item1")
        await queue.put("item2")

        # This should trigger backpressure
        with pytest.raises(BackpressureError):
            await queue.put("item3")

    @pytest.mark.asyncio
    async def test_put_over_threshold_with_shed_oldest_policy(self) -> None:
        """Test putting items over threshold with shed oldest policy."""
        policy: ShedOldestPolicy[Any] = ShedOldestPolicy(
            threshold=2, agent_id="agent_1"
        )
        queue = BackpressureQueue(policy)

        # Fill to threshold
        await queue.put("item1")
        await queue.put("item2")

        # This should shed oldest and add new
        await queue.put("item3")

        assert queue.qsize() == 2
        items = []
        while not queue.empty():
            items.append(await queue.get())

        assert items == ["item2", "item3"]  # item1 was shed

    @pytest.mark.asyncio
    async def test_put_over_threshold_with_drop_newest_policy(self) -> None:
        """Test putting items over threshold with drop newest policy."""
        policy: DropNewestPolicy[Any] = DropNewestPolicy(
            threshold=2, agent_id="agent_1"
        )
        queue = BackpressureQueue(policy)

        # Fill to threshold
        await queue.put("item1")
        await queue.put("item2")

        # This should drop the new item
        await queue.put("item3")

        assert queue.qsize() == 2
        items = []
        while not queue.empty():
            items.append(await queue.get())

        assert items == ["item1", "item2"]  # item3 was dropped

    @pytest.mark.asyncio
    async def test_get_from_queue(self) -> None:
        """Test getting items from queue."""
        policy: DeferPolicy[Any] = DeferPolicy(threshold=5, agent_id="agent_1")
        queue = BackpressureQueue(policy)

        await queue.put("item1")
        await queue.put("item2")

        item1 = await queue.get()
        item2 = await queue.get()

        assert item1 == "item1"
        assert item2 == "item2"
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_get_from_empty_queue_raises_error(self) -> None:
        """Test getting from empty queue raises QueueEmpty."""
        policy: DeferPolicy[Any] = DeferPolicy(threshold=5, agent_id="agent_1")
        queue = BackpressureQueue(policy)

        with pytest.raises(asyncio.QueueEmpty):
            await queue.get()

    @pytest.mark.asyncio
    async def test_maxsize_limit(self) -> None:
        """Test maxsize limit is enforced."""
        policy: ShedOldestPolicy[Any] = ShedOldestPolicy(
            threshold=10, agent_id="agent_1"
        )  # High threshold
        queue = BackpressureQueue(policy, maxsize=3)  # But low maxsize

        # Fill to maxsize
        await queue.put("item1")
        await queue.put("item2")
        await queue.put("item3")

        # This should trigger overflow handling due to maxsize
        await queue.put("item4")

        assert queue.qsize() == 3
        items = []
        while not queue.empty():
            items.append(await queue.get())

        assert items == ["item2", "item3", "item4"]  # item1 was shed

    @pytest.mark.asyncio
    async def test_full_property(self) -> None:
        """Test full property with maxsize."""
        policy: DeferPolicy[Any] = DeferPolicy(threshold=10, agent_id="agent_1")
        queue = BackpressureQueue(policy, maxsize=2)

        assert not queue.full()

        await queue.put("item1")
        assert not queue.full()

        await queue.put("item2")
        assert queue.full()

    @pytest.mark.asyncio
    async def test_full_property_unlimited_maxsize(self) -> None:
        """Test full property with unlimited maxsize."""
        policy: ShedOldestPolicy[Any] = ShedOldestPolicy(
            threshold=200, agent_id="agent_1"
        )  # Use shed policy with high threshold
        queue = BackpressureQueue(policy, maxsize=0)  # Unlimited

        for i in range(100):
            await queue.put(f"item{i}")

        assert not queue.full()  # Never full with maxsize=0

    @pytest.mark.asyncio
    async def test_queue_size_tracking(self) -> None:
        """Test queue size tracking methods."""
        policy: DeferPolicy[Any] = DeferPolicy(threshold=10, agent_id="agent_1")
        queue = BackpressureQueue(policy)

        assert queue.qsize() == 0
        assert queue.empty()

        await queue.put("item1")
        assert queue.qsize() == 1
        assert not queue.empty()

        await queue.put("item2")
        assert queue.qsize() == 2

        await queue.get()
        assert queue.qsize() == 1

        await queue.get()
        assert queue.qsize() == 0
        assert queue.empty()
