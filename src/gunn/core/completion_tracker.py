"""Action completion tracking for intent-to-effect confirmation.

This module provides the ActionCompletionTracker class that tracks the completion
of intents as they are processed into effects and applied to the world state.
"""

import asyncio

from gunn.schemas.types import Effect


class ActionCompletionTracker:
    """Tracks action completion and provides confirmation to agents.

    This class manages the lifecycle of actions from intent submission to effect
    application, allowing agents to wait for confirmation that their actions have
    been processed and applied to the world state.

    Requirements addressed:
    - 16.1: Track intent-to-effect completion
    - 16.3: Wait for effect completion with timeout handling
    - 16.4: Provide completion confirmation to agents
    """

    def __init__(self, default_timeout: float = 30.0):
        """Initialize the action completion tracker.

        Args:
            default_timeout: Default timeout in seconds for action completion
        """
        self._pending_actions: dict[str, asyncio.Future[Effect]] = {}
        self._default_timeout = default_timeout
        self._timeout_tasks: dict[str, asyncio.Task] = {}

    def register_action(
        self, req_id: str, timeout: float | None = None
    ) -> asyncio.Future[Effect]:
        """Register an action and return future for completion.

        Args:
            req_id: Request ID of the action to track
            timeout: Optional timeout override (uses default if None)

        Returns:
            Future that will be resolved when the action completes

        Raises:
            ValueError: If req_id is empty or already registered
        """
        if not req_id.strip():
            raise ValueError("req_id cannot be empty")

        if req_id in self._pending_actions:
            raise ValueError(f"Action {req_id} is already registered")

        future: asyncio.Future[Effect] = asyncio.Future()
        self._pending_actions[req_id] = future

        # Set timeout
        timeout_value = timeout if timeout is not None else self._default_timeout
        timeout_task = asyncio.create_task(
            self._timeout_action(req_id, future, timeout_value)
        )
        self._timeout_tasks[req_id] = timeout_task

        return future

    async def complete_action(self, req_id: str, result: Effect) -> None:
        """Mark action as completed with result.

        Args:
            req_id: Request ID of the completed action
            result: The Effect that was applied
        """
        if req_id in self._pending_actions:
            future = self._pending_actions.pop(req_id)

            # Cancel timeout task
            if req_id in self._timeout_tasks:
                timeout_task = self._timeout_tasks.pop(req_id)
                timeout_task.cancel()

            if not future.done():
                future.set_result(result)

    async def fail_action(self, req_id: str, error: Exception) -> None:
        """Mark action as failed with error.

        Args:
            req_id: Request ID of the failed action
            error: The exception that caused the failure
        """
        if req_id in self._pending_actions:
            future = self._pending_actions.pop(req_id)

            # Cancel timeout task
            if req_id in self._timeout_tasks:
                timeout_task = self._timeout_tasks.pop(req_id)
                timeout_task.cancel()

            if not future.done():
                future.set_exception(error)

    async def wait_for_completion(
        self, req_id: str, timeout: float | None = None
    ) -> Effect:
        """Wait for an action to complete.

        Args:
            req_id: Request ID to wait for
            timeout: Optional timeout override

        Returns:
            The Effect that was applied

        Raises:
            asyncio.TimeoutError: If timeout is reached
            ValueError: If req_id is not registered
        """
        if req_id not in self._pending_actions:
            raise ValueError(f"Action {req_id} is not registered")

        future = self._pending_actions[req_id]

        if timeout is not None:
            return await asyncio.wait_for(future, timeout=timeout)
        else:
            return await future

    async def _timeout_action(
        self, req_id: str, future: asyncio.Future, timeout: float
    ) -> None:
        """Timeout an action if it takes too long.

        Args:
            req_id: Request ID of the action
            future: Future to timeout
            timeout: Timeout duration in seconds
        """
        try:
            await asyncio.sleep(timeout)
            if not future.done():
                future.set_exception(
                    TimeoutError(f"Action {req_id} timed out after {timeout}s")
                )
                self._pending_actions.pop(req_id, None)
                self._timeout_tasks.pop(req_id, None)
        except asyncio.CancelledError:
            # Timeout was cancelled because action completed
            pass

    def is_pending(self, req_id: str) -> bool:
        """Check if an action is still pending.

        Args:
            req_id: Request ID to check

        Returns:
            True if the action is pending, False otherwise
        """
        return req_id in self._pending_actions

    def get_pending_count(self) -> int:
        """Get the number of pending actions.

        Returns:
            Number of actions currently pending
        """
        return len(self._pending_actions)

    def cancel_all(self) -> None:
        """Cancel all pending actions."""
        for req_id in list(self._pending_actions.keys()):
            future = self._pending_actions.pop(req_id)
            if not future.done():
                future.cancel()

            # Cancel timeout task
            if req_id in self._timeout_tasks:
                timeout_task = self._timeout_tasks.pop(req_id)
                timeout_task.cancel()
