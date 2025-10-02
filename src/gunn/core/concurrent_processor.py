"""Concurrent intent processing for Gunn orchestrator.

This module provides enhanced intent processing capabilities including
batch submission, concurrent execution, and deterministic ordering
for multi-agent scenarios.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from gunn.schemas.types import Effect, Intent
from gunn.utils.telemetry import async_performance_timer, get_logger

logger = get_logger(__name__)


class ProcessingMode(str, Enum):
    """Intent processing modes."""

    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    DETERMINISTIC_CONCURRENT = "deterministic_concurrent"


class BatchResult(BaseModel):
    """Result of batch intent processing."""

    model_config = {"arbitrary_types_allowed": True}

    effects: list[Effect] = Field(description="All effects generated from the batch")
    errors: dict[str, Exception] = Field(description="Errors by intent req_id")
    processing_time: float = Field(description="Total processing time in seconds")
    metadata: dict[str, Any] = Field(description="Additional processing metadata")


class ConcurrentProcessingConfig(BaseModel):
    """Configuration for concurrent intent processing."""

    max_concurrent_intents: int = Field(default=100, ge=1, le=1000)
    default_timeout: float = Field(default=30.0, gt=0.0)
    enable_deterministic_mode: bool = Field(default=True)
    processing_mode: ProcessingMode = Field(default=ProcessingMode.SEQUENTIAL)

    # Performance tuning
    batch_size_threshold: int = Field(default=5, ge=1)
    semaphore_size: int = Field(default=50, ge=1)


class ConcurrentIntentProcessor:
    """Handles concurrent intent processing logic."""

    def __init__(self, config: ConcurrentProcessingConfig):
        """Initialize concurrent processor.

        Args:
            config: Configuration for concurrent processing
        """
        self.config = config
        self.processing_semaphore = asyncio.Semaphore(config.semaphore_size)
        self._intent_processor: Callable[[Intent], Awaitable[list[Effect]]] | None = (
            None
        )

    def set_intent_processor(
        self, processor: Callable[[Intent], Awaitable[list[Effect]]]
    ):
        """Set the function used to process individual intents.

        Args:
            processor: Function that processes a single intent and returns effects
        """
        self._intent_processor = processor

    async def process_batch(
        self,
        intents: list[Intent],
        mode: ProcessingMode | None = None,
        timeout: float | None = None,
    ) -> BatchResult:
        """Process batch of intents according to specified mode.

        Args:
            intents: List of intents to process
            mode: Processing mode (defaults to config default)
            timeout: Timeout for concurrent operations

        Returns:
            BatchResult containing effects, errors, and metadata

        Raises:
            ValueError: If no intent processor is set
            asyncio.TimeoutError: If processing exceeds timeout
        """
        if not self._intent_processor:
            raise ValueError(
                "Intent processor not set. Call set_intent_processor() first."
            )

        if not intents:
            return BatchResult(
                effects=[],
                errors={},
                processing_time=0.0,
                metadata={"mode": "empty", "intent_count": 0},
            )

        # Use provided mode or config default
        processing_mode = mode or self.config.processing_mode
        processing_timeout = timeout or self.config.default_timeout

        async with async_performance_timer("batch_intent_processing", logger=logger):
            if processing_mode == ProcessingMode.SEQUENTIAL:
                return await self._process_sequential(intents)
            elif processing_mode == ProcessingMode.CONCURRENT:
                return await self._process_concurrent(intents, processing_timeout)
            elif processing_mode == ProcessingMode.DETERMINISTIC_CONCURRENT:
                return await self._process_deterministic_concurrent(
                    intents, processing_timeout
                )
            else:
                raise ValueError(f"Unknown processing mode: {processing_mode}")

    async def _process_sequential(self, intents: list[Intent]) -> BatchResult:
        """Process intents sequentially (maintains current behavior).

        Args:
            intents: List of intents to process

        Returns:
            BatchResult with sequential processing results
        """
        effects = []
        errors = {}
        start_time = time.time()

        for intent in intents:
            req_id = intent.get("req_id", f"unknown_{id(intent)}")
            try:
                intent_effects = await self._intent_processor(intent)
                effects.extend(intent_effects)
                logger.debug(
                    f"Processed intent {req_id}: {len(intent_effects)} effects"
                )
            except Exception as e:
                errors[req_id] = e
                logger.warning(f"Intent {req_id} failed: {e}")

        processing_time = time.time() - start_time

        return BatchResult(
            effects=effects,
            errors=errors,
            processing_time=processing_time,
            metadata={
                "mode": "sequential",
                "intent_count": len(intents),
                "success_count": len(intents) - len(errors),
                "error_count": len(errors),
            },
        )

    async def _process_concurrent(
        self, intents: list[Intent], timeout: float
    ) -> BatchResult:
        """Process intents concurrently.

        Args:
            intents: List of intents to process
            timeout: Timeout for concurrent operations

        Returns:
            BatchResult with concurrent processing results

        Raises:
            asyncio.TimeoutError: If processing exceeds timeout
        """
        start_time = time.time()

        # Create tasks for all intents
        tasks = []
        req_ids = []

        for intent in intents:
            req_id = intent.get("req_id", f"unknown_{id(intent)}")
            req_ids.append(req_id)

            task = asyncio.create_task(self._process_single_intent_safe(intent, req_id))
            tasks.append(task)

        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
            )
        except TimeoutError:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

            logger.error(f"Batch processing timed out after {timeout}s")
            raise

        # Collect results
        effects = []
        errors = {}

        for req_id, result in zip(req_ids, results, strict=False):
            if isinstance(result, Exception):
                errors[req_id] = result
            else:
                effects.extend(result)

        processing_time = time.time() - start_time

        return BatchResult(
            effects=effects,
            errors=errors,
            processing_time=processing_time,
            metadata={
                "mode": "concurrent",
                "intent_count": len(intents),
                "success_count": len(intents) - len(errors),
                "error_count": len(errors),
                "timeout": timeout,
            },
        )

    async def _process_deterministic_concurrent(
        self, intents: list[Intent], timeout: float
    ) -> BatchResult:
        """Process intents concurrently with deterministic result ordering.

        Args:
            intents: List of intents to process
            timeout: Timeout for concurrent operations

        Returns:
            BatchResult with deterministically ordered results
        """
        # Sort intents by agent_id for deterministic processing order
        sorted_intents = sorted(intents, key=lambda x: x.get("agent_id", ""))

        # Process concurrently
        result = await self._process_concurrent(sorted_intents, timeout)

        # Sort effects by source agent for deterministic output
        result.effects.sort(key=lambda e: e.get("source_id", ""))

        # Update metadata
        result.metadata["mode"] = "deterministic_concurrent"
        result.metadata["deterministic_ordering"] = True

        return result

    async def _process_single_intent_safe(
        self, intent: Intent, req_id: str
    ) -> list[Effect]:
        """Process single intent with error isolation and semaphore control.

        Args:
            intent: Intent to process
            req_id: Request ID for logging

        Returns:
            List of effects generated by the intent

        Raises:
            Exception: Any exception from intent processing (for error collection)
        """
        async with self.processing_semaphore:
            try:
                async with async_performance_timer(
                    "single_intent_processing",
                    req_id=req_id,
                    agent_id=intent.get("agent_id"),
                    logger=logger,
                ):
                    effects = await self._intent_processor(intent)
                    logger.debug(f"Intent {req_id} processed: {len(effects)} effects")
                    return effects
            except Exception as e:
                logger.error(f"Intent {req_id} processing failed: {e}")
                raise

    def should_use_concurrent_processing(self, intent_count: int) -> bool:
        """Determine if concurrent processing should be used based on batch size.

        Args:
            intent_count: Number of intents in the batch

        Returns:
            True if concurrent processing is recommended
        """
        return intent_count >= self.config.batch_size_threshold

    def get_recommended_mode(self, intent_count: int) -> ProcessingMode:
        """Get recommended processing mode based on batch characteristics.

        Args:
            intent_count: Number of intents in the batch

        Returns:
            Recommended processing mode
        """
        if intent_count < self.config.batch_size_threshold:
            return ProcessingMode.SEQUENTIAL
        elif self.config.enable_deterministic_mode:
            return ProcessingMode.DETERMINISTIC_CONCURRENT
        else:
            return ProcessingMode.CONCURRENT

    def get_processing_stats(self) -> dict[str, Any]:
        """Get statistics about concurrent processing performance.

        Returns:
            Dictionary containing processing statistics
        """
        return {
            "config": self.config.model_dump(),
            "semaphore_available": self.processing_semaphore._value,
            "semaphore_total": self.config.semaphore_size,
            "semaphore_utilization": 1.0
            - (self.processing_semaphore._value / self.config.semaphore_size),
        }
