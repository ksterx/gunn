"""Quota management and rate limiting for multi-agent systems.

This module provides token bucket rate limiting and quota enforcement
to prevent agents from overwhelming the system with too many requests.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from gunn.utils.errors import QuotaExceededError
from gunn.utils.telemetry import get_logger


@dataclass
class QuotaPolicy:
    """Configuration for quota enforcement.

    Attributes:
        intents_per_minute: Maximum intents per agent per minute
        tokens_per_minute: Maximum tokens per agent per minute
        burst_size: Maximum burst size (defaults to rate if not specified)
        refill_interval_seconds: How often to refill tokens
    """

    intents_per_minute: int = 60
    tokens_per_minute: int = 10000
    burst_size: int | None = None
    refill_interval_seconds: float = 1.0

    def __post_init__(self):
        """Set default burst size if not specified."""
        if self.burst_size is None:
            # Allow burst up to 10% of per-minute rate
            self.burst_size = max(1, self.intents_per_minute // 6)


class TokenBucket:
    """Token bucket algorithm for rate limiting.

    Implements a classic token bucket with configurable capacity,
    refill rate, and burst handling. Thread-safe for concurrent access.

    Requirements addressed:
    - 18.1: Per-agent rate limiting with token bucket
    - 10.2: Quota enforcement in backpressure policies
    """

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        refill_interval: float = 1.0,
    ):
        """Initialize token bucket.

        Args:
            capacity: Maximum number of tokens (burst size)
            refill_rate: Tokens added per refill interval
            refill_interval: Time between refills in seconds
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_interval = refill_interval

        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        async with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            return False

    async def wait_for_tokens(
        self, tokens: float = 1.0, timeout: float | None = None
    ) -> bool:
        """Wait until tokens are available.

        Args:
            tokens: Number of tokens needed
            timeout: Maximum time to wait in seconds

        Returns:
            True if tokens were acquired, False if timeout
        """
        start_time = time.monotonic()

        while True:
            if await self.consume(tokens):
                return True

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False

            # Wait for next refill
            await asyncio.sleep(self.refill_interval)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill

        if elapsed >= self.refill_interval:
            # Calculate tokens to add
            intervals = elapsed / self.refill_interval
            tokens_to_add = intervals * self.refill_rate

            # Add tokens up to capacity
            self._tokens = min(self.capacity, self._tokens + tokens_to_add)
            self._last_refill = now

    async def get_available_tokens(self) -> float:
        """Get current number of available tokens.

        Returns:
            Number of tokens currently available
        """
        async with self._lock:
            self._refill()
            return self._tokens

    async def reset(self) -> None:
        """Reset bucket to full capacity."""
        async with self._lock:
            self._tokens = self.capacity
            self._last_refill = time.monotonic()


class QuotaController:
    """Manages quotas and rate limiting for multiple agents.

    Provides per-agent token buckets for intent and token quotas,
    with configurable policies and enforcement strategies.

    Requirements addressed:
    - 18.1: QuotaController with TokenBucket for per-agent rate limiting
    - 18.4: Configurable quota policies
    - 10.3: QUOTA_EXCEEDED error surfacing
    """

    def __init__(self, policy: QuotaPolicy | None = None):
        """Initialize quota controller.

        Args:
            policy: Quota policy configuration
        """
        self.policy = policy or QuotaPolicy()

        # Per-agent token buckets
        self._intent_buckets: dict[str, TokenBucket] = {}
        self._token_buckets: dict[str, TokenBucket] = {}

        # Statistics
        self._quota_exceeded_count: dict[str, int] = defaultdict(int)
        self._total_intents: dict[str, int] = defaultdict(int)
        self._total_tokens: dict[str, int] = defaultdict(int)

        self._lock = asyncio.Lock()
        self._logger = get_logger("gunn.quota")

    def _get_intent_bucket(self, agent_id: str) -> TokenBucket:
        """Get or create intent bucket for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Token bucket for intent rate limiting
        """
        if agent_id not in self._intent_buckets:
            # Convert per-minute rate to per-second
            refill_rate = self.policy.intents_per_minute / 60.0
            capacity = self.policy.burst_size or refill_rate

            self._intent_buckets[agent_id] = TokenBucket(
                capacity=capacity,
                refill_rate=refill_rate,
                refill_interval=self.policy.refill_interval_seconds,
            )

        return self._intent_buckets[agent_id]

    def _get_token_bucket(self, agent_id: str) -> TokenBucket:
        """Get or create token bucket for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Token bucket for token rate limiting
        """
        if agent_id not in self._token_buckets:
            # Convert per-minute rate to per-second
            refill_rate = self.policy.tokens_per_minute / 60.0
            capacity = self.policy.tokens_per_minute / 6.0  # 10 second burst

            self._token_buckets[agent_id] = TokenBucket(
                capacity=capacity,
                refill_rate=refill_rate,
                refill_interval=self.policy.refill_interval_seconds,
            )

        return self._token_buckets[agent_id]

    async def check_intent_quota(self, agent_id: str, wait: bool = False) -> None:
        """Check if agent can submit an intent.

        Args:
            agent_id: Agent identifier
            wait: If True, wait for quota; if False, raise immediately

        Raises:
            QuotaExceededError: If quota is exceeded and wait=False
        """
        bucket = self._get_intent_bucket(agent_id)

        if wait:
            success = await bucket.wait_for_tokens(1.0, timeout=5.0)
            if not success:
                self._quota_exceeded_count[agent_id] += 1
                self._logger.warning(
                    "Intent quota exceeded after waiting",
                    agent_id=agent_id,
                    quota_type="intents_per_minute",
                    limit=self.policy.intents_per_minute,
                )
                raise QuotaExceededError(
                    agent_id=agent_id,
                    quota_type="intents_per_minute",
                    limit=self.policy.intents_per_minute,
                )
        else:
            success = await bucket.consume(1.0)
            if not success:
                self._quota_exceeded_count[agent_id] += 1
                self._logger.warning(
                    "Intent quota exceeded",
                    agent_id=agent_id,
                    quota_type="intents_per_minute",
                    limit=self.policy.intents_per_minute,
                )
                raise QuotaExceededError(
                    agent_id=agent_id,
                    quota_type="intents_per_minute",
                    limit=self.policy.intents_per_minute,
                )

        self._total_intents[agent_id] += 1

    async def check_token_quota(
        self,
        agent_id: str,
        tokens: int,
        wait: bool = False,
    ) -> None:
        """Check if agent can consume tokens.

        Args:
            agent_id: Agent identifier
            tokens: Number of tokens to consume
            wait: If True, wait for quota; if False, raise immediately

        Raises:
            QuotaExceededError: If quota is exceeded and wait=False
        """
        bucket = self._get_token_bucket(agent_id)

        if wait:
            success = await bucket.wait_for_tokens(float(tokens), timeout=5.0)
            if not success:
                self._quota_exceeded_count[agent_id] += 1
                self._logger.warning(
                    "Token quota exceeded after waiting",
                    agent_id=agent_id,
                    quota_type="tokens_per_minute",
                    limit=self.policy.tokens_per_minute,
                    requested=tokens,
                )
                raise QuotaExceededError(
                    agent_id=agent_id,
                    quota_type="tokens_per_minute",
                    limit=self.policy.tokens_per_minute,
                )
        else:
            success = await bucket.consume(float(tokens))
            if not success:
                self._quota_exceeded_count[agent_id] += 1
                self._logger.warning(
                    "Token quota exceeded",
                    agent_id=agent_id,
                    quota_type="tokens_per_minute",
                    limit=self.policy.tokens_per_minute,
                    requested=tokens,
                )
                raise QuotaExceededError(
                    agent_id=agent_id,
                    quota_type="tokens_per_minute",
                    limit=self.policy.tokens_per_minute,
                )

        self._total_tokens[agent_id] += tokens

    async def get_available_intents(self, agent_id: str) -> float:
        """Get available intent quota for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Number of intents currently available
        """
        bucket = self._get_intent_bucket(agent_id)
        return await bucket.get_available_tokens()

    async def get_available_tokens(self, agent_id: str) -> float:
        """Get available token quota for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Number of tokens currently available
        """
        bucket = self._get_token_bucket(agent_id)
        return await bucket.get_available_tokens()

    async def reset_agent_quota(self, agent_id: str) -> None:
        """Reset quota for specific agent.

        Args:
            agent_id: Agent identifier
        """
        if agent_id in self._intent_buckets:
            await self._intent_buckets[agent_id].reset()
        if agent_id in self._token_buckets:
            await self._token_buckets[agent_id].reset()

        self._logger.info("Agent quota reset", agent_id=agent_id)

    def get_stats(self) -> dict[str, Any]:
        """Get quota statistics.

        Returns:
            Dictionary with quota statistics
        """
        return {
            "policy": {
                "intents_per_minute": self.policy.intents_per_minute,
                "tokens_per_minute": self.policy.tokens_per_minute,
                "burst_size": self.policy.burst_size,
            },
            "total_intents": dict(self._total_intents),
            "total_tokens": dict(self._total_tokens),
            "quota_exceeded_count": dict(self._quota_exceeded_count),
            "tracked_agents": len(self._intent_buckets),
        }

    async def remove_agent(self, agent_id: str) -> None:
        """Remove agent and clean up resources.

        Args:
            agent_id: Agent identifier
        """
        async with self._lock:
            if agent_id in self._intent_buckets:
                del self._intent_buckets[agent_id]
            if agent_id in self._token_buckets:
                del self._token_buckets[agent_id]

            # Clear statistics
            if agent_id in self._quota_exceeded_count:
                del self._quota_exceeded_count[agent_id]
            if agent_id in self._total_intents:
                del self._total_intents[agent_id]
            if agent_id in self._total_tokens:
                del self._total_tokens[agent_id]

            self._logger.info("Agent removed from quota controller", agent_id=agent_id)
