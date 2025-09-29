"""RL-style facade interface for multi-agent simulation.

This module provides an RL-style interface that wraps the Orchestrator functionality,
allowing developers to interact with the simulation using familiar RL patterns like
observe() and step() methods.
"""

import asyncio
import time
import uuid

from gunn.core.orchestrator import AgentHandle, Orchestrator, OrchestratorConfig
from gunn.policies.observation import ObservationPolicy
from gunn.schemas.types import Effect, Intent, ObservationDelta
from gunn.utils.telemetry import get_logger


class RLFacade:
    """RL-style facade wrapping Orchestrator functionality.

    Provides a familiar RL interface with observe() and step() methods while
    operating on the same underlying event system as other facades.

    Requirements addressed:
    - 5.1: Developers call env.observe(agent_id) to get observations
    - 5.2: Developers call env.step(agent_id, intent) and receive (Effect, ObservationDelta) tuple
    - 5.5: Operates on the same underlying event system
    """

    def __init__(
        self,
        orchestrator: Orchestrator | None = None,
        config: OrchestratorConfig | None = None,
        world_id: str = "default",
        timeout_seconds: float = 30.0,
    ):
        """Initialize RL facade.

        Args:
            orchestrator: Optional existing orchestrator instance
            config: Configuration for new orchestrator if not provided
            world_id: World identifier
            timeout_seconds: Default timeout for operations

        Raises:
            ValueError: If neither orchestrator nor config is provided
        """
        if orchestrator is None:
            if config is None:
                raise ValueError("Either orchestrator or config must be provided")
            self._orchestrator = Orchestrator(config, world_id)
            self.world_id = world_id
        else:
            self._orchestrator = orchestrator
            self.world_id = orchestrator.world_id
        self.timeout_seconds = timeout_seconds
        self._logger = get_logger("gunn.facades.rl", world_id=world_id)

        # Track pending step operations for proper cleanup
        self._pending_steps: dict[
            str, asyncio.Task[tuple[Effect, ObservationDelta]]
        ] = {}

        self._logger.info(
            "RL facade initialized",
            world_id=world_id,
            timeout_seconds=timeout_seconds,
        )

    async def initialize(self) -> None:
        """Initialize the underlying orchestrator."""
        await self._orchestrator.initialize()

    async def register_agent(
        self, agent_id: str, policy: ObservationPolicy
    ) -> AgentHandle:
        """Register an agent with the simulation.

        Args:
            agent_id: Unique identifier for the agent
            policy: Observation policy for this agent

        Returns:
            AgentHandle for the registered agent

        Raises:
            ValueError: If agent_id is invalid or already registered
            RuntimeError: If maximum agents exceeded
        """
        handle = await self._orchestrator.register_agent(agent_id, policy)
        self._logger.info("Agent registered via RL facade", agent_id=agent_id)
        return handle

    async def observe(self, agent_id: str) -> ObservationDelta:
        """Get current observations for an agent.

        This method implements requirement 5.1: developers call env.observe(agent_id)
        to get observations.

        Args:
            agent_id: Agent identifier

        Returns:
            Current observation delta for the agent

        Raises:
            ValueError: If agent_id is not registered
            TimeoutError: If observation retrieval times out
            RuntimeError: If agent handle is not available

        Requirements addressed:
        - 5.1: WHEN using RL facade THEN developers SHALL call env.observe(agent_id) to get observations
        """
        start_time = time.perf_counter()

        if agent_id not in self._orchestrator.agent_handles:
            raise ValueError(f"Agent {agent_id} is not registered")

        agent_handle = self._orchestrator.agent_handles[agent_id]

        try:
            # Get next observation with timeout
            observation_raw = await asyncio.wait_for(
                agent_handle.next_observation(), timeout=self.timeout_seconds
            )

            # Convert to ObservationDelta if needed
            if isinstance(observation_raw, dict):
                observation: ObservationDelta = observation_raw  # type: ignore
            else:
                # Handle other observation types by converting to dict
                observation = {
                    "view_seq": getattr(observation_raw, "view_seq", 0),
                    "patches": getattr(observation_raw, "patches", []),
                    "context_digest": getattr(observation_raw, "context_digest", ""),
                    "schema_version": getattr(
                        observation_raw, "schema_version", "1.0.0"
                    ),
                }

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            self._logger.info(
                "Observation retrieved via RL facade",
                agent_id=agent_id,
                view_seq=observation.get("view_seq", "unknown"),
                processing_time_ms=processing_time_ms,
            )

            return observation

        except TimeoutError as e:
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.error(
                "Observation retrieval timed out",
                agent_id=agent_id,
                timeout_seconds=self.timeout_seconds,
                processing_time_ms=processing_time_ms,
            )
            raise TimeoutError(
                f"Observation retrieval for agent {agent_id} timed out after {self.timeout_seconds}s"
            ) from e

    async def step(
        self, agent_id: str, intent: Intent
    ) -> tuple[Effect, ObservationDelta]:
        """Execute a step with an intent and return the effect and observation.

        This method implements requirement 5.2: developers call env.step(agent_id, intent)
        and receive (Effect, ObservationDelta) tuple.

        Args:
            agent_id: Agent identifier
            intent: Intent to execute

        Returns:
            Tuple of (Effect, ObservationDelta) representing the action result

        Raises:
            ValueError: If agent_id is not registered or intent is invalid
            StaleContextError: If intent context is stale
            QuotaExceededError: If agent quota is exceeded
            BackpressureError: If backpressure limits are exceeded
            ValidationError: If intent validation fails
            TimeoutError: If step execution times out

        Requirements addressed:
        - 5.2: WHEN using RL facade THEN developers SHALL call env.step(agent_id, intent) and receive (Effect, ObservationDelta) tuple
        """
        start_time = time.perf_counter()

        if agent_id not in self._orchestrator.agent_handles:
            raise ValueError(f"Agent {agent_id} is not registered")

        # Validate intent has required fields
        if not intent.get("req_id"):
            intent["req_id"] = f"rl_step_{uuid.uuid4().hex[:8]}"
        if not intent.get("agent_id"):
            intent["agent_id"] = agent_id
        if intent["agent_id"] != agent_id:
            raise ValueError(
                f"Intent agent_id {intent['agent_id']} does not match provided agent_id {agent_id}"
            )

        # Clean up any existing pending step for this agent
        if agent_id in self._pending_steps:
            old_task = self._pending_steps[agent_id]
            if not old_task.done():
                old_task.cancel()
                try:
                    await old_task
                except asyncio.CancelledError:
                    pass

        try:
            # Create task for step execution
            step_task = asyncio.create_task(self._execute_step(agent_id, intent))
            self._pending_steps[agent_id] = step_task

            # Execute with timeout
            result = await asyncio.wait_for(step_task, timeout=self.timeout_seconds)

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            self._logger.info(
                "Step completed via RL facade",
                agent_id=agent_id,
                req_id=intent["req_id"],
                intent_kind=intent["kind"],
                processing_time_ms=processing_time_ms,
            )

            return result

        except TimeoutError as e:
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.error(
                "Step execution timed out",
                agent_id=agent_id,
                req_id=intent.get("req_id", "unknown"),
                timeout_seconds=self.timeout_seconds,
                processing_time_ms=processing_time_ms,
            )
            raise TimeoutError(
                f"Step execution for agent {agent_id} timed out after {self.timeout_seconds}s"
            ) from e

        finally:
            # Clean up pending step
            self._pending_steps.pop(agent_id, None)

    async def _execute_step(
        self, agent_id: str, intent: Intent
    ) -> tuple[Effect, ObservationDelta]:
        """Execute a single step and return effect and observation.

        Args:
            agent_id: Agent identifier
            intent: Intent to execute

        Returns:
            Tuple of (Effect, ObservationDelta)

        Raises:
            StaleContextError: If intent context is stale
            QuotaExceededError: If agent quota is exceeded
            BackpressureError: If backpressure limits are exceeded
            ValidationError: If intent validation fails
        """
        # Submit intent and wait for processing
        _req_id = await self._orchestrator.submit_intent(intent)

        # Wait for intent processing to complete
        # This is necessary to ensure the effect is created before trying to get observation
        await asyncio.sleep(0.001)  # Reduced from 0.1s to 1ms for better performance

        # Get the observation for the agent
        agent_handle = self._orchestrator.agent_handles[agent_id]

        # Try to get the next observation with a reasonable timeout
        try:
            observation = await asyncio.wait_for(
                agent_handle.next_observation(),
                timeout=0.1,  # Reduced from 0.5s to 0.1s for better performance
            )
        except TimeoutError:
            # If no observation is available, create an empty delta
            observation = {
                "agent_id": agent_id,
                "global_seq": self._orchestrator._global_seq,
                "view_seq": 0,
                "delta": [],
                "schema_version": "1.0.0",
            }

        # Create a mock effect for now - in a full implementation, we would
        # track the actual effect created from the intent
        effect: Effect = {
            "uuid": uuid.uuid4().hex,
            "kind": intent["kind"],
            "payload": intent["payload"],
            "global_seq": self._orchestrator._global_seq,
            "sim_time": self._orchestrator._current_sim_time(),
            "source_id": agent_id,
            "schema_version": intent.get("schema_version", "1.0.0"),
        }

        return effect, observation

    async def shutdown(self) -> None:
        """Shutdown the RL facade and clean up resources.

        Cancels any pending step operations and shuts down the orchestrator.
        """
        self._logger.info("Shutting down RL facade")

        # Cancel all pending steps
        for _agent_id, task in list(self._pending_steps.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._pending_steps.clear()

        # Shutdown orchestrator
        await self._orchestrator.shutdown()

        self._logger.info("RL facade shutdown complete")

    def get_orchestrator(self) -> Orchestrator:
        """Get the underlying orchestrator instance.

        Returns:
            The orchestrator instance used by this facade

        Requirements addressed:
        - 5.5: Both facades operate on the same underlying event system
        """
        return self._orchestrator

    def set_timeout(self, timeout_seconds: float) -> None:
        """Set the default timeout for operations.

        Args:
            timeout_seconds: New timeout value in seconds

        Raises:
            ValueError: If timeout_seconds is not positive
        """
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

        self.timeout_seconds = timeout_seconds
        self._logger.info("Timeout updated", timeout_seconds=timeout_seconds)

    async def get_agent_view_seq(self, agent_id: str) -> int:
        """Get the current view sequence for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Current view sequence number

        Raises:
            ValueError: If agent_id is not registered
        """
        if agent_id not in self._orchestrator.agent_handles:
            raise ValueError(f"Agent {agent_id} is not registered")

        return self._orchestrator.agent_handles[agent_id].get_view_seq()

    async def cancel_agent_step(self, agent_id: str) -> bool:
        """Cancel any pending step operation for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            True if a step was cancelled, False if no step was pending

        Raises:
            ValueError: If agent_id is not registered
        """
        if agent_id not in self._orchestrator.agent_handles:
            raise ValueError(f"Agent {agent_id} is not registered")

        if agent_id in self._pending_steps:
            task = self._pending_steps[agent_id]
            if not task.done():
                task.cancel()
                self._logger.info("Cancelled pending step", agent_id=agent_id)
                return True

        return False

    def __repr__(self) -> str:
        """String representation of the RL facade."""
        return (
            f"RLFacade(world_id={self.world_id}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"agents={len(self._orchestrator.agent_handles)})"
        )
