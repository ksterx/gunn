"""Message-oriented facade interface for multi-agent simulation.

This module provides a message-oriented interface that wraps the Orchestrator functionality,
allowing developers to interact with the simulation using event-driven patterns like
emit() for broadcasting and subscription-based message delivery.
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import ObservationPolicy
from gunn.schemas.types import EffectDraft
from gunn.utils.telemetry import get_logger

# Type aliases for message handling
MessageHandler = Callable[[str, dict[str, Any]], None]
AsyncMessageHandler = Callable[[str, dict[str, Any]], Any]


class MessageSubscription:
    """Represents a message subscription for an agent."""

    def __init__(
        self,
        agent_id: str,
        message_types: set[str] | None = None,
        handler: MessageHandler | AsyncMessageHandler | None = None,
    ):
        """Initialize a message subscription.

        Args:
            agent_id: Agent identifier
            message_types: Set of message types to subscribe to (None = all types)
            handler: Optional message handler function
        """
        self.agent_id = agent_id
        self.message_types = message_types or set()
        self.handler = handler
        self.active = True

    def matches(self, message_type: str) -> bool:
        """Check if this subscription matches a message type.

        Args:
            message_type: Message type to check

        Returns:
            True if subscription matches, False otherwise
        """
        if not self.active:
            return False
        if not self.message_types:  # Empty set means subscribe to all
            return True
        return message_type in self.message_types


class MessageFacade:
    """Message-oriented facade wrapping Orchestrator functionality.

    Provides an event-driven interface with emit() for broadcasting and
    subscription-based message delivery while operating on the same underlying
    event system as other facades.

    Requirements addressed:
    - 5.3: Developers call env.emit() to broadcast events through ObservationPolicy filtering
    - 5.4: Agents receive messages according to their observation policies
    - 5.5: Operates on the same underlying event system
    """

    def __init__(
        self,
        orchestrator: Orchestrator | None = None,
        config: OrchestratorConfig | None = None,
        world_id: str = "default",
        timeout_seconds: float = 30.0,
    ):
        """Initialize message facade.

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
        self._logger = get_logger("gunn.facades.message", world_id=world_id)

        # Message subscription management
        self._subscriptions: dict[str, list[MessageSubscription]] = {}
        self._message_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._subscription_lock = asyncio.Lock()

        # Background tasks for message delivery
        self._delivery_tasks: dict[str, asyncio.Task[None]] = {}
        self._shutdown_event = asyncio.Event()

        self._logger.info(
            "Message facade initialized",
            world_id=world_id,
            timeout_seconds=timeout_seconds,
        )

    async def initialize(self) -> None:
        """Initialize the underlying orchestrator and start message delivery."""
        await self._orchestrator.initialize()
        await self._start_message_delivery()

    async def register_agent(self, agent_id: str, policy: ObservationPolicy) -> None:
        """Register an agent with the simulation.

        Args:
            agent_id: Unique identifier for the agent
            policy: Observation policy for this agent

        Raises:
            ValueError: If agent_id is invalid or already registered
            RuntimeError: If maximum agents exceeded
        """
        await self._orchestrator.register_agent(agent_id, policy)

        # Initialize message queue for this agent
        async with self._subscription_lock:
            if agent_id not in self._message_queues:
                self._message_queues[agent_id] = asyncio.Queue()
                self._subscriptions[agent_id] = []

                # Start message delivery task for this agent
                self._delivery_tasks[agent_id] = asyncio.create_task(
                    self._message_delivery_loop(agent_id)
                )

        self._logger.info("Agent registered via message facade", agent_id=agent_id)

    async def emit(
        self,
        message_type: str,
        payload: dict[str, Any],
        source_id: str,
        schema_version: str = "1.0.0",
    ) -> None:
        """Broadcast an event through observation policies.

        This method implements requirement 5.3: developers call env.emit() to broadcast
        events through ObservationPolicy filtering.

        Args:
            message_type: Type of message being emitted
            payload: Message payload data
            source_id: Identifier of the message source
            schema_version: Schema version for the message

        Raises:
            ValueError: If message_type or source_id is invalid
            BackpressureError: If backpressure limits are exceeded
            ValidationError: If message validation fails
            TimeoutError: If emission times out

        Requirements addressed:
        - 5.3: WHEN using message facade THEN developers SHALL call env.emit() to broadcast events through ObservationPolicy filtering
        """
        start_time = time.perf_counter()

        if not message_type.strip():
            raise ValueError("message_type cannot be empty")
        if not source_id.strip():
            raise ValueError("source_id cannot be empty")

        # Create effect draft for broadcasting
        draft: EffectDraft = {
            "kind": message_type,
            "payload": payload,
            "source_id": source_id,
            "schema_version": schema_version,
        }

        try:
            # Broadcast through orchestrator's event system
            await asyncio.wait_for(
                self._orchestrator.broadcast_event(draft), timeout=self.timeout_seconds
            )

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            self._logger.info(
                "Message emitted via message facade",
                message_type=message_type,
                source_id=source_id,
                payload_size=len(str(payload)),
                processing_time_ms=processing_time_ms,
            )

        except TimeoutError as e:
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.error(
                "Message emission timed out",
                message_type=message_type,
                source_id=source_id,
                timeout_seconds=self.timeout_seconds,
                processing_time_ms=processing_time_ms,
            )
            raise TimeoutError(
                f"Message emission timed out after {self.timeout_seconds}s"
            ) from e

    async def subscribe(
        self,
        agent_id: str,
        message_types: set[str] | None = None,
        handler: MessageHandler | AsyncMessageHandler | None = None,
    ) -> MessageSubscription:
        """Subscribe an agent to specific message types.

        This method implements requirement 5.4: agents receive messages according
        to their observation policies.

        Args:
            agent_id: Agent identifier
            message_types: Set of message types to subscribe to (None = all types)
            handler: Optional message handler function

        Returns:
            MessageSubscription object for managing the subscription

        Raises:
            ValueError: If agent_id is not registered

        Requirements addressed:
        - 5.4: WHEN using message facade THEN agents SHALL receive messages according to their observation policies
        """
        if agent_id not in self._orchestrator.agent_handles:
            raise ValueError(f"Agent {agent_id} is not registered")

        subscription = MessageSubscription(agent_id, message_types, handler)

        async with self._subscription_lock:
            if agent_id not in self._subscriptions:
                self._subscriptions[agent_id] = []
            self._subscriptions[agent_id].append(subscription)

        self._logger.info(
            "Agent subscribed to messages",
            agent_id=agent_id,
            message_types=list(message_types) if message_types else "all",
            has_handler=handler is not None,
        )

        return subscription

    async def unsubscribe(self, subscription: MessageSubscription) -> None:
        """Remove a message subscription.

        Args:
            subscription: Subscription to remove
        """
        async with self._subscription_lock:
            agent_subscriptions = self._subscriptions.get(subscription.agent_id, [])
            if subscription in agent_subscriptions:
                subscription.active = False
                agent_subscriptions.remove(subscription)

        self._logger.info(
            "Agent unsubscribed from messages",
            agent_id=subscription.agent_id,
        )

    async def get_messages(
        self, agent_id: str, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """Get pending messages for an agent.

        Args:
            agent_id: Agent identifier
            timeout: Optional timeout for waiting for messages

        Returns:
            List of pending messages

        Raises:
            ValueError: If agent_id is not registered
            TimeoutError: If timeout is exceeded
        """
        if agent_id not in self._orchestrator.agent_handles:
            raise ValueError(f"Agent {agent_id} is not registered")

        if agent_id not in self._message_queues:
            return []

        messages = []
        queue = self._message_queues[agent_id]
        timeout_time = time.time() + (timeout or self.timeout_seconds)

        try:
            # Get all immediately available messages
            while not queue.empty():
                try:
                    message = queue.get_nowait()
                    messages.append(message)
                except asyncio.QueueEmpty:
                    break

            # If no messages and timeout specified, wait for at least one
            if not messages and timeout is not None:
                remaining_timeout = max(0, timeout_time - time.time())
                if remaining_timeout > 0:
                    message = await asyncio.wait_for(
                        queue.get(), timeout=remaining_timeout
                    )
                    messages.append(message)

        except TimeoutError as e:
            if timeout is not None:
                raise TimeoutError(f"No messages received within {timeout}s") from e

        return messages

    async def wait_for_message(
        self,
        agent_id: str,
        message_type: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Wait for a specific message for an agent.

        Args:
            agent_id: Agent identifier
            message_type: Optional specific message type to wait for
            timeout: Optional timeout for waiting

        Returns:
            The received message

        Raises:
            ValueError: If agent_id is not registered
            TimeoutError: If timeout is exceeded
        """
        if agent_id not in self._orchestrator.agent_handles:
            raise ValueError(f"Agent {agent_id} is not registered")

        if agent_id not in self._message_queues:
            raise ValueError(f"No message queue for agent {agent_id}")

        queue = self._message_queues[agent_id]
        timeout_time = time.time() + (timeout or self.timeout_seconds)

        while True:
            remaining_timeout = timeout_time - time.time() if timeout else None

            try:
                message = await asyncio.wait_for(queue.get(), timeout=remaining_timeout)

                # Check if this is the message type we're waiting for
                if message_type is None or message.get("type") == message_type:
                    return message
                else:
                    # Put message back and continue waiting
                    await queue.put(message)

            except TimeoutError as e:
                raise TimeoutError(
                    f"No message of type '{message_type}' received within {timeout or self.timeout_seconds}s"
                ) from e

    async def _start_message_delivery(self) -> None:
        """Start message delivery for all registered agents."""
        async with self._subscription_lock:
            for agent_id in self._orchestrator.agent_handles:
                if agent_id not in self._message_queues:
                    self._message_queues[agent_id] = asyncio.Queue()
                    self._subscriptions[agent_id] = []

                if agent_id not in self._delivery_tasks:
                    self._delivery_tasks[agent_id] = asyncio.create_task(
                        self._message_delivery_loop(agent_id)
                    )

    async def _message_delivery_loop(self, agent_id: str) -> None:
        """Background loop for delivering messages to an agent.

        Args:
            agent_id: Agent identifier
        """
        self._logger.info("Message delivery loop started", agent_id=agent_id)

        try:
            agent_handle = self._orchestrator.agent_handles[agent_id]

            while not self._shutdown_event.is_set():
                try:
                    # Get next observation from orchestrator
                    observation = await asyncio.wait_for(
                        agent_handle.next_observation(),
                        timeout=1.0,  # Short timeout to check shutdown regularly
                    )

                    # Convert observation to message format
                    message = await self._observation_to_message(agent_id, observation)

                    if message:
                        # Check subscriptions and deliver message
                        await self._deliver_message_to_agent(agent_id, message)

                except TimeoutError:
                    # Normal timeout, continue loop
                    continue
                except Exception as e:
                    self._logger.error(
                        "Error in message delivery loop",
                        agent_id=agent_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    # Brief pause before retrying
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            self._logger.info("Message delivery loop cancelled", agent_id=agent_id)
        except Exception as e:
            self._logger.error(
                "Message delivery loop failed",
                agent_id=agent_id,
                error=str(e),
                error_type=type(e).__name__,
            )
        finally:
            self._logger.info("Message delivery loop ended", agent_id=agent_id)

    async def _observation_to_message(
        self, agent_id: str, observation: Any
    ) -> dict[str, Any] | None:
        """Convert an observation to a message format.

        Args:
            agent_id: Agent identifier
            observation: Observation data

        Returns:
            Message dictionary or None if no message should be generated
        """
        try:
            # Handle different observation formats
            if isinstance(observation, dict):
                obs_dict = observation
            else:
                # Convert observation object to dict
                obs_dict = {
                    "view_seq": getattr(observation, "view_seq", 0),
                    "patches": getattr(observation, "patches", []),
                    "context_digest": getattr(observation, "context_digest", ""),
                    "schema_version": getattr(observation, "schema_version", "1.0.0"),
                }

            # Create message from observation
            message = {
                "type": "observation_update",
                "agent_id": agent_id,
                "timestamp": time.time(),
                "data": obs_dict,
            }

            return message

        except Exception as e:
            self._logger.error(
                "Failed to convert observation to message",
                agent_id=agent_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    async def _deliver_message_to_agent(
        self, agent_id: str, message: dict[str, Any]
    ) -> None:
        """Deliver a message to an agent based on their subscriptions.

        Args:
            agent_id: Agent identifier
            message: Message to deliver
        """
        message_type = message.get("type", "unknown")

        async with self._subscription_lock:
            subscriptions = self._subscriptions.get(agent_id, [])
            matching_subscriptions = [
                sub for sub in subscriptions if sub.matches(message_type)
            ]

        if not matching_subscriptions:
            # No subscriptions match, skip delivery
            return

        # Add to message queue
        if agent_id in self._message_queues:
            try:
                await self._message_queues[agent_id].put(message)
            except Exception as e:
                self._logger.error(
                    "Failed to queue message",
                    agent_id=agent_id,
                    message_type=message_type,
                    error=str(e),
                )

        # Call handlers for matching subscriptions
        for subscription in matching_subscriptions:
            if subscription.handler:
                try:
                    if asyncio.iscoroutinefunction(subscription.handler):
                        await subscription.handler(agent_id, message)
                    else:
                        subscription.handler(agent_id, message)
                except Exception as e:
                    self._logger.error(
                        "Message handler failed",
                        agent_id=agent_id,
                        message_type=message_type,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

    async def shutdown(self) -> None:
        """Shutdown the message facade and clean up resources.

        Cancels all delivery tasks and shuts down the orchestrator.
        """
        self._logger.info("Shutting down message facade")

        # Signal shutdown to all delivery loops
        self._shutdown_event.set()

        # Cancel all delivery tasks
        for _agent_id, task in list(self._delivery_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._delivery_tasks.clear()

        # Clear message queues
        async with self._subscription_lock:
            self._message_queues.clear()
            self._subscriptions.clear()

        # Shutdown orchestrator
        await self._orchestrator.shutdown()

        self._logger.info("Message facade shutdown complete")

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

    def get_agent_subscriptions(self, agent_id: str) -> list[MessageSubscription]:
        """Get all active subscriptions for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of active subscriptions

        Raises:
            ValueError: If agent_id is not registered
        """
        if agent_id not in self._orchestrator.agent_handles:
            raise ValueError(f"Agent {agent_id} is not registered")

        return [sub for sub in self._subscriptions.get(agent_id, []) if sub.active]

    def get_message_queue_size(self, agent_id: str) -> int:
        """Get the current size of an agent's message queue.

        Args:
            agent_id: Agent identifier

        Returns:
            Number of pending messages

        Raises:
            ValueError: If agent_id is not registered
        """
        if agent_id not in self._orchestrator.agent_handles:
            raise ValueError(f"Agent {agent_id} is not registered")

        if agent_id not in self._message_queues:
            return 0

        return self._message_queues[agent_id].qsize()

    def __repr__(self) -> str:
        """String representation of the message facade."""
        return (
            f"MessageFacade(world_id={self.world_id}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"agents={len(self._orchestrator.agent_handles)}, "
            f"active_subscriptions={sum(len(subs) for subs in self._subscriptions.values())})"
        )
