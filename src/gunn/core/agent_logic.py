"""Abstract base class for asynchronous agent behavior.

This module provides the AsyncAgentLogic interface that defines how agents
process observations and generate intents in an asynchronous observe-think-act loop.
"""

from abc import ABC, abstractmethod

from gunn.schemas.messages import View
from gunn.schemas.types import Intent


class AsyncAgentLogic(ABC):
    """Abstract base class for implementing asynchronous agent behavior.

    This interface defines how agents should process observations and generate
    intents in an asynchronous observe-think-act loop. Implementations should
    handle the "think" phase of the loop, taking current observations and
    deciding what action (if any) to take next.

    Requirements addressed:
    - 14.3: Agent thinks using LLM to generate next action based on observations
    - 14.5: Agents operate independently without synchronization barriers
    - 14.8: Agents coordinate through observed actions and communication
    """

    @abstractmethod
    async def process_observation(
        self, observation: View, agent_id: str
    ) -> Intent | list[Intent] | None:
        """Process current observation and return intent(s) to execute, or None to wait.

        This method is called by the agent's async loop whenever a new observation
        is available. The implementation should analyze the observation and decide
        whether to generate intent(s) (actions) or wait for more information.

        **New in v2.0**: Can now return multiple intents for concurrent execution
        (e.g., physical action + communication).

        Args:
            observation: Current view of the world state for this agent
            agent_id: ID of the agent processing this observation

        Returns:
            - Single Intent to execute
            - List of Intents to execute concurrently
            - None if no action should be taken

        Examples:
            >>> # Single action (backward compatible)
            >>> return move_intent
            
            >>> # Multiple concurrent actions (new feature)
            >>> return [move_intent, speak_intent]
            
            >>> # No action
            >>> return None

        Raises:
            Exception: Implementation-specific errors during processing
        """
        pass

    async def on_loop_start(self, agent_id: str) -> None:
        """Called when the agent's async loop starts.

        Override this method to perform any initialization needed when
        the agent begins its observe-think-act loop.

        Args:
            agent_id: ID of the agent starting the loop
        """
        pass

    async def on_loop_stop(self, agent_id: str) -> None:
        """Called when the agent's async loop stops.

        Override this method to perform any cleanup needed when
        the agent stops its observe-think-act loop.

        Args:
            agent_id: ID of the agent stopping the loop
        """
        pass

    async def on_error(self, agent_id: str, error: Exception) -> bool:
        """Called when an error occurs during observation processing.

        Override this method to handle errors that occur during the
        observe-think-act loop. Return True to continue the loop,
        False to stop it.

        Args:
            agent_id: ID of the agent that encountered the error
            error: The exception that occurred

        Returns:
            True to continue the loop, False to stop it
        """
        return False  # Default: stop on error


class SimpleAgentLogic(AsyncAgentLogic):
    """Simple example implementation of AsyncAgentLogic.

    This is a basic implementation that can be used for testing or as
    a starting point for more complex agent behaviors.
    """

    def __init__(self, action_probability: float = 0.1):
        """Initialize simple agent logic.

        Args:
            action_probability: Probability of taking an action on each observation
        """
        self.action_probability = action_probability
        self._observation_count = 0

    async def process_observation(
        self, observation: View, agent_id: str
    ) -> Intent | None:
        """Simple logic that occasionally generates a speak intent."""
        import random
        import uuid

        self._observation_count += 1

        # Occasionally generate a speak intent
        if random.random() < self.action_probability:
            return {
                "kind": "Speak",
                "payload": {
                    "text": f"Hello from {agent_id} (observation #{self._observation_count})",
                    "agent_id": agent_id,
                },
                "context_seq": observation.view_seq,
                "req_id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }

        return None  # No action

    async def on_loop_start(self, agent_id: str) -> None:
        """Reset observation count when loop starts."""
        self._observation_count = 0

    async def on_error(self, agent_id: str, error: Exception) -> bool:
        """Log error and continue loop."""
        print(f"Agent {agent_id} encountered error: {error}")
        return True  # Continue on error
