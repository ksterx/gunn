"""
Battle agent implementation using AsyncAgentLogic.

This module defines the BattleAgent class that integrates with Gunn's
async agent framework to provide independent, asynchronous decision-making
for each agent in the battle simulation.
"""

import asyncio
import logging
from typing import Any

from gunn.core.agent_logic import AsyncAgentLogic

from ..shared.models import Agent, BattleWorldState
from ..shared.schemas import AgentDecision
from .ai_decision import AIDecisionMaker

logger = logging.getLogger(__name__)


class BattleAgent(AsyncAgentLogic):
    """Battle agent with AI decision-making capabilities.

    This agent operates independently with its own async loop, making
    decisions based on observations and submitting intents to the orchestrator.
    Each agent runs on its own timeline, allowing for realistic asynchronous
    behavior where faster agents can act more frequently than slower ones.
    """

    def __init__(
        self,
        agent_id: str,
        ai_decision_maker: AIDecisionMaker,
        world_state: BattleWorldState,
        agent_handle=None,
    ):
        """Initialize battle agent.

        Args:
            agent_id: Unique identifier for this agent
            ai_decision_maker: AI decision maker for generating actions
            world_state: Shared world state reference for context
            agent_handle: Optional agent handle for submitting intents (set later)
        """
        self.agent_id = agent_id
        self.ai_decision_maker = ai_decision_maker
        self.world_state = world_state
        self.agent_handle = agent_handle

        # Agent-specific state
        self.observations_processed = 0
        self.decisions_made = 0
        self.last_decision: AgentDecision | None = None
        self.current_view_seq = 0  # Track current view sequence number

        logger.info(f"Initialized BattleAgent {agent_id}")

    async def process_observation(
        self, observation: dict[str, Any], agent_id: str
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Process observation and generate action decision with concurrent intents.

        This method now supports returning multiple intents for concurrent execution,
        allowing agents to perform physical actions and communicate simultaneously.

        Args:
            observation: Current observation from the orchestrator
            agent_id: ID of the agent processing the observation

        Returns:
            - Single intent dict for single action (backward compatible)
            - List of intent dicts for concurrent actions (e.g., move + speak)
            - None to wait without taking action
        """
        try:
            self.observations_processed += 1

            # Convert View object to dict format if needed
            if hasattr(observation, "model_dump"):
                observation_dict = observation.model_dump()
            elif hasattr(observation, "__dict__"):
                observation_dict = observation.__dict__
            else:
                observation_dict = observation

            # Update current view_seq from observation (View object)
            # AgentHandle.get_current_observation() now returns View with view_seq
            if hasattr(observation, "view_seq"):
                self.current_view_seq = observation.view_seq
            elif isinstance(observation_dict, dict):
                self.current_view_seq = observation_dict.get("view_seq", 0)

            logger.debug(
                f"Agent {self.agent_id} received observation with view_seq={self.current_view_seq}"
            )

            # Get agent data from world state
            agent_data = self.world_state.agents.get(self.agent_id)
            if not agent_data:
                logger.warning(f"Agent {self.agent_id} not found in world state")
                return None

            # Check if agent is alive
            if not agent_data.is_alive():
                logger.info(f"Agent {self.agent_id} is dead, stopping decision-making")
                return None

            # Build team context for AI decision maker
            team_context = self._build_team_context(agent_data)

            # Generate AI decision
            decision = await self.ai_decision_maker.make_decision(
                self.agent_id, observation_dict, self.world_state, team_context
            )

            self.last_decision = decision
            self.decisions_made += 1

            logger.debug(
                f"Agent {self.agent_id} made decision: {decision.primary_action.action_type}"
            )

            # Get latest observation to get most up-to-date view_seq
            if self.agent_handle:
                try:
                    latest_obs = await self.agent_handle.get_current_observation()
                    if hasattr(latest_obs, "view_seq"):
                        self.current_view_seq = latest_obs.view_seq
                        logger.debug(
                            f"Agent {self.agent_id} updated to latest view_seq={self.current_view_seq} before intent submission"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to get latest observation for {self.agent_id}: {e}"
                    )

            # Convert decision to intent(s) - NEW: Support concurrent actions
            intents = []

            # Primary action intent
            primary_intent = self._action_to_intent(decision.primary_action)
            if primary_intent:
                intents.append(primary_intent)

            # Communication intent (if present)
            if decision.communication:
                comm_intent = self._communication_to_intent(decision.communication)
                if comm_intent:
                    intents.append(comm_intent)

            # Return based on number of intents
            if len(intents) == 0:
                return None
            elif len(intents) == 1:
                logger.debug(
                    f"Agent {self.agent_id} returning single intent: {intents[0]['kind']}"
                )
                return intents[0]
            else:
                logger.info(
                    f"Agent {self.agent_id} returning {len(intents)} concurrent intents: "
                    f"{[i['kind'] for i in intents]}"
                )
                return intents

        except Exception as e:
            logger.error(f"Error processing observation for {self.agent_id}: {e}")
            return None

    def _build_team_context(self, agent: Agent) -> dict[str, Any]:
        """Build team context for AI decision making.

        Args:
            agent: Current agent data

        Returns:
            Team context dictionary
        """
        teammates = [
            aid
            for aid, a in self.world_state.agents.items()
            if a.team == agent.team and aid != self.agent_id and a.is_alive()
        ]

        return {
            "team": agent.team,
            "teammates": teammates,
            "strategy": "coordinate with teammates and eliminate enemies",
        }

    def _communication_to_intent(
        self, communication: "CommunicateAction"
    ) -> dict[str, Any] | None:
        """Create a communication intent dictionary from CommunicateAction.

        Args:
            communication: CommunicateAction from AI decision

        Returns:
            Intent dict for Speak action, or None if creation fails
        """
        import time
        import uuid

        try:
            current_seq = self.current_view_seq

            intent = {
                "kind": "Speak",
                "payload": {
                    "message": communication.message,
                    "urgency": communication.urgency,
                },
                "context_seq": current_seq,
                "req_id": f"{self.agent_id}_{time.time()}_speak_{uuid.uuid4().hex[:8]}",
                "agent_id": self.agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }
            logger.info(
                f"Agent {self.agent_id} creating communication intent: {communication.message[:50]}..."
            )
            return intent

        except Exception as e:
            logger.warning(
                f"Failed to create communication intent for {self.agent_id}: {e}"
            )
            return None

    def _action_to_intent(self, action) -> dict[str, Any] | None:
        """Convert primary action to Gunn intent.

        Args:
            action: Primary action (MoveAction, AttackAction, etc.)

        Returns:
            Intent dictionary for submission to orchestrator, or None if invalid
        """
        import time
        import uuid

        from ..shared.schemas import (
            AttackAction,
            HealAction,
            MoveAction,
            RepairAction,
        )

        timestamp = time.time()
        base_req_id = f"{self.agent_id}_{timestamp}"

        # Create primary action intent with current view_seq
        if isinstance(action, MoveAction):
            return {
                "kind": "Move",
                "payload": {
                    "to": action.target_position,  # Gunn expects 'to' field
                    "reason": action.reason,
                },
                "context_seq": self.current_view_seq,
                "req_id": f"{base_req_id}_move_{uuid.uuid4().hex[:8]}",
                "agent_id": self.agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }
        elif isinstance(action, AttackAction):
            return {
                "kind": "Attack",
                "payload": {
                    "target_agent_id": action.target_agent_id,
                    "reason": action.reason,
                },
                "context_seq": self.current_view_seq,
                "req_id": f"{base_req_id}_attack_{uuid.uuid4().hex[:8]}",
                "agent_id": self.agent_id,
                "priority": 1,
                "schema_version": "1.0.0",
            }
        elif isinstance(action, HealAction):
            return {
                "kind": "Heal",
                "payload": {
                    "target_agent_id": action.target_agent_id or self.agent_id,
                    "reason": action.reason,
                },
                "context_seq": self.current_view_seq,
                "req_id": f"{base_req_id}_heal_{uuid.uuid4().hex[:8]}",
                "agent_id": self.agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }
        elif isinstance(action, RepairAction):
            return {
                "kind": "Repair",
                "payload": {"reason": action.reason},
                "context_seq": self.current_view_seq,
                "req_id": f"{base_req_id}_repair_{uuid.uuid4().hex[:8]}",
                "agent_id": self.agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }
        else:
            logger.warning(f"Unknown action type: {type(action)}")
            return None

    async def on_loop_start(self, agent_id: str) -> None:
        """Called when agent loop starts.

        Args:
            agent_id: ID of the agent starting the loop
        """
        logger.info(f"Agent {agent_id} loop starting")

    async def on_loop_stop(self, agent_id: str) -> None:
        """Called when agent loop stops.

        Args:
            agent_id: ID of the agent stopping the loop
        """
        logger.info(
            f"Agent {agent_id} loop stopped. "
            f"Observations: {self.observations_processed}, "
            f"Decisions: {self.decisions_made}"
        )

    async def on_error(self, agent_id: str, error: Exception) -> bool:
        """Called when an error occurs in the agent loop.

        Args:
            agent_id: ID of the agent that encountered the error
            error: The exception that occurred

        Returns:
            True to continue the loop, False to stop
        """
        logger.error(f"Agent {agent_id} encountered error: {error}")
        return True  # Continue running despite errors

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dictionary containing agent statistics
        """
        return {
            "agent_id": self.agent_id,
            "observations_processed": self.observations_processed,
            "decisions_made": self.decisions_made,
            "last_decision": (
                {
                    "action_type": self.last_decision.primary_action.action_type,
                    "confidence": self.last_decision.confidence,
                }
                if self.last_decision
                else None
            ),
        }
