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
    ) -> dict[str, Any] | None:
        """Process observation and generate action decision.

        This method is called by the async agent loop whenever new observations
        are available. It uses the AI decision maker to generate strategic
        decisions based on the current battlefield situation.

        Args:
            observation: Current observation from the orchestrator
            agent_id: ID of the agent processing the observation

        Returns:
            Intent dict to submit, or None to wait
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

            # Update current view_seq from observation
            # First try to get from observation object
            if hasattr(observation, "view_seq"):
                self.current_view_seq = observation.view_seq
            elif isinstance(observation_dict, dict):
                self.current_view_seq = observation_dict.get("view_seq", 0)

            # Also sync with agent_handle's view_seq
            if self.agent_handle and hasattr(self.agent_handle, "view_seq"):
                # Update agent_handle's view_seq to match observation
                self.agent_handle.view_seq = self.current_view_seq

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
                        # Also update agent_handle's view_seq
                        self.agent_handle.view_seq = latest_obs.view_seq
                        logger.debug(
                            f"Agent {self.agent_id} updated to latest view_seq={self.current_view_seq} before intent submission"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to get latest observation for {self.agent_id}: {e}"
                    )

            # Convert decision to intent with current view_seq
            intent = self._decision_to_intent(decision)

            # Handle communication separately if present (as background task)
            if decision.communication and self.agent_handle:
                asyncio.create_task(self._send_communication(decision.communication))

            return intent

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

    async def _send_communication(self, communication: "CommunicateAction") -> None:
        """Send team communication through orchestrator.

        Args:
            communication: Communication action from AI decision
        """
        import time
        import uuid

        try:
            # Get latest observation to get most up-to-date view_seq
            current_seq = self.current_view_seq
            if self.agent_handle:
                try:
                    latest_obs = await self.agent_handle.get_current_observation()
                    if hasattr(latest_obs, "view_seq"):
                        current_seq = latest_obs.view_seq
                        # Also update agent_handle's view_seq
                        self.agent_handle.view_seq = latest_obs.view_seq
                        logger.debug(
                            f"Agent {self.agent_id} using latest view_seq={current_seq} for communication"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to get latest observation for communication: {e}"
                    )

            # Create communication intent
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
                f"Agent {self.agent_id} sending message: {communication.message[:50]}..."
            )

            # Submit intent through agent handle
            if self.agent_handle:
                await self.agent_handle.submit_intent(intent)
                logger.debug(f"Communication intent submitted for {self.agent_id}")
            else:
                logger.warning(f"No agent handle available for {self.agent_id}")

        except Exception as e:
            logger.warning(f"Failed to send communication for {self.agent_id}: {e}")

    def _decision_to_intent(self, decision: AgentDecision) -> dict[str, Any]:
        """Convert AI decision to Gunn intent.

        Args:
            decision: AI-generated decision

        Returns:
            Intent dictionary for submission to orchestrator
        """
        import time
        import uuid

        from ..shared.schemas import (
            AttackAction,
            HealAction,
            MoveAction,
            RepairAction,
        )

        action = decision.primary_action
        timestamp = time.time()
        base_req_id = f"{self.agent_id}_{timestamp}"

        # Create primary action intent with current view_seq
        if isinstance(action, MoveAction):
            intent = {
                "kind": "Move",
                "payload": {
                    "to": action.target_position,  # Gunn expects 'to' field
                    "reason": action.reason,
                },
                "context_seq": self.current_view_seq,  # Use current view_seq
                "req_id": f"{base_req_id}_move_{uuid.uuid4().hex[:8]}",
                "agent_id": self.agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }
        elif isinstance(action, AttackAction):
            intent = {
                "kind": "Attack",
                "payload": {
                    "target_agent_id": action.target_agent_id,
                    "reason": action.reason,
                },
                "context_seq": self.current_view_seq,  # Use current view_seq
                "req_id": f"{base_req_id}_attack_{uuid.uuid4().hex[:8]}",
                "agent_id": self.agent_id,
                "priority": 1,
                "schema_version": "1.0.0",
            }
        elif isinstance(action, HealAction):
            intent = {
                "kind": "Heal",
                "payload": {
                    "target_agent_id": action.target_agent_id or self.agent_id,
                    "reason": action.reason,
                },
                "context_seq": self.current_view_seq,  # Use current view_seq
                "req_id": f"{base_req_id}_heal_{uuid.uuid4().hex[:8]}",
                "agent_id": self.agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }
        elif isinstance(action, RepairAction):
            intent = {
                "kind": "Repair",
                "payload": {"reason": action.reason},
                "context_seq": self.current_view_seq,  # Use current view_seq
                "req_id": f"{base_req_id}_repair_{uuid.uuid4().hex[:8]}",
                "agent_id": self.agent_id,
                "priority": 0,
                "schema_version": "1.0.0",
            }
        else:
            logger.warning(f"Unknown action type: {type(action)}")
            return None

        # Note: Communication is handled separately if needed
        # For now, focusing on primary action only
        return intent

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
