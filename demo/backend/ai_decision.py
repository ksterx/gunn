"""
OpenAI structured output integration for AI decision making.

This module handles communication with OpenAI's API using structured
outputs to generate valid agent decisions with proper error handling
and fallback mechanisms.
"""

import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI

from ..shared.errors import AIDecisionError, OpenAIAPIError, ValidationError
from ..shared.models import Agent, BattleWorldState
from ..shared.schemas import (
    AgentDecision,
    AttackAction,
    HealAction,
    MoveAction,
    RepairAction,
)
from .error_handler import BattleErrorHandler

logger = logging.getLogger(__name__)


class AIDecisionMaker:
    """Handles AI decision making using OpenAI structured outputs with comprehensive error handling."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4.1-mini"):
        """
        Initialize the AI decision maker.

        Args:
            api_key: OpenAI API key. If None, will use environment variable.
            model: OpenAI model to use for structured outputs.
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.request_timeout = 30.0
        self.max_retries = 2
        self.error_handler = BattleErrorHandler()

        # Import performance monitor here to avoid circular imports
        from .performance_monitor import performance_monitor

        self.performance_monitor = performance_monitor

    async def make_decision(
        self,
        agent_id: str,
        observation: dict[str, Any],
        world_state: BattleWorldState,
        team_context: dict[str, Any] | None = None,
    ) -> AgentDecision:
        """
        Generate a structured decision for an agent based on their observation.

        Args:
            agent_id: ID of the agent making the decision
            observation: Current observation data from Gunn
            world_state: Current battle world state
            team_context: Additional team strategy context

        Returns:
            AgentDecision with primary action and optional communication
        """
        try:
            # Monitor decision making performance
            async with self.performance_monitor.monitor_decision_making(agent_id):
                agent = world_state.agents.get(agent_id)
                if not agent:
                    error = AIDecisionError(
                        message=f"Agent {agent_id} not found in world state",
                        agent_id=agent_id,
                        severity="high",
                    )
                    return await self.error_handler.handle_ai_decision_error(
                        error, agent_id, world_state
                    )

                system_prompt = self._build_system_prompt(
                    agent, world_state, team_context
                )
                user_prompt = self._build_observation_prompt(
                    agent, observation, world_state
                )

                # Attempt to get structured decision from OpenAI with error handling
                for attempt in range(self.max_retries + 1):
                    try:
                        response = await asyncio.wait_for(
                            self.client.beta.chat.completions.parse(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt},
                                ],
                                response_format=AgentDecision,
                                temperature=0.7,
                                max_tokens=1000,
                            ),
                            timeout=self.request_timeout,
                        )

                        if response.choices and response.choices[0].message.parsed:
                            decision = response.choices[0].message.parsed

                            # Validate the decision before returning
                            is_valid, validation_error = await self.validate_decision(
                                decision, agent, world_state
                            )

                            if not is_valid:
                                validation_err = ValidationError(
                                    message=f"Invalid decision generated: {validation_error}",
                                    agent_id=agent_id,
                                    validation_type="decision_validation",
                                    invalid_data=decision.model_dump(),
                                )
                                logger.warning(
                                    f"Invalid decision for {agent_id}: {validation_error}"
                                )
                                return (
                                    await self.error_handler.handle_ai_decision_error(
                                        validation_err, agent_id, world_state, attempt
                                    )
                                )

                            logger.info(
                                f"Generated valid decision for {agent_id}: {decision.primary_action.action_type}"
                            )
                            return decision
                        else:
                            logger.warning(
                                f"Empty response from OpenAI for {agent_id}, attempt {attempt + 1}"
                            )

                    except TimeoutError:
                        timeout_error = OpenAIAPIError(
                            message=f"Request timeout after {self.request_timeout}s",
                            agent_id=agent_id,
                            retry_count=attempt,
                            context={"timeout": self.request_timeout},
                        )
                        logger.warning(
                            f"OpenAI request timeout for {agent_id}, attempt {attempt + 1}"
                        )

                        if attempt >= self.max_retries:
                            return await self.error_handler.handle_ai_decision_error(
                                timeout_error, agent_id, world_state, attempt
                            )

                    except Exception as e:
                        api_error = OpenAIAPIError(
                            message=str(e),
                            agent_id=agent_id,
                            retry_count=attempt,
                            api_error=e,
                        )
                        logger.warning(
                            f"OpenAI API error for {agent_id}, attempt {attempt + 1}: {e}"
                        )

                        if attempt >= self.max_retries:
                            return await self.error_handler.handle_ai_decision_error(
                                api_error, agent_id, world_state, attempt
                            )

                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

                # All attempts failed, create final fallback
                final_error = OpenAIAPIError(
                    message="All OpenAI API attempts failed",
                    agent_id=agent_id,
                    retry_count=self.max_retries,
                )
                return await self.error_handler.handle_ai_decision_error(
                    final_error, agent_id, world_state, self.max_retries
                )

        except Exception as e:
            unexpected_error = AIDecisionError(
                message=f"Unexpected error in decision making: {e!s}",
                agent_id=agent_id,
                api_error=e,
                severity="high",
            )
            logger.error(f"Unexpected error in make_decision for {agent_id}: {e}")
            return await self.error_handler.handle_ai_decision_error(
                unexpected_error, agent_id, world_state
            )

    def _build_system_prompt(
        self,
        agent: Agent,
        world_state: BattleWorldState,
        team_context: dict[str, Any] | None = None,
    ) -> str:
        """Build system prompt with agent role and team strategy."""
        teammates = [
            aid
            for aid, a in world_state.agents.items()
            if a.team == agent.team and aid != agent.agent_id
        ]

        team_info = team_context or {}
        team_strategy = team_info.get(
            "strategy", "coordinate with teammates and eliminate enemies"
        )

        return f"""You are {agent.agent_id}, a tactical combat agent on {agent.team}.

Your teammates are: {", ".join(teammates) if teammates else "none"}

COMBAT RULES:
- Health: 0-100 points, you die at 0
- Weapons degrade with use: excellent → good → damaged → broken
- Broken weapons cannot attack and must be repaired at your team's forge
- Healing takes time and leaves you vulnerable
- You can only communicate with teammates (enemies cannot hear you)
- Vision range is limited - you may not see all enemies
- Attack range: {agent.attack_range}, Vision range: {agent.vision_range}

STRATEGIC PRIORITIES:
1. Survive and eliminate enemy team
2. Coordinate with teammates through communication
3. Maintain weapon condition by visiting forge when needed
4. Use tactical positioning and cover

TEAM STRATEGY: {team_strategy}

DECISION MAKING:
- Consider your health ({agent.health}/100), weapon condition ({agent.weapon_condition.value}), and position
- Communicate important information to teammates
- Balance aggression with survival
- Adapt to changing battlefield conditions
- If weapon is broken, prioritize getting to your team's forge

Make tactical decisions that help your team win while keeping yourself alive."""

    def _build_observation_prompt(
        self, agent: Agent, observation: dict[str, Any], world_state: BattleWorldState
    ) -> str:
        """Build observation prompt from current game state."""
        visible_entities = observation.get("visible_entities", {})

        # Extract information about visible entities
        teammates = []
        enemies = []
        map_locations = []

        for entity_id, entity in visible_entities.items():
            if entity.get("type") == "map_location":
                map_locations.append(
                    f"- {entity_id}: {entity.get('location_type')} at {entity.get('position')}"
                )
            elif "agent" in str(entity_id) and entity_id != agent.agent_id:
                if entity.get("team") == agent.team:
                    teammates.append(
                        f"- {entity_id}: Health {entity.get('health', '?')}, "
                        f"Position {entity.get('position', '?')}, Status {entity.get('status', '?')}"
                    )
                else:
                    enemies.append(
                        f"- {entity_id}: Position {entity.get('position', '?')}, "
                        f"Status {entity.get('status', '?')}"
                    )

        # Get recent team communications
        recent_messages = world_state.get_recent_team_messages(agent.team, 5)
        team_comms = []
        for msg in recent_messages:
            if msg.sender_id != agent.agent_id:  # Don't show own messages
                team_comms.append(f'- {msg.sender_id}: "{msg.message}" ({msg.urgency})')

        prompt = f"""CURRENT SITUATION:

YOUR STATUS:
- Health: {agent.health}/100
- Position: {agent.position}
- Weapon Condition: {agent.weapon_condition.value}
- Status: {agent.status.value}
- Team: {agent.team}

TEAMMATES VISIBLE:
{chr(10).join(teammates) if teammates else "- None visible"}

ENEMIES VISIBLE:
{chr(10).join(enemies) if enemies else "- None visible"}

MAP LOCATIONS:
{chr(10).join(map_locations) if map_locations else "- None visible"}

RECENT TEAM COMMUNICATIONS:
{chr(10).join(team_comms) if team_comms else "- No recent messages"}

GAME STATUS: {world_state.game_status} (Time: {world_state.game_time:.1f}s)

Based on this situation, make your tactical decision. Consider both immediate actions and team communication.
Remember: You can perform one primary action AND optionally send a team message simultaneously."""

        return prompt

    def get_error_statistics(self) -> dict[str, Any]:
        """Get error handling statistics from the error handler."""
        return self.error_handler.get_error_statistics()

    def reset_error_tracking(self) -> None:
        """Reset error tracking in the error handler."""
        self.error_handler.reset_error_tracking()

    async def validate_decision(
        self, decision: AgentDecision, agent: Agent, world_state: BattleWorldState
    ) -> tuple[bool, str]:
        """
        Validate that a decision is legal given current game state.

        Args:
            decision: The decision to validate
            agent: The agent making the decision
            world_state: Current world state

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            action = decision.primary_action

            # Basic agent state checks
            if not agent.is_alive():
                return False, "Agent is not alive"

            # Validate specific action types
            if isinstance(action, AttackAction):
                if not agent.can_attack():
                    return False, "Agent cannot attack (weapon broken or agent dead)"

                target = world_state.agents.get(action.target_agent_id)
                if not target:
                    return False, f"Target agent {action.target_agent_id} not found"

                if target.team == agent.team:
                    return False, "Cannot attack teammate"

                if not target.is_alive():
                    return False, "Target is already dead"

            elif isinstance(action, HealAction):
                if action.target_agent_id:
                    target = world_state.agents.get(action.target_agent_id)
                    if not target:
                        return False, f"Heal target {action.target_agent_id} not found"

                    if target.team != agent.team:
                        return False, "Cannot heal enemy agent"

            elif isinstance(action, RepairAction):
                if agent.weapon_condition.value == "excellent":
                    return False, "Weapon is already in excellent condition"

            elif isinstance(action, MoveAction):
                x, y = action.target_position
                if x < 0 or y < 0 or x > 1000 or y > 1000:
                    return False, "Target position is out of bounds"

            # Validate communication if present
            if decision.communication:
                if len(decision.communication.message.strip()) == 0:
                    return False, "Communication message cannot be empty"

            return True, "Decision is valid"

        except Exception as e:
            return False, f"Validation error: {e!s}"

    async def batch_make_decisions(
        self,
        agent_observations: dict[str, dict[str, Any]],
        world_state: BattleWorldState,
        team_contexts: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, AgentDecision]:
        """
        Generate decisions for multiple agents concurrently.

        Args:
            agent_observations: Map of agent_id to observation data
            world_state: Current battle world state
            team_contexts: Optional team-specific context for each agent

        Returns:
            Map of agent_id to AgentDecision
        """
        team_contexts = team_contexts or {}

        # Monitor concurrent processing performance
        async with self.performance_monitor.monitor_concurrent_processing(
            len(agent_observations)
        ):
            # Create tasks for concurrent decision making
            tasks = []
            agent_ids = []

            for agent_id, observation in agent_observations.items():
                team_context = team_contexts.get(agent_id)
                task = self.make_decision(
                    agent_id, observation, world_state, team_context
                )
                tasks.append(task)
                agent_ids.append(agent_id)

            # Execute all decisions concurrently with error handling
            try:
                decisions = await asyncio.gather(*tasks, return_exceptions=True)

                # Separate successful results from errors
                successful_results = {}
                errors = {}

                for agent_id, decision in zip(agent_ids, decisions, strict=False):
                    if isinstance(decision, Exception):
                        errors[agent_id] = decision
                    else:
                        successful_results[agent_id] = decision

                # Handle any errors that occurred
                if errors:
                    logger.warning(
                        f"Batch decision errors for agents: {list(errors.keys())}"
                    )
                    combined_results = (
                        await self.error_handler.handle_concurrent_processing_error(
                            errors, successful_results
                        )
                    )

                    # Convert error results back to decisions
                    for agent_id, result in combined_results.items():
                        if (
                            isinstance(result, dict)
                            and result.get("status") == "fallback"
                        ):
                            # Create fallback decision for failed agent
                            combined_results[
                                agent_id
                            ] = await self.error_handler.handle_ai_decision_error(
                                Exception(result.get("error", "Unknown error")),
                                agent_id,
                                world_state,
                            )

                    return combined_results

                return successful_results

            except Exception as e:
                batch_error = AIDecisionError(
                    message=f"Critical error in batch decision making: {e!s}",
                    agent_id="batch_operation",
                    api_error=e,
                    severity="critical",
                )
                logger.error(f"Critical error in batch decision making: {e}")

                # Return fallback decisions for all agents
                fallback_results = {}
                for agent_id in agent_ids:
                    fallback_results[
                        agent_id
                    ] = await self.error_handler.handle_ai_decision_error(
                        batch_error, agent_id, world_state
                    )

                return fallback_results
