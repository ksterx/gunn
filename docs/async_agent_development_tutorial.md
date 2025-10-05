# Asynchronous Agent Development Tutorial

## Introduction

This tutorial teaches you how to develop asynchronous agents for the Gunn multi-agent simulation system. You'll learn how to create agents that operate independently, respond to observations, collaborate with other agents, and leverage all the system's guarantees.

## Prerequisites

- Python 3.13+
- Basic understanding of async/await in Python
- Familiarity with the Gunn core concepts (Orchestrator, AgentHandle, ObservationPolicy)

## Table of Contents

1. [Basic Agent Structure](#basic-agent-structure)
2. [Implementing AsyncAgentLogic](#implementing-asyncagentlogic)
3. [Processing Observations](#processing-observations)
4. [Generating Intents](#generating-intents)
5. [Handling Spatial Awareness](#handling-spatial-awareness)
6. [Building Conversational Agents](#building-conversational-agents)
7. [Implementing Collaborative Behavior](#implementing-collaborative-behavior)
8. [Using System Guarantees](#using-system-guarantees)
9. [Error Handling and Recovery](#error-handling-and-recovery)
10. [Testing and Debugging](#testing-and-debugging)

## Basic Agent Structure

Every asynchronous agent in Gunn follows the observe-think-act loop pattern:

```python
from gunn import AsyncAgentLogic, Orchestrator, OrchestratorConfig
from gunn.schemas.messages import View
from gunn.schemas.types import Intent
import uuid

class MyAgent(AsyncAgentLogic):
    """Basic asynchronous agent implementation."""
    
    def __init__(self, name: str):
        self.name = name
        self.action_count = 0
    
    async def process_observation(self, observation: View, agent_id: str) -> Intent | None:
        """Process observation and return intent, or None to wait."""
        self.action_count += 1
        
        # Analyze observation
        visible_entities = observation.visible_entities
        
        # Decide on action
        if self._should_act(visible_entities):
            return self._generate_intent(agent_id, observation)
        
        return None  # No action this time
    
    def _should_act(self, visible_entities: dict) -> bool:
        """Determine if agent should take action."""
        # Your decision logic here
        return True
    
    def _generate_intent(self, agent_id: str, observation: View) -> Intent:
        """Generate an intent based on observation."""
        return {
            "kind": "Speak",
            "payload": {"text": f"Hello from {self.name}!", "agent_id": agent_id},
            "context_seq": observation.view_seq,
            "req_id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "priority": 0,
            "schema_version": "1.0.0"
        }
```

## Implementing AsyncAgentLogic

The `AsyncAgentLogic` interface requires implementing `process_observation`:

```python
from abc import ABC, abstractmethod

class AsyncAgentLogic(ABC):
    """Abstract base class for asynchronous agent behavior."""
    
    @abstractmethod
    async def process_observation(self, observation: View, agent_id: str) -> Intent | None:
        """Process observation and return intent, or None to wait.
        
        Args:
            observation: Current view of the world state
            agent_id: ID of this agent
            
        Returns:
            Intent to execute, or None to wait for next observation
        """
        pass
```

### Optional Lifecycle Hooks

You can override these methods for additional control:

```python
class MyAgent(AsyncAgentLogic):
    async def on_loop_start(self, agent_id: str) -> None:
        """Called when agent loop starts."""
        print(f"Agent {agent_id} starting...")
    
    async def on_loop_stop(self, agent_id: str) -> None:
        """Called when agent loop stops."""
        print(f"Agent {agent_id} stopping...")
    
    async def on_error(self, agent_id: str, error: Exception) -> bool:
        """Handle errors during processing.
        
        Returns:
            True to continue, False to stop the loop
        """
        print(f"Error in agent {agent_id}: {error}")
        return True  # Continue on error
```

## Processing Observations

Observations contain filtered world state based on your observation policy:

```python
async def process_observation(self, observation: View, agent_id: str) -> Intent | None:
    """Process observation with detailed analysis."""
    
    # Extract visible entities
    visible_entities = observation.visible_entities
    
    # Find other agents
    other_agents = [
        (entity_id, entity_data)
        for entity_id, entity_data in visible_entities.items()
        if entity_id != agent_id and entity_data.get("type") == "agent"
    ]
    
    # Find landmarks
    landmarks = [
        (entity_id, entity_data)
        for entity_id, entity_data in visible_entities.items()
        if entity_data.get("type") == "landmark"
    ]
    
    # Check for recent messages
    messages = [
        entity_data
        for entity_data in visible_entities.values()
        if entity_data.get("type") == "message"
    ]
    
    # Make decision based on what we observe
    if messages:
        return self._respond_to_message(messages[-1], agent_id, observation)
    elif other_agents:
        return self._interact_with_agents(other_agents, agent_id, observation)
    elif landmarks:
        return self._explore_landmark(landmarks[0], agent_id, observation)
    
    return None  # Wait for more interesting observations
```

## Generating Intents

Intents are the actions your agent wants to perform:

### Speak Intent

```python
def _generate_speak_intent(self, text: str, agent_id: str, observation: View) -> Intent:
    """Generate a speak intent."""
    return {
        "kind": "Speak",
        "payload": {
            "text": text,
            "agent_id": agent_id,
            "timestamp": time.time()
        },
        "context_seq": observation.view_seq,
        "req_id": str(uuid.uuid4()),
        "agent_id": agent_id,
        "priority": 0,
        "schema_version": "1.0.0"
    }
```

### Move Intent

```python
def _generate_move_intent(
    self,
    target_position: list[float],
    agent_id: str,
    observation: View
) -> Intent:
    """Generate a move intent."""
    # Get current position
    current_pos = observation.visible_entities.get(agent_id, {}).get("position", [0, 0, 0])
    
    return {
        "kind": "Move",
        "payload": {
            "from": current_pos,
            "to": target_position,
            "agent_id": agent_id,
            "speed": 5.0  # Optional movement speed
        },
        "context_seq": observation.view_seq,
        "req_id": str(uuid.uuid4()),
        "agent_id": agent_id,
        "priority": 0,
        "schema_version": "1.0.0"
    }
```

### Custom Intent

```python
def _generate_custom_intent(
    self,
    action_type: str,
    payload: dict,
    agent_id: str,
    observation: View,
    priority: int = 0
) -> Intent:
    """Generate a custom intent."""
    return {
        "kind": "Custom",
        "payload": {
            "action_type": action_type,
            **payload,
            "agent_id": agent_id
        },
        "context_seq": observation.view_seq,
        "req_id": str(uuid.uuid4()),
        "agent_id": agent_id,
        "priority": priority,
        "schema_version": "1.0.0"
    }
```

## Handling Spatial Awareness

Agents can use spatial information to make decisions:

```python
class SpatialAgent(AsyncAgentLogic):
    """Agent with spatial awareness."""
    
    def __init__(self, name: str, vision_range: float = 30.0):
        self.name = name
        self.vision_range = vision_range
        self.current_position = [0.0, 0.0, 0.0]
    
    async def process_observation(self, observation: View, agent_id: str) -> Intent | None:
        """Process observation with spatial awareness."""
        # Update current position
        if agent_id in observation.visible_entities:
            self.current_position = observation.visible_entities[agent_id].get(
                "position", self.current_position
            )
        
        # Find nearby agents
        nearby_agents = self._find_nearby_agents(observation, agent_id)
        
        if nearby_agents:
            # Move towards the nearest agent
            nearest = min(nearby_agents, key=lambda x: x[1])  # (agent_id, distance)
            return self._move_towards(nearest[0], observation, agent_id)
        else:
            # Explore randomly
            return self._explore_randomly(agent_id, observation)
    
    def _find_nearby_agents(self, observation: View, agent_id: str) -> list[tuple[str, float]]:
        """Find nearby agents with distances."""
        nearby = []
        
        for entity_id, entity_data in observation.visible_entities.items():
            if entity_id == agent_id or entity_data.get("type") != "agent":
                continue
            
            distance = entity_data.get("distance", float('inf'))
            if distance <= self.vision_range:
                nearby.append((entity_id, distance))
        
        return nearby
    
    def _move_towards(self, target_id: str, observation: View, agent_id: str) -> Intent:
        """Move towards target agent."""
        target_pos = observation.visible_entities[target_id].get("position", [0, 0, 0])
        
        # Move halfway towards target
        new_pos = [
            (self.current_position[i] + target_pos[i]) / 2
            for i in range(3)
        ]
        
        return self._generate_move_intent(new_pos, agent_id, observation)
    
    def _explore_randomly(self, agent_id: str, observation: View) -> Intent:
        """Move to random nearby location."""
        import random
        
        new_pos = [
            self.current_position[0] + random.uniform(-10, 10),
            self.current_position[1] + random.uniform(-10, 10),
            0.0
        ]
        
        return self._generate_move_intent(new_pos, agent_id, observation)
```

## Building Conversational Agents

Use the `ConversationalAgent` class or build your own:

```python
from gunn.core.conversational_agent import ConversationalAgent, LLMResponse

class MyLLMClient:
    """Custom LLM client for agent decision-making."""
    
    async def generate_response(
        self,
        context: str,
        personality: str,
        max_tokens: int = 150,
        temperature: float = 0.7
    ) -> LLMResponse:
        """Generate response based on context and personality."""
        # Your LLM integration here
        # This could call OpenAI, Anthropic, or any other LLM service
        
        return LLMResponse(
            action_type="speak",
            text="Generated response based on context",
            reasoning="Responding to conversation"
        )

# Use the conversational agent
llm_client = MyLLMClient()
agent = ConversationalAgent(
    llm_client=llm_client,
    personality="helpful and friendly",
    name="Alice",
    conversation_distance=30.0
)
```

## Implementing Collaborative Behavior

Agents can collaborate by observing and responding to each other:

```python
class CollaborativeAgent(AsyncAgentLogic):
    """Agent that detects and responds to collaboration opportunities."""
    
    def __init__(self, name: str):
        self.name = name
        self.collaboration_threshold = 0.5
        self.active_collaborations = set()
    
    async def process_observation(self, observation: View, agent_id: str) -> Intent | None:
        """Process observation and detect collaboration opportunities."""
        # Check for collaboration requests
        collab_requests = self._detect_collaboration_requests(observation)
        
        if collab_requests:
            # Respond to collaboration request
            return self._respond_to_collaboration(collab_requests[0], agent_id, observation)
        
        # Check if we should initiate collaboration
        if self._should_initiate_collaboration(observation):
            return self._initiate_collaboration(agent_id, observation)
        
        return None
    
    def _detect_collaboration_requests(self, observation: View) -> list[dict]:
        """Detect collaboration requests in observations."""
        requests = []
        
        for entity_data in observation.visible_entities.values():
            if entity_data.get("type") == "message":
                text = entity_data.get("text", "").lower()
                if any(word in text for word in ["collaborate", "help", "together", "coordinate"]):
                    requests.append(entity_data)
        
        return requests
    
    def _should_initiate_collaboration(self, observation: View) -> bool:
        """Determine if we should initiate collaboration."""
        # Count nearby agents
        nearby_agents = sum(
            1 for entity_data in observation.visible_entities.values()
            if entity_data.get("type") == "agent"
        )
        
        # Initiate if we have enough nearby agents
        return nearby_agents >= 2 and len(self.active_collaborations) == 0
    
    def _respond_to_collaboration(
        self,
        request: dict,
        agent_id: str,
        observation: View
    ) -> Intent:
        """Respond to collaboration request."""
        return self._generate_speak_intent(
            f"{self.name} here! I'd love to collaborate. What's the plan?",
            agent_id,
            observation
        )
    
    def _initiate_collaboration(self, agent_id: str, observation: View) -> Intent:
        """Initiate collaboration with nearby agents."""
        return self._generate_speak_intent(
            f"Hey everyone! I think we could work together on this. Anyone interested?",
            agent_id,
            observation
        )
```

## Using System Guarantees

### Delivery Guarantees

```python
class ReliableAgent(AsyncAgentLogic):
    """Agent that uses delivery guarantees."""
    
    async def process_observation(self, observation: View, agent_id: str) -> Intent | None:
        """Process observation with delivery tracking."""
        # The system automatically handles delivery tracking
        # You can access delivery_id from ObservationDelta if needed
        
        # Process observation normally
        intent = self._generate_intent(agent_id, observation)
        
        # Acknowledge observation (optional, done automatically by AgentHandle)
        # await self.agent_handle.ack_observation(observation.delivery_id)
        
        return intent
```

### Completion Confirmation

```python
class ConfirmingAgent(AsyncAgentLogic):
    """Agent that waits for action completion."""
    
    async def process_observation(self, observation: View, agent_id: str) -> Intent | None:
        """Process observation with completion confirmation."""
        # Generate intent
        intent = self._generate_intent(agent_id, observation)
        
        # The AgentHandle automatically waits for completion
        # You can customize timeout in OrchestratorConfig
        
        return intent
```

### Intelligent Staleness Detection

```python
class StaleAwareAgent(AsyncAgentLogic):
    """Agent that benefits from intelligent staleness detection."""
    
    async def process_observation(self, observation: View, agent_id: str) -> Intent | None:
        """Process observation with staleness awareness."""
        # Generate intent based on current observation
        intent = self._generate_move_intent(target_pos, agent_id, observation)
        
        # The system automatically detects if the intent becomes stale
        # due to relevant changes in the world state
        # Stale intents are cancelled and agent gets new observation
        
        return intent
```

## Error Handling and Recovery

```python
class RobustAgent(AsyncAgentLogic):
    """Agent with comprehensive error handling."""
    
    def __init__(self, name: str):
        self.name = name
        self.error_count = 0
        self.max_errors = 10
    
    async def process_observation(self, observation: View, agent_id: str) -> Intent | None:
        """Process observation with error handling."""
        try:
            # Your processing logic
            return self._generate_intent(agent_id, observation)
        
        except ValueError as e:
            # Handle specific errors
            print(f"Value error in {self.name}: {e}")
            return None
        
        except Exception as e:
            # Handle unexpected errors
            print(f"Unexpected error in {self.name}: {e}")
            self.error_count += 1
            
            if self.error_count >= self.max_errors:
                raise  # Stop agent after too many errors
            
            return None
    
    async def on_error(self, agent_id: str, error: Exception) -> bool:
        """Handle errors at loop level."""
        print(f"Loop error in {agent_id}: {error}")
        
        # Decide whether to continue
        if isinstance(error, KeyboardInterrupt):
            return False  # Stop on keyboard interrupt
        
        return True  # Continue on other errors
```

## Testing and Debugging

### Unit Testing

```python
import pytest
from gunn.schemas.messages import View, WorldState

@pytest.mark.asyncio
async def test_agent_responds_to_greeting():
    """Test that agent responds to greetings."""
    agent = MyAgent("TestAgent")
    
    # Create mock observation with greeting
    observation = View(
        agent_id="test_agent",
        view_seq=1,
        visible_entities={
            "other_agent": {
                "type": "message",
                "text": "Hello!",
                "sender": "other_agent"
            }
        },
        visible_relationships={},
        context_digest="test"
    )
    
    # Process observation
    intent = await agent.process_observation(observation, "test_agent")
    
    # Verify response
    assert intent is not None
    assert intent["kind"] == "Speak"
    assert "hello" in intent["payload"]["text"].lower()
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_agent_in_orchestrator():
    """Test agent in full orchestrator environment."""
    # Create orchestrator
    config = OrchestratorConfig(use_in_memory_dedup=True)
    orchestrator = Orchestrator(config)
    await orchestrator.initialize()
    
    try:
        # Create and register agent
        agent_logic = MyAgent("TestAgent")
        policy = SimpleObservationPolicy()
        handle = await orchestrator.register_agent("test_agent", policy)
        
        # Start agent loop
        loop_task = asyncio.create_task(handle.run_async_loop(agent_logic))
        
        # Let it run briefly
        await asyncio.sleep(1.0)
        
        # Stop agent
        handle.stop_async_loop()
        await loop_task
        
        # Verify agent performed actions
        assert agent_logic.action_count > 0
    
    finally:
        await orchestrator.shutdown()
```

### Debugging Tips

1. **Add Logging**
```python
import structlog

logger = structlog.get_logger()

async def process_observation(self, observation: View, agent_id: str) -> Intent | None:
    logger.info("processing_observation",
                agent_id=agent_id,
                view_seq=observation.view_seq,
                visible_entities=len(observation.visible_entities))
    # ... rest of processing
```

2. **Track State**
```python
class DebuggableAgent(AsyncAgentLogic):
    def __init__(self, name: str):
        self.name = name
        self.observation_history = []
        self.intent_history = []
    
    async def process_observation(self, observation: View, agent_id: str) -> Intent | None:
        # Track observations
        self.observation_history.append(observation)
        
        # Generate intent
        intent = self._generate_intent(agent_id, observation)
        
        # Track intents
        if intent:
            self.intent_history.append(intent)
        
        return intent
```

3. **Use Breakpoints**
```python
async def process_observation(self, observation: View, agent_id: str) -> Intent | None:
    # Set breakpoint for debugging
    import pdb; pdb.set_trace()
    
    # Your processing logic
    return self._generate_intent(agent_id, observation)
```

## Best Practices

1. **Keep Processing Fast**: Avoid long-running operations in `process_observation`
2. **Handle None Gracefully**: Return `None` when no action is needed
3. **Use Context Seq**: Always include current `view_seq` in intents
4. **Generate Unique Request IDs**: Use `uuid.uuid4()` for each intent
5. **Implement Error Handling**: Use `on_error` to handle exceptions gracefully
6. **Test Independently**: Unit test agent logic before integration testing
7. **Log Important Events**: Use structured logging for debugging
8. **Monitor Performance**: Track action counts and timing
9. **Respect Spatial Constraints**: Check distances before interacting
10. **Be Collaborative**: Respond to other agents' actions and messages

## Next Steps

- Review the [comprehensive async multi-agent demo](async_multi_agent_demo.md)
- Explore [observation policy customization](intelligent_staleness.md)
- Learn about [delivery guarantees](delivery_guarantees.md)
- Study [completion confirmation](completion_confirmation.md)
- Understand [replay invariance](replay_invariance.md)

## Conclusion

You now have the knowledge to build sophisticated asynchronous agents for the Gunn multi-agent simulation system. Start with simple agents and gradually add more complex behaviors as you become comfortable with the patterns.

Remember: the key to successful asynchronous agents is to keep them independent, responsive, and collaborative. Let the system handle the guarantees while you focus on agent behavior.
