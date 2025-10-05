# Collaborative Behavior Patterns

## Overview

The collaborative behavior patterns module provides helper methods and patterns for detecting collaboration opportunities, implementing coordination patterns, and enabling emergent collaborative behavior without explicit synchronization.

## Key Components

### CollaborationDetector

Detects collaboration opportunities from agent observations:

- **Spatial Clustering**: Identifies when multiple agents are in close proximity
- **Task Collaboration**: Detects tasks that require multiple agents
- **Group Conversations**: Identifies active multi-agent conversations
- **Helping Opportunities**: Detects when agents need assistance
- **Resource Sharing**: Identifies shared resource coordination opportunities

```python
from gunn.core.collaborative_patterns import CollaborationDetector

detector = CollaborationDetector(proximity_threshold=15.0)
opportunities = detector.detect_opportunities(observation, agent_id)

for opp in opportunities:
    print(f"{opp.opportunity_type}: {opp.description}")
    print(f"Priority: {opp.priority}/10")
    print(f"Involved agents: {opp.involved_agents}")
```

### CoordinationPatternTracker

Tracks active coordination patterns between agents:

- **Pattern Management**: Start, update, and complete coordination patterns
- **Participant Tracking**: Track which agents are coordinating
- **Pattern History**: Maintain history of completed patterns
- **Partner Discovery**: Find agents coordinating with a given agent

```python
from gunn.core.collaborative_patterns import CoordinationPatternTracker

tracker = CoordinationPatternTracker()

# Start a coordination pattern
pattern_id = tracker.start_pattern(
    pattern_type="following",
    initiator="agent_a",
    participants=["agent_a", "agent_b"],
    metadata={"distance": 5.0}
)

# Check if agent is coordinating
is_coordinating = tracker.is_agent_coordinating("agent_a")

# Get coordination partners
partners = tracker.get_coordination_partners("agent_a")

# Update pattern status
tracker.update_pattern(pattern_id, status="completed")
```

### Helper Functions

#### detect_following_pattern

Detects if one agent is following another:

```python
from gunn.core.collaborative_patterns import detect_following_pattern

is_following = detect_following_pattern(
    observation,
    follower_id="agent_a",
    leader_id="agent_b",
    distance_threshold=5.0
)
```

#### suggest_collaborative_action

Suggests an action to participate in a collaboration opportunity:

```python
from gunn.core.collaborative_patterns import suggest_collaborative_action

suggestion = suggest_collaborative_action(
    opportunity,
    agent_id="agent_a",
    agent_position=(0.0, 0.0, 0.0)
)

print(f"Action: {suggestion['action_type']}")
print(f"Reasoning: {suggestion['reasoning']}")
if 'text' in suggestion:
    print(f"Message: {suggestion['text']}")
```

## Collaboration Opportunity Types

### 1. Spatial Clustering

Detected when multiple agents are in close proximity.

**Use Cases:**
- Initiating group conversations
- Coordinating movement
- Forming teams

**Example:**
```python
# Opportunity detected when 3+ agents are within 15 units
{
    "opportunity_type": "spatial_clustering",
    "description": "3 agents are in close proximity",
    "involved_agents": ["agent_a", "agent_b", "agent_c"],
    "location": (5.0, 5.0, 0.0),
    "priority": 5
}
```

### 2. Task Collaboration

Detected when tasks require multiple agents to complete.

**Use Cases:**
- Collaborative problem solving
- Multi-agent task completion
- Skill complementarity

**Example:**
```python
# Opportunity detected for collaborative task
{
    "opportunity_type": "task_collaboration",
    "description": "Task 'Build structure' requires collaboration",
    "involved_agents": ["agent_a", "agent_b"],
    "priority": 7,
    "metadata": {
        "task_id": "task_1",
        "difficulty": "hard"
    }
}
```

### 3. Group Conversation

Detected when multiple agents are actively communicating.

**Use Cases:**
- Joining ongoing discussions
- Multi-party negotiations
- Information sharing

**Example:**
```python
# Opportunity detected for active conversation
{
    "opportunity_type": "group_conversation",
    "description": "Active conversation with 2 participants",
    "involved_agents": ["agent_a", "agent_b", "agent_c"],
    "priority": 6,
    "metadata": {
        "speaker_count": 2,
        "topics": ["Hello everyone!", "Hi there!"]
    }
}
```

### 4. Helping Behavior

Detected when agents express need for assistance.

**Use Cases:**
- Providing assistance
- Unblocking stuck agents
- Skill sharing

**Example:**
```python
# Opportunity detected when agent needs help
{
    "opportunity_type": "helping",
    "description": "Agent agent_b may need assistance: stuck",
    "involved_agents": ["agent_a", "agent_b"],
    "priority": 8,
    "metadata": {
        "target_agent": "agent_b",
        "help_reason": "I'm stuck and need help!"
    }
}
```

### 5. Resource Sharing

Detected when shared resources are available for coordination.

**Use Cases:**
- Coordinated resource access
- Resource allocation
- Shared facility management

**Example:**
```python
# Opportunity detected for resource sharing
{
    "opportunity_type": "resource_sharing",
    "description": "Resource 'Information Hub' available for sharing",
    "involved_agents": ["agent_a", "agent_b"],
    "location": (2.0, 2.0, 0.0),
    "priority": 5,
    "metadata": {
        "resource_id": "resource_1",
        "resource_type": "resource"
    }
}
```

## Coordination Pattern Types

### Following Pattern

One agent follows another agent's movements.

**Characteristics:**
- Spatial proximity maintained
- Leader-follower relationship
- Dynamic distance tracking

**Example:**
```python
pattern_id = tracker.start_pattern(
    pattern_type="following",
    initiator="agent_follower",
    participants=["agent_follower", "agent_leader"],
    metadata={"distance": 5.0}
)
```

### Helping Pattern

One agent assists another agent with a task or problem.

**Characteristics:**
- Helper-helpee relationship
- Task-oriented coordination
- Completion-based lifecycle

**Example:**
```python
pattern_id = tracker.start_pattern(
    pattern_type="helping",
    initiator="agent_helper",
    participants=["agent_helper", "agent_helpee"],
    metadata={"help_reason": "stuck on task"}
)
```

### Group Conversation Pattern

Multiple agents engage in a shared conversation.

**Characteristics:**
- Multi-party participation
- Topic-based coordination
- Dynamic participant list

**Example:**
```python
pattern_id = tracker.start_pattern(
    pattern_type="group_conversation",
    initiator="agent_a",
    participants=["agent_a", "agent_b", "agent_c"],
    metadata={"topic": "task planning"}
)
```

### Task Collaboration Pattern

Multiple agents coordinate to complete a shared task.

**Characteristics:**
- Goal-oriented coordination
- Skill complementarity
- Progress tracking

**Example:**
```python
pattern_id = tracker.start_pattern(
    pattern_type="task_collaboration",
    initiator="agent_a",
    participants=["agent_a", "agent_b"],
    metadata={"task_id": "task_1", "difficulty": "hard"}
)
```

### Resource Sharing Pattern

Multiple agents coordinate access to a shared resource.

**Characteristics:**
- Resource-centric coordination
- Access scheduling
- Conflict avoidance

**Example:**
```python
pattern_id = tracker.start_pattern(
    pattern_type="resource_sharing",
    initiator="agent_a",
    participants=["agent_a", "agent_b"],
    metadata={"resource_id": "resource_1"}
)
```

## Integration with Agents

### Using CollaborationDetector in Agent Logic

```python
from gunn.core.conversational_agent import ConversationalAgent
from gunn.core.collaborative_patterns import (
    CollaborationDetector,
    suggest_collaborative_action
)

class CollaborativeAgent(ConversationalAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = CollaborationDetector(proximity_threshold=20.0)
    
    async def process_observation(self, observation, agent_id):
        # Detect collaboration opportunities
        opportunities = self.detector.detect_opportunities(observation, agent_id)
        
        if opportunities:
            # Get action suggestion for highest priority opportunity
            opp = max(opportunities, key=lambda o: o.priority)
            suggestion = suggest_collaborative_action(
                opp, agent_id, self.current_position
            )
            
            # Convert suggestion to intent
            if suggestion["action_type"] == "speak":
                return self._create_speak_intent(
                    suggestion["text"], observation.view_seq, agent_id
                )
        
        # Fall back to default behavior
        return await super().process_observation(observation, agent_id)
```

### Tracking Coordination Patterns

```python
from gunn.core.collaborative_patterns import CoordinationPatternTracker

class CoordinatingAgent(ConversationalAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = CoordinationPatternTracker()
        self.active_patterns = []
    
    async def process_observation(self, observation, agent_id):
        # Check if already coordinating
        if self.tracker.is_agent_coordinating(agent_id):
            partners = self.tracker.get_coordination_partners(agent_id)
            print(f"Currently coordinating with: {partners}")
        
        # Process observation normally
        return await super().process_observation(observation, agent_id)
```

## Requirements Addressed

This module addresses the following requirements:

- **3.6**: Multi-agent task coordination without explicit synchronization
- **4.6**: Collaborative opportunities through observation
- **14.9**: Coordination through observed actions and communication

## Design Principles

### 1. Observation-Based Detection

All collaboration opportunities are detected through observation, not explicit coordination protocols:

- Agents observe the world state
- Detector analyzes observations for patterns
- Opportunities emerge from observed state

### 2. No Explicit Synchronization

Coordination happens without synchronization barriers:

- Agents make independent decisions
- Coordination emerges from shared observations
- No blocking or waiting for other agents

### 3. Emergent Behavior

Collaborative behaviors emerge naturally:

- Simple detection rules lead to complex behaviors
- Multiple patterns can coexist
- Scales naturally with agent count

### 4. Flexible Patterns

Coordination patterns are flexible and extensible:

- New pattern types can be added easily
- Patterns can be customized per use case
- Metadata supports pattern-specific information

## Examples

See `examples/collaborative_behavior_demo.py` for a comprehensive demonstration of all collaborative behavior patterns.

## Testing

Unit tests: `src/gunn/core/test_collaborative_patterns.py`
Integration tests: `tests/integration/test_collaborative_behavior.py`

Run tests:
```bash
uv run pytest src/gunn/core/test_collaborative_patterns.py -v
uv run pytest tests/integration/test_collaborative_behavior.py -v
```

## Future Enhancements

Potential future enhancements:

1. **Machine Learning Integration**: Learn collaboration patterns from agent behavior
2. **Dynamic Thresholds**: Adjust detection thresholds based on context
3. **Pattern Prediction**: Predict likely collaboration opportunities
4. **Performance Metrics**: Track collaboration effectiveness
5. **Conflict Resolution**: Handle competing collaboration opportunities
6. **Hierarchical Patterns**: Support nested coordination patterns
