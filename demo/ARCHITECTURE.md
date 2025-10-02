# Architecture Documentation

This document explains the architectural design decisions behind the Gunn multi-agent battle demo, providing insight into why specific patterns and technologies were chosen.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architectural Principles](#architectural-principles)
3. [Component Design](#component-design)
4. [Technology Choices](#technology-choices)
5. [Design Patterns](#design-patterns)
6. [Performance Considerations](#performance-considerations)
7. [Security and Reliability](#security-and-reliability)
8. [Future Extensibility](#future-extensibility)

## System Overview

### High-Level Architecture

The battle demo follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Pygame Renderer│  │   UI Components │  │ Event Handler│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────┐
│                    Backend Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   FastAPI Server│  │  Game Manager   │  │ WebSocket   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Simulation Layer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │Battle Orchestrator│ │ Battle Mechanics│  │AI Decision  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Gunn Core                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Orchestrator  │  │ Observation     │  │ Event Log   │ │
│  │                 │  │ Policies        │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Design Philosophy

The architecture is built around several key principles:

1. **Separation of Concerns**: Each layer has a distinct responsibility
2. **Event-Driven Design**: Components communicate through events and effects
3. **Deterministic Behavior**: Reproducible simulations through controlled randomness
4. **Scalable Concurrency**: Efficient handling of multiple agents
5. **Educational Value**: Clear patterns for learning multi-agent systems

## Architectural Principles

### 1. Layered Architecture

**Decision**: Use a strict layered architecture with unidirectional dependencies.

**Rationale**:
- **Maintainability**: Clear boundaries make the system easier to understand and modify
- **Testability**: Each layer can be tested independently
- **Reusability**: Lower layers can be reused in different contexts
- **Educational Value**: Demonstrates proper software architecture patterns

**Implementation**:
```python
# Frontend only depends on Backend API
frontend → backend_api

# Backend depends on Simulation layer
backend → simulation_layer

# Simulation layer depends on Gunn Core
simulation_layer → gunn_core

# No circular dependencies allowed
```

### 2. Event-Driven Communication

**Decision**: Use events and effects for component communication instead of direct method calls.

**Rationale**:
- **Decoupling**: Components don't need direct references to each other
- **Auditability**: All interactions are logged as events
- **Replay Capability**: Events can be replayed for debugging
- **Extensibility**: New components can subscribe to existing events

**Implementation**:
```python
# Instead of direct calls:
# agent.take_damage(25)

# Use effects:
effect = {
    "kind": "AgentDamaged",
    "payload": {"agent_id": "agent_1", "damage": 25},
    "source_id": "combat_system"
}
await orchestrator.emit_effect(effect)
```

### 3. Immutable State with Controlled Mutations

**Decision**: Use immutable data structures with controlled mutation points.

**Rationale**:
- **Concurrency Safety**: Immutable data prevents race conditions
- **Debugging**: State changes are explicit and traceable
- **Determinism**: Controlled mutations ensure reproducible behavior
- **Performance**: Structural sharing reduces memory usage

**Implementation**:
```python
# State is immutable
@dataclass(frozen=True)
class Agent:
    agent_id: str
    health: int
    position: tuple[float, float]

# Mutations go through controlled channels
def apply_damage(agent: Agent, damage: int) -> Agent:
    return dataclasses.replace(agent, health=max(0, agent.health - damage))
```

## Component Design

### 1. Battle Orchestrator

**Purpose**: Wraps Gunn's orchestrator to provide game-specific functionality.

**Design Decisions**:

- **Composition over Inheritance**: Wraps rather than extends Gunn's orchestrator
- **Single Responsibility**: Only handles orchestration, delegates game logic
- **Configuration Driven**: Behavior controlled through configuration objects

```python
class BattleOrchestrator:
    def __init__(self, config: BattleConfig):
        # Compose Gunn orchestrator rather than inherit
        self.orchestrator = Orchestrator(config.to_gunn_config())
        self.battle_mechanics = BattleMechanics(config)
        self.ai_decision_maker = AIDecisionMaker(config)
```

**Rationale**:
- Keeps game-specific logic separate from Gunn's core
- Allows easy swapping of underlying orchestrator
- Maintains clear API boundaries

### 2. Observation Policies

**Purpose**: Control what information each agent can observe.

**Design Decisions**:

- **Strategy Pattern**: Different policies for different game modes
- **Team-Based Filtering**: Separate visibility rules for teammates vs enemies
- **Configurable Parameters**: Vision range, communication range, etc.

```python
class BattleObservationPolicy(ObservationPolicy):
    def __init__(self, team: str, vision_range: float):
        self.team = team
        self.vision_range = vision_range
    
    def filter_world_state(self, world_state: WorldState, agent_id: str) -> View:
        # Implement team-based filtering logic
        pass
```

**Rationale**:
- Demonstrates Gunn's partial observation capabilities
- Provides realistic fog of war mechanics
- Shows how to implement custom observation logic

### 3. AI Decision Making

**Purpose**: Generate intelligent agent decisions using OpenAI's structured outputs.

**Design Decisions**:

- **Structured Schemas**: Use Pydantic models for type safety
- **Fallback Mechanisms**: Handle API failures gracefully
- **Concurrent Processing**: Make decisions for multiple agents simultaneously

```python
class AIDecisionMaker:
    async def make_decision(self, agent_id: str, observation: dict) -> AgentDecision:
        try:
            # Use OpenAI structured outputs
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=AgentDecision
            )
            return response.choices[0].message.parsed
        except Exception:
            # Fallback to safe default
            return self._create_fallback_decision(agent_id)
```

**Rationale**:
- Demonstrates integration with modern AI APIs
- Shows how to handle AI failures in production systems
- Provides educational example of structured AI outputs

### 4. Battle Mechanics

**Purpose**: Implement game rules and combat calculations.

**Design Decisions**:

- **Pure Functions**: Combat calculations are stateless
- **Configurable Parameters**: Damage, healing, etc. can be tuned
- **Deterministic Randomness**: Use seeded random for reproducibility

```python
class BattleMechanics:
    def calculate_damage(self, attacker: Agent, target: Agent, distance: float) -> int:
        # Pure function - no side effects
        base_damage = self.config.attack_damage
        # Apply modifiers based on weapon condition, distance, etc.
        return int(base_damage * modifiers)
```

**Rationale**:
- Makes testing easier with pure functions
- Allows easy balancing through configuration
- Demonstrates proper game mechanics design

## Technology Choices

### 1. FastAPI for Backend

**Decision**: Use FastAPI instead of Flask, Django, or other frameworks.

**Rationale**:
- **Async Support**: Native async/await support for concurrent operations
- **Type Safety**: Automatic validation using Pydantic models
- **Documentation**: Auto-generated OpenAPI documentation
- **Performance**: High performance with minimal overhead
- **WebSocket Support**: Built-in WebSocket support for real-time updates

**Trade-offs**:
- ✅ Modern async patterns
- ✅ Excellent type safety
- ✅ Great documentation
- ❌ Newer ecosystem (fewer plugins)
- ❌ Learning curve for developers new to async

### 2. Pygame for Frontend

**Decision**: Use Pygame instead of web-based frontend or other game engines.

**Rationale**:
- **Simplicity**: Easy to set up and understand
- **Educational Value**: Shows how to build custom visualizations
- **Performance**: Direct hardware access for smooth rendering
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Python Integration**: No language switching required

**Trade-offs**:
- ✅ Simple to understand and modify
- ✅ Good performance for 2D graphics
- ✅ No web browser dependencies
- ❌ Limited to desktop platforms
- ❌ More basic UI capabilities than web frameworks

### 3. OpenAI Structured Outputs

**Decision**: Use OpenAI's structured outputs instead of text parsing or other AI APIs.

**Rationale**:
- **Type Safety**: Guaranteed valid JSON responses
- **Reliability**: No parsing errors or malformed responses
- **Performance**: Faster than text parsing approaches
- **Developer Experience**: Easy to work with typed responses

**Trade-offs**:
- ✅ Reliable structured responses
- ✅ Excellent type safety
- ✅ Good performance
- ❌ Vendor lock-in to OpenAI
- ❌ Requires paid API access

### 4. Pydantic for Data Models

**Decision**: Use Pydantic v2 for all data models and validation.

**Rationale**:
- **Type Safety**: Runtime type checking and validation
- **Performance**: Fast serialization/deserialization
- **Integration**: Works well with FastAPI and OpenAI
- **Documentation**: Self-documenting schemas

**Trade-offs**:
- ✅ Excellent type safety and validation
- ✅ Great performance
- ✅ Good ecosystem integration
- ❌ Learning curve for complex validation
- ❌ Can be verbose for simple models

## Design Patterns

### 1. Observer Pattern for Effects

**Implementation**: Effects are broadcast to all interested components.

```python
class EffectProcessor:
    def __init__(self):
        self.handlers = {
            "AgentDamaged": [self.update_health, self.check_death, self.log_damage],
            "TeamMessage": [self.store_message, self.broadcast_to_team]
        }
    
    async def process_effect(self, effect: Effect):
        handlers = self.handlers.get(effect["kind"], [])
        await asyncio.gather(*[handler(effect) for handler in handlers])
```

**Benefits**:
- Easy to add new effect handlers
- Decoupled components
- Extensible system

### 2. Strategy Pattern for Observation Policies

**Implementation**: Different observation strategies for different game modes.

```python
class ObservationPolicyFactory:
    @staticmethod
    def create_policy(game_mode: str, team: str) -> ObservationPolicy:
        if game_mode == "full_visibility":
            return FullVisibilityPolicy()
        elif game_mode == "fog_of_war":
            return BattleObservationPolicy(team)
        else:
            raise ValueError(f"Unknown game mode: {game_mode}")
```

**Benefits**:
- Easy to switch between different visibility rules
- Testable in isolation
- Configurable behavior

### 3. Command Pattern for Intents

**Implementation**: Agent actions are represented as command objects.

```python
@dataclass
class MoveIntent:
    agent_id: str
    target_position: tuple[float, float]
    
    def execute(self, world_state: WorldState) -> list[Effect]:
        # Implement movement logic
        pass

@dataclass
class AttackIntent:
    agent_id: str
    target_id: str
    
    def execute(self, world_state: WorldState) -> list[Effect]:
        # Implement attack logic
        pass
```

**Benefits**:
- Uniform interface for all actions
- Easy to queue and batch operations
- Supports undo/redo functionality

### 4. Factory Pattern for AI Decisions

**Implementation**: Create different types of AI decision makers.

```python
class AIDecisionFactory:
    @staticmethod
    def create_decision_maker(ai_type: str, config: dict) -> AIDecisionMaker:
        if ai_type == "openai":
            return OpenAIDecisionMaker(config)
        elif ai_type == "random":
            return RandomDecisionMaker(config)
        elif ai_type == "scripted":
            return ScriptedDecisionMaker(config)
        else:
            raise ValueError(f"Unknown AI type: {ai_type}")
```

**Benefits**:
- Easy to test with different AI types
- Supports A/B testing
- Configurable AI behavior

## Performance Considerations

### 1. Concurrent Agent Processing

**Challenge**: Process multiple agents efficiently without blocking.

**Solution**: Use asyncio for concurrent processing with proper coordination.

```python
async def process_agents_concurrently(self, agent_ids: list[str]):
    # Create tasks for all agents
    tasks = [self._process_single_agent(agent_id) for agent_id in agent_ids]
    
    # Process with timeout and error handling
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    for agent_id, result in zip(agent_ids, results):
        if isinstance(result, Exception):
            logger.error(f"Agent {agent_id} processing failed: {result}")
            # Use fallback behavior
```

**Benefits**:
- Scales with number of agents
- Handles failures gracefully
- Maintains responsiveness

### 2. Memory Management

**Challenge**: Prevent memory leaks in long-running simulations.

**Solution**: Implement bounded collections and periodic cleanup.

```python
class BoundedHistory:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.items = deque(maxlen=max_size)
    
    def add(self, item):
        self.items.append(item)
        # Automatically removes oldest items when full
```

**Benefits**:
- Prevents unbounded memory growth
- Maintains recent history for debugging
- Configurable memory usage

### 3. Efficient State Updates

**Challenge**: Update world state efficiently for real-time performance.

**Solution**: Use incremental updates and dirty tracking.

```python
class WorldStateManager:
    def __init__(self):
        self.dirty_entities = set()
    
    def update_entity(self, entity_id: str, changes: dict):
        # Only update changed fields
        entity = self.entities[entity_id]
        for key, value in changes.items():
            if getattr(entity, key) != value:
                setattr(entity, key, value)
                self.dirty_entities.add(entity_id)
    
    def get_updates(self) -> dict:
        # Return only changed entities
        updates = {eid: self.entities[eid] for eid in self.dirty_entities}
        self.dirty_entities.clear()
        return updates
```

**Benefits**:
- Reduces network traffic
- Improves rendering performance
- Scales with number of changes, not total entities

## Security and Reliability

### 1. Input Validation

**Implementation**: Validate all inputs at API boundaries.

```python
class GameAction(BaseModel):
    action_type: Literal["move", "attack", "heal", "repair", "communicate"]
    agent_id: str = Field(pattern=r"^[a-zA-Z0-9_]+$")
    payload: dict = Field(max_length=1000)
    
    @field_validator("agent_id")
    def validate_agent_exists(cls, v, info):
        # Validate agent exists and is controlled by current player
        return v
```

**Benefits**:
- Prevents injection attacks
- Ensures data consistency
- Provides clear error messages

### 2. Error Handling

**Implementation**: Comprehensive error handling with graceful degradation.

```python
async def safe_ai_decision(self, agent_id: str) -> AgentDecision:
    try:
        return await self.ai_decision_maker.make_decision(agent_id, observation)
    except openai.RateLimitError:
        # Wait and retry
        await asyncio.sleep(1.0)
        return await self._retry_decision(agent_id)
    except openai.APIError:
        # Use fallback decision
        return self._create_fallback_decision(agent_id)
    except Exception as e:
        # Log error and use safe default
        logger.error(f"Unexpected error in AI decision: {e}")
        return self._create_safe_default_decision(agent_id)
```

**Benefits**:
- System continues running despite errors
- Provides debugging information
- Maintains user experience

### 3. Rate Limiting

**Implementation**: Protect against API abuse and resource exhaustion.

```python
class RateLimiter:
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
    
    async def acquire(self) -> bool:
        now = time.time()
        
        # Remove old requests
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        
        # Check if we can make a new request
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
```

**Benefits**:
- Prevents resource exhaustion
- Protects external APIs
- Maintains system stability

## Future Extensibility

### 1. Plugin Architecture

**Design**: Support for pluggable components.

```python
class PluginManager:
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, name: str, plugin: Plugin):
        self.plugins[name] = plugin
    
    async def execute_hooks(self, hook_name: str, *args, **kwargs):
        for plugin in self.plugins.values():
            if hasattr(plugin, hook_name):
                await getattr(plugin, hook_name)(*args, **kwargs)
```

**Benefits**:
- Easy to add new features
- Supports third-party extensions
- Maintains core system simplicity

### 2. Configuration System

**Design**: Hierarchical configuration with environment overrides.

```python
class ConfigManager:
    def __init__(self):
        self.config = self._load_default_config()
        self._apply_environment_overrides()
        self._apply_file_overrides()
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
```

**Benefits**:
- Easy deployment configuration
- Supports different environments
- Maintains backward compatibility

### 3. Modular Game Mechanics

**Design**: Separate game mechanics into composable modules.

```python
class GameMechanicsRegistry:
    def __init__(self):
        self.mechanics = {}
    
    def register(self, name: str, mechanic: GameMechanic):
        self.mechanics[name] = mechanic
    
    def get_mechanics_for_action(self, action_type: str) -> list[GameMechanic]:
        return [m for m in self.mechanics.values() if m.handles(action_type)]
```

**Benefits**:
- Easy to add new game rules
- Supports different game modes
- Maintains clean separation of concerns

This architecture provides a solid foundation for building sophisticated multi-agent simulations while maintaining clarity, performance, and extensibility. The design decisions are driven by the need to demonstrate Gunn's capabilities while providing educational value for developers learning multi-agent systems.