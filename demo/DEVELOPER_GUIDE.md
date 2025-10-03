# Gunn Integration Developer Guide

This guide explains how the multi-agent battle demo integrates with Gunn's core simulation framework, providing patterns and examples for building your own multi-agent applications.

## Table of Contents

1. [Gunn Core Concepts](#gunn-core-concepts)
2. [Integration Architecture](#integration-architecture)
3. [Observation Policies](#observation-policies)
4. [Intent Processing](#intent-processing)
5. [Effect Handling](#effect-handling)
6. [World State Management](#world-state-management)
7. [Performance Considerations](#performance-considerations)
8. [Best Practices](#best-practices)

## Gunn Core Concepts

### Orchestrator
The central coordinator that manages agent interactions, world state, and event processing.

```python
from gunn import Orchestrator, OrchestratorConfig

# Configure the orchestrator
config = OrchestratorConfig(
    max_agents=6,                    # Maximum concurrent agents
    staleness_threshold=1,           # View staleness tolerance
    debounce_ms=50.0,               # Intent debouncing
    deadline_ms=3000.0,             # Processing deadline
    token_budget=500,               # Resource allocation
    backpressure_policy="defer",    # Overload handling
    default_priority=0              # Default agent priority
)

orchestrator = Orchestrator(config, world_id="battle_demo")
```

### World State
Represents the current state of the simulation environment.

```python
from gunn.schemas.messages import WorldState

# World state contains:
# - entities: Dict of all simulation entities
# - spatial_index: Spatial positioning data
# - metadata: Additional context information
# - view_seq: Version tracking for updates
```

### Agents and Views
Agents receive filtered views of the world state based on observation policies.

```python
from gunn.schemas.messages import View

# Each agent receives a personalized view containing:
# - visible_entities: Entities the agent can observe
# - visible_relationships: Relationships between entities
# - context_digest: Hash for change detection
```

## Integration Architecture

### 1. Orchestrator Wrapper

The `BattleOrchestrator` class wraps Gunn's orchestrator to provide game-specific functionality:

```python
class BattleOrchestrator:
    """Wrapper around Gunn orchestrator for battle simulation"""
    
    def __init__(self):
        self.config = OrchestratorConfig(
            max_agents=6,
            staleness_threshold=1,
            debounce_ms=50.0,
            deadline_ms=3000.0,
            token_budget=500,
            backpressure_policy="defer",
            default_priority=0
        )
        
        self.orchestrator = Orchestrator(self.config, world_id="battle_demo")
        self.world_state = BattleWorldState()
        self.ai_decision_maker = AIDecisionMaker()
        self.battle_mechanics = BattleMechanics()
        
    async def initialize(self):
        """Initialize orchestrator and set up battle world"""
        await self.orchestrator.initialize()
        await self._setup_battle_world()
        await self._register_agents()
```

### 2. Agent Registration

Each agent is registered with a team-specific observation policy:

```python
async def _register_agents(self):
    """Register agents with team-based observation policies"""
    for agent_id, agent in self.world_state.agents.items():
        # Create team-specific observation policy
        policy = BattleObservationPolicy(
            team=agent.team,
            vision_range=agent.vision_range,
            communication_range=agent.communication_range
        )
        
        # Register with Gunn orchestrator
        await self.orchestrator.register_agent(agent_id, policy)
```

### 3. World State Synchronization

Keep Gunn's world state synchronized with your game state:

```python
async def _sync_world_state(self):
    """Synchronize battle state with Gunn's world state"""
    gunn_entities = {}
    spatial_index = {}
    
    # Add agents to Gunn's world state
    for agent_id, agent in self.world_state.agents.items():
        gunn_entities[agent_id] = {
            "type": "agent",
            "team": agent.team,
            "position": agent.position,
            "health": agent.health,
            "status": agent.status.value,
            "weapon_condition": agent.weapon_condition.value,
            "vision_range": agent.vision_range,
            "attack_range": agent.attack_range
        }
        spatial_index[agent_id] = (*agent.position, 0.0)
    
    # Add map locations
    for loc_id, location in self.world_state.map_locations.items():
        gunn_entities[loc_id] = {
            "type": "map_location",
            "location_type": location.location_type.value,
            "position": location.position,
            "radius": location.radius
        }
        spatial_index[loc_id] = (*location.position, 0.0)
    
    # Update Gunn's world state
    self.orchestrator.world_state.entities = gunn_entities
    self.orchestrator.world_state.spatial_index = spatial_index
    self.orchestrator.world_state.metadata = {
        "team_scores": self.world_state.team_scores,
        "game_time": self.world_state.game_time,
        "game_status": self.world_state.game_status,
        "team_communications": self.world_state.last_communication
    }
```

## Observation Policies

### Team-Based Filtering

The `BattleObservationPolicy` implements team-based visibility with fog of war:

```python
class BattleObservationPolicy(ObservationPolicy):
    """Team-based observation policy with fog of war"""
    
    def __init__(self, team: str, vision_range: float = 30.0, 
                 communication_range: float = 50.0):
        self.team = team
        self.vision_range = vision_range
        self.communication_range = communication_range
    
    def filter_world_state(self, world_state: WorldState, agent_id: str) -> View:
        """Filter world state based on team membership and vision"""
        agent = world_state.entities.get(agent_id)
        if not agent:
            return self._empty_view(agent_id)
        
        visible_entities = {}
        agent_pos = world_state.spatial_index.get(agent_id, (0, 0, 0))
        
        # Always see teammates
        for entity_id, entity in world_state.entities.items():
            if entity.get("team") == self.team:
                visible_entities[entity_id] = entity
            elif entity.get("type") == "agent":
                # Check vision range for enemies
                entity_pos = world_state.spatial_index.get(entity_id, (0, 0, 0))
                distance = self._calculate_distance(agent_pos, entity_pos)
                
                if distance <= self.vision_range:
                    # Partial information about enemies
                    visible_entities[entity_id] = {
                        "type": "agent",
                        "team": entity.get("team"),
                        "position": entity.get("position"),
                        "status": entity.get("status"),
                        # Hide detailed information like health/weapon condition
                    }
        
        # Always see map locations
        for entity_id, entity in world_state.entities.items():
            if entity.get("type") == "map_location":
                visible_entities[entity_id] = entity
        
        # Include team communications
        team_messages = world_state.metadata.get("team_communications", {}).get(self.team, [])
        if team_messages:
            visible_entities["team_communications"] = {
                "type": "communications",
                "messages": team_messages[-10:]  # Last 10 messages
            }
        
        return View(
            agent_id=agent_id,
            view_seq=0,  # Will be set by Gunn
            visible_entities=visible_entities,
            visible_relationships={},
            context_digest=""  # Will be calculated by Gunn
        )
```

### Communication Filtering

Control which communication effects agents can observe:

```python
def should_observe_communication(self, effect: Effect, agent_id: str) -> bool:
    """Determine if agent should see a communication effect"""
    if effect["kind"] != "TeamMessage":
        return True  # Non-communication effects follow normal rules
    
    # Only show team messages to team members
    sender_team = effect["payload"].get("sender_team")
    agent_team = self.team
    
    return sender_team == agent_team
```

## Intent Processing

### Concurrent Decision Making

Process multiple agent decisions simultaneously:

```python
async def _process_concurrent_decisions(self) -> list[Effect]:
    """Process all agent decisions concurrently"""
    active_agents = [
        agent_id for agent_id, agent in self.world_state.agents.items()
        if agent.status == AgentStatus.ALIVE
    ]
    
    # Create decision tasks for all active agents
    decision_tasks = [
        self._process_agent_decision(agent_id)
        for agent_id in active_agents
    ]
    
    # Wait for all decisions with timeout
    try:
        decisions = await asyncio.wait_for(
            asyncio.gather(*decision_tasks, return_exceptions=True),
            timeout=3.0
        )
    except asyncio.TimeoutError:
        # Handle timeout with fallback decisions
        decisions = [self._create_fallback_decision(agent_id) for agent_id in active_agents]
    
    # Convert decisions to intents and submit to Gunn
    all_effects = []
    for agent_id, decision in zip(active_agents, decisions):
        if isinstance(decision, Exception):
            decision = self._create_fallback_decision(agent_id)
        
        intents = self._decision_to_intents(agent_id, decision)
        for intent in intents:
            effects = await self.orchestrator.submit_intent(intent)
            all_effects.extend(effects)
    
    return all_effects
```

### Intent Creation

Convert AI decisions into Gunn intents:

```python
def _decision_to_intents(self, agent_id: str, decision: AgentDecision) -> list[Intent]:
    """Convert AI decision to Gunn intents"""
    intents = []
    
    # Primary action intent
    primary_intent = {
        "intent_id": f"{agent_id}_{uuid.uuid4().hex[:8]}",
        "agent_id": agent_id,
        "action": decision.primary_action.action_type,
        "payload": decision.primary_action.model_dump(),
        "priority": 0,
        "source_id": "ai_decision_maker",
        "schema_version": "1.0.0"
    }
    intents.append(primary_intent)
    
    # Communication intent (if present)
    if decision.communication:
        comm_intent = {
            "intent_id": f"{agent_id}_comm_{uuid.uuid4().hex[:8]}",
            "agent_id": agent_id,
            "action": "communicate",
            "payload": decision.communication.model_dump(),
            "priority": 1,  # Higher priority for communication
            "source_id": "ai_decision_maker",
            "schema_version": "1.0.0"
        }
        intents.append(comm_intent)
    
    return intents
```

## Effect Handling

### Effect Processing Pipeline

Handle effects generated by Gunn:

```python
async def _process_effects(self, effects: list[Effect]):
    """Process effects and update world state"""
    for effect in effects:
        effect_kind = effect["kind"]
        payload = effect["payload"]
        
        if effect_kind == "AgentDamaged":
            await self._handle_agent_damaged(payload)
        elif effect_kind == "AgentDied":
            await self._handle_agent_died(payload)
        elif effect_kind == "AgentHealed":
            await self._handle_agent_healed(payload)
        elif effect_kind == "WeaponDegraded":
            await self._handle_weapon_degraded(payload)
        elif effect_kind == "WeaponRepaired":
            await self._handle_weapon_repaired(payload)
        elif effect_kind == "TeamMessage":
            await self._handle_team_message(payload)
        elif effect_kind == "AgentMoved":
            await self._handle_agent_moved(payload)
        else:
            logger.warning(f"Unknown effect kind: {effect_kind}")
    
    # Sync updated state back to Gunn
    await self._sync_world_state()
```

### Specific Effect Handlers

```python
async def _handle_agent_damaged(self, payload: dict):
    """Handle agent damage effect"""
    target_id = payload["target_id"]
    new_health = payload["new_health"]
    
    if target_id in self.world_state.agents:
        agent = self.world_state.agents[target_id]
        agent.health = new_health
        
        if new_health <= 0:
            agent.status = AgentStatus.DEAD
            self._check_win_condition()

async def _handle_team_message(self, payload: dict):
    """Handle team communication effect"""
    sender_team = payload["sender_team"]
    message_data = {
        "sender_id": payload["sender_id"],
        "message": payload["message"],
        "urgency": payload["urgency"],
        "timestamp": payload["timestamp"]
    }
    
    # Store in team communication history
    if sender_team not in self.world_state.last_communication:
        self.world_state.last_communication[sender_team] = []
    
    self.world_state.last_communication[sender_team].append(message_data)
    
    # Keep only last 10 messages per team
    if len(self.world_state.last_communication[sender_team]) > 10:
        self.world_state.last_communication[sender_team] = \
            self.world_state.last_communication[sender_team][-10:]
```

## World State Management

### State Consistency

Maintain consistency between your game state and Gunn's world state:

```python
class WorldStateManager:
    """Manages synchronization between game and Gunn world states"""
    
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        
    async def update_entity(self, entity_id: str, updates: dict):
        """Update a single entity in both states"""
        # Update Gunn's world state
        if entity_id in self.orchestrator.world_state.entities:
            self.orchestrator.world_state.entities[entity_id].update(updates)
        
        # Update spatial index if position changed
        if "position" in updates:
            pos = updates["position"]
            self.orchestrator.world_state.spatial_index[entity_id] = (*pos, 0.0)
    
    async def add_entity(self, entity_id: str, entity_data: dict):
        """Add new entity to both states"""
        self.orchestrator.world_state.entities[entity_id] = entity_data
        
        if "position" in entity_data:
            pos = entity_data["position"]
            self.orchestrator.world_state.spatial_index[entity_id] = (*pos, 0.0)
    
    async def remove_entity(self, entity_id: str):
        """Remove entity from both states"""
        self.orchestrator.world_state.entities.pop(entity_id, None)
        self.orchestrator.world_state.spatial_index.pop(entity_id, None)
```

## Performance Considerations

### Concurrent Processing

Optimize for concurrent agent processing:

```python
async def _process_agents_concurrently(self):
    """Process agents with proper concurrency control"""
    # Group agents by team for better cache locality
    team_agents = {}
    for agent_id, agent in self.world_state.agents.items():
        if agent.status == AgentStatus.ALIVE:
            if agent.team not in team_agents:
                team_agents[agent.team] = []
            team_agents[agent.team].append(agent_id)
    
    # Process teams concurrently, agents within teams sequentially
    team_tasks = []
    for team, agent_ids in team_agents.items():
        task = self._process_team_agents(team, agent_ids)
        team_tasks.append(task)
    
    await asyncio.gather(*team_tasks)

async def _process_team_agents(self, team: str, agent_ids: list[str]):
    """Process agents within a team"""
    for agent_id in agent_ids:
        await self._process_single_agent(agent_id)
```

### Memory Management

Manage memory usage for long-running simulations:

```python
class MemoryManager:
    """Manages memory usage for long-running simulations"""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        
    def cleanup_old_data(self, world_state: BattleWorldState):
        """Clean up old data to prevent memory leaks"""
        # Limit communication history
        for team in world_state.last_communication:
            messages = world_state.last_communication[team]
            if len(messages) > self.max_history_size:
                world_state.last_communication[team] = messages[-self.max_history_size:]
        
        # Clean up completed effects
        # (Implementation depends on your effect storage strategy)
```

## Best Practices

### 1. Error Handling

Always handle errors gracefully in multi-agent systems:

```python
async def _safe_agent_processing(self, agent_id: str):
    """Process agent with comprehensive error handling"""
    try:
        decision = await self.ai_decision_maker.make_decision(agent_id, observation)
        intents = self._decision_to_intents(agent_id, decision)
        
        for intent in intents:
            try:
                effects = await self.orchestrator.submit_intent(intent)
                await self._process_effects(effects)
            except Exception as e:
                logger.error(f"Intent processing failed for {agent_id}: {e}")
                # Continue with other intents
                
    except Exception as e:
        logger.error(f"Agent processing failed for {agent_id}: {e}")
        # Use fallback decision
        fallback_decision = self._create_fallback_decision(agent_id)
        # Process fallback...
```

### 2. Deterministic Behavior

Ensure deterministic behavior for reproducible simulations:

```python
def _ensure_deterministic_ordering(self, agent_ids: list[str]) -> list[str]:
    """Ensure consistent agent processing order"""
    return sorted(agent_ids)  # Always process in alphabetical order

async def _process_concurrent_intents(self, intents: list[Intent]):
    """Process intents with deterministic ordering"""
    # Sort intents by agent_id for consistent processing order
    sorted_intents = sorted(intents, key=lambda x: x["agent_id"])
    
    effects = []
    for intent in sorted_intents:
        intent_effects = await self.orchestrator.submit_intent(intent)
        effects.extend(intent_effects)
    
    return effects
```

### 3. Monitoring and Telemetry

Use Gunn's telemetry features for monitoring:

```python
from gunn.utils.telemetry import PerformanceTimer, setup_logging

# Set up structured logging
setup_logging(level="INFO", redact_pii=True)

# Use performance timers
async def _timed_decision_making(self, agent_id: str):
    """Make decision with performance monitoring"""
    with PerformanceTimer(f"decision_making.{agent_id}"):
        decision = await self.ai_decision_maker.make_decision(agent_id, observation)
        return decision
```

### 4. Configuration Management

Make your integration configurable:

```python
@dataclass
class BattleConfig:
    """Configuration for battle simulation"""
    max_agents: int = 6
    vision_range: float = 30.0
    attack_range: float = 15.0
    movement_speed: float = 5.0
    ai_model: str = "gpt-4.1-mini"
    decision_timeout: float = 3.0
    
    # Gunn orchestrator config
    staleness_threshold: int = 1
    debounce_ms: float = 50.0
    deadline_ms: float = 3000.0
    token_budget: int = 500

class BattleOrchestrator:
    def __init__(self, config: BattleConfig):
        self.config = config
        
        # Create Gunn config from battle config
        orchestrator_config = OrchestratorConfig(
            max_agents=config.max_agents,
            staleness_threshold=config.staleness_threshold,
            debounce_ms=config.debounce_ms,
            deadline_ms=config.deadline_ms,
            token_budget=config.token_budget,
            backpressure_policy="defer",
            default_priority=0
        )
        
        self.orchestrator = Orchestrator(orchestrator_config, world_id="battle_demo")
```

## Integration Checklist

When integrating Gunn into your own project, ensure you:

- [ ] Configure the orchestrator with appropriate limits and policies
- [ ] Implement observation policies that match your game's visibility rules
- [ ] Create proper intent/effect conversion logic
- [ ] Maintain world state synchronization between your game and Gunn
- [ ] Handle errors gracefully with fallback behaviors
- [ ] Implement deterministic processing for reproducible results
- [ ] Add monitoring and telemetry for performance tracking
- [ ] Test concurrent agent processing thoroughly
- [ ] Document your integration patterns for future developers

This guide provides the foundation for building sophisticated multi-agent simulations with Gunn. Use the battle demo as a reference implementation and adapt these patterns to your specific use case.