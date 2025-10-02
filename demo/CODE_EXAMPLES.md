# Code Examples and Educational Patterns

This document provides educational code examples demonstrating key patterns used in the Gunn multi-agent battle demo. These examples are designed to help developers understand how to implement similar functionality in their own projects.

## Table of Contents

1. [Gunn Integration Patterns](#gunn-integration-patterns)
2. [Observation Policy Implementation](#observation-policy-implementation)
3. [AI Decision Making with OpenAI](#ai-decision-making-with-openai)
4. [Concurrent Agent Processing](#concurrent-agent-processing)
5. [Effect Processing and World State Updates](#effect-processing-and-world-state-updates)
6. [Error Handling Patterns](#error-handling-patterns)
7. [Real-time Frontend Integration](#real-time-frontend-integration)
8. [Performance Monitoring](#performance-monitoring)

## Gunn Integration Patterns

### Basic Orchestrator Setup

```python
from gunn import Orchestrator, OrchestratorConfig
from gunn.policies.observation import ObservationPolicy

# Configure Gunn orchestrator for your simulation
config = OrchestratorConfig(
    max_agents=6,                    # Maximum concurrent agents
    staleness_threshold=1,           # View staleness tolerance
    debounce_ms=50.0,               # Intent debouncing (milliseconds)
    deadline_ms=3000.0,             # Processing deadline
    token_budget=500,               # Resource allocation per agent
    backpressure_policy="defer",    # How to handle overload
    default_priority=0              # Default agent priority
)

orchestrator = Orchestrator(config, world_id="my_simulation")
await orchestrator.initialize()
```

### Agent Registration with Custom Policies

```python
class MyObservationPolicy(ObservationPolicy):
    def __init__(self, visibility_range: float):
        super().__init__(PolicyConfig(distance_limit=visibility_range))
        self.visibility_range = visibility_range
    
    def filter_world_state(self, world_state: WorldState, agent_id: str) -> View:
        # Implement your custom filtering logic
        visible_entities = {}
        agent_pos = world_state.spatial_index.get(agent_id, (0, 0, 0))
        
        for entity_id, entity in world_state.entities.items():
            entity_pos = world_state.spatial_index.get(entity_id, (0, 0, 0))
            distance = self._calculate_distance(agent_pos, entity_pos)
            
            if distance <= self.visibility_range:
                visible_entities[entity_id] = entity
        
        return View(
            agent_id=agent_id,
            view_seq=0,
            visible_entities=visible_entities,
            visible_relationships={},
            context_digest=""
        )

# Register agents with custom policies
for agent_id in agent_ids:
    policy = MyObservationPolicy(visibility_range=30.0)
    await orchestrator.register_agent(agent_id, policy)
```

### World State Synchronization

```python
async def sync_world_state(orchestrator: Orchestrator, game_state: MyGameState):
    """Synchronize your game state with Gunn's world state."""
    gunn_entities = {}
    spatial_index = {}
    
    # Convert your entities to Gunn format
    for entity_id, entity in game_state.entities.items():
        gunn_entities[entity_id] = entity.to_dict()
        if hasattr(entity, 'position'):
            spatial_index[entity_id] = (*entity.position, 0.0)
    
    # Update Gunn's world state
    orchestrator.world_state.entities = gunn_entities
    orchestrator.world_state.spatial_index = spatial_index
    orchestrator.world_state.metadata = {
        "game_time": game_state.current_time,
        "custom_data": game_state.custom_metadata
    }
```

## Observation Policy Implementation

### Team-Based Visibility with Fog of War

```python
class TeamObservationPolicy(ObservationPolicy):
    """Example of team-based observation with fog of war mechanics."""
    
    def __init__(self, team: str, vision_range: float = 30.0):
        super().__init__(PolicyConfig(distance_limit=vision_range))
        self.team = team
        self.vision_range = vision_range
    
    def filter_world_state(self, world_state: WorldState, agent_id: str) -> View:
        agent_pos = world_state.spatial_index.get(agent_id, (0, 0, 0))
        visible_entities = {}
        
        for entity_id, entity in world_state.entities.items():
            if self._should_see_entity(entity_id, entity, agent_id, agent_pos, world_state):
                # Apply information filtering based on relationship
                filtered_entity = self._filter_entity_info(entity, agent_id)
                visible_entities[entity_id] = filtered_entity
        
        return View(
            agent_id=agent_id,
            view_seq=0,
            visible_entities=visible_entities,
            visible_relationships={},
            context_digest=self._generate_digest(visible_entities)
        )
    
    def _should_see_entity(self, entity_id: str, entity: dict, 
                          agent_id: str, agent_pos: tuple, world_state: WorldState) -> bool:
        # Always see self
        if entity_id == agent_id:
            return True
        
        # Always see teammates
        if entity.get("team") == self.team:
            return True
        
        # Check vision range for others
        entity_pos = world_state.spatial_index.get(entity_id, (0, 0, 0))
        distance = self._calculate_distance(agent_pos, entity_pos)
        return distance <= self.vision_range
    
    def _filter_entity_info(self, entity: dict, observer_id: str) -> dict:
        """Filter entity information based on observer's permissions."""
        if entity.get("team") == self.team:
            # Full information for teammates
            return entity
        else:
            # Limited information for enemies (fog of war)
            return {
                "id": entity.get("id"),
                "position": entity.get("position"),
                "status": entity.get("status"),
                # Hide detailed stats like health, equipment, etc.
            }
```

### Communication Filtering

```python
def should_observe_communication(self, effect: Effect, agent_id: str) -> bool:
    """Control which communication effects an agent can observe."""
    if effect.get("kind") != "Communication":
        return True  # Non-communication effects use normal rules
    
    # Get sender and receiver information
    sender_team = effect.get("payload", {}).get("sender_team")
    receiver_team = effect.get("payload", {}).get("receiver_team")
    
    # Team-only communication
    if receiver_team and receiver_team != self.team:
        return False
    
    # Broadcast to team
    if sender_team == self.team:
        return True
    
    return False
```

## AI Decision Making with OpenAI

### Structured Output Schema Definition

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class MoveAction(BaseModel):
    action_type: Literal["move"] = "move"
    target_position: tuple[float, float] = Field(description="Target coordinates")
    reason: str = Field(description="Strategic reasoning for this move")

class AttackAction(BaseModel):
    action_type: Literal["attack"] = "attack"
    target_id: str = Field(description="ID of target to attack")
    reason: str = Field(description="Why this target was chosen")

class AgentDecision(BaseModel):
    """Complete decision with primary action and optional communication."""
    primary_action: MoveAction | AttackAction  # Union of possible actions
    communication: Optional[str] = Field(None, description="Optional team message")
    confidence: float = Field(ge=0.0, le=1.0, description="Decision confidence")
    reasoning: str = Field(description="Strategic assessment")
```

### AI Decision Making with Error Handling

```python
class AIDecisionMaker:
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.timeout = 30.0
        self.max_retries = 3
    
    async def make_decision(self, agent_id: str, observation: dict) -> AgentDecision:
        """Generate structured decision with comprehensive error handling."""
        
        system_prompt = self._build_system_prompt(agent_id)
        user_prompt = self._build_observation_prompt(observation)
        
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format=AgentDecision,
                        temperature=0.7,
                        max_tokens=800
                    ),
                    timeout=self.timeout
                )
                
                if response.choices and response.choices[0].message.parsed:
                    decision = response.choices[0].message.parsed
                    
                    # Validate decision before returning
                    if self._validate_decision(decision, observation):
                        return decision
                    else:
                        logger.warning(f"Invalid decision generated for {agent_id}")
                
            except asyncio.TimeoutError:
                logger.warning(f"OpenAI timeout for {agent_id}, attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"OpenAI error for {agent_id}: {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        # All attempts failed - return safe fallback
        return self._create_fallback_decision(agent_id, observation)
    
    def _create_fallback_decision(self, agent_id: str, observation: dict) -> AgentDecision:
        """Create a safe fallback decision when AI fails."""
        # Simple fallback: move to center of map
        return AgentDecision(
            primary_action=MoveAction(
                target_position=(100.0, 100.0),
                reason="Fallback decision due to AI error"
            ),
            confidence=0.1,
            reasoning="Using fallback behavior due to AI decision failure"
        )
```

### Batch Decision Processing

```python
async def batch_make_decisions(self, agent_observations: dict[str, dict]) -> dict[str, AgentDecision]:
    """Process multiple agent decisions concurrently."""
    
    # Create tasks for all agents
    tasks = []
    agent_ids = []
    
    for agent_id, observation in agent_observations.items():
        task = self.make_decision(agent_id, observation)
        tasks.append(task)
        agent_ids.append(agent_id)
    
    # Execute concurrently with error handling
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from errors
        decisions = {}
        for agent_id, result in zip(agent_ids, results):
            if isinstance(result, Exception):
                # Create fallback for failed agents
                decisions[agent_id] = self._create_fallback_decision(
                    agent_id, agent_observations[agent_id]
                )
            else:
                decisions[agent_id] = result
        
        return decisions
        
    except Exception as e:
        logger.error(f"Critical error in batch processing: {e}")
        # Return fallbacks for all agents
        return {
            agent_id: self._create_fallback_decision(agent_id, obs)
            for agent_id, obs in agent_observations.items()
        }
```

## Concurrent Agent Processing

### Deterministic Concurrent Processing

```python
async def process_agents_concurrently(self, agent_ids: list[str]) -> list[Effect]:
    """Process multiple agents with deterministic ordering."""
    
    # Sort agent IDs for consistent processing order
    sorted_agent_ids = sorted(agent_ids)
    
    # Generate decisions concurrently
    decision_tasks = [
        self._process_single_agent(agent_id) 
        for agent_id in sorted_agent_ids
    ]
    
    try:
        # Wait for all decisions with timeout
        decisions = await asyncio.wait_for(
            asyncio.gather(*decision_tasks, return_exceptions=True),
            timeout=5.0
        )
        
        # Convert decisions to intents
        all_intents = []
        current_time = time.time()
        
        for agent_id, decision in zip(sorted_agent_ids, decisions):
            if isinstance(decision, Exception):
                logger.warning(f"Agent {agent_id} decision failed: {decision}")
                continue
            
            # Convert to Gunn intents
            intents = self._decision_to_intents(agent_id, decision, current_time)
            all_intents.extend(intents)
        
        # Submit all intents simultaneously for true concurrency
        if all_intents:
            effects = await self.orchestrator.submit_intents(all_intents, current_time)
            return effects
        
        return []
        
    except asyncio.TimeoutError:
        logger.error("Agent processing timeout - using fallback decisions")
        return await self._handle_processing_timeout(sorted_agent_ids)
```

### Intent Conversion with Simultaneous Actions

```python
def _decision_to_intents(self, agent_id: str, decision: AgentDecision, sim_time: float) -> list[dict]:
    """Convert AI decision to Gunn intents supporting multiple simultaneous actions."""
    intents = []
    
    # Primary action intent
    primary_intent = {
        "kind": decision.primary_action.action_type.title(),
        "payload": decision.primary_action.model_dump(),
        "agent_id": agent_id,
        "sim_time": sim_time,
        "req_id": f"{agent_id}_{sim_time}_{uuid.uuid4().hex[:8]}",
        "priority": 0,
        "schema_version": "1.0.0"
    }
    intents.append(primary_intent)
    
    # Communication intent (if present)
    if decision.communication:
        comm_intent = {
            "kind": "Communication",
            "payload": {
                "message": decision.communication,
                "sender_id": agent_id,
                "team_only": True
            },
            "agent_id": agent_id,
            "sim_time": sim_time,
            "req_id": f"{agent_id}_comm_{sim_time}_{uuid.uuid4().hex[:8]}",
            "priority": 1,  # Higher priority for communication
            "schema_version": "1.0.0"
        }
        intents.append(comm_intent)
    
    return intents
```

## Effect Processing and World State Updates

### Effect Handler Pattern

```python
class EffectProcessor:
    """Processes effects and updates world state accordingly."""
    
    def __init__(self):
        self.handlers = {
            "AgentDamaged": self._handle_agent_damaged,
            "AgentMoved": self._handle_agent_moved,
            "AgentDied": self._handle_agent_died,
            "Communication": self._handle_communication,
            "WeaponDegraded": self._handle_weapon_degraded,
        }
    
    async def process_effects(self, effects: list[Effect], world_state: GameState) -> None:
        """Process a batch of effects and update world state."""
        for effect in effects:
            effect_kind = effect.get("kind")
            handler = self.handlers.get(effect_kind)
            
            if handler:
                try:
                    await handler(effect, world_state)
                except Exception as e:
                    logger.error(f"Error processing {effect_kind}: {e}")
            else:
                logger.warning(f"No handler for effect kind: {effect_kind}")
    
    async def _handle_agent_damaged(self, effect: Effect, world_state: GameState) -> None:
        """Handle agent damage effect."""
        payload = effect["payload"]
        target_id = payload["target_id"]
        damage = payload["damage"]
        
        if target_id in world_state.agents:
            agent = world_state.agents[target_id]
            agent.health = max(0, agent.health - damage)
            
            if agent.health <= 0:
                agent.status = AgentStatus.DEAD
                await self._check_win_condition(world_state)
    
    async def _handle_communication(self, effect: Effect, world_state: GameState) -> None:
        """Handle team communication effect."""
        payload = effect["payload"]
        sender_id = payload["sender_id"]
        message = payload["message"]
        
        # Add to team communication history
        sender = world_state.agents.get(sender_id)
        if sender:
            team_messages = world_state.team_communications.setdefault(sender.team, [])
            team_messages.append({
                "sender_id": sender_id,
                "message": message,
                "timestamp": world_state.game_time
            })
            
            # Keep only recent messages
            if len(team_messages) > 50:
                world_state.team_communications[sender.team] = team_messages[-50:]
```

### State Validation and Consistency

```python
class WorldStateValidator:
    """Validates world state consistency after updates."""
    
    def validate_state(self, world_state: GameState) -> list[str]:
        """Validate world state and return list of issues found."""
        issues = []
        
        # Check agent health bounds
        for agent_id, agent in world_state.agents.items():
            if agent.health < 0 or agent.health > 100:
                issues.append(f"Agent {agent_id} has invalid health: {agent.health}")
            
            if agent.health <= 0 and agent.status != AgentStatus.DEAD:
                issues.append(f"Agent {agent_id} has 0 health but is not marked as dead")
        
        # Check position bounds
        for agent_id, agent in world_state.agents.items():
            x, y = agent.position
            if x < 0 or y < 0 or x > 1000 or y > 1000:
                issues.append(f"Agent {agent_id} position out of bounds: {agent.position}")
        
        # Check team balance
        team_counts = {}
        for agent in world_state.agents.values():
            if agent.status == AgentStatus.ALIVE:
                team_counts[agent.team] = team_counts.get(agent.team, 0) + 1
        
        if len(team_counts) == 1:
            # Only one team has living agents - game should be over
            if world_state.game_status == "active":
                issues.append("Game should be over but status is still active")
        
        return issues
```

## Error Handling Patterns

### Hierarchical Error Handling

```python
class GameError(Exception):
    """Base class for game-specific errors."""
    def __init__(self, message: str, agent_id: str = None, severity: str = "medium"):
        super().__init__(message)
        self.agent_id = agent_id
        self.severity = severity
        self.timestamp = time.time()

class AIDecisionError(GameError):
    """Error in AI decision making process."""
    def __init__(self, message: str, agent_id: str, api_error: Exception = None):
        super().__init__(message, agent_id, "high")
        self.api_error = api_error

class ValidationError(GameError):
    """Error in data validation."""
    def __init__(self, message: str, invalid_data: dict = None):
        super().__init__(message, severity="medium")
        self.invalid_data = invalid_data

class ErrorHandler:
    """Centralized error handling with recovery strategies."""
    
    def __init__(self):
        self.error_counts = {}
        self.recovery_strategies = {
            AIDecisionError: self._handle_ai_error,
            ValidationError: self._handle_validation_error,
            Exception: self._handle_generic_error
        }
    
    async def handle_error(self, error: Exception, context: dict = None) -> dict:
        """Handle error with appropriate recovery strategy."""
        error_type = type(error)
        
        # Track error frequency
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Find appropriate handler
        handler = None
        for error_class, error_handler in self.recovery_strategies.items():
            if isinstance(error, error_class):
                handler = error_handler
                break
        
        if handler:
            return await handler(error, context)
        else:
            return await self._handle_generic_error(error, context)
    
    async def _handle_ai_error(self, error: AIDecisionError, context: dict) -> dict:
        """Handle AI decision errors with fallback decisions."""
        agent_id = error.agent_id
        
        # Create safe fallback decision
        fallback_decision = {
            "action": "move",
            "target": (100, 100),  # Move to center
            "reason": f"Fallback due to AI error: {str(error)}"
        }
        
        logger.warning(f"AI error for {agent_id}, using fallback: {error}")
        
        return {
            "status": "recovered",
            "fallback_decision": fallback_decision,
            "error_message": str(error)
        }
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    """Circuit breaker for external API calls."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e

# Usage example
ai_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30.0)

async def make_ai_decision_with_breaker(agent_id: str, observation: dict):
    try:
        return await ai_circuit_breaker.call(
            ai_decision_maker.make_decision, agent_id, observation
        )
    except Exception:
        # Circuit breaker is open or call failed
        return create_fallback_decision(agent_id)
```

## Real-time Frontend Integration

### WebSocket State Synchronization

```python
class GameStateManager:
    """Manages real-time game state synchronization with frontend."""
    
    def __init__(self):
        self.websocket_connections: list[WebSocket] = []
        self.last_broadcast_state = None
        self.state_lock = asyncio.Lock()
    
    async def add_connection(self, websocket: WebSocket):
        """Add new WebSocket connection."""
        self.websocket_connections.append(websocket)
        
        # Send current state to new connection
        if self.last_broadcast_state:
            await self._send_to_connection(websocket, {
                "type": "full_state",
                "data": self.last_broadcast_state
            })
    
    async def broadcast_state_update(self, world_state: GameState):
        """Broadcast state update to all connected clients."""
        async with self.state_lock:
            # Calculate state delta for efficiency
            state_data = self._serialize_state(world_state)
            
            if self.last_broadcast_state:
                delta = self._calculate_state_delta(self.last_broadcast_state, state_data)
                message = {"type": "state_delta", "data": delta}
            else:
                message = {"type": "full_state", "data": state_data}
            
            # Broadcast to all connections
            disconnected = []
            for websocket in self.websocket_connections:
                try:
                    await self._send_to_connection(websocket, message)
                except Exception:
                    disconnected.append(websocket)
            
            # Remove disconnected clients
            for ws in disconnected:
                self.websocket_connections.remove(ws)
            
            self.last_broadcast_state = state_data
    
    def _calculate_state_delta(self, old_state: dict, new_state: dict) -> dict:
        """Calculate minimal delta between states."""
        delta = {}
        
        # Check for changed agents
        old_agents = old_state.get("agents", {})
        new_agents = new_state.get("agents", {})
        
        changed_agents = {}
        for agent_id, agent_data in new_agents.items():
            if agent_id not in old_agents or old_agents[agent_id] != agent_data:
                changed_agents[agent_id] = agent_data
        
        if changed_agents:
            delta["agents"] = changed_agents
        
        # Check other state changes
        for key in ["game_time", "game_status", "team_scores"]:
            if old_state.get(key) != new_state.get(key):
                delta[key] = new_state.get(key)
        
        return delta
```

### Efficient Rendering Updates

```python
class RenderManager:
    """Manages efficient rendering updates for real-time visualization."""
    
    def __init__(self, screen):
        self.screen = screen
        self.dirty_regions = []
        self.last_render_time = 0
        self.target_fps = 60
        self.frame_time = 1.0 / self.target_fps
    
    def mark_dirty(self, rect: pygame.Rect):
        """Mark a screen region as needing redraw."""
        self.dirty_regions.append(rect)
    
    def render_frame(self, world_state: GameState) -> bool:
        """Render frame with dirty region optimization."""
        current_time = time.time()
        
        # Throttle rendering to target FPS
        if current_time - self.last_render_time < self.frame_time:
            return False
        
        if self.dirty_regions:
            # Only redraw dirty regions
            for rect in self.dirty_regions:
                self._render_region(rect, world_state)
            
            # Update only dirty regions
            pygame.display.update(self.dirty_regions)
            self.dirty_regions.clear()
        else:
            # Full screen update if no dirty regions
            self._render_full_screen(world_state)
            pygame.display.flip()
        
        self.last_render_time = current_time
        return True
    
    def _render_region(self, rect: pygame.Rect, world_state: GameState):
        """Render a specific screen region."""
        # Clip rendering to the dirty region
        self.screen.set_clip(rect)
        
        # Render background
        self.screen.fill((50, 50, 50), rect)
        
        # Render entities that intersect with this region
        for agent in world_state.agents.values():
            agent_rect = self._get_agent_rect(agent)
            if agent_rect.colliderect(rect):
                self._render_agent(agent)
        
        # Remove clipping
        self.screen.set_clip(None)
```

## Performance Monitoring

### Custom Performance Metrics

```python
class PerformanceMonitor:
    """Monitors and tracks performance metrics for the simulation."""
    
    def __init__(self):
        self.metrics = {
            "decision_times": [],
            "processing_times": [],
            "frame_times": [],
            "memory_usage": [],
            "api_call_counts": {},
            "error_counts": {}
        }
        self.start_time = time.time()
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operation performance."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self._record_metric(f"{operation_name}_duration", duration)
            self._record_metric(f"{operation_name}_memory_delta", memory_delta)
    
    def _record_metric(self, metric_name: str, value: float):
        """Record a performance metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            "value": value,
            "timestamp": time.time()
        })
        
        # Keep only recent metrics (last 1000 entries)
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]
    
    def get_performance_summary(self) -> dict:
        """Get summary of performance metrics."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                recent_values = [v["value"] for v in values[-100:]]  # Last 100 samples
                summary[metric_name] = {
                    "count": len(values),
                    "recent_avg": sum(recent_values) / len(recent_values),
                    "recent_min": min(recent_values),
                    "recent_max": max(recent_values)
                }
        
        return summary

# Usage example
performance_monitor = PerformanceMonitor()

async def process_game_tick():
    with performance_monitor.monitor_operation("game_tick"):
        # Process agents
        with performance_monitor.monitor_operation("agent_decisions"):
            decisions = await make_agent_decisions()
        
        # Process effects
        with performance_monitor.monitor_operation("effect_processing"):
            effects = await process_effects(decisions)
        
        # Update display
        with performance_monitor.monitor_operation("rendering"):
            update_display()
```

These examples demonstrate the key patterns used throughout the battle demo. They show how to integrate with Gunn's core systems while maintaining clean, maintainable, and performant code. Use these patterns as starting points for your own multi-agent simulations.