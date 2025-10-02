# Gunn Concurrent Intent Processing Specification

## Overview

This specification defines enhancements to Gunn's orchestrator to support concurrent intent processing, which is essential for real-time multi-agent simulations where multiple agents need to act simultaneously.

## Current State

The current Gunn orchestrator processes intents sequentially:

```python
async def submit_intent(self, intent: Intent) -> list[Effect]:
    """Process a single intent sequentially"""
    # Validate intent
    # Apply effects
    # Update world state
    # Return effects
```

## Requirements

### Functional Requirements

1. **Batch Intent Submission**: Support submitting multiple intents simultaneously
2. **Deterministic Concurrent Processing**: Ensure reproducible results despite concurrent execution
3. **Configurable Processing Modes**: Allow sequential, concurrent, or deterministic concurrent processing
4. **Error Isolation**: Failures in one intent should not affect others in the same batch
5. **Performance Optimization**: Concurrent processing should improve throughput for multi-agent scenarios

### Non-Functional Requirements

1. **Backward Compatibility**: Existing single-intent submission must continue to work
2. **Thread Safety**: Concurrent operations must be safe
3. **Memory Efficiency**: Batch processing should not significantly increase memory usage
4. **Latency**: Concurrent processing should reduce overall latency for batch operations

## Design

### API Design

```python
class Orchestrator:
    async def submit_intents_batch(
        self,
        intents: list[Intent],
        sim_time: float,
        processing_mode: ProcessingMode = ProcessingMode.SEQUENTIAL,
        timeout: float | None = None
    ) -> BatchResult:
        """Submit multiple intents with configurable processing mode"""
        pass

class ProcessingMode(Enum):
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent" 
    DETERMINISTIC_CONCURRENT = "deterministic_concurrent"

class BatchResult:
    """Result of batch intent processing"""
    effects: list[Effect]
    errors: dict[str, Exception]  # intent_id -> error
    processing_time: float
    metadata: dict[str, Any]
```

### Processing Modes

#### Sequential Mode (Default)
- Process intents one by one in submission order
- Maintains exact current behavior
- Guarantees deterministic results

#### Concurrent Mode
- Process intents simultaneously using asyncio
- Fastest processing but non-deterministic ordering
- Suitable for scenarios where order doesn't matter

#### Deterministic Concurrent Mode
- Process intents concurrently but sort results deterministically
- Balance between performance and reproducibility
- Sort by agent_id or other stable criteria

### Implementation Architecture

```python
class ConcurrentIntentProcessor:
    """Handles concurrent intent processing logic"""
    
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.processing_semaphore = asyncio.Semaphore(100)  # Configurable
    
    async def process_batch(
        self,
        intents: list[Intent],
        mode: ProcessingMode,
        timeout: float | None = None
    ) -> BatchResult:
        """Process batch of intents according to specified mode"""
        
        if mode == ProcessingMode.SEQUENTIAL:
            return await self._process_sequential(intents)
        elif mode == ProcessingMode.CONCURRENT:
            return await self._process_concurrent(intents, timeout)
        elif mode == ProcessingMode.DETERMINISTIC_CONCURRENT:
            return await self._process_deterministic_concurrent(intents, timeout)
    
    async def _process_sequential(self, intents: list[Intent]) -> BatchResult:
        """Process intents sequentially (current behavior)"""
        effects = []
        errors = {}
        start_time = time.time()
        
        for intent in intents:
            try:
                intent_effects = await self.orchestrator._process_single_intent(intent)
                effects.extend(intent_effects)
            except Exception as e:
                errors[intent["req_id"]] = e
        
        return BatchResult(
            effects=effects,
            errors=errors,
            processing_time=time.time() - start_time,
            metadata={"mode": "sequential", "intent_count": len(intents)}
        )
    
    async def _process_concurrent(
        self, 
        intents: list[Intent], 
        timeout: float | None = None
    ) -> BatchResult:
        """Process intents concurrently"""
        start_time = time.time()
        
        # Create tasks for all intents
        tasks = []
        for intent in intents:
            task = asyncio.create_task(
                self._process_single_intent_safe(intent)
            )
            tasks.append((intent["req_id"], task))
        
        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for _, task in tasks:
                task.cancel()
            raise
        
        # Collect results
        effects = []
        errors = {}
        
        for (req_id, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                errors[req_id] = result
            else:
                effects.extend(result)
        
        return BatchResult(
            effects=effects,
            errors=errors,
            processing_time=time.time() - start_time,
            metadata={"mode": "concurrent", "intent_count": len(intents)}
        )
    
    async def _process_deterministic_concurrent(
        self,
        intents: list[Intent],
        timeout: float | None = None
    ) -> BatchResult:
        """Process intents concurrently with deterministic result ordering"""
        
        # Sort intents by agent_id for deterministic processing order
        sorted_intents = sorted(intents, key=lambda x: x["agent_id"])
        
        # Process concurrently
        result = await self._process_concurrent(sorted_intents, timeout)
        
        # Sort effects by source agent for deterministic output
        result.effects.sort(key=lambda e: e.get("source_id", ""))
        
        result.metadata["mode"] = "deterministic_concurrent"
        return result
    
    async def _process_single_intent_safe(self, intent: Intent) -> list[Effect]:
        """Process single intent with error isolation"""
        async with self.processing_semaphore:
            try:
                return await self.orchestrator._process_single_intent(intent)
            except Exception as e:
                # Log error but don't let it propagate
                logger.error(f"Intent processing failed: {intent['req_id']}: {e}")
                raise
```

### Configuration

```python
class ConcurrentProcessingConfig(BaseModel):
    """Configuration for concurrent intent processing"""
    
    max_concurrent_intents: int = Field(default=100, ge=1, le=1000)
    default_timeout: float = Field(default=30.0, gt=0.0)
    enable_deterministic_mode: bool = Field(default=True)
    processing_mode: ProcessingMode = Field(default=ProcessingMode.SEQUENTIAL)
    
    # Performance tuning
    batch_size_threshold: int = Field(default=5, ge=1)  # When to use concurrent processing
    semaphore_size: int = Field(default=50, ge=1)  # Concurrent operation limit

class OrchestratorConfig(BaseModel):
    # ... existing fields ...
    concurrent_processing: ConcurrentProcessingConfig = Field(
        default_factory=ConcurrentProcessingConfig
    )
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Add `ConcurrentIntentProcessor` class
2. Implement `ProcessingMode` enum and `BatchResult` class
3. Add configuration support
4. Implement sequential mode (no behavior change)

### Phase 2: Concurrent Processing
1. Implement concurrent mode
2. Add timeout handling
3. Implement error isolation
4. Add performance monitoring

### Phase 3: Deterministic Mode
1. Implement deterministic concurrent processing
2. Add result ordering logic
3. Ensure reproducible behavior

### Phase 4: Integration and Testing
1. Integrate with existing orchestrator
2. Add comprehensive tests
3. Performance benchmarking
4. Documentation and examples

## Testing Strategy

### Unit Tests
```python
class TestConcurrentProcessing:
    async def test_sequential_mode_unchanged(self):
        """Ensure sequential mode maintains exact current behavior"""
        pass
    
    async def test_concurrent_mode_performance(self):
        """Verify concurrent mode improves performance"""
        pass
    
    async def test_deterministic_mode_reproducibility(self):
        """Ensure deterministic mode produces consistent results"""
        pass
    
    async def test_error_isolation(self):
        """Verify errors in one intent don't affect others"""
        pass
    
    async def test_timeout_handling(self):
        """Test timeout behavior in concurrent processing"""
        pass
```

### Integration Tests
```python
class TestBattleScenarios:
    async def test_simultaneous_agent_actions(self):
        """Test multiple agents acting simultaneously"""
        intents = [
            create_move_intent("agent_1", (10, 10)),
            create_attack_intent("agent_2", "agent_3"),
            create_heal_intent("agent_4", "agent_5")
        ]
        
        result = await orchestrator.submit_intents_batch(
            intents, 
            sim_time=1.0,
            processing_mode=ProcessingMode.DETERMINISTIC_CONCURRENT
        )
        
        assert len(result.effects) > 0
        assert len(result.errors) == 0
        # Verify deterministic ordering
```

### Performance Tests
```python
class TestPerformance:
    async def test_concurrent_vs_sequential_performance(self):
        """Compare performance of concurrent vs sequential processing"""
        
        # Create large batch of intents
        intents = [create_test_intent(f"agent_{i}") for i in range(100)]
        
        # Test sequential
        start = time.time()
        seq_result = await orchestrator.submit_intents_batch(
            intents, mode=ProcessingMode.SEQUENTIAL
        )
        sequential_time = time.time() - start
        
        # Test concurrent
        start = time.time()
        conc_result = await orchestrator.submit_intents_batch(
            intents, mode=ProcessingMode.CONCURRENT
        )
        concurrent_time = time.time() - start
        
        # Concurrent should be faster for large batches
        assert concurrent_time < sequential_time
        assert len(seq_result.effects) == len(conc_result.effects)
```

## Migration Guide

### For Existing Applications

No changes required for existing applications using single intent submission:

```python
# This continues to work unchanged
effects = await orchestrator.submit_intent(intent)
```

### For New Applications

Applications can opt into batch processing:

```python
# New batch API
result = await orchestrator.submit_intents_batch(
    intents=[intent1, intent2, intent3],
    sim_time=current_time,
    processing_mode=ProcessingMode.DETERMINISTIC_CONCURRENT
)

# Handle results
for effect in result.effects:
    await process_effect(effect)

# Handle errors
for intent_id, error in result.errors.items():
    logger.error(f"Intent {intent_id} failed: {error}")
```

## Performance Expectations

### Throughput Improvements
- **Sequential**: Baseline performance (current behavior)
- **Concurrent**: 2-5x improvement for I/O bound operations
- **Deterministic Concurrent**: 1.5-3x improvement (overhead from sorting)

### Latency Characteristics
- **Sequential**: Linear increase with batch size
- **Concurrent**: Constant latency regardless of batch size (up to semaphore limit)
- **Deterministic Concurrent**: Slight overhead from result ordering

### Memory Usage
- Minimal increase in memory usage
- Temporary storage for concurrent task results
- Configurable semaphore limits memory growth

## Future Enhancements

1. **Priority-based Processing**: Process high-priority intents first
2. **Adaptive Batching**: Automatically batch intents based on load
3. **Streaming Results**: Return effects as they're processed
4. **Custom Ordering**: Allow custom result ordering functions
5. **Distributed Processing**: Support for multi-node processing

This specification provides a comprehensive approach to adding concurrent intent processing to Gunn while maintaining backward compatibility and ensuring deterministic behavior when needed.