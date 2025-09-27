# Product Overview

**gunn** (群) is a multi-agent simulation core that provides a controlled interface for agent-environment interaction, supporting both single and multi-agent settings.

## Core Purpose

The system enables multiple AI agents to interact in a shared environment with:
- **Partial observation** - agents see only what they should based on distance, relationships, and policies
- **Concurrent execution** - multiple agents can act simultaneously without blocking
- **Intelligent interruption** - agents can interrupt and regenerate responses when new relevant information arrives
- **Event-driven architecture** - unified core with deterministic ordering and complete audit trails

## Key Features

- **Dual API facades**: Both RL-style (`env.step()`) and message-oriented (`env.emit()`) interfaces
- **Real-time streaming**: Token-level streaming with sub-100ms cancellation response
- **External integration**: Unity, Unreal, and web API adapters for rich interactive experiences
- **Deterministic replay**: Complete event logs enable debugging and analysis
- **Multi-tenant security**: Proper isolation and access controls for production use

## Target Use Cases

- Multi-agent conversations with natural interruption patterns
- Spatial simulations with distance-based observation
- Game engine integration for interactive experiences
- Research environments requiring deterministic, auditable agent behavior
