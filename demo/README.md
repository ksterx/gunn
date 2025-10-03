# Gunn Multi-Agent Battle Demo

A comprehensive demonstration of Gunn's multi-agent simulation capabilities through a 2D real-time strategy battle simulation.

## Overview

This demo showcases two teams of 3 CPU agents each engaging in tactical combat while demonstrating Gunn's key features:

- **Multi-agent orchestration** with deterministic behavior
- **Team-based observation policies** with fog of war
- **OpenAI structured outputs** for AI decision-making
- **Real-time visualization** with Pygame
- **FastAPI backend** with WebSocket support
- **Concurrent agent processing** and coordination

## Architecture

```
demo/
├── backend/           # FastAPI server and game logic
│   ├── server.py      # REST API and WebSocket endpoints
│   ├── game_manager.py # Game state management
│   ├── battle_mechanics.py # Combat calculations
│   ├── ai_decision.py # OpenAI integration
│   └── gunn_integration.py # Gunn orchestrator wrapper
├── frontend/          # Pygame visualization
│   ├── __init__.py    # Package initialization
│   ├── __main__.py    # Command-line entry point
│   ├── renderer.py    # Main BattleRenderer class with visualization
│   └── README.md      # Frontend documentation
└── shared/            # Common models and utilities
    ├── models.py      # Pydantic data models
    ├── schemas.py     # OpenAI structured output schemas
    ├── enums.py       # Game enumerations
    ├── constants.py   # Configuration constants
    └── utils.py       # Helper functions
```

## Installation

### Prerequisites

- Python 3.13+
- uv package manager
- OpenAI API key (for AI decision making)

### Setup

1. **Install demo dependencies:**
   ```bash
   uv sync --group demo
   ```

2. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

3. **Verify installation:**
   ```bash
   uv run --group demo python -c "import demo; print('Demo package installed successfully')"
   ```

## Usage

### Running the Complete Demo

1. **Start the backend server:**
   ```bash
   uv run --group demo python -m demo.backend.server
   ```

2. **Launch the frontend (in a separate terminal):**
   ```bash
   # Using the launcher script
   cd demo && python run_frontend.py
   
   # Or using the module directly
   uv run --group demo python -m demo.frontend
   ```

3. **Watch the battle simulation:**
   - The game will auto-start with two teams of 3 agents each
   - Agents will make decisions using OpenAI's structured outputs
   - Real-time visualization shows agent movements, combat, and team communication
   - The simulation ends when one team eliminates the other

### Development Mode

For development and testing individual components:

```bash
# Run backend only
uv run --group demo python -m demo.backend.server --dev

# Run frontend with custom backend URL
BACKEND_URL=http://192.168.1.100:8000 python demo/run_frontend.py

# Test AI decision making
uv run --group demo python -m demo.backend.ai_decision --test
```

## Game Mechanics

### Combat System
- **Health:** Agents start with 100 health points
- **Weapons:** Degrade from excellent → good → damaged → broken with use
- **Repair:** Broken weapons must be repaired at team forges
- **Healing:** Agents can heal themselves or teammates (takes time)
- **Vision:** Limited vision range with fog of war for enemy positions

### Team Coordination
- **Communication:** Team-only messaging with urgency levels
- **Observation:** Agents only see teammates and enemies within vision range
- **Strategy:** AI agents coordinate through structured communication

### Win Conditions
- Eliminate all enemy agents
- Game timeout (5 minutes) - team with more surviving agents wins

## Configuration

Key settings can be modified in `demo/shared/constants.py`:

```python
GAME_CONFIG = {
    "attack_damage": 25,        # Base damage per attack
    "heal_amount": 30,          # Health restored per heal
    "movement_speed": 5.0,      # Units per second
    "vision_range": 30.0,       # Agent vision distance
    "attack_range": 15.0,       # Maximum attack distance
    "agents_per_team": 3,       # Number of agents per team
    "max_game_duration": 300.0, # Game timeout in seconds
}
```

## API Endpoints

The backend exposes the following REST API:

- `POST /api/game/start` - Start a new battle simulation
- `GET /api/game/state` - Get current game state
- `GET /api/game/status` - Get game status and statistics
- `WebSocket /ws/game` - Real-time game state updates

## Development

### Adding New Features

1. **New AI Actions:** Add to `demo/shared/schemas.py` and update decision logic
2. **Game Mechanics:** Modify `demo/backend/battle_mechanics.py`
3. **UI Elements:** Add components to `demo/frontend/ui_components.py`
4. **Configuration:** Update `demo/shared/constants.py`

### Testing

```bash
# Run all demo tests
uv run --group demo pytest demo/

# Test specific components
uv run --group demo pytest demo/backend/tests/
uv run --group demo pytest demo/frontend/tests/
```

### Code Quality

```bash
# Format code
uv run ruff format demo/

# Check linting
uv run ruff check demo/

# Type checking
uv run mypy demo/
```

## Troubleshooting

### Common Issues

1. **OpenAI API Errors:**
   - Verify your API key is set correctly
   - Check your OpenAI account has sufficient credits
   - Ensure you have access to the gpt-4.1 model

2. **Pygame Display Issues:**
   - Install system dependencies for Pygame
   - On Linux: `sudo apt-get install python3-pygame`
   - On macOS: Pygame should work with the pip installation

3. **WebSocket Connection Errors:**
   - Ensure the backend server is running before starting the frontend
   - Check firewall settings if running on different machines

4. **Performance Issues:**
   - Reduce the number of agents per team in configuration
   - Lower the frame rate in display settings
   - Use a faster OpenAI model for quicker decisions

### Getting Help

- Check the [Gunn documentation](../docs/) for core concepts
- Review the [implementation tasks](../.kiro/specs/multi-agent-battle-demo/tasks.md) for development progress
- Open an issue in the repository for bugs or feature requests

## Educational Value

This demo serves as a comprehensive example of:

- **Multi-agent system design** with Gunn's orchestration
- **Real-time AI decision making** with structured outputs
- **Partial observation** and information filtering
- **Team coordination** and communication patterns
- **Event-driven architecture** with deterministic behavior
- **Frontend-backend separation** with real-time updates

## Documentation

Comprehensive documentation is available to help you understand and extend the demo:

- **[Developer Guide](DEVELOPER_GUIDE.md)** - Detailed explanation of Gunn integration patterns
- **[Architecture Documentation](ARCHITECTURE.md)** - Design decisions and architectural patterns
- **[Code Examples](CODE_EXAMPLES.md)** - Educational code examples and patterns
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Solutions for common issues
- **[Development Guide](DEVELOPMENT.md)** - Setup and development workflow
- **[Initialization System](INITIALIZATION_SYSTEM.md)** - Game initialization patterns
- **[Error Handling Summary](ERROR_HANDLING_SUMMARY.md)** - Error handling strategies

Use this demo as a reference for building your own multi-agent simulations with Gunn.