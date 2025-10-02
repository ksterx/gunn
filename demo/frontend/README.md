# Battle Demo Frontend

This directory contains the Pygame-based frontend for the Gunn multi-agent battle simulation demo. The frontend provides real-time visualization of the battle simulation with agent rendering, team colors, health bars, weapon condition indicators, and map locations.

## Features

### Visual Components
- **Agent Rendering**: Agents are displayed as colored circles with team colors (blue for Team A, red for Team B)
- **Health Bars**: Visual health indicators above each agent showing current health status
- **Weapon Condition**: Small colored indicators below agents showing weapon condition (green=excellent, yellow=good, orange=damaged, red=broken)
- **Map Locations**: Forges and strategic points are rendered with team-specific colors and labels
- **Vision Range**: Debug mode shows agent vision ranges as translucent circles

### User Interface
- **Game Status Panel**: Shows current game status, time, and connection status
- **Team Scores**: Real-time team scores and statistics
- **Agent Statistics**: Live agent count, health totals, and weapon conditions per team
- **Team Communications**: Display of recent team messages with urgency indicators
- **Error Handling**: Clear error messages and recovery options

### Real-time Updates
- **WebSocket Connection**: Real-time game state updates from the backend
- **Auto-reconnection**: Automatic reconnection on connection loss
- **State Synchronization**: Consistent game state between frontend and backend

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume game |
| `ESC` | Quit application |
| `D` | Toggle debug information display |
| `C` | Toggle team communication display |
| `R` | Reset game |
| `F5` | Refresh game state from backend |

## Installation

The frontend requires pygame and other dependencies. Install using uv with the demo group:

```bash
# Install demo dependencies including pygame
uv sync --group demo

# Or install pygame directly
uv add pygame
```

## Usage

### Quick Start

Run the frontend with default settings:

```bash
# From the demo directory
python run_frontend.py

# Or run the frontend module directly
python -m frontend
```

### Custom Configuration

```bash
# Specify backend URL
BACKEND_URL=http://192.168.1.100:8000 python run_frontend.py

# Or use command line arguments
python -m frontend.renderer --backend-url http://localhost:8000 --window-size 1024x768
```

### Backend Connection

The frontend connects to the backend API server at `http://localhost:8000` by default. Make sure the backend is running before starting the frontend:

```bash
# Start backend (in another terminal)
cd demo
python -m backend.server

# Then start frontend
python run_frontend.py
```

## Architecture

### Components

- **BattleRenderer**: Main renderer class handling visualization and user input
- **Coordinate System**: Converts between world coordinates and screen pixels
- **Network Layer**: HTTP client and WebSocket connection for backend communication
- **Event System**: Pygame event handling for user input and controls

### Data Flow

1. **Initialization**: Connect to backend WebSocket and fetch initial game state
2. **Real-time Updates**: Receive game state updates via WebSocket
3. **Rendering**: Convert game state to visual representation
4. **User Input**: Handle keyboard input and send control commands to backend
5. **Error Handling**: Display errors and attempt recovery

### Coordinate System

The frontend uses a coordinate conversion system to map world coordinates to screen pixels:

- **World Space**: Game coordinates (0,0) to (200,100) representing the battle map
- **Screen Space**: Pixel coordinates with UI panels and margins
- **Scaling**: Automatic scaling to fit the window size while preserving aspect ratio

## Configuration

### Window Settings

```python
# Default window size
WINDOW_SIZE = (800, 600)

# World dimensions (from game config)
WORLD_WIDTH = 200.0
WORLD_HEIGHT = 100.0
```

### Visual Settings

```python
# Team colors
TEAM_COLORS = {
    "team_a": (0, 100, 255),    # Blue
    "team_b": (255, 100, 0),    # Red
}

# Agent rendering
AGENT_RADIUS = 12  # pixels
HEALTH_BAR_WIDTH = 24  # pixels
WEAPON_INDICATOR_RADIUS = 3  # pixels
```

### Network Settings

```python
# Backend connection
BACKEND_URL = "http://localhost:8000"
WEBSOCKET_URL = "ws://localhost:8000/ws"

# Connection timeouts
WEBSOCKET_TIMEOUT = 30.0  # seconds
HTTP_TIMEOUT = 10.0  # seconds
```

## Troubleshooting

### Common Issues

1. **"pygame not found"**
   ```bash
   uv add pygame
   ```

2. **"Connection refused"**
   - Make sure the backend server is running
   - Check the backend URL configuration
   - Verify firewall settings

3. **"WebSocket connection failed"**
   - Backend may not support WebSockets
   - Check network connectivity
   - Try refreshing with F5

4. **Poor performance**
   - Reduce window size
   - Disable debug mode (press D)
   - Close other applications

### Debug Mode

Enable debug mode by pressing `D` to see:
- Agent vision ranges
- Coordinate information
- Performance metrics
- Network status details

### Logging

The frontend uses Python's logging module. Increase verbosity:

```bash
# Enable debug logging
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from frontend.renderer import main
import asyncio
asyncio.run(main())
"
```

## Development

### Testing

Run the frontend tests:

```bash
# Run frontend-specific tests
uv run pytest demo/tests/test_frontend_renderer.py -v

# Run with coverage
uv run pytest demo/tests/test_frontend_renderer.py --cov=demo.frontend
```

### Code Structure

```
frontend/
├── __init__.py          # Package initialization
├── __main__.py          # Command-line entry point
├── renderer.py          # Main BattleRenderer class
└── README.md           # This file
```

### Adding Features

To add new visual features:

1. Add rendering method to `BattleRenderer`
2. Call from `render_frame()` method
3. Add any new controls to `handle_events()`
4. Update tests in `test_frontend_renderer.py`

### Performance Optimization

- Use pygame's dirty rectangle updates for better performance
- Implement sprite groups for efficient rendering
- Cache rendered surfaces when possible
- Profile with pygame's built-in profiling tools

## Integration

The frontend integrates with the backend through:

- **REST API**: Initial game state and control commands
- **WebSocket**: Real-time game state updates
- **Shared Models**: Pydantic models for data consistency

See the backend documentation for API details and the shared models documentation for data structures.