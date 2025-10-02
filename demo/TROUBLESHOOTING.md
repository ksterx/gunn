# Troubleshooting Guide

This guide helps resolve common issues when running the Gunn multi-agent battle demo.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [OpenAI API Problems](#openai-api-problems)
3. [Backend Server Issues](#backend-server-issues)
4. [Frontend Display Problems](#frontend-display-problems)
5. [Performance Issues](#performance-issues)
6. [Network and Connectivity](#network-and-connectivity)
7. [Development and Testing](#development-and-testing)
8. [Debugging Tools](#debugging-tools)

## Installation Issues

### Problem: `uv` command not found

**Symptoms:**
```bash
bash: uv: command not found
```

**Solution:**
Install uv package manager:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

### Problem: Demo dependencies not installing

**Symptoms:**
```bash
ERROR: No such group: demo
```

**Solution:**
Ensure you're in the project root and the demo group is defined:

```bash
# Check if you're in the right directory
ls pyproject.toml

# Install demo dependencies
uv sync --group demo

# If still failing, try installing all groups
uv sync --all-groups
```

### Problem: Python version compatibility

**Symptoms:**
```bash
ERROR: Python 3.13+ required
```

**Solution:**
Install Python 3.13 or newer:

```bash
# Using uv to manage Python versions
uv python install 3.13

# Or check your current version
python --version
```

## OpenAI API Problems

### Problem: Missing API key

**Symptoms:**
```bash
openai.AuthenticationError: No API key provided
```

**Solution:**
Set your OpenAI API key:

```bash
# Set environment variable
export OPENAI_API_KEY="your-api-key-here"

# Or create a .env file in the demo directory
echo "OPENAI_API_KEY=your-api-key-here" > demo/.env
```

### Problem: API rate limits

**Symptoms:**
```bash
openai.RateLimitError: Rate limit exceeded
```

**Solution:**
1. **Reduce concurrent agents:**
   ```python
   # In demo/shared/constants.py
   GAME_CONFIG = {
       "agents_per_team": 2,  # Reduce from 3 to 2
   }
   ```

2. **Add delays between requests:**
   ```python
   # In demo/backend/ai_decision.py
   await asyncio.sleep(0.5)  # Add delay between API calls
   ```

3. **Use a different model:**
   ```python
   # In demo/backend/ai_decision.py
   model = "gpt-3.5-turbo"  # Instead of gpt-4o
   ```

### Problem: Model access denied

**Symptoms:**
```bash
openai.PermissionDeniedError: You don't have access to this model
```

**Solution:**
1. **Check your OpenAI plan** - GPT-4 requires a paid plan
2. **Use an available model:**
   ```python
   # In demo/backend/ai_decision.py
   model = "gpt-3.5-turbo"  # More widely available
   ```

### Problem: API timeout errors

**Symptoms:**
```bash
openai.APITimeoutError: Request timed out
```

**Solution:**
1. **Increase timeout:**
   ```python
   # In demo/backend/ai_decision.py
   client = AsyncOpenAI(timeout=30.0)  # Increase from default
   ```

2. **Add retry logic:**
   ```python
   for attempt in range(3):
       try:
           response = await client.beta.chat.completions.parse(...)
           break
       except openai.APITimeoutError:
           if attempt == 2:
               raise
           await asyncio.sleep(2 ** attempt)
   ```

## Backend Server Issues

### Problem: Port already in use

**Symptoms:**
```bash
OSError: [Errno 48] Address already in use
```

**Solution:**
1. **Kill existing process:**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   
   # Kill the process
   kill -9 <PID>
   ```

2. **Use a different port:**
   ```bash
   # Start server on different port
   uvicorn demo.backend.server:app --port 8001
   ```

### Problem: Import errors

**Symptoms:**
```bash
ModuleNotFoundError: No module named 'demo'
```

**Solution:**
1. **Ensure you're using uv:**
   ```bash
   uv run --group demo python -m demo.backend.server
   ```

2. **Check PYTHONPATH:**
   ```bash
   # Add project root to PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Install in development mode:**
   ```bash
   uv pip install -e .
   ```

### Problem: Database connection errors

**Symptoms:**
```bash
sqlite3.OperationalError: database is locked
```

**Solution:**
1. **Close other connections:**
   ```bash
   # Remove lock files
   rm -f *.db-shm *.db-wal
   ```

2. **Use in-memory database for testing:**
   ```python
   # In demo/backend/server.py
   database_url = "sqlite:///:memory:"
   ```

## Frontend Display Problems

### Problem: Pygame not displaying

**Symptoms:**
- Black screen or no window appears
- `pygame.error: No available video device`

**Solution:**
1. **Install system dependencies:**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install python3-pygame libsdl2-dev
   
   # On macOS
   brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
   
   # On Windows
   # Pygame should work with pip installation
   ```

2. **Set display environment:**
   ```bash
   # On Linux with X11
   export DISPLAY=:0
   
   # For headless systems, use virtual display
   sudo apt-get install xvfb
   xvfb-run -a python demo/run_frontend.py
   ```

### Problem: Font rendering issues

**Symptoms:**
- Text not displaying correctly
- Font errors in console

**Solution:**
1. **Install system fonts:**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install fonts-dejavu-core
   
   # On macOS
   # System fonts should be available
   ```

2. **Use fallback fonts:**
   ```python
   # In demo/frontend/renderer.py
   try:
       font = pygame.font.Font("arial.ttf", 12)
   except:
       font = pygame.font.Font(None, 12)  # Use default font
   ```

### Problem: Performance/frame rate issues

**Symptoms:**
- Choppy animation
- High CPU usage
- Slow response

**Solution:**
1. **Reduce frame rate:**
   ```python
   # In demo/frontend/renderer.py
   clock.tick(30)  # Reduce from 60 FPS to 30 FPS
   ```

2. **Optimize rendering:**
   ```python
   # Only update changed areas
   pygame.display.update(dirty_rects)
   
   # Use convert() for surfaces
   surface = surface.convert()
   ```

## Performance Issues

### Problem: Slow AI decision making

**Symptoms:**
- Long delays between agent actions
- Timeout errors

**Solution:**
1. **Use faster OpenAI model:**
   ```python
   model = "gpt-3.5-turbo"  # Faster than GPT-4
   ```

2. **Reduce decision complexity:**
   ```python
   # Simplify system prompt
   # Reduce max_tokens in API call
   max_tokens = 200  # Reduce from 1000
   ```

3. **Cache similar decisions:**
   ```python
   # Implement decision caching
   decision_cache = {}
   cache_key = hash(observation_summary)
   if cache_key in decision_cache:
       return decision_cache[cache_key]
   ```

### Problem: Memory usage growing over time

**Symptoms:**
- Increasing RAM usage during long simulations
- Eventually crashes with out of memory

**Solution:**
1. **Limit history size:**
   ```python
   # In demo/shared/models.py
   MAX_COMMUNICATION_HISTORY = 50  # Limit message history
   ```

2. **Clean up old data:**
   ```python
   # Periodically clean up old effects and observations
   if len(self.effect_history) > 1000:
       self.effect_history = self.effect_history[-500:]
   ```

3. **Use memory profiling:**
   ```bash
   # Install memory profiler
   uv add memory-profiler
   
   # Profile memory usage
   mprof run python demo/run_frontend.py
   mprof plot
   ```

## Network and Connectivity

### Problem: WebSocket connection failures

**Symptoms:**
```bash
WebSocketException: Connection failed
```

**Solution:**
1. **Check backend is running:**
   ```bash
   curl http://localhost:8000/api/game/status
   ```

2. **Verify WebSocket endpoint:**
   ```python
   # Test WebSocket connection
   import websockets
   
   async def test_connection():
       uri = "ws://localhost:8000/ws/game"
       async with websockets.connect(uri) as websocket:
           print("Connected successfully")
   ```

3. **Check firewall settings:**
   ```bash
   # On macOS
   sudo pfctl -d  # Disable firewall temporarily
   
   # On Linux
   sudo ufw allow 8000
   ```

### Problem: CORS errors in browser

**Symptoms:**
```bash
Access to fetch at 'http://localhost:8000' from origin 'null' has been blocked by CORS policy
```

**Solution:**
Update CORS settings in backend:

```python
# In demo/backend/server.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Development and Testing

### Problem: Tests failing

**Symptoms:**
```bash
FAILED tests/test_battle_mechanics.py::test_attack_calculation
```

**Solution:**
1. **Run tests with verbose output:**
   ```bash
   uv run --group demo pytest -v demo/tests/
   ```

2. **Run specific test:**
   ```bash
   uv run --group demo pytest demo/tests/test_battle_mechanics.py::test_attack_calculation -v
   ```

3. **Check test dependencies:**
   ```bash
   # Ensure test dependencies are installed
   uv sync --group demo --group test
   ```

### Problem: Type checking errors

**Symptoms:**
```bash
mypy: error: Cannot find implementation or library stub
```

**Solution:**
1. **Install type stubs:**
   ```bash
   uv add types-requests types-pygame
   ```

2. **Update mypy configuration:**
   ```toml
   # In pyproject.toml
   [tool.mypy]
   ignore_missing_imports = true
   ```

### Problem: Linting errors

**Symptoms:**
```bash
ruff: F401 'unused import'
```

**Solution:**
1. **Auto-fix issues:**
   ```bash
   uv run ruff check --fix demo/
   ```

2. **Format code:**
   ```bash
   uv run ruff format demo/
   ```

## Debugging Tools

### Enable Debug Logging

```python
# In demo/backend/server.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use structured logging
from gunn.utils.telemetry import setup_logging
setup_logging(level="DEBUG", redact_pii=False)
```

### Monitor Performance

```python
# Add performance monitoring
from gunn.utils.telemetry import PerformanceTimer

with PerformanceTimer("decision_making"):
    decision = await make_decision(agent_id, observation)
```

### Debug WebSocket Messages

```python
# In demo/frontend/renderer.py
async def debug_websocket():
    async with websockets.connect("ws://localhost:8000/ws/game") as websocket:
        async for message in websocket:
            print(f"Received: {message}")
```

### Profile Code Performance

```bash
# Install profiling tools
uv add py-spy

# Profile running process
py-spy record -o profile.svg -- python demo/run_frontend.py

# View profile
open profile.svg
```

### Memory Debugging

```python
# Track memory usage
import tracemalloc

tracemalloc.start()

# ... run your code ...

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

## Getting Additional Help

If you're still experiencing issues:

1. **Check the logs:**
   - Backend logs: Look for errors in the server console
   - Frontend logs: Check the terminal running the frontend
   - System logs: Check system logs for hardware/driver issues

2. **Create a minimal reproduction:**
   - Isolate the problem to the smallest possible example
   - Test with minimal configuration

3. **Check system resources:**
   - Monitor CPU, memory, and network usage
   - Ensure sufficient system resources are available

4. **Update dependencies:**
   ```bash
   uv sync --upgrade
   ```

5. **Report issues:**
   - Include error messages, logs, and system information
   - Provide steps to reproduce the problem
   - Mention your operating system and Python version

Remember to check the [Developer Guide](DEVELOPER_GUIDE.md) for integration patterns and the main [README](README.md) for basic setup instructions.