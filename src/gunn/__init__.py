"""gunn - Multi-agent simulation core.

gunn (群) provides a controlled interface for agent-environment interaction,
supporting both single and multi-agent settings with partial observation,
concurrent execution, and intelligent interruption capabilities.
"""

__version__ = "0.1.0"

# Core exports
from .core import EventLog, EventLogEntry

__all__ = [
    "EventLog",
    "EventLogEntry",
    "__version__",
]
