"""gunn - Multi-agent simulation core.

gunn (群) provides a controlled interface for agent-environment interaction,
supporting both single and multi-agent settings with partial observation,
concurrent execution, and intelligent interruption capabilities.
"""

__version__ = "0.1.0"

# Core exports
from .core import (
    AgentHandle,
    DefaultEffectValidator,
    EffectValidator,
    EventLog,
    EventLogEntry,
    Orchestrator,
    OrchestratorConfig,
)
from .core.concurrent_processor import (
    BatchResult,
    ConcurrentIntentProcessor,
    ConcurrentProcessingConfig,
    ProcessingMode,
)

# Facade exports
from .facades import MessageFacade, RLFacade

__all__ = [
    "AgentHandle",
    "BatchResult",
    "ConcurrentIntentProcessor",
    "ConcurrentProcessingConfig",
    "DefaultEffectValidator",
    "EffectValidator",
    "EventLog",
    "EventLogEntry",
    "MessageFacade",
    "Orchestrator",
    "OrchestratorConfig",
    "ProcessingMode",
    "RLFacade",
    "__version__",
]
