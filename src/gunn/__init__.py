"""gunn - Multi-agent simulation core.

gunn (群) provides a controlled interface for agent-environment interaction,
supporting both single and multi-agent settings with partial observation,
concurrent execution, and intelligent interruption capabilities.
"""

__version__ = "0.1.0"

# Core exports
from .core import (
    AgentHandle,
    AsyncAgentLogic,
    DefaultEffectValidator,
    EffectValidator,
    EventLog,
    EventLogEntry,
    Orchestrator,
    OrchestratorConfig,
    SimpleAgentLogic,
)
from .core.collaborative_agent import CollaborativeAgent, SpecializedCollaborativeAgent
from .core.collaborative_behavior import (
    CollaborationOpportunity,
    CollaborationType,
    CollaborativeBehaviorManager,
    CoordinationPattern,
    create_collaborative_intent,
)
from .core.collaborative_patterns import (
    CollaborationDetector,
    CoordinationPatternTracker,
    detect_following_pattern,
    suggest_collaborative_action,
)
from .core.concurrent_processor import (
    BatchResult,
    ConcurrentIntentProcessor,
    ConcurrentProcessingConfig,
    ProcessingMode,
)
from .core.conversational_agent import (
    ConversationalAgent,
    ConversationMemory,
    LLMClient,
    LLMResponse,
    MockLLMClient,
)

# Facade exports
from .facades import MessageFacade, RLFacade

__all__ = [
    "AgentHandle",
    "AsyncAgentLogic",
    "BatchResult",
    "CollaborationDetector",
    "CollaborationOpportunity",
    "CollaborationType",
    "CollaborativeAgent",
    "CollaborativeBehaviorManager",
    "ConcurrentIntentProcessor",
    "ConcurrentProcessingConfig",
    "ConversationMemory",
    "ConversationalAgent",
    "CoordinationPattern",
    "CoordinationPatternTracker",
    "DefaultEffectValidator",
    "EffectValidator",
    "EventLog",
    "EventLogEntry",
    "LLMClient",
    "LLMResponse",
    "MessageFacade",
    "MockLLMClient",
    "Orchestrator",
    "OrchestratorConfig",
    "ProcessingMode",
    "RLFacade",
    "SimpleAgentLogic",
    "SpecializedCollaborativeAgent",
    "__version__",
    "create_collaborative_intent",
    "detect_following_pattern",
    "suggest_collaborative_action",
]
