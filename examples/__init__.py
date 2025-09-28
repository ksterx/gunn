"""Example applications and demos for the gunn multi-agent simulation core.

This package contains comprehensive examples demonstrating all the features
of the gunn simulation system, including:

- A/B/C conversation demo with interruption and regeneration
- 2D spatial simulation with movement and distance-based observation
- Unity integration demo (placeholder for real-time agent interaction)
- Performance benchmark scenarios for load testing
- Comprehensive documentation and tutorials

These examples serve as both demonstrations and integration tests,
showing how to use the gunn system in various scenarios.
"""

__version__ = "0.1.0"

# Example categories
CONVERSATION_EXAMPLES = [
    "abc_conversation_demo",
    "interruption_demo",
]

SPATIAL_EXAMPLES = [
    "spatial_2d_demo",
    "movement_demo",
    "distance_observation_demo",
]

INTEGRATION_EXAMPLES = [
    "unity_integration_demo",
    "web_adapter_demo",
]

PERFORMANCE_EXAMPLES = [
    "load_test_demo",
    "benchmark_scenarios",
]

ALL_EXAMPLES = (
    CONVERSATION_EXAMPLES
    + SPATIAL_EXAMPLES
    + INTEGRATION_EXAMPLES
    + PERFORMANCE_EXAMPLES
)

__all__ = [
    "CONVERSATION_EXAMPLES",
    "SPATIAL_EXAMPLES",
    "INTEGRATION_EXAMPLES",
    "PERFORMANCE_EXAMPLES",
    "ALL_EXAMPLES",
]
