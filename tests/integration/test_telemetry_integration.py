"""Integration tests for telemetry with orchestrator."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.policies.observation import ObservationPolicy, PolicyConfig
from gunn.schemas.messages import View, WorldState
from gunn.schemas.types import Effect, EffectDraft, Intent
from gunn.utils.telemetry import (
    bandwidth_monitor,
    setup_logging,
    setup_tracing,
    system_monitor,
)


class TestTelemetryIntegration:
    """Test telemetry integration with orchestrator."""

    @pytest.fixture
    def config(self):
        """Create orchestrator configuration."""
        return OrchestratorConfig(
            max_agents=10,
            staleness_threshold=5,
            debounce_ms=100.0,
            deadline_ms=5000.0,
            token_budget=1000,
            backpressure_policy="defer",
            default_priority=10,
            use_in_memory_dedup=True,
            dedup_ttl_minutes=5,
            max_dedup_entries=1000,
            dedup_warmup_minutes=1,
            max_queue_depth=100,
            max_log_entries=1000,
            view_cache_size=100,
            compaction_threshold=500,
            snapshot_interval=100,
            max_snapshots=5,
            memory_check_interval_seconds=30.0,
            auto_compaction_enabled=True,
        )

    @pytest.fixture
    def observation_policy(self):
        """Create observation policy."""

        class TestObservationPolicy(ObservationPolicy):
            def filter_world_state(
                self, world_state: WorldState, agent_id: str
            ) -> View:
                return View(
                    agent_id=agent_id,
                    view_seq=0,
                    visible_entities={},
                    visible_relationships={},
                    context_digest="test_digest",
                )

            def should_observe_event(
                self, effect: Effect, agent_id: str, world_state: WorldState
            ) -> bool:
                return True

        config = PolicyConfig(
            distance_limit=100.0,
            relationship_filter=[],
            field_visibility={},
        )
        policy = TestObservationPolicy(config)
        policy.latency_model = MagicMock()
        policy.latency_model.calculate_delay.return_value = 0.01
        return policy

    @pytest.fixture
    async def orchestrator(self, config):
        """Create orchestrator instance."""
        orchestrator = Orchestrator(config, world_id="test_world")
        yield orchestrator
        await orchestrator.shutdown()

    def test_setup_logging_and_tracing(self):
        """Test that logging and tracing can be set up without errors."""
        setup_logging("DEBUG", enable_pii_redaction=True)
        setup_tracing("test_service", enable_fastapi_instrumentation=False)

    @pytest.mark.asyncio
    async def test_orchestrator_telemetry_initialization(self, orchestrator):
        """Test that orchestrator initializes telemetry correctly."""
        # Orchestrator should have logger and tracer
        assert orchestrator._logger is not None
        assert orchestrator._tracer is not None

    @pytest.mark.asyncio
    async def test_agent_registration_telemetry(self, orchestrator, observation_policy):
        """Test telemetry during agent registration."""
        # Mock the latency model
        observation_policy.latency_model.calculate_delay.return_value = 0.01

        # Register an agent
        agent_handle = await orchestrator.register_agent(
            "test_agent", observation_policy
        )

        assert agent_handle is not None
        assert agent_handle.agent_id == "test_agent"

        # Verify agent count is tracked
        assert orchestrator.get_agent_count() == 1

    @pytest.mark.asyncio
    async def test_intent_submission_telemetry(self, orchestrator, observation_policy):
        """Test telemetry during intent submission."""
        # Mock the latency model
        observation_policy.latency_model.calculate_delay.return_value = 0.01

        # Register an agent
        await orchestrator.register_agent("test_agent", observation_policy)

        # Submit an intent
        intent = Intent(
            kind="Speak",
            payload={"text": "Hello world"},
            context_seq=0,
            req_id="test_req_1",
            agent_id="test_agent",
            priority=10,
            schema_version="1.0.0",
        )

        req_id = await orchestrator.submit_intent(intent)
        assert req_id == "test_req_1"

        # Allow some time for processing
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_broadcast_event_telemetry(self, orchestrator, observation_policy):
        """Test telemetry during event broadcasting."""
        # Mock the latency model
        observation_policy.latency_model.calculate_delay.return_value = 0.01

        # Register an agent
        await orchestrator.register_agent("test_agent", observation_policy)

        # Broadcast an event
        effect_draft = EffectDraft(
            kind="MessageEmitted",
            payload={"text": "Test message"},
            source_id="test_agent",
            schema_version="1.0.0",
        )

        await orchestrator.broadcast_event(effect_draft)

        # Verify global sequence was updated
        assert orchestrator.get_latest_seq() > 0

    @pytest.mark.asyncio
    async def test_observation_delivery_telemetry(
        self, orchestrator, observation_policy
    ):
        """Test telemetry during observation delivery."""
        # Mock the latency model
        observation_policy.latency_model.calculate_delay.return_value = 0.01

        # Mock the methods with patch
        with (
            patch.object(observation_policy, "should_observe_event", return_value=True),
            patch.object(observation_policy, "filter_world_state") as mock_filter,
        ):
            mock_view = MagicMock()
            mock_view.view_seq = 1
            mock_view.visible_entities = {}
            mock_view.visible_relationships = {}
            mock_view.context_digest = "test_digest"
            mock_filter.return_value = mock_view

            # Register an agent
            agent_handle = await orchestrator.register_agent(
                "test_agent", observation_policy
            )

            # Broadcast an event
            effect_draft = EffectDraft(
                kind="MessageEmitted",
                payload={"text": "Test message"},
                source_id="test_agent",
                schema_version="1.0.0",
            )

            await orchestrator.broadcast_event(effect_draft)

            # Get the observation (this should trigger telemetry)
            observation = await agent_handle.next_observation()
            assert observation is not None

    @pytest.mark.asyncio
    async def test_system_monitoring(self):
        """Test system monitoring functionality."""
        # Record some system metrics
        memory_bytes = system_monitor.record_memory_usage("test_component")
        cpu_percent = system_monitor.record_cpu_usage("test_component")

        assert memory_bytes >= 0
        assert cpu_percent >= 0

        # Get system stats
        stats = system_monitor.get_system_stats()
        assert "process" in stats
        assert "system" in stats

    def test_bandwidth_monitoring(self):
        """Test bandwidth monitoring functionality."""
        # Record patch bandwidth
        bandwidth_monitor.record_patch_bandwidth("test_agent", 1024, 5, False)

        # Record fallback bandwidth
        bandwidth_monitor.record_patch_bandwidth("test_agent", 10240, 1000, True)

        # Record general data transfer
        bandwidth_monitor.record_data_transfer("inbound", "web_adapter", 2048)

    @pytest.mark.asyncio
    async def test_performance_impact(self, orchestrator, observation_policy):
        """Test that telemetry doesn't significantly impact performance."""
        # Mock the latency model
        observation_policy.latency_model.calculate_delay.return_value = 0.001

        # Mock should_observe_event to return False to skip observation processing
        with patch.object(
            observation_policy, "should_observe_event", return_value=False
        ):
            # Register an agent
            await orchestrator.register_agent("test_agent", observation_policy)

            # Measure performance of multiple operations (reduced to avoid quota limits)
            start_time = time.perf_counter()

            for i in range(10):
                # Submit intents
                intent = Intent(
                    kind="Speak",
                    payload={"text": f"Message {i}"},
                    context_seq=i,
                    req_id=f"test_req_{i}",
                    agent_id="test_agent",
                    priority=10,
                    schema_version="1.0.0",
                )

                await orchestrator.submit_intent(intent)

                # Broadcast events
                effect_draft = EffectDraft(
                    kind="MessageEmitted",
                    payload={"text": f"Event {i}"},
                    source_id="test_agent",
                    schema_version="1.0.0",
                )

                await orchestrator.broadcast_event(effect_draft)

            end_time = time.perf_counter()
            duration = end_time - start_time

            # Should complete 20 operations (10 intents + 10 broadcasts) in reasonable time
            assert duration < 2.0, (
                f"Operations took too long with telemetry: {duration:.3f}s"
            )

            # Allow processing to complete
            await asyncio.sleep(0.5)

    @pytest.mark.asyncio
    async def test_error_telemetry(self, orchestrator, observation_policy):
        """Test telemetry during error conditions."""
        # Mock the latency model
        observation_policy.latency_model.calculate_delay.return_value = 0.01

        # Register an agent
        await orchestrator.register_agent("test_agent", observation_policy)

        # Submit an invalid intent (missing required field)
        with pytest.raises(ValueError):
            invalid_intent = Intent(
                kind="Speak",
                payload={"text": "Hello"},
                context_seq=0,
                # Missing req_id
                agent_id="test_agent",
                priority=10,
                schema_version="1.0.0",
            )
            await orchestrator.submit_intent(invalid_intent)

        # Submit a stale intent
        with pytest.raises(Exception):  # noqa: B017
            stale_intent = Intent(
                kind="Speak",
                payload={"text": "Hello"},
                context_seq=-1000,  # Very stale
                req_id="stale_req",
                agent_id="test_agent",
                priority=10,
                schema_version="1.0.0",
            )
            await orchestrator.submit_intent(stale_intent)

    def test_metrics_collection(self):
        """Test that metrics are being collected."""
        # Import telemetry to ensure metrics are registered

        # Verify metrics are registered
        metric_names = set()
        for collector in REGISTRY._collector_to_names.keys():
            if hasattr(collector, "_name"):
                metric_names.add(collector._name)

        # Check that our key metrics are present
        expected_metrics = {
            "gunn_operations",
            "gunn_operation_duration_seconds",
            "gunn_intents_processed",
            "gunn_conflicts",
        }

        for metric in expected_metrics:
            assert any(metric in name for name in metric_names), (
                f"Metric {metric} not found in {metric_names}"
            )
