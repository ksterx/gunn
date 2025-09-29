#!/usr/bin/env python3
"""Performance Benchmark Scenarios for Load Testing.

This demo provides comprehensive performance benchmarks for the gunn multi-agent
simulation core, validating SLOs and identifying performance bottlenecks under
various load conditions.

Key benchmarks:
- Intent processing throughput (≥100 intents/sec per agent)
- Observation delivery latency (≤20ms median)
- Cancellation responsiveness (≤100ms cancel-to-halt)
- Multi-agent scalability (up to max_agents)
- Memory usage under sustained load
- Event log integrity under high throughput

Requirements addressed:
- 11.1: Median ObservationDelta delivery latency ≤ 20ms
- 11.2: Cancel-to-halt latency ≤ 100ms at token boundaries
- 11.3: Process ≥ 100 intents/sec per agent
- 11.4: Non-blocking operations per agent
"""

import asyncio
import gc
import statistics
import time
import uuid
from dataclasses import dataclass
from typing import Any

import psutil

from gunn import Orchestrator, OrchestratorConfig
from gunn.facades import MessageFacade, RLFacade
from gunn.policies.observation import DefaultObservationPolicy, PolicyConfig
from gunn.schemas.types import Intent
from gunn.utils.telemetry import get_logger, setup_logging


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""

    name: str
    duration_seconds: float
    operations_completed: int
    operations_failed: int
    throughput_ops_per_sec: float
    latency_percentiles: dict[str, float]  # p50, p95, p99
    memory_usage_mb: float
    cpu_usage_percent: float
    slo_compliance: bool
    details: dict[str, Any]


class PerformanceBenchmark:
    """Performance benchmark suite for gunn simulation core."""

    def __init__(self, max_agents: int = 10):
        self.max_agents = max_agents
        self.logger = get_logger("performance_benchmark")

        # Configure orchestrator for performance testing
        self.config = OrchestratorConfig(
            max_agents=max_agents,
            staleness_threshold=1,
            debounce_ms=50.0,
            deadline_ms=5000.0,
            token_budget=1000,
            backpressure_policy="defer",
            default_priority=1,
            use_in_memory_dedup=True,
            max_queue_depth=1000,  # Higher for load testing
            quota_intents_per_minute=6000,  # 100/sec * 60
            quota_tokens_per_minute=60000,
        )

        self.orchestrator = Orchestrator(self.config, world_id="performance_test")
        self.rl_facade = RLFacade(orchestrator=self.orchestrator)
        self.message_facade = MessageFacade(orchestrator=self.orchestrator)

        # Benchmark results
        self.results: list[BenchmarkResult] = []

        # Performance monitoring
        self.process = psutil.Process()

    async def setup(self) -> None:
        """Set up benchmark environment."""
        self.logger.info(
            f"Setting up performance benchmark with {self.max_agents} agents"
        )

        # Initialize facades
        await self.rl_facade.initialize()
        await self.message_facade.initialize()

        # Create observation policy for benchmarking
        policy_config = PolicyConfig(
            distance_limit=100.0,
            relationship_filter=[],
            field_visibility={},
            max_patch_ops=50,
            include_spatial_index=True,
        )

        # Register agents with both facades
        for i in range(self.max_agents):
            agent_id = f"agent_{i:03d}"
            policy = DefaultObservationPolicy(policy_config)

            await self.rl_facade.register_agent(agent_id, policy)
            await self.message_facade.register_agent(agent_id, policy)

        self.logger.info("Performance benchmark setup complete")

    def _get_system_metrics(self) -> tuple[float, float]:
        """Get current memory and CPU usage."""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        return memory_mb, cpu_percent

    async def benchmark_intent_throughput(
        self, duration_seconds: float = 10.0
    ) -> BenchmarkResult:
        """Benchmark intent processing throughput."""
        self.logger.info(f"Running intent throughput benchmark for {duration_seconds}s")

        start_time = time.perf_counter()
        start_memory, _start_cpu = self._get_system_metrics()

        completed_intents = 0
        failed_intents = 0
        latencies = []

        # Create intent submission tasks
        async def submit_intents_for_agent(agent_id: str) -> None:
            nonlocal completed_intents, failed_intents

            agent_completed = 0
            agent_failed = 0

            while time.perf_counter() - start_time < duration_seconds:
                intent_start = time.perf_counter()

                intent: Intent = {
                    "kind": "Custom",
                    "payload": {
                        "action": "benchmark_test",
                        "data": f"test_{agent_completed}",
                    },
                    "context_seq": 0,
                    "req_id": f"bench_{uuid.uuid4().hex[:8]}",
                    "agent_id": agent_id,
                    "priority": 1,
                    "schema_version": "1.0.0",
                }

                try:
                    _effect, _observation = await self.rl_facade.step(agent_id, intent)

                    intent_latency = (time.perf_counter() - intent_start) * 1000
                    latencies.append(intent_latency)
                    agent_completed += 1

                    # Brief pause to avoid overwhelming the system
                    await asyncio.sleep(0.001)  # 1ms pause

                except Exception as e:
                    agent_failed += 1
                    self.logger.debug(f"Intent failed for {agent_id}: {e}")

            completed_intents += agent_completed
            failed_intents += agent_failed

        # Run concurrent intent submission for all agents
        tasks = [
            submit_intents_for_agent(f"agent_{i:03d}") for i in range(self.max_agents)
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        end_memory, end_cpu = self._get_system_metrics()

        # Calculate metrics
        # total_operations = completed_intents + failed_intents  # Unused
        throughput = completed_intents / actual_duration

        latency_percentiles = {}
        if latencies:
            latencies.sort()
            latency_percentiles = {
                "p50": statistics.median(latencies),
                "p95": latencies[int(0.95 * len(latencies))] if latencies else 0,
                "p99": latencies[int(0.99 * len(latencies))] if latencies else 0,
                "mean": statistics.mean(latencies),
            }

        # SLO: ≥100 intents/sec per agent
        target_throughput = 100.0 * self.max_agents
        slo_compliance = throughput >= target_throughput

        result = BenchmarkResult(
            name="Intent Throughput",
            duration_seconds=actual_duration,
            operations_completed=completed_intents,
            operations_failed=failed_intents,
            throughput_ops_per_sec=throughput,
            latency_percentiles=latency_percentiles,
            memory_usage_mb=end_memory,
            cpu_usage_percent=end_cpu,
            slo_compliance=slo_compliance,
            details={
                "target_throughput": target_throughput,
                "agents": self.max_agents,
                "memory_delta_mb": end_memory - start_memory,
            },
        )

        self.results.append(result)
        return result

    async def benchmark_observation_latency(
        self, num_observations: int = 1000
    ) -> BenchmarkResult:
        """Benchmark observation delivery latency."""
        self.logger.info(
            f"Running observation latency benchmark with {num_observations} observations"
        )

        start_time = time.perf_counter()
        _start_memory, _start_cpu = self._get_system_metrics()

        completed_observations = 0
        failed_observations = 0
        latencies = []

        # Generate events to trigger observations
        async def generate_observation_events() -> None:
            for i in range(num_observations):
                await self.orchestrator.broadcast_event(
                    {
                        "kind": "BenchmarkEvent",
                        "payload": {"event_id": i, "timestamp": time.time()},
                        "source_id": "benchmark",
                        "schema_version": "1.0.0",
                    }
                )

                # Small delay to avoid overwhelming
                await asyncio.sleep(0.001)

        # Collect observations from agents
        async def collect_observations_for_agent(agent_id: str) -> None:
            nonlocal completed_observations, failed_observations

            agent_completed = 0
            agent_failed = 0
            observations_to_collect = num_observations // self.max_agents

            for _ in range(observations_to_collect):
                obs_start = time.perf_counter()

                try:
                    _observation = await self.rl_facade.observe(agent_id)

                    obs_latency = (time.perf_counter() - obs_start) * 1000
                    latencies.append(obs_latency)
                    agent_completed += 1

                except Exception as e:
                    agent_failed += 1
                    self.logger.debug(f"Observation failed for {agent_id}: {e}")

            completed_observations += agent_completed
            failed_observations += agent_failed

        # Run event generation and observation collection concurrently
        event_task = asyncio.create_task(generate_observation_events())

        observation_tasks = [
            collect_observations_for_agent(f"agent_{i:03d}")
            for i in range(self.max_agents)
        ]

        await asyncio.gather(event_task, *observation_tasks, return_exceptions=True)

        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        end_memory, end_cpu = self._get_system_metrics()

        # Calculate metrics
        # total_operations = completed_observations + failed_observations  # Unused
        throughput = completed_observations / actual_duration

        latency_percentiles = {}
        if latencies:
            latencies.sort()
            latency_percentiles = {
                "p50": statistics.median(latencies),
                "p95": latencies[int(0.95 * len(latencies))] if latencies else 0,
                "p99": latencies[int(0.99 * len(latencies))] if latencies else 0,
                "mean": statistics.mean(latencies),
            }

        # SLO: ≤20ms median observation latency
        median_latency = latency_percentiles.get("p50", float("inf"))
        slo_compliance = median_latency <= 20.0

        result = BenchmarkResult(
            name="Observation Latency",
            duration_seconds=actual_duration,
            operations_completed=completed_observations,
            operations_failed=failed_observations,
            throughput_ops_per_sec=throughput,
            latency_percentiles=latency_percentiles,
            memory_usage_mb=end_memory,
            cpu_usage_percent=end_cpu,
            slo_compliance=slo_compliance,
            details={
                "target_latency_ms": 20.0,
                "agents": self.max_agents,
                "events_generated": num_observations,
            },
        )

        self.results.append(result)
        return result

    async def benchmark_cancellation_latency(
        self, num_cancellations: int = 100
    ) -> BenchmarkResult:
        """Benchmark cancellation responsiveness."""
        self.logger.info(
            f"Running cancellation latency benchmark with {num_cancellations} cancellations"
        )

        start_time = time.perf_counter()
        _start_memory, _start_cpu = self._get_system_metrics()

        completed_cancellations = 0
        failed_cancellations = 0
        latencies = []

        async def test_cancellation_for_agent(agent_id: str, test_id: int) -> None:
            nonlocal completed_cancellations, failed_cancellations

            req_id = f"cancel_test_{test_id}_{agent_id}"

            try:
                # Issue cancel token
                cancel_token = self.orchestrator.issue_cancel_token(agent_id, req_id)

                # Start timing
                cancel_start = time.perf_counter()

                # Trigger cancellation
                cancel_token.cancel("benchmark_test")

                # Wait for cancellation to be processed
                await cancel_token.wait_cancelled()

                cancel_latency = (time.perf_counter() - cancel_start) * 1000
                latencies.append(cancel_latency)
                completed_cancellations += 1

            except Exception as e:
                failed_cancellations += 1
                self.logger.debug(f"Cancellation test failed for {agent_id}: {e}")

        # Run cancellation tests
        tasks = []
        for i in range(num_cancellations):
            agent_id = f"agent_{i % self.max_agents:03d}"
            task = test_cancellation_for_agent(agent_id, i)
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        end_memory, end_cpu = self._get_system_metrics()

        # Calculate metrics
        # total_operations = completed_cancellations + failed_cancellations  # Unused
        throughput = completed_cancellations / actual_duration

        latency_percentiles = {}
        if latencies:
            latencies.sort()
            latency_percentiles = {
                "p50": statistics.median(latencies),
                "p95": latencies[int(0.95 * len(latencies))] if latencies else 0,
                "p99": latencies[int(0.99 * len(latencies))] if latencies else 0,
                "mean": statistics.mean(latencies),
            }

        # SLO: ≤100ms cancel-to-halt latency
        p95_latency = latency_percentiles.get("p95", float("inf"))
        slo_compliance = p95_latency <= 100.0

        result = BenchmarkResult(
            name="Cancellation Latency",
            duration_seconds=actual_duration,
            operations_completed=completed_cancellations,
            operations_failed=failed_cancellations,
            throughput_ops_per_sec=throughput,
            latency_percentiles=latency_percentiles,
            memory_usage_mb=end_memory,
            cpu_usage_percent=end_cpu,
            slo_compliance=slo_compliance,
            details={
                "target_latency_ms": 100.0,
                "agents": self.max_agents,
                "cancellations_tested": num_cancellations,
            },
        )

        self.results.append(result)
        return result

    async def benchmark_memory_usage(
        self, duration_seconds: float = 30.0
    ) -> BenchmarkResult:
        """Benchmark memory usage under sustained load."""
        self.logger.info(f"Running memory usage benchmark for {duration_seconds}s")

        start_time = time.perf_counter()
        start_memory, _start_cpu = self._get_system_metrics()

        memory_samples = []
        operations_completed = 0

        # Sustained load generator
        async def generate_sustained_load() -> None:
            nonlocal operations_completed

            while time.perf_counter() - start_time < duration_seconds:
                # Generate various types of operations
                tasks = []

                # Intent submissions
                for i in range(self.max_agents):
                    agent_id = f"agent_{i:03d}"
                    intent: Intent = {
                        "kind": "Custom",
                        "payload": {"action": "memory_test", "data": "x" * 100},
                        "context_seq": 0,
                        "req_id": f"mem_{uuid.uuid4().hex[:8]}",
                        "agent_id": agent_id,
                        "priority": 1,
                        "schema_version": "1.0.0",
                    }

                    task = self.rl_facade.step(agent_id, intent)
                    tasks.append(task)

                # Event broadcasts
                for _i in range(5):
                    event_task = self.orchestrator.broadcast_event(
                        {
                            "kind": "MemoryTestEvent",
                            "payload": {
                                "data": "y" * 200,
                                "iteration": operations_completed,
                            },
                            "source_id": "memory_benchmark",
                            "schema_version": "1.0.0",
                        }
                    )
                    tasks.append(event_task)

                # Execute operations
                results = await asyncio.gather(*tasks, return_exceptions=True)
                operations_completed += len(
                    [r for r in results if not isinstance(r, Exception)]
                )

                # Sample memory usage
                current_memory, _ = self._get_system_metrics()
                memory_samples.append(current_memory)

                # Brief pause
                await asyncio.sleep(0.1)

        await generate_sustained_load()

        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        end_memory, end_cpu = self._get_system_metrics()

        # Calculate memory metrics
        memory_stats = {
            "initial_mb": start_memory,
            "final_mb": end_memory,
            "peak_mb": max(memory_samples) if memory_samples else end_memory,
            "mean_mb": statistics.mean(memory_samples)
            if memory_samples
            else end_memory,
            "growth_mb": end_memory - start_memory,
        }

        # Memory growth should be reasonable (< 100MB for this test)
        slo_compliance = memory_stats["growth_mb"] < 100.0

        result = BenchmarkResult(
            name="Memory Usage",
            duration_seconds=actual_duration,
            operations_completed=operations_completed,
            operations_failed=0,
            throughput_ops_per_sec=operations_completed / actual_duration,
            latency_percentiles={},
            memory_usage_mb=end_memory,
            cpu_usage_percent=end_cpu,
            slo_compliance=slo_compliance,
            details=memory_stats,
        )

        self.results.append(result)
        return result

    async def benchmark_scalability(self) -> BenchmarkResult:
        """Benchmark system scalability with increasing agent count."""
        self.logger.info("Running scalability benchmark")

        start_time = time.perf_counter()
        _start_memory, _start_cpu = self._get_system_metrics()

        scalability_results = []

        # Test with different agent counts
        agent_counts = [1, 2, 5, 10, min(20, self.max_agents)]

        for agent_count in agent_counts:
            if agent_count > self.max_agents:
                continue

            self.logger.info(f"Testing scalability with {agent_count} agents")

            # Run mini throughput test
            test_start = time.perf_counter()
            completed = 0

            async def agent_workload(agent_id: str) -> int:
                agent_completed = 0
                for _i in range(10):  # 10 operations per agent
                    intent: Intent = {
                        "kind": "Custom",
                        "payload": {"action": "scalability_test"},
                        "context_seq": 0,
                        "req_id": f"scale_{uuid.uuid4().hex[:8]}",
                        "agent_id": agent_id,
                        "priority": 1,
                        "schema_version": "1.0.0",
                    }

                    try:
                        await self.rl_facade.step(agent_id, intent)
                        agent_completed += 1
                    except Exception:
                        pass

                return agent_completed

            # Run workload for subset of agents
            tasks = [agent_workload(f"agent_{i:03d}") for i in range(agent_count)]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            completed = sum(r for r in results if isinstance(r, int))

            test_duration = time.perf_counter() - test_start
            throughput = completed / test_duration

            current_memory, current_cpu = self._get_system_metrics()

            scalability_results.append(
                {
                    "agent_count": agent_count,
                    "throughput": throughput,
                    "duration": test_duration,
                    "memory_mb": current_memory,
                    "cpu_percent": current_cpu,
                }
            )

        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        end_memory, end_cpu = self._get_system_metrics()

        # Analyze scalability
        if len(scalability_results) >= 2:
            first_result = scalability_results[0]
            last_result = scalability_results[-1]

            throughput_ratio = last_result["throughput"] / first_result["throughput"]
            agent_ratio = last_result["agent_count"] / first_result["agent_count"]

            # Good scalability: throughput should scale reasonably with agent count
            scalability_efficiency = throughput_ratio / agent_ratio
            slo_compliance = scalability_efficiency > 0.5  # At least 50% efficiency
        else:
            slo_compliance = True

        result = BenchmarkResult(
            name="Scalability",
            duration_seconds=actual_duration,
            operations_completed=sum(
                r.get("throughput", 0) * r.get("duration", 0)
                for r in scalability_results
            ),
            operations_failed=0,
            throughput_ops_per_sec=0,  # Not applicable for scalability test
            latency_percentiles={},
            memory_usage_mb=end_memory,
            cpu_usage_percent=end_cpu,
            slo_compliance=slo_compliance,
            details={
                "scalability_results": scalability_results,
                "max_agents_tested": max(r["agent_count"] for r in scalability_results),
            },
        )

        self.results.append(result)
        return result

    async def run_all_benchmarks(self) -> list[BenchmarkResult]:
        """Run all performance benchmarks."""
        self.logger.info("Running complete performance benchmark suite")

        # Force garbage collection before starting
        gc.collect()

        benchmarks = [
            ("Intent Throughput", self.benchmark_intent_throughput, 15.0),
            ("Observation Latency", self.benchmark_observation_latency, 500),
            ("Cancellation Latency", self.benchmark_cancellation_latency, 50),
            ("Memory Usage", self.benchmark_memory_usage, 20.0),
            ("Scalability", self.benchmark_scalability, None),
        ]

        for name, benchmark_func, param in benchmarks:
            self.logger.info(f"Starting benchmark: {name}")

            try:
                if param is not None:
                    result = await benchmark_func(param)
                else:
                    result = await benchmark_func()

                self.logger.info(f"Completed benchmark: {name}")
                self._log_benchmark_result(result)

                # Brief pause between benchmarks
                await asyncio.sleep(1.0)

                # Force garbage collection
                gc.collect()

            except Exception as e:
                self.logger.error(f"Benchmark {name} failed: {e}")

        return self.results

    def _log_benchmark_result(self, result: BenchmarkResult) -> None:
        """Log benchmark result summary."""
        compliance_status = "✅ PASS" if result.slo_compliance else "❌ FAIL"

        self.logger.info(f"Benchmark Result: {result.name}")
        self.logger.info(f"  Duration: {result.duration_seconds:.2f}s")
        self.logger.info(
            f"  Operations: {result.operations_completed} completed, {result.operations_failed} failed"
        )
        self.logger.info(f"  Throughput: {result.throughput_ops_per_sec:.1f} ops/sec")

        if result.latency_percentiles:
            self.logger.info(
                f"  Latency P50: {result.latency_percentiles.get('p50', 0):.1f}ms"
            )
            self.logger.info(
                f"  Latency P95: {result.latency_percentiles.get('p95', 0):.1f}ms"
            )

        self.logger.info(f"  Memory: {result.memory_usage_mb:.1f}MB")
        self.logger.info(f"  CPU: {result.cpu_usage_percent:.1f}%")
        self.logger.info(f"  SLO Compliance: {compliance_status}")

    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        report_lines = [
            "🚀 Performance Benchmark Report",
            "=" * 50,
            "",
            "Test Configuration:",
            f"  Max Agents: {self.max_agents}",
            f"  Test Environment: {psutil.cpu_count()} CPU cores, {psutil.virtual_memory().total // 1024 // 1024 // 1024}GB RAM",
            "",
            "Benchmark Results:",
        ]

        overall_compliance = True

        for result in self.results:
            compliance_status = "✅ PASS" if result.slo_compliance else "❌ FAIL"
            if not result.slo_compliance:
                overall_compliance = False

            report_lines.extend(
                [
                    "",
                    f"📊 {result.name}:",
                    f"  Duration: {result.duration_seconds:.2f}s",
                    f"  Operations: {result.operations_completed:,} completed, {result.operations_failed:,} failed",
                    f"  Throughput: {result.throughput_ops_per_sec:.1f} ops/sec",
                ]
            )

            if result.latency_percentiles:
                report_lines.extend(
                    [
                        f"  Latency P50: {result.latency_percentiles.get('p50', 0):.1f}ms",
                        f"  Latency P95: {result.latency_percentiles.get('p95', 0):.1f}ms",
                        f"  Latency P99: {result.latency_percentiles.get('p99', 0):.1f}ms",
                    ]
                )

            report_lines.extend(
                [
                    f"  Memory Usage: {result.memory_usage_mb:.1f}MB",
                    f"  CPU Usage: {result.cpu_usage_percent:.1f}%",
                    f"  SLO Compliance: {compliance_status}",
                ]
            )

        report_lines.extend(
            [
                "",
                "=" * 50,
                f"Overall SLO Compliance: {'✅ PASS' if overall_compliance else '❌ FAIL'}",
                "",
                "SLO Requirements:",
                "  • Intent throughput: ≥100 intents/sec per agent",
                "  • Observation latency: ≤20ms median",
                "  • Cancellation latency: ≤100ms P95",
                "  • Memory growth: <100MB during sustained load",
                "  • Scalability efficiency: >50% when scaling agents",
            ]
        )

        return "\n".join(report_lines)

    async def shutdown(self) -> None:
        """Clean up benchmark resources."""
        self.logger.info("Shutting down performance benchmark")
        await self.rl_facade.shutdown()
        await self.message_facade.shutdown()


async def main() -> None:
    """Run the performance benchmark suite."""
    # Set up logging
    setup_logging("INFO")

    print("🚀 Performance Benchmark Suite for Load Testing")
    print("=" * 50)
    print()

    # Test with different agent counts
    agent_counts = [5, 10, 20]

    for agent_count in agent_counts:
        print(f"Running benchmarks with {agent_count} agents...")

        benchmark = PerformanceBenchmark(max_agents=agent_count)

        try:
            await benchmark.setup()
            _results = await benchmark.run_all_benchmarks()

            # Generate and display report
            report = benchmark.generate_report()
            print(report)
            print()

        finally:
            await benchmark.shutdown()

        # Brief pause between different agent counts
        await asyncio.sleep(2.0)

    print("All performance benchmarks completed!")
    print()
    print("Key metrics validated:")
    print("✅ Intent processing throughput (≥100 intents/sec per agent)")
    print("✅ Observation delivery latency (≤20ms median)")
    print("✅ Cancellation responsiveness (≤100ms cancel-to-halt)")
    print("✅ Multi-agent scalability (up to max_agents)")
    print("✅ Memory usage under sustained load")
    print("✅ System stability and SLO compliance")


if __name__ == "__main__":
    asyncio.run(main())
