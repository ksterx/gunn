"""
Performance monitoring and optimization for the battle demo.

This module provides comprehensive performance monitoring including:
- Decision making latency tracking
- Concurrent processing metrics
- API response time monitoring
- WebSocket update performance
- Frame rate monitoring for frontend
- Memory usage tracking for long-running simulations
- Real-time performance optimization
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, Summary

from gunn.utils.telemetry import (
    get_logger,
    system_monitor,
)

# Configure logging
logger = logging.getLogger(__name__)

# Demo-specific performance metrics
DECISION_MAKING_LATENCY = Histogram(
    "battle_demo_decision_latency_seconds",
    "AI decision making latency per agent",
    ["agent_id", "decision_type"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

CONCURRENT_PROCESSING_TIME = Histogram(
    "battle_demo_concurrent_processing_seconds",
    "Time to process all agents concurrently",
    ["agent_count"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

API_RESPONSE_TIME = Histogram(
    "battle_demo_api_response_seconds",
    "API endpoint response times",
    ["endpoint", "method", "status_code"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

WEBSOCKET_UPDATE_LATENCY = Histogram(
    "battle_demo_websocket_update_seconds",
    "WebSocket update broadcast latency",
    ["update_type"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

FRAME_RATE_GAUGE = Gauge("battle_demo_frame_rate_fps", "Frontend frame rate in FPS")

MEMORY_USAGE_TREND = Gauge(
    "battle_demo_memory_usage_bytes",
    "Memory usage trend over time",
    ["component", "measurement_type"],
)

GAME_LOOP_PERFORMANCE = Summary(
    "battle_demo_game_loop_duration_seconds", "Game loop iteration performance"
)

AGENT_PROCESSING_QUEUE_SIZE = Gauge(
    "battle_demo_agent_queue_size",
    "Number of agents waiting for processing",
    ["queue_type"],
)

PERFORMANCE_ALERTS = Counter(
    "battle_demo_performance_alerts_total",
    "Performance alerts triggered",
    ["alert_type", "severity"],
)

OPTIMIZATION_ACTIONS = Counter(
    "battle_demo_optimization_actions_total",
    "Optimization actions taken",
    ["action_type", "trigger"],
)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""

    timestamp: float
    decision_latencies: dict[str, float] = field(default_factory=dict)
    concurrent_processing_time: float = 0.0
    api_response_times: dict[str, float] = field(default_factory=dict)
    websocket_latency: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    frame_rate: float = 0.0
    active_agents: int = 0
    queue_sizes: dict[str, int] = field(default_factory=dict)


@dataclass
class PerformanceThresholds:
    """Performance thresholds for alerting and optimization."""

    max_decision_latency: float = 5.0  # seconds
    max_concurrent_processing_time: float = 10.0  # seconds
    max_api_response_time: float = 1.0  # seconds
    max_websocket_latency: float = 0.1  # seconds
    min_frame_rate: float = 30.0  # FPS
    max_memory_usage: float = 1024 * 1024 * 1024  # 1GB
    max_cpu_usage: float = 80.0  # percent
    max_queue_size: int = 100


class PerformanceOptimizer:
    """Handles performance optimization actions."""

    def __init__(self):
        self.logger = get_logger("battle_demo.performance.optimizer")
        self.optimization_history: deque = deque(maxlen=100)

    async def optimize_decision_making(self, avg_latency: float) -> bool:
        """Optimize AI decision making performance."""
        if avg_latency > 3.0:
            self.logger.warning(
                "High decision making latency detected",
                avg_latency=avg_latency,
                optimization="reducing_model_complexity",
            )

            OPTIMIZATION_ACTIONS.labels(
                action_type="reduce_model_complexity", trigger="high_decision_latency"
            ).inc()

            # Record optimization action
            self.optimization_history.append(
                {
                    "timestamp": time.time(),
                    "action": "reduce_model_complexity",
                    "trigger": "high_decision_latency",
                    "value": avg_latency,
                }
            )

            return True
        return False

    async def optimize_memory_usage(self, memory_bytes: float) -> bool:
        """Optimize memory usage when thresholds are exceeded."""
        if memory_bytes > 512 * 1024 * 1024:  # 512MB
            self.logger.warning(
                "High memory usage detected",
                memory_bytes=memory_bytes,
                optimization="garbage_collection",
            )

            # Trigger garbage collection
            import gc

            collected = gc.collect()

            OPTIMIZATION_ACTIONS.labels(
                action_type="garbage_collection", trigger="high_memory_usage"
            ).inc()

            self.optimization_history.append(
                {
                    "timestamp": time.time(),
                    "action": "garbage_collection",
                    "trigger": "high_memory_usage",
                    "collected_objects": collected,
                    "memory_before": memory_bytes,
                }
            )

            return True
        return False

    async def optimize_concurrent_processing(
        self, processing_time: float, agent_count: int
    ) -> bool:
        """Optimize concurrent processing performance."""
        if processing_time > 5.0 and agent_count > 3:
            self.logger.warning(
                "Slow concurrent processing detected",
                processing_time=processing_time,
                agent_count=agent_count,
                optimization="batch_size_reduction",
            )

            OPTIMIZATION_ACTIONS.labels(
                action_type="batch_size_reduction", trigger="slow_concurrent_processing"
            ).inc()

            self.optimization_history.append(
                {
                    "timestamp": time.time(),
                    "action": "batch_size_reduction",
                    "trigger": "slow_concurrent_processing",
                    "processing_time": processing_time,
                    "agent_count": agent_count,
                }
            )

            return True
        return False


class BattlePerformanceMonitor:
    """Comprehensive performance monitoring for the battle demo."""

    def __init__(self, monitoring_interval: float = 1.0):
        """
        Initialize the performance monitor.

        Args:
            monitoring_interval: How often to collect performance metrics (seconds)
        """
        self.monitoring_interval = monitoring_interval
        self.logger = get_logger("battle_demo.performance")
        self.thresholds = PerformanceThresholds()
        self.optimizer = PerformanceOptimizer()

        # Performance data storage
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 measurements
        self.decision_latencies: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.api_response_times: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: asyncio.Task | None = None

        # Performance counters
        self.total_decisions_processed = 0
        self.total_api_requests = 0
        self.total_websocket_updates = 0

        # Alert state
        self.active_alerts: dict[str, float] = {}  # alert_type -> timestamp

    async def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False

        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that collects metrics periodically."""
        try:
            while self.monitoring_active:
                try:
                    # Collect current metrics
                    metrics = await self._collect_metrics()

                    # Store metrics
                    self.metrics_history.append(metrics)

                    # Check thresholds and trigger alerts
                    await self._check_performance_thresholds(metrics)

                    # Update Prometheus metrics
                    await self._update_prometheus_metrics(metrics)

                    # Perform optimizations if needed
                    await self._perform_optimizations(metrics)

                except Exception as e:
                    self.logger.error("Error in monitoring loop", error=str(e))

                await asyncio.sleep(self.monitoring_interval)

        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error("Monitoring loop failed", error=str(e))

    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        current_time = time.time()

        # Collect system metrics
        memory_usage = system_monitor.record_memory_usage("battle_demo")
        cpu_usage = system_monitor.record_cpu_usage("battle_demo")

        # Calculate average decision latencies
        avg_decision_latencies = {}
        for agent_id, latencies in self.decision_latencies.items():
            if latencies:
                avg_decision_latencies[agent_id] = sum(latencies) / len(latencies)

        # Calculate average API response times
        avg_api_times = {}
        for endpoint, times in self.api_response_times.items():
            if times:
                avg_api_times[endpoint] = sum(times) / len(times)

        return PerformanceMetrics(
            timestamp=current_time,
            decision_latencies=avg_decision_latencies,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            api_response_times=avg_api_times,
        )

    async def _check_performance_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check performance thresholds and trigger alerts."""
        current_time = time.time()

        # Check decision latency thresholds
        for agent_id, latency in metrics.decision_latencies.items():
            if latency > self.thresholds.max_decision_latency:
                alert_key = f"high_decision_latency_{agent_id}"
                if alert_key not in self.active_alerts:
                    self.active_alerts[alert_key] = current_time
                    PERFORMANCE_ALERTS.labels(
                        alert_type="high_decision_latency", severity="warning"
                    ).inc()

                    self.logger.warning(
                        "High decision latency alert",
                        agent_id=agent_id,
                        latency=latency,
                        threshold=self.thresholds.max_decision_latency,
                    )

        # Check memory usage
        if metrics.memory_usage > self.thresholds.max_memory_usage:
            alert_key = "high_memory_usage"
            if alert_key not in self.active_alerts:
                self.active_alerts[alert_key] = current_time
                PERFORMANCE_ALERTS.labels(
                    alert_type="high_memory_usage", severity="critical"
                ).inc()

                self.logger.error(
                    "High memory usage alert",
                    memory_bytes=metrics.memory_usage,
                    threshold=self.thresholds.max_memory_usage,
                )

        # Check CPU usage
        if metrics.cpu_usage > self.thresholds.max_cpu_usage:
            alert_key = "high_cpu_usage"
            if alert_key not in self.active_alerts:
                self.active_alerts[alert_key] = current_time
                PERFORMANCE_ALERTS.labels(
                    alert_type="high_cpu_usage", severity="warning"
                ).inc()

                self.logger.warning(
                    "High CPU usage alert",
                    cpu_percent=metrics.cpu_usage,
                    threshold=self.thresholds.max_cpu_usage,
                )

        # Clear resolved alerts (older than 60 seconds)
        resolved_alerts = [
            alert
            for alert, timestamp in self.active_alerts.items()
            if current_time - timestamp > 60.0
        ]
        for alert in resolved_alerts:
            del self.active_alerts[alert]

    async def _update_prometheus_metrics(self, metrics: PerformanceMetrics) -> None:
        """Update Prometheus metrics with current values."""
        # Update memory and CPU gauges
        MEMORY_USAGE_TREND.labels(
            component="battle_demo", measurement_type="current"
        ).set(metrics.memory_usage)

        # Update decision latency metrics
        for agent_id, latency in metrics.decision_latencies.items():
            DECISION_MAKING_LATENCY.labels(
                agent_id=agent_id, decision_type="ai_decision"
            ).observe(latency)

    async def _perform_optimizations(self, metrics: PerformanceMetrics) -> None:
        """Perform automatic optimizations based on metrics."""
        # Optimize decision making if needed
        if metrics.decision_latencies:
            avg_latency = sum(metrics.decision_latencies.values()) / len(
                metrics.decision_latencies
            )
            await self.optimizer.optimize_decision_making(avg_latency)

        # Optimize memory usage if needed
        await self.optimizer.optimize_memory_usage(metrics.memory_usage)

    @asynccontextmanager
    async def monitor_decision_making(
        self, agent_id: str, decision_type: str = "ai_decision"
    ):
        """Context manager to monitor AI decision making latency."""
        start_time = time.perf_counter()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            latency = end_time - start_time

            # Record latency
            self.decision_latencies[agent_id].append(latency)
            self.total_decisions_processed += 1

            # Update Prometheus metrics
            DECISION_MAKING_LATENCY.labels(
                agent_id=agent_id, decision_type=decision_type
            ).observe(latency)

            self.logger.debug(
                "Decision making completed",
                agent_id=agent_id,
                latency_seconds=latency,
                decision_type=decision_type,
            )

    @asynccontextmanager
    async def monitor_concurrent_processing(self, agent_count: int):
        """Context manager to monitor concurrent agent processing."""
        start_time = time.perf_counter()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            processing_time = end_time - start_time

            # Update metrics
            CONCURRENT_PROCESSING_TIME.labels(agent_count=str(agent_count)).observe(
                processing_time
            )

            # Check for optimization opportunities
            await self.optimizer.optimize_concurrent_processing(
                processing_time, agent_count
            )

            self.logger.debug(
                "Concurrent processing completed",
                agent_count=agent_count,
                processing_time_seconds=processing_time,
            )

    @asynccontextmanager
    async def monitor_api_request(self, endpoint: str, method: str):
        """Context manager to monitor API request performance."""
        start_time = time.perf_counter()
        status_code = "200"  # Default success

        try:
            yield
        except Exception:
            status_code = "500"  # Error status
            raise
        finally:
            end_time = time.perf_counter()
            response_time = end_time - start_time

            # Record response time
            endpoint_key = f"{method}_{endpoint}"
            self.api_response_times[endpoint_key].append(response_time)
            self.total_api_requests += 1

            # Update Prometheus metrics
            API_RESPONSE_TIME.labels(
                endpoint=endpoint, method=method, status_code=status_code
            ).observe(response_time)

            self.logger.debug(
                "API request completed",
                endpoint=endpoint,
                method=method,
                response_time_seconds=response_time,
                status_code=status_code,
            )

    @asynccontextmanager
    async def monitor_websocket_update(self, update_type: str = "game_state"):
        """Context manager to monitor WebSocket update performance."""
        start_time = time.perf_counter()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            latency = end_time - start_time

            self.total_websocket_updates += 1

            # Update Prometheus metrics
            WEBSOCKET_UPDATE_LATENCY.labels(update_type=update_type).observe(latency)

            self.logger.debug(
                "WebSocket update completed",
                update_type=update_type,
                latency_seconds=latency,
            )

    def record_frame_rate(self, fps: float) -> None:
        """Record frontend frame rate."""
        FRAME_RATE_GAUGE.set(fps)

        # Check frame rate threshold
        if fps < self.thresholds.min_frame_rate:
            alert_key = "low_frame_rate"
            current_time = time.time()

            if alert_key not in self.active_alerts:
                self.active_alerts[alert_key] = current_time
                PERFORMANCE_ALERTS.labels(
                    alert_type="low_frame_rate", severity="warning"
                ).inc()

                self.logger.warning(
                    "Low frame rate detected",
                    fps=fps,
                    threshold=self.thresholds.min_frame_rate,
                )

    def record_queue_size(self, queue_type: str, size: int) -> None:
        """Record agent processing queue size."""
        AGENT_PROCESSING_QUEUE_SIZE.labels(queue_type=queue_type).set(size)

        # Check queue size threshold
        if size > self.thresholds.max_queue_size:
            alert_key = f"large_queue_{queue_type}"
            current_time = time.time()

            if alert_key not in self.active_alerts:
                self.active_alerts[alert_key] = current_time
                PERFORMANCE_ALERTS.labels(
                    alert_type="large_queue", severity="warning"
                ).inc()

                self.logger.warning(
                    "Large processing queue detected",
                    queue_type=queue_type,
                    size=size,
                    threshold=self.thresholds.max_queue_size,
                )

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {"status": "no_data"}

        latest_metrics = self.metrics_history[-1]

        # Calculate averages over recent history
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements

        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)

        # Calculate decision latency statistics
        all_latencies = []
        for metrics in recent_metrics:
            all_latencies.extend(metrics.decision_latencies.values())

        decision_stats = {}
        if all_latencies:
            decision_stats = {
                "avg_latency": sum(all_latencies) / len(all_latencies),
                "max_latency": max(all_latencies),
                "min_latency": min(all_latencies),
                "count": len(all_latencies),
            }

        return {
            "timestamp": latest_metrics.timestamp,
            "system": {
                "memory_usage_bytes": latest_metrics.memory_usage,
                "cpu_usage_percent": latest_metrics.cpu_usage,
                "avg_memory_bytes": avg_memory,
                "avg_cpu_percent": avg_cpu,
            },
            "decision_making": decision_stats,
            "counters": {
                "total_decisions": self.total_decisions_processed,
                "total_api_requests": self.total_api_requests,
                "total_websocket_updates": self.total_websocket_updates,
            },
            "active_alerts": list(self.active_alerts.keys()),
            "optimization_history": list(self.optimizer.optimization_history)[
                -5:
            ],  # Last 5 optimizations
            "thresholds": {
                "max_decision_latency": self.thresholds.max_decision_latency,
                "max_memory_usage": self.thresholds.max_memory_usage,
                "max_cpu_usage": self.thresholds.max_cpu_usage,
                "min_frame_rate": self.thresholds.min_frame_rate,
            },
        }

    async def run_performance_test(self, duration: float = 60.0) -> dict[str, Any]:
        """Run a comprehensive performance test."""
        self.logger.info(f"Starting performance test for {duration} seconds")

        test_start = time.time()
        initial_metrics = await self._collect_metrics()

        # Wait for test duration
        await asyncio.sleep(duration)

        final_metrics = await self._collect_metrics()
        test_end = time.time()

        # Calculate test results
        memory_delta = final_metrics.memory_usage - initial_metrics.memory_usage
        cpu_avg = sum(
            m.cpu_usage for m in self.metrics_history if m.timestamp >= test_start
        ) / max(1, len([m for m in self.metrics_history if m.timestamp >= test_start]))

        test_results = {
            "test_duration": test_end - test_start,
            "initial_memory": initial_metrics.memory_usage,
            "final_memory": final_metrics.memory_usage,
            "memory_delta": memory_delta,
            "avg_cpu_during_test": cpu_avg,
            "decisions_processed": self.total_decisions_processed,
            "api_requests_processed": self.total_api_requests,
            "websocket_updates_sent": self.total_websocket_updates,
            "alerts_triggered": len(self.active_alerts),
            "optimizations_performed": len(self.optimizer.optimization_history),
        }

        self.logger.info("Performance test completed", **test_results)
        return test_results


# Global performance monitor instance
performance_monitor = BattlePerformanceMonitor()
