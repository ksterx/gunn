#!/usr/bin/env python3
"""
Performance monitoring demonstration for the battle demo.

This script demonstrates the comprehensive performance monitoring system
including decision making latency, concurrent processing, API monitoring,
frame rate tracking, and optimization features.
"""

import asyncio
import logging
import os
import sys
import time

# Add the project root to Python path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from demo.backend.performance_monitor import BattlePerformanceMonitor
from demo.frontend.performance_monitor import FrontendPerformanceMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_backend_monitoring():
    """Demonstrate backend performance monitoring capabilities."""
    print("🔍 Backend Performance Monitoring Demo")
    print("=" * 50)

    monitor = BattlePerformanceMonitor(monitoring_interval=0.1)

    # Start monitoring
    await monitor.start_monitoring()

    try:
        print("📊 Testing decision making latency monitoring...")

        # Simulate AI decision making for multiple agents
        agent_ids = [f"agent_{i}" for i in range(6)]

        for round_num in range(3):
            print(
                f"  Round {round_num + 1}: Processing decisions for {len(agent_ids)} agents"
            )

            # Monitor concurrent processing
            async with monitor.monitor_concurrent_processing(len(agent_ids)):
                decision_tasks = []

                for agent_id in agent_ids:

                    async def make_decision(aid):
                        async with monitor.monitor_decision_making(aid):
                            # Simulate variable decision making time
                            delay = 0.05 + (0.02 * (hash(aid) % 5))
                            await asyncio.sleep(delay)

                    decision_tasks.append(make_decision(agent_id))

                await asyncio.gather(*decision_tasks)

        print("📡 Testing API request monitoring...")

        # Simulate API requests
        api_endpoints = [
            ("/api/game/state", "GET"),
            ("/api/game/stats", "GET"),
            ("/api/performance/summary", "GET"),
        ]

        for endpoint, method in api_endpoints:
            async with monitor.monitor_api_request(endpoint, method):
                # Simulate API processing
                await asyncio.sleep(0.01)
            print(f"  ✓ {method} {endpoint}")

        print("🌐 Testing WebSocket update monitoring...")

        # Simulate WebSocket updates
        update_types = ["game_state", "agent_update", "team_message"]

        for update_type in update_types:
            async with monitor.monitor_websocket_update(update_type):
                # Simulate WebSocket broadcast
                await asyncio.sleep(0.005)
            print(f"  ✓ {update_type} update")

        # Wait a bit for monitoring to collect data
        await asyncio.sleep(0.2)

        print("📈 Performance Summary:")
        summary = monitor.get_performance_summary()

        if summary.get("status") != "no_data":
            print(
                f"  • Total decisions processed: {summary['counters']['total_decisions']}"
            )
            print(
                f"  • Total API requests: {summary['counters']['total_api_requests']}"
            )
            print(
                f"  • Total WebSocket updates: {summary['counters']['total_websocket_updates']}"
            )

            if summary.get("decision_making"):
                dm = summary["decision_making"]
                print(f"  • Average decision latency: {dm['avg_latency']:.3f}s")
                print(f"  • Max decision latency: {dm['max_latency']:.3f}s")

            if summary["active_alerts"]:
                print(f"  ⚠️  Active alerts: {', '.join(summary['active_alerts'])}")
            else:
                print("  ✅ No performance alerts")

        print("🎯 Testing performance optimization...")

        # Test optimization triggers
        optimizer = monitor.optimizer

        # Test decision making optimization
        high_latency = 4.0
        optimized = await optimizer.optimize_decision_making(high_latency)
        if optimized:
            print(
                f"  ✓ Decision making optimization triggered (latency: {high_latency}s)"
            )

        # Test memory optimization
        high_memory = 600 * 1024 * 1024  # 600MB
        optimized = await optimizer.optimize_memory_usage(high_memory)
        if optimized:
            print(
                f"  ✓ Memory optimization triggered (usage: {high_memory / (1024 * 1024):.0f}MB)"
            )

        print("✅ Backend monitoring demonstration complete!")

    finally:
        await monitor.stop_monitoring()


def demonstrate_frontend_monitoring():
    """Demonstrate frontend performance monitoring capabilities."""
    print("\n🎮 Frontend Performance Monitoring Demo")
    print("=" * 50)

    monitor = FrontendPerformanceMonitor(target_fps=60.0)

    print("🖼️  Testing frame rate monitoring...")

    # Simulate frame rendering
    for frame in range(30):  # 30 frames
        frame_start = monitor.start_frame()

        # Simulate variable frame processing time
        if frame < 10:
            processing_time = 0.016  # Good performance (60 FPS)
        elif frame < 20:
            processing_time = 0.033  # Moderate performance (30 FPS)
        else:
            processing_time = 0.050  # Poor performance (20 FPS)

        time.sleep(processing_time)

        monitor.end_frame(
            frame_start_time=frame_start,
            render_time=processing_time * 0.8,
            event_time=processing_time * 0.1,
            network_time=processing_time * 0.1,
            total_agents=6,
            visible_agents=6,
            ui_elements=15,
        )

    # Force FPS calculation
    monitor._calculate_fps()

    print("📊 Frame Rate Analysis:")
    print(f"  • Current FPS: {monitor.get_current_fps():.1f}")
    print(f"  • Average FPS (last 10): {monitor.get_average_fps(10):.1f}")
    print(f"  • Total frames rendered: {monitor.frame_count}")

    print("🔧 Testing adaptive quality adjustment...")

    # Enable adaptive quality
    monitor.set_adaptive_quality(True)
    initial_quality = monitor.current_quality_level

    # Simulate sustained low performance
    for _ in range(15):  # Enough to trigger optimization
        monitor._check_performance_optimization(15.0)  # Very low FPS

    final_quality = monitor.current_quality_level

    print(f"  • Initial quality level: {initial_quality:.2f}")
    print(f"  • Final quality level: {final_quality:.2f}")
    print(
        f"  • Quality reduction: {((initial_quality - final_quality) / initial_quality * 100):.1f}%"
    )

    # Show optimization effects
    print("🎯 Optimization Effects:")
    print(
        f"  • Debug rendering: {'enabled' if monitor.should_render_debug_info() else 'disabled'}"
    )
    print(
        f"  • Particle effects: {'enabled' if monitor.should_render_particle_effects() else 'disabled'}"
    )
    print(
        f"  • Smooth animations: {'enabled' if monitor.should_use_smooth_animations() else 'disabled'}"
    )
    print(f"  • Max visible agents: {monitor.get_max_visible_agents()}")

    print("📈 Performance Summary:")
    summary = monitor.get_performance_summary()

    fps_stats = summary["fps"]
    print(f"  • Target FPS: {fps_stats['target']}")
    print(f"  • Current FPS: {fps_stats['current']:.1f}")
    print(f"  • Average FPS: {fps_stats['average']:.1f}")

    if summary.get("frame_times_ms"):
        ft = summary["frame_times_ms"]
        print(f"  • Average frame time: {ft['avg']:.1f}ms")
        print(f"  • 95th percentile frame time: {ft['p95']:.1f}ms")

    print("✅ Frontend monitoring demonstration complete!")


async def demonstrate_integration():
    """Demonstrate integration between backend and frontend monitoring."""
    print("\n🔗 Integration Demo")
    print("=" * 50)

    backend_monitor = BattlePerformanceMonitor(monitoring_interval=0.1)
    frontend_monitor = FrontendPerformanceMonitor(target_fps=60.0)

    print("🚀 Starting integrated performance monitoring...")

    await backend_monitor.start_monitoring()

    try:
        # Simulate a complete game cycle
        for cycle in range(5):
            print(f"  Cycle {cycle + 1}: Simulating complete game loop")

            # Backend: AI decisions
            async with backend_monitor.monitor_concurrent_processing(6):
                decision_tasks = []
                for i in range(6):

                    async def make_decision(agent_id):
                        async with backend_monitor.monitor_decision_making(agent_id):
                            await asyncio.sleep(0.02)  # Realistic decision time

                    decision_tasks.append(make_decision(f"agent_{i}"))

                await asyncio.gather(*decision_tasks)

            # Backend: API and WebSocket
            async with backend_monitor.monitor_api_request("/api/game/state", "GET"):
                await asyncio.sleep(0.005)

            async with backend_monitor.monitor_websocket_update("game_state"):
                await asyncio.sleep(0.002)

            # Frontend: Frame rendering
            frame_start = frontend_monitor.start_frame()
            time.sleep(0.016)  # 60 FPS target
            frontend_monitor.end_frame(
                frame_start_time=frame_start,
                render_time=0.012,
                event_time=0.002,
                network_time=0.002,
                total_agents=6,
                visible_agents=6,
                ui_elements=15,
            )

        # Wait for monitoring data collection
        await asyncio.sleep(0.2)

        print("📊 Integrated Performance Report:")

        # Backend metrics
        backend_summary = backend_monitor.get_performance_summary()
        if backend_summary.get("status") != "no_data":
            print("  Backend Metrics:")
            print(
                f"    • Decisions processed: {backend_summary['counters']['total_decisions']}"
            )
            print(
                f"    • API requests: {backend_summary['counters']['total_api_requests']}"
            )
            print(
                f"    • WebSocket updates: {backend_summary['counters']['total_websocket_updates']}"
            )

            if backend_summary.get("decision_making"):
                dm = backend_summary["decision_making"]
                print(f"    • Avg decision latency: {dm['avg_latency']:.3f}s")

        # Frontend metrics
        frontend_monitor._calculate_fps()
        frontend_summary = frontend_monitor.get_performance_summary()

        print("  Frontend Metrics:")
        print(f"    • Current FPS: {frontend_summary['fps']['current']:.1f}")
        print(f"    • Total frames: {frontend_summary['counters']['total_frames']}")
        print(
            f"    • Quality level: {frontend_summary['quality']['current_level']:.2f}"
        )

        # Performance validation
        print("🎯 Performance Validation:")

        # Check real-time requirements
        if backend_summary.get("decision_making", {}).get("avg_latency", 0) < 2.0:
            print("    ✅ Decision latency meets real-time requirements (< 2s)")
        else:
            print("    ❌ Decision latency exceeds real-time requirements")

        if frontend_summary["fps"]["current"] >= 30.0:
            print("    ✅ Frame rate meets minimum requirements (≥ 30 FPS)")
        else:
            print("    ❌ Frame rate below minimum requirements")

        if not backend_summary.get("active_alerts", []):
            print("    ✅ No performance alerts active")
        else:
            print(
                f"    ⚠️  Active alerts: {', '.join(backend_summary['active_alerts'])}"
            )

        print("✅ Integration demonstration complete!")

    finally:
        await backend_monitor.stop_monitoring()


async def main():
    """Run the complete performance monitoring demonstration."""
    print("🚀 Battle Demo Performance Monitoring System")
    print("=" * 60)
    print("This demo showcases comprehensive performance monitoring")
    print("including latency tracking, optimization, and real-time metrics.")
    print()

    try:
        # Run demonstrations
        await demonstrate_backend_monitoring()
        demonstrate_frontend_monitoring()
        await demonstrate_integration()

        print("\n🎉 Performance Monitoring Demo Complete!")
        print("=" * 60)
        print("Key Features Demonstrated:")
        print("  ✓ AI decision making latency tracking")
        print("  ✓ Concurrent processing performance monitoring")
        print("  ✓ API response time measurement")
        print("  ✓ WebSocket update latency tracking")
        print("  ✓ Frontend frame rate monitoring")
        print("  ✓ Adaptive quality adjustment")
        print("  ✓ Performance optimization triggers")
        print("  ✓ Real-time performance validation")
        print("  ✓ Comprehensive performance reporting")
        print()
        print("The system is ready for production use with real-time")
        print("performance requirements validation and automatic optimization!")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
