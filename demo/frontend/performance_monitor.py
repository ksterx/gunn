"""
Frontend performance monitoring for the Pygame renderer.

This module provides frame rate monitoring, rendering optimization,
and performance metrics collection for the battle demo frontend.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import pygame
import requests

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """Frame rendering performance metrics."""

    timestamp: float
    frame_time: float
    fps: float
    render_time: float
    event_processing_time: float
    network_update_time: float
    total_agents: int
    visible_agents: int
    ui_elements: int


class FrontendPerformanceMonitor:
    """Performance monitoring for the Pygame frontend."""

    def __init__(
        self, target_fps: float = 60.0, backend_url: str = "http://localhost:8000"
    ):
        """
        Initialize frontend performance monitor.

        Args:
            target_fps: Target frame rate for optimization
            backend_url: Backend URL for reporting metrics
        """
        self.target_fps = target_fps
        self.backend_url = backend_url.rstrip("/")

        # Performance tracking
        self.frame_times: deque = deque(maxlen=100)  # Last 100 frame times
        self.fps_history: deque = deque(maxlen=60)  # Last 60 FPS measurements
        self.render_times: deque = deque(maxlen=100)

        # Frame timing
        self.last_frame_time = time.perf_counter()
        self.frame_count = 0
        self.fps_calculation_interval = 1.0  # Calculate FPS every second
        self.last_fps_calculation = time.perf_counter()

        # Performance optimization
        self.adaptive_quality = True
        self.current_quality_level = 1.0  # 1.0 = full quality, 0.5 = half quality, etc.
        self.min_quality_level = 0.3
        self.quality_adjustment_threshold = (
            5.0  # Adjust if FPS drops below target for 5 seconds
        )
        self.low_fps_duration = 0.0

        # Rendering optimization flags
        self.enable_debug_rendering = True
        self.enable_particle_effects = True
        self.enable_smooth_animations = True
        self.max_visible_agents = 20  # Limit for performance

        # Network performance
        self.network_update_times: deque = deque(maxlen=50)
        self.last_network_update = time.perf_counter()

        # Metrics reporting
        self.metrics_reporting_interval = 5.0  # Report to backend every 5 seconds
        self.last_metrics_report = time.perf_counter()

        logger.info(
            f"Frontend performance monitor initialized (target FPS: {target_fps})"
        )

    def start_frame(self) -> float:
        """Mark the start of a new frame and return frame start time."""
        current_time = time.perf_counter()

        # Calculate frame time
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time

        self.frame_count += 1

        return current_time

    def end_frame(
        self,
        frame_start_time: float,
        render_time: float,
        event_time: float = 0.0,
        network_time: float = 0.0,
        total_agents: int = 0,
        visible_agents: int = 0,
        ui_elements: int = 0,
    ) -> None:
        """
        Mark the end of a frame and record performance metrics.

        Args:
            frame_start_time: Time when frame started (from start_frame())
            render_time: Time spent rendering
            event_time: Time spent processing events
            network_time: Time spent on network updates
            total_agents: Total number of agents in game
            visible_agents: Number of agents currently visible
            ui_elements: Number of UI elements rendered
        """
        current_time = time.perf_counter()
        total_frame_time = current_time - frame_start_time

        # Record render time
        self.render_times.append(render_time)

        # Record network update time
        if network_time > 0:
            self.network_update_times.append(network_time)

        # Calculate FPS periodically
        if current_time - self.last_fps_calculation >= self.fps_calculation_interval:
            self._calculate_fps()
            self.last_fps_calculation = current_time

        # Create frame metrics
        fps = self.get_current_fps()
        metrics = FrameMetrics(
            timestamp=current_time,
            frame_time=total_frame_time,
            fps=fps,
            render_time=render_time,
            event_processing_time=event_time,
            network_update_time=network_time,
            total_agents=total_agents,
            visible_agents=visible_agents,
            ui_elements=ui_elements,
        )

        # Check for performance optimization needs
        self._check_performance_optimization(fps)

        # Report metrics to backend periodically
        if current_time - self.last_metrics_report >= self.metrics_reporting_interval:
            asyncio.create_task(self._report_metrics_to_backend(metrics))
            self.last_metrics_report = current_time

    def _calculate_fps(self) -> None:
        """Calculate current FPS based on recent frame times."""
        if len(self.frame_times) < 2:
            return

        # Calculate FPS from average frame time
        recent_frame_times = list(self.frame_times)[-30:]  # Last 30 frames
        avg_frame_time = sum(recent_frame_times) / len(recent_frame_times)

        if avg_frame_time > 0:
            fps = 1.0 / avg_frame_time
            self.fps_history.append(fps)

    def get_current_fps(self) -> float:
        """Get current FPS measurement."""
        if not self.fps_history:
            return 0.0
        return self.fps_history[-1]

    def get_average_fps(self, samples: int = 10) -> float:
        """Get average FPS over recent samples."""
        if not self.fps_history:
            return 0.0

        recent_fps = list(self.fps_history)[-samples:]
        return sum(recent_fps) / len(recent_fps)

    def _check_performance_optimization(self, current_fps: float) -> None:
        """Check if performance optimization is needed."""
        if not self.adaptive_quality:
            return

        # Track low FPS duration
        if current_fps < self.target_fps * 0.8:  # 80% of target FPS
            self.low_fps_duration += (
                1.0 / 60.0
            )  # Assume 60 FPS for duration calculation
        else:
            self.low_fps_duration = 0.0

        # Reduce quality if FPS is consistently low
        if self.low_fps_duration > self.quality_adjustment_threshold:
            if self.current_quality_level > self.min_quality_level:
                old_quality = self.current_quality_level
                self.current_quality_level = max(
                    self.min_quality_level, self.current_quality_level * 0.8
                )

                logger.warning(
                    f"Reducing rendering quality due to low FPS: "
                    f"{old_quality:.2f} -> {self.current_quality_level:.2f} "
                    f"(FPS: {current_fps:.1f})"
                )

                # Adjust rendering settings
                self._apply_quality_settings()

                self.low_fps_duration = 0.0

        # Increase quality if FPS is consistently high
        elif current_fps > self.target_fps * 1.1 and self.current_quality_level < 1.0:
            old_quality = self.current_quality_level
            self.current_quality_level = min(1.0, self.current_quality_level * 1.1)

            logger.info(
                f"Increasing rendering quality due to good FPS: "
                f"{old_quality:.2f} -> {self.current_quality_level:.2f} "
                f"(FPS: {current_fps:.1f})"
            )

            self._apply_quality_settings()

    def _apply_quality_settings(self) -> None:
        """Apply current quality settings to rendering options."""
        # Adjust rendering features based on quality level
        if self.current_quality_level < 0.5:
            self.enable_debug_rendering = False
            self.enable_particle_effects = False
            self.enable_smooth_animations = False
            self.max_visible_agents = 10
        elif self.current_quality_level < 0.8:
            self.enable_debug_rendering = False
            self.enable_particle_effects = True
            self.enable_smooth_animations = False
            self.max_visible_agents = 15
        else:
            self.enable_debug_rendering = True
            self.enable_particle_effects = True
            self.enable_smooth_animations = True
            self.max_visible_agents = 20

    def should_render_debug_info(self) -> bool:
        """Check if debug information should be rendered."""
        return self.enable_debug_rendering and self.current_quality_level >= 0.8

    def should_render_particle_effects(self) -> bool:
        """Check if particle effects should be rendered."""
        return self.enable_particle_effects and self.current_quality_level >= 0.5

    def should_use_smooth_animations(self) -> bool:
        """Check if smooth animations should be used."""
        return self.enable_smooth_animations and self.current_quality_level >= 0.8

    def get_max_visible_agents(self) -> int:
        """Get maximum number of agents to render for performance."""
        return self.max_visible_agents

    def get_render_scale(self) -> float:
        """Get rendering scale factor based on quality level."""
        return self.current_quality_level

    async def _report_metrics_to_backend(self, metrics: FrameMetrics) -> None:
        """Report performance metrics to the backend."""
        try:
            # Prepare metrics data
            metrics_data = {
                "timestamp": metrics.timestamp,
                "fps": metrics.fps,
                "frame_time_ms": metrics.frame_time * 1000,
                "render_time_ms": metrics.render_time * 1000,
                "event_processing_time_ms": metrics.event_processing_time * 1000,
                "network_update_time_ms": metrics.network_update_time * 1000,
                "total_agents": metrics.total_agents,
                "visible_agents": metrics.visible_agents,
                "ui_elements": metrics.ui_elements,
                "quality_level": self.current_quality_level,
                "adaptive_quality_enabled": self.adaptive_quality,
            }

            # Send to backend (non-blocking)
            response = requests.post(
                f"{self.backend_url}/api/performance/frontend",
                json=metrics_data,
                timeout=1.0,  # Short timeout to avoid blocking rendering
            )

            if response.status_code != 200:
                logger.warning(
                    f"Failed to report metrics to backend: {response.status_code}"
                )

        except requests.exceptions.RequestException:
            # Don't log every network error to avoid spam
            pass
        except Exception as e:
            logger.error(f"Error reporting metrics to backend: {e}")

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        current_fps = self.get_current_fps()
        avg_fps = self.get_average_fps()

        # Calculate percentiles for frame times
        frame_times_ms = [ft * 1000 for ft in self.frame_times]
        render_times_ms = [rt * 1000 for rt in self.render_times]

        frame_time_stats = {}
        render_time_stats = {}

        if frame_times_ms:
            sorted_frame_times = sorted(frame_times_ms)
            frame_time_stats = {
                "avg": sum(frame_times_ms) / len(frame_times_ms),
                "min": min(frame_times_ms),
                "max": max(frame_times_ms),
                "p50": sorted_frame_times[len(sorted_frame_times) // 2],
                "p95": sorted_frame_times[int(len(sorted_frame_times) * 0.95)],
                "p99": sorted_frame_times[int(len(sorted_frame_times) * 0.99)],
            }

        if render_times_ms:
            sorted_render_times = sorted(render_times_ms)
            render_time_stats = {
                "avg": sum(render_times_ms) / len(render_times_ms),
                "min": min(render_times_ms),
                "max": max(render_times_ms),
                "p50": sorted_render_times[len(sorted_render_times) // 2],
                "p95": sorted_render_times[int(len(sorted_render_times) * 0.95)],
                "p99": sorted_render_times[int(len(sorted_render_times) * 0.99)],
            }

        # Network performance stats
        network_stats = {}
        if self.network_update_times:
            network_times_ms = [nt * 1000 for nt in self.network_update_times]
            network_stats = {
                "avg_update_time_ms": sum(network_times_ms) / len(network_times_ms),
                "max_update_time_ms": max(network_times_ms),
                "update_count": len(self.network_update_times),
            }

        return {
            "fps": {
                "current": current_fps,
                "average": avg_fps,
                "target": self.target_fps,
                "samples": len(self.fps_history),
            },
            "frame_times_ms": frame_time_stats,
            "render_times_ms": render_time_stats,
            "network": network_stats,
            "quality": {
                "current_level": self.current_quality_level,
                "adaptive_enabled": self.adaptive_quality,
                "low_fps_duration": self.low_fps_duration,
                "debug_rendering": self.enable_debug_rendering,
                "particle_effects": self.enable_particle_effects,
                "smooth_animations": self.enable_smooth_animations,
                "max_visible_agents": self.max_visible_agents,
            },
            "counters": {
                "total_frames": self.frame_count,
                "metrics_samples": len(self.frame_times),
            },
        }

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.frame_times.clear()
        self.fps_history.clear()
        self.render_times.clear()
        self.network_update_times.clear()

        self.frame_count = 0
        self.low_fps_duration = 0.0
        self.current_quality_level = 1.0

        # Reset optimization settings
        self._apply_quality_settings()

        logger.info("Performance metrics reset")

    def set_adaptive_quality(self, enabled: bool) -> None:
        """Enable or disable adaptive quality adjustment."""
        self.adaptive_quality = enabled

        if not enabled:
            # Reset to full quality when disabled
            self.current_quality_level = 1.0
            self._apply_quality_settings()

        logger.info(f"Adaptive quality {'enabled' if enabled else 'disabled'}")

    def set_target_fps(self, fps: float) -> None:
        """Set target FPS for optimization."""
        self.target_fps = fps
        logger.info(f"Target FPS set to {fps}")

    def force_quality_level(self, quality: float) -> None:
        """Force a specific quality level (0.0 to 1.0)."""
        self.current_quality_level = max(0.0, min(1.0, quality))
        self._apply_quality_settings()
        logger.info(f"Quality level forced to {self.current_quality_level:.2f}")


class RenderingOptimizer:
    """Handles rendering optimizations for better performance."""

    def __init__(self, performance_monitor: FrontendPerformanceMonitor):
        self.performance_monitor = performance_monitor

        # Optimization caches
        self.surface_cache: dict[str, pygame.Surface] = {}
        self.font_cache: dict[tuple[str, int], pygame.font.Font] = {}

        # Culling settings
        self.enable_frustum_culling = True
        self.culling_margin = 50  # pixels outside screen to still render

    def should_render_agent(
        self,
        agent_position: tuple[float, float],
        screen_bounds: tuple[int, int, int, int],
    ) -> bool:
        """Check if an agent should be rendered based on screen culling."""
        if not self.enable_frustum_culling:
            return True

        x, y = agent_position
        screen_x, screen_y, screen_w, screen_h = screen_bounds

        # Add margin for smooth transitions
        margin = self.culling_margin

        return (
            x >= screen_x - margin
            and x <= screen_x + screen_w + margin
            and y >= screen_y - margin
            and y <= screen_y + screen_h + margin
        )

    def get_cached_surface(self, key: str, create_func) -> pygame.Surface:
        """Get a cached surface or create it if not exists."""
        if key not in self.surface_cache:
            self.surface_cache[key] = create_func()
        return self.surface_cache[key]

    def get_cached_font(self, font_name: str, size: int) -> pygame.font.Font:
        """Get a cached font or create it if not exists."""
        key = (font_name, size)
        if key not in self.font_cache:
            try:
                self.font_cache[key] = pygame.font.Font(font_name, size)
            except:
                self.font_cache[key] = pygame.font.SysFont("arial", size)
        return self.font_cache[key]

    def clear_caches(self) -> None:
        """Clear all rendering caches."""
        self.surface_cache.clear()
        self.font_cache.clear()
        logger.info("Rendering caches cleared")

    def optimize_agent_rendering(
        self, agents: list[Any], screen_bounds: tuple[int, int, int, int]
    ) -> list[Any]:
        """Optimize agent list for rendering based on performance settings."""
        # Apply frustum culling
        if self.enable_frustum_culling:
            visible_agents = [
                agent
                for agent in agents
                if self.should_render_agent(agent.position, screen_bounds)
            ]
        else:
            visible_agents = agents

        # Limit number of agents based on performance
        max_agents = self.performance_monitor.get_max_visible_agents()
        if len(visible_agents) > max_agents:
            # Sort by distance to center and take closest ones
            center_x, center_y = screen_bounds[2] // 2, screen_bounds[3] // 2

            def distance_to_center(agent):
                dx = agent.position[0] - center_x
                dy = agent.position[1] - center_y
                return dx * dx + dy * dy

            visible_agents.sort(key=distance_to_center)
            visible_agents = visible_agents[:max_agents]

        return visible_agents
