"""
Profiler — Research-grade benchmarking for edge deployment.
============================================================
Tracks per-component latency, peak memory, CPU temperature,
and produces structured output for paper/report inclusion.
"""

import time
import os
import csv
import logging
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ProfileSample:
    """Single profiling measurement."""

    timestamp: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    fps: float
    memory_mb: float
    cpu_temp_c: Optional[float]
    num_detections: int
    frame_index: int


class Profiler:
    """
    Lightweight profiler designed for Raspberry Pi.

    Reads memory from /proc/self/status (Linux) and
    CPU temperature from /sys/class/thermal (Pi-specific).

    Usage
    -----
    >>> profiler = Profiler(window_size=30)
    >>> profiler.record(result)  # DetectionResult
    >>> print(profiler.summary())
    >>> profiler.export_csv("benchmark_results.csv")
    """

    def __init__(self, window_size: int = 60, export_dir: Optional[Path] = None):
        self.window_size = window_size
        self.export_dir = export_dir or Path(".")
        self._samples: list[ProfileSample] = []
        self._fps_window: deque[float] = deque(maxlen=window_size)
        self._last_time: Optional[float] = None
        self._is_linux = platform.system() == "Linux"
        self._is_pi = self._detect_raspberry_pi()

    def _detect_raspberry_pi(self) -> bool:
        """Check if running on a Raspberry Pi."""
        try:
            with open("/proc/device-tree/model", "r") as f:
                return "raspberry pi" in f.read().lower()
        except (FileNotFoundError, PermissionError):
            return False

    def get_memory_mb(self) -> float:
        """Get current process RSS in MB. Linux-specific via /proc."""
        if not self._is_linux:
            try:
                import psutil
                return psutil.Process().memory_info().rss / (1024 * 1024)
            except ImportError:
                return 0.0

        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # Value is in kB
                        return int(line.split()[1]) / 1024.0
        except (FileNotFoundError, ValueError, IndexError):
            pass
        return 0.0

    def get_cpu_temp(self) -> Optional[float]:
        """Get CPU temperature in Celsius. Pi-specific."""
        if not self._is_linux:
            return None

        # Method 1: thermal_zone (works on Pi and most Linux)
        thermal_path = "/sys/class/thermal/thermal_zone0/temp"
        try:
            with open(thermal_path, "r") as f:
                return int(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError):
            pass

        # Method 2: vcgencmd (Pi-specific)
        if self._is_pi:
            try:
                result = os.popen("vcgencmd measure_temp").readline()
                return float(result.replace("temp=", "").replace("'C\n", ""))
            except (ValueError, OSError):
                pass

        return None

    def record(self, result) -> ProfileSample:
        """
        Record a profiling sample from a DetectionResult.

        Parameters
        ----------
        result : DetectionResult
            Detection result with timing information.

        Returns
        -------
        ProfileSample
            Recorded sample.
        """
        now = time.time()

        # Compute rolling FPS
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                self._fps_window.append(1.0 / dt)
        self._last_time = now

        fps = sum(self._fps_window) / len(self._fps_window) if self._fps_window else 0

        sample = ProfileSample(
            timestamp=now,
            preprocess_ms=result.preprocess_ms,
            inference_ms=result.inference_ms,
            postprocess_ms=result.postprocess_ms,
            total_ms=result.total_ms,
            fps=fps,
            memory_mb=self.get_memory_mb(),
            cpu_temp_c=self.get_cpu_temp(),
            num_detections=result.count,
            frame_index=result.frame_index,
        )
        self._samples.append(sample)
        return sample

    @property
    def rolling_fps(self) -> float:
        if not self._fps_window:
            return 0.0
        return sum(self._fps_window) / len(self._fps_window)

    def summary(self) -> dict:
        """
        Compute summary statistics across all recorded samples.

        Returns
        -------
        dict
            Statistics including latency percentiles, peak memory,
            average FPS, and temperature range.
        """
        if not self._samples:
            return {"error": "No samples recorded"}

        import numpy as np

        inference_times = np.array([s.inference_ms for s in self._samples])
        total_times = np.array([s.total_ms for s in self._samples])
        memory_values = np.array([s.memory_mb for s in self._samples])
        fps_values = np.array([s.fps for s in self._samples])

        temps = [s.cpu_temp_c for s in self._samples if s.cpu_temp_c is not None]

        return {
            "num_samples": len(self._samples),
            "inference_ms": {
                "mean": float(np.mean(inference_times)),
                "std": float(np.std(inference_times)),
                "p50": float(np.percentile(inference_times, 50)),
                "p95": float(np.percentile(inference_times, 95)),
                "p99": float(np.percentile(inference_times, 99)),
                "min": float(np.min(inference_times)),
                "max": float(np.max(inference_times)),
            },
            "total_ms": {
                "mean": float(np.mean(total_times)),
                "p50": float(np.percentile(total_times, 50)),
                "p95": float(np.percentile(total_times, 95)),
            },
            "fps": {
                "mean": float(np.mean(fps_values[fps_values > 0]))
                if np.any(fps_values > 0)
                else 0,
                "min": float(np.min(fps_values[fps_values > 0]))
                if np.any(fps_values > 0)
                else 0,
                "max": float(np.max(fps_values)) if len(fps_values) > 0 else 0,
            },
            "memory_mb": {
                "mean": float(np.mean(memory_values)),
                "peak": float(np.max(memory_values)),
            },
            "cpu_temp_c": {
                "mean": float(np.mean(temps)) if temps else None,
                "max": float(np.max(temps)) if temps else None,
            },
        }

    def summary_table(self) -> str:
        """Format summary as a markdown table for research papers."""
        stats = self.summary()
        if "error" in stats:
            return stats["error"]

        lines = [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Samples | {stats['num_samples']} |",
            f"| Inference (mean) | {stats['inference_ms']['mean']:.1f} ms |",
            f"| Inference (p50) | {stats['inference_ms']['p50']:.1f} ms |",
            f"| Inference (p95) | {stats['inference_ms']['p95']:.1f} ms |",
            f"| Inference (p99) | {stats['inference_ms']['p99']:.1f} ms |",
            f"| Total pipeline (mean) | {stats['total_ms']['mean']:.1f} ms |",
            f"| FPS (mean) | {stats['fps']['mean']:.2f} |",
            f"| Memory (peak) | {stats['memory_mb']['peak']:.1f} MB |",
        ]
        if stats["cpu_temp_c"]["max"] is not None:
            lines.append(
                f"| CPU Temp (max) | {stats['cpu_temp_c']['max']:.1f} °C |"
            )
        return "\n".join(lines)

    def export_csv(self, filename: str = "profile_results.csv") -> Path:
        """Export all samples to CSV."""
        out_path = self.export_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "frame_index", "preprocess_ms", "inference_ms",
                "postprocess_ms", "total_ms", "fps", "memory_mb",
                "cpu_temp_c", "num_detections",
            ])
            for s in self._samples:
                writer.writerow([
                    f"{s.timestamp:.3f}",
                    s.frame_index,
                    f"{s.preprocess_ms:.2f}",
                    f"{s.inference_ms:.2f}",
                    f"{s.postprocess_ms:.2f}",
                    f"{s.total_ms:.2f}",
                    f"{s.fps:.2f}",
                    f"{s.memory_mb:.1f}",
                    f"{s.cpu_temp_c:.1f}" if s.cpu_temp_c else "",
                    s.num_detections,
                ])

        logger.info(f"Profile data exported to: {out_path}")
        return out_path

    def reset(self) -> None:
        """Clear all recorded samples."""
        self._samples.clear()
        self._fps_window.clear()
        self._last_time = None
