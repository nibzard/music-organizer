"""
Memory usage monitoring utilities for performance tracking
"""

import gc
import os
import sys
import time
import threading
import tracemalloc
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore


@dataclass
class MemorySnapshot:
    """A snapshot of memory usage at a point in time"""
    timestamp: float
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Percentage of memory used
    python_mb: float  # Python-specific memory usage
    tracemalloc_current: float = 0  # Current tracemalloc usage
    tracemalloc_peak: float = 0  # Peak tracemalloc usage


@dataclass
class MemoryStats:
    """Aggregated memory statistics over time"""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    peak_rss: float = 0
    peak_python: float = 0
    peak_tracemalloc: float = 0
    avg_rss: float = 0
    total_time: float = 0

    def add_snapshot(self, snapshot: MemorySnapshot):
        """Add a memory snapshot and update statistics"""
        self.snapshots.append(snapshot)
        self.peak_rss = max(self.peak_rss, snapshot.rss_mb)
        self.peak_python = max(self.peak_python, snapshot.python_mb)
        self.peak_tracemalloc = max(self.peak_tracemalloc, snapshot.tracemalloc_peak)

    def finalize(self):
        """Finalize statistics collection"""
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time

        if self.snapshots:
            self.avg_rss = sum(s.rss_mb for s in self.snapshots) / len(self.snapshots)


class MemoryMonitor:
    """Monitor memory usage during application execution"""

    def __init__(self, interval: float = 0.5, enable_tracemalloc: bool = False):
        self.interval = interval
        self.enable_tracemalloc = enable_tracemalloc
        self.sample_interval = interval
        self.snapshots: List[MemorySnapshot] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.monitoring = False
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[MemorySnapshot], None]] = []
        self.stats = MemoryStats()

    def start(self):
        """Start memory monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self.monitoring = True
        self.start_time = time.time()
        self.snapshots = []
        self.stats = MemoryStats()
        self.end_time = None

        if self.enable_tracemalloc:
            tracemalloc.start()

        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def take_snapshot(self) -> MemorySnapshot:
        """Take a single memory snapshot."""
        snapshot = self._get_snapshot()

        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            snapshot.tracemalloc_current = current / 1024 / 1024
            snapshot.tracemalloc_peak = peak / 1024 / 1024

        self.snapshots.append(snapshot)
        self.stats.add_snapshot(snapshot)
        return snapshot

    def reset(self):
        """Reset monitor state."""
        self.start_time = None
        self.end_time = None
        self.snapshots = []
        self.stats = MemoryStats()
        self.monitoring = False

    def stop(self):
        """Stop memory monitoring and return statistics"""
        if not self._monitoring:
            return self.stats

        self._monitoring = False
        self.monitoring = False
        self.end_time = time.time()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            # Add final snapshot with tracemalloc data
            snapshot = self._get_snapshot()
            snapshot.tracemalloc_current = current / 1024 / 1024
            snapshot.tracemalloc_peak = peak / 1024 / 1024
            self.stats.add_snapshot(snapshot)
            tracemalloc.stop()

        self.stats.finalize()
        return self.stats

    def add_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Add a callback to be called with each memory snapshot"""
        self._callbacks.append(callback)

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            snapshot = self._get_snapshot()
            self.snapshots.append(snapshot)
            self.stats.add_snapshot(snapshot)

            # Call callbacks
            for callback in self._callbacks:
                try:
                    callback(snapshot)
                except Exception:
                    pass

            time.sleep(self.sample_interval)

    def _take_snapshot(self) -> MemorySnapshot:
        """Alias for _get_snapshot - used by tests."""
        return self._get_snapshot()

    def _get_snapshot(self) -> MemorySnapshot:
        """Get current memory snapshot"""
        if psutil is not None:
            # Try to get memory info from psutil if available
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            rss_mb = memory_info.rss / 1024 / 1024
            vms_mb = memory_info.vms / 1024 / 1024
            percent = memory_percent
        else:
            # Fallback to resource module (less accurate)
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)

            # Max RSS is in KB on Linux
            rss_mb = usage.ru_maxrss / 1024
            vms_mb = rss_mb  # Not available
            percent = 0  # Not available

        # Python-specific memory
        gc.collect()  # Force garbage collection for accurate reading
        python_objects = sys.gettotalrefcount() if hasattr(sys, 'gettotalrefcount') else 0
        python_mb = python_objects * 8 / 1024 / 1024  # Rough estimate

        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            percent=percent,
            python_mb=python_mb
        )

    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage as a dictionary (alias for get_current_usage)"""
        return self.get_current_usage()

    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage as a dictionary"""
        snapshot = self._get_snapshot()
        return {
            "rss_mb": snapshot.rss_mb,
            "vms_mb": snapshot.vms_mb,
            "percent": snapshot.percent,
            "python_mb": snapshot.python_mb
        }

    def check_memory_limit(self, limit_mb: float) -> bool:
        """Check if current memory usage exceeds limit"""
        current = self.get_current_usage()
        return current["rss_mb"] > limit_mb


class MemoryProfiler:
    """Context manager for profiling memory usage of a code block"""

    def __init__(self, name: str = "Code Block", enable_tracemalloc: bool = True):
        self.name = name
        self.enable_tracemalloc = enable_tracemalloc
        self.stats: Optional[MemoryStats] = None

    def __enter__(self):
        self.monitor = get_global_monitor()
        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stats = self.monitor.stop()
        self.print_summary()

    def get_stats(self) -> MemoryStats:
        """Get the collected statistics."""
        return self.stats

    def print_summary(self):
        """Print a summary of memory usage"""
        if not self.stats:
            return

        print(f"\n📊 Memory Profile: {self.name}")
        print("-" * 50)
        print(f"Duration: {self.stats.total_time:.2f}s")
        print(f"Peak RSS: {self.stats.peak_rss:.2f} MB")
        print(f"Avg RSS: {self.stats.avg_rss:.2f} MB")
        print(f"Peak Python: {self.stats.peak_python:.2f} MB")

        if self.enable_tracemalloc and self.stats.peak_tracemalloc > 0:
            print(f"Peak Tracemalloc: {self.stats.peak_tracemalloc:.2f} MB")

    def save_report(self, filepath: Path):
        """Save detailed memory report to file"""
        if not self.stats:
            return

        report = {
            "name": self.name,
            "duration": self.stats.total_time,
            "peak_rss_mb": self.stats.peak_rss,
            "avg_rss_mb": self.stats.avg_rss,
            "peak_python_mb": self.stats.peak_python,
            "peak_tracemalloc_mb": self.stats.peak_tracemalloc,
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "rss_mb": s.rss_mb,
                    "vms_mb": s.vms_mb,
                    "percent": s.percent,
                    "python_mb": s.python_mb,
                    "tracemalloc_current": s.tracemalloc_current,
                    "tracemalloc_peak": s.tracemalloc_peak
                }
                for s in self.stats.snapshots
            ]
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)


# Convenience decorators
def profile_memory(name: Optional[str] = None, enable_tracemalloc: bool = False):
    """Decorator to profile memory usage of a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler_name = name or f"{func.__module__}.{func.__name__}"
            with MemoryProfiler(profiler_name, enable_tracemalloc=enable_tracemalloc):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class MonitorMemoryContext:
    """Context manager for monitoring memory usage and detecting limit exceeded."""

    def __init__(self, limit_mb: float, check_interval: float = 1.0):
        self.limit_mb = limit_mb
        self.check_interval = check_interval

    def __enter__(self):
        self.monitor = get_global_monitor()
        self.monitor.sample_interval = self.check_interval
        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop()
        # Check if memory exceeded limit
        if hasattr(self.monitor, 'snapshots') and self.monitor.snapshots:
            peak_rss = max(s.rss_mb for s in self.monitor.snapshots)
            if peak_rss > self.limit_mb:
                raise MemoryError(f"Memory limit exceeded: {peak_rss:.1f} MB > {self.limit_mb} MB")


def monitor_memory(limit_mb: float, check_interval: float = 1.0):
    """Context manager to monitor memory usage and raise if limit exceeded."""
    return MonitorMemoryContext(limit_mb, check_interval)


# Global memory monitor instance
_global_monitor: Optional[MemoryMonitor] = None


def get_global_monitor() -> MemoryMonitor:
    """Get or create the global memory monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor


def reset_global_monitor():
    """Reset the global monitor instance (for testing)."""
    global _global_monitor
    _global_monitor = None


def start_global_monitoring(enable_tracemalloc: bool = False):
    """Start global memory monitoring"""
    monitor = get_global_monitor()
    monitor.enable_tracemalloc = enable_tracemalloc
    monitor.start()


def stop_global_monitoring() -> MemoryStats:
    """Stop global monitoring and return stats"""
    monitor = get_global_monitor()
    return monitor.stop()


def get_memory_pressure() -> float:
    """Get current memory pressure (0-1, where 1 is high pressure)"""
    if psutil is not None:
        virtual = psutil.virtual_memory()
        return virtual.percent / 100.0
    return 0.5  # Unknown pressure


def should_use_streaming(file_count: int, memory_limit_mb: float = 100) -> bool:
    """Determine if streaming should be used based on file count and memory"""
    # Rough estimate: use streaming when file count exceeds ~2x memory limit (in MB)
    # E.g., 500 files > 100MB * 2 = 200, so use streaming
    return file_count > memory_limit_mb * 2 or get_memory_pressure() > 0.8