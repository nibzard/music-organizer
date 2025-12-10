"""
Performance testing utilities for plugins.

This module provides tools for profiling and benchmarking plugin performance,
including memory usage, execution time, and throughput measurements.
"""

import time
import asyncio
import threading
import tracemalloc
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
import gc

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from ...models.audio_file import AudioFile
from ..base import Plugin


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    execution_time: float
    memory_usage: int  # in bytes
    peak_memory: int  # in bytes
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    throughput: Optional[float] = None  # items per second


@dataclass
class BenchmarkResult:
    """Results of a benchmark test."""
    test_name: str
    metrics: PerformanceMetrics
    iterations: int
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    memory_trend: List[int]


class PerformanceProfiler:
    """
    Performance profiler for plugin testing.

    Measures execution time, memory usage, CPU usage, and other metrics.
    """

    def __init__(self):
        """Initialize performance profiler."""
        if PSUTIL_AVAILABLE:
            self.current_process = psutil.Process()
        else:
            self.current_process = None
        self.start_time: Optional[float] = None
        self.start_memory: Optional[int] = None
        self.peak_memory: int = 0

    @contextmanager
    def profile(self):
        """Context manager for profiling code execution."""
        # Start profiling
        tracemalloc.start()
        gc.collect()  # Clear garbage collector

        if PSUTIL_AVAILABLE and self.current_process:
            start_memory = self.current_process.memory_info().rss
            start_cpu_time = self.current_process.cpu_percent()
        else:
            start_memory = 0
            start_cpu_time = 0

        start_time = time.time()

        try:
            yield self
        finally:
            # End profiling
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            if PSUTIL_AVAILABLE and self.current_process:
                end_cpu_time = self.current_process.cpu_percent()
                end_memory = self.current_process.memory_info().rss
            else:
                end_cpu_time = 0
                end_memory = 0

            self.execution_time = end_time - start_time
            self.memory_usage = end_memory - start_memory
            self.peak_memory = peak
            self.cpu_percent = end_cpu_time - start_cpu_time

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return PerformanceMetrics(
            execution_time=getattr(self, 'execution_time', 0),
            memory_usage=getattr(self, 'memory_usage', 0),
            peak_memory=getattr(self, 'peak_memory', 0),
            cpu_percent=getattr(self, 'cpu_percent', 0),
            success=True
        )

    def measure_method(self, plugin: Plugin, method_name: str, *args, **kwargs) -> PerformanceMetrics:
        """
        Measure performance of a specific plugin method.

        Args:
            plugin: Plugin instance
            method_name: Name of method to measure
            *args: Arguments to pass to method
            **kwargs: Keyword arguments to pass to method

        Returns:
            PerformanceMetrics with measurement results
        """
        try:
            method = getattr(plugin, method_name)
            if not method:
                return PerformanceMetrics(
                    execution_time=0,
                    memory_usage=0,
                    peak_memory=0,
                    cpu_percent=0,
                    success=False,
                    error_message=f"Method {method_name} not found"
                )

            with self.profile():
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method(*args, **kwargs))
                else:
                    method(*args, **kwargs)

            return self.get_current_metrics()

        except Exception as e:
            return PerformanceMetrics(
                execution_time=0,
                memory_usage=0,
                peak_memory=0,
                cpu_percent=0,
                success=False,
                error_message=str(e)
            )

    def measure_batch_performance(
        self,
        plugin: Plugin,
        method_name: str,
        items: List[Any],
        batch_size: int = 10
    ) -> List[PerformanceMetrics]:
        """
        Measure performance of batch operations.

        Args:
            plugin: Plugin instance
            method_name: Name of method to measure
            items: List of items to process
            batch_size: Size of each batch

        Returns:
            List of PerformanceMetrics for each batch
        """
        metrics = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            with self.profile():
                method = getattr(plugin, method_name)
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method(batch))
                else:
                    method(batch)

            metrics.append(self.get_current_metrics())

        return metrics


class MemoryProfiler:
    """
    Specialized memory profiler for detailed memory analysis.
    """

    def __init__(self):
        """Initialize memory profiler."""
        self.memory_snapshots: List[tuple] = []

    @contextmanager
    def track_memory(self):
        """Context manager for tracking memory usage over time."""
        tracemalloc.start()
        self.memory_snapshots = []

        def capture_snapshot():
            current, peak = tracemalloc.get_traced_memory()
            self.memory_snapshots.append((time.time(), current, peak))

        # Start capturing snapshots
        snapshot_thread = threading.Thread(target=self._capture_snapshots, args=(capture_snapshot,))
        snapshot_thread.daemon = True
        snapshot_thread.start()

        try:
            yield self
        finally:
            snapshot_thread.join(timeout=1)
            tracemalloc.stop()

    def _capture_snapshots(self, capture_func: Callable, interval: float = 0.1):
        """Capture memory snapshots at intervals."""
        while True:
            capture_func()
            time.sleep(interval)

    def get_memory_trend(self) -> List[int]:
        """Get memory usage trend over time."""
        return [snapshot[1] for snapshot in self.memory_snapshots]

    def get_peak_memory_trend(self) -> List[int]:
        """Get peak memory trend over time."""
        return [snapshot[2] for snapshot in self.memory_snapshots]


def benchmark_plugin(
    plugin: Plugin,
    test_files: List[AudioFile],
    iterations: int = 5,
    warmup_iterations: int = 2
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark plugin performance with test data.

    Args:
        plugin: Plugin to benchmark
        test_files: List of test audio files
        iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations (not counted)

    Returns:
        Dictionary of benchmark results for each method
    """
    results = {}
    profiler = PerformanceProfiler()

    # Warmup phase
    for _ in range(warmup_iterations):
        if hasattr(plugin, 'enhance_metadata'):
            if test_files:
                asyncio.run(plugin.enhance_metadata(test_files[0]))
        if hasattr(plugin, 'classify'):
            if test_files:
                asyncio.run(plugin.classify(test_files[0]))

    # Benchmark different methods based on plugin type
    if hasattr(plugin, 'enhance_metadata'):
        results['enhance_metadata'] = _benchmark_method(
            profiler, plugin, 'enhance_metadata', test_files, iterations
        )

        # Test batch processing if available
        if hasattr(plugin, 'batch_enhance'):
            results['batch_enhance'] = _benchmark_method(
                profiler, plugin, 'batch_enhance', test_files, iterations
            )

    if hasattr(plugin, 'classify'):
        results['classify'] = _benchmark_method(
            profiler, plugin, 'classify', test_files, iterations
        )

        # Test batch classification if available
        if hasattr(plugin, 'batch_classify'):
            results['batch_classify'] = _benchmark_method(
                profiler, plugin, 'batch_classify', test_files, iterations
            )

    if hasattr(plugin, 'generate_target_path'):
        results['generate_target_path'] = _benchmark_method(
            profiler, plugin, 'generate_target_path', test_files, iterations
        )

    if hasattr(plugin, 'generate_filename'):
        results['generate_filename'] = _benchmark_method(
            profiler, plugin, 'generate_filename', test_files, iterations
        )

    return results


def _benchmark_method(
    profiler: PerformanceProfiler,
    plugin: Plugin,
    method_name: str,
    test_files: List[AudioFile],
    iterations: int
) -> BenchmarkResult:
    """Benchmark a specific method."""
    execution_times = []
    memory_usage = []
    success_count = 0

    for _ in range(iterations):
        for test_file in test_files:
            metrics = profiler.measure_method(plugin, method_name, test_file)
            execution_times.append(metrics.execution_time)
            memory_usage.append(metrics.memory_usage)
            if metrics.success:
                success_count += 1

    # Calculate statistics
    avg_time = sum(execution_times) / len(execution_times)
    min_time = min(execution_times)
    max_time = max(execution_times)

    # Calculate standard deviation
    variance = sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)
    std_dev = variance ** 0.5

    # Calculate throughput (items per second)
    throughput = len(test_files) * iterations / sum(execution_times) if sum(execution_times) > 0 else 0

    # Get representative metrics
    sample_metrics = profiler.measure_method(plugin, method_name, test_files[0])

    # Create updated metrics with average values
    updated_metrics = PerformanceMetrics(
        execution_time=avg_time,
        memory_usage=sum(memory_usage) / len(memory_usage),
        peak_memory=sample_metrics.peak_memory,
        cpu_percent=sample_metrics.cpu_percent,
        success=sample_metrics.success,
        error_message=sample_metrics.error_message,
        throughput=len(test_files) * iterations / (avg_time * len(test_files)) if avg_time > 0 else 0
    )

    return BenchmarkResult(
        test_name=method_name,
        metrics=updated_metrics,
        iterations=len(test_files) * iterations,
        avg_time=avg_time,
        min_time=min_time,
        max_time=max_time,
        std_dev=std_dev,
        memory_trend=memory_usage
    )


def profile_memory_usage(
    plugin: Plugin,
    method_name: str,
    test_files: List[AudioFile]
) -> Dict[str, Any]:
    """
    Profile memory usage of a plugin method.

    Args:
        plugin: Plugin to profile
        method_name: Method name to profile
        test_files: Test files to process

    Returns:
        Dictionary with memory profiling results
    """
    profiler = MemoryProfiler()
    method = getattr(plugin, method_name)

    with profiler.track_memory():
        if asyncio.iscoroutinefunction(method):
            for test_file in test_files:
                asyncio.run(method(test_file))
        else:
            for test_file in test_files:
                method(test_file)

    memory_trend = profiler.get_memory_trend()
    peak_trend = profiler.get_peak_memory_trend()

    return {
        'method': method_name,
        'files_processed': len(test_files),
        'memory_trend': memory_trend,
        'peak_trend': peak_trend,
        'initial_memory': memory_trend[0] if memory_trend else 0,
        'final_memory': memory_trend[-1] if memory_trend else 0,
        'peak_memory': max(peak_trend) if peak_trend else 0,
        'memory_growth': (memory_trend[-1] - memory_trend[0]) if memory_trend and len(memory_trend) > 1 else 0
    }


def measure_throughput(
    plugin: Plugin,
    method_name: str,
    test_files: List[AudioFile],
    duration: float = 10.0
) -> Dict[str, Any]:
    """
    Measure plugin throughput over a fixed duration.

    Args:
        plugin: Plugin to test
        method_name: Method to test
        test_files: Test files to process
        duration: Duration in seconds to run test

    Returns:
        Dictionary with throughput measurements
    """
    method = getattr(plugin, method_name)
    if not method:
        return {'error': f'Method {method_name} not found'}

    start_time = time.time()
    processed_files = 0
    errors = 0

    file_queue = iter(test_files)

    while time.time() - start_time < duration:
        try:
            test_file = next(file_queue)
        except StopIteration:
            # Reset iterator if we run out of files
            file_queue = iter(test_files)
            test_file = next(file_queue)

        try:
            if asyncio.iscoroutinefunction(method):
                asyncio.run(method(test_file))
            else:
                method(test_file)
            processed_files += 1
        except Exception:
            errors += 1

    actual_duration = time.time() - start_time
    throughput = processed_files / actual_duration

    return {
        'method': method_name,
        'duration': actual_duration,
        'files_processed': processed_files,
        'errors': errors,
        'throughput': throughput,
        'avg_time_per_file': actual_duration / processed_files if processed_files > 0 else 0
    }


def compare_plugin_performance(
    plugins: List[Plugin],
    method_name: str,
    test_files: List[AudioFile]
) -> Dict[str, Any]:
    """
    Compare performance of multiple plugins.

    Args:
        plugins: List of plugins to compare
        method_name: Method name to compare
        test_files: Test files to use

    Returns:
        Dictionary with comparison results
    """
    comparison_results = {}

    for plugin in plugins:
        plugin_name = plugin.info.name if hasattr(plugin, 'info') else str(plugin)
        plugin_results = benchmark_plugin(plugin, test_files, iterations=3)

        if method_name in plugin_results:
            comparison_results[plugin_name] = {
                'avg_time': plugin_results[method_name].avg_time,
                'memory_usage': plugin_results[method_name].metrics.memory_usage,
                'throughput': plugin_results[method_name].metrics.throughput,
                'success_rate': sum(1 for r in plugin_results[method_name].execution_times
                                 if r > 0) / len(plugin_results[method_name].execution_times)
            }

    # Find best performers
    if comparison_results:
        fastest = min(comparison_results.items(), key=lambda x: x[1]['avg_time'])
        lowest_memory = min(comparison_results.items(), key=lambda x: x[1]['memory_usage'])
        highest_throughput = max(comparison_results.items(), key=lambda x: x[1]['throughput'])

        return {
            'results': comparison_results,
            'fastest': {'plugin': fastest[0], 'time': fastest[1]['avg_time']},
            'lowest_memory': {'plugin': lowest_memory[0], 'memory': lowest_memory[1]['memory_usage']},
            'highest_throughput': {'plugin': highest_throughput[0], 'throughput': highest_throughput[1]['throughput']}
        }

    return {'results': {}, 'message': 'No results to compare'}


def generate_performance_report(benchmark_results: Dict[str, BenchmarkResult]) -> str:
    """
    Generate a human-readable performance report.

    Args:
        benchmark_results: Results from benchmark_plugin

    Returns:
        Formatted performance report string
    """
    report = []
    report.append("Plugin Performance Report")
    report.append("=" * 50)
    report.append("")

    for method_name, result in benchmark_results.items():
        report.append(f"Method: {method_name}")
        report.append("-" * 30)
        report.append(f"Iterations: {result.iterations}")
        report.append(f"Average time: {result.avg_time:.4f}s")
        report.append(f"Min time: {result.min_time:.4f}s")
        report.append(f"Max time: {result.max_time:.4f}s")
        report.append(f"Std deviation: {result.std_dev:.4f}s")
        report.append(f"Memory usage: {result.metrics.memory_usage / 1024 / 1024:.2f} MB")
        report.append(f"Peak memory: {result.metrics.peak_memory / 1024 / 1024:.2f} MB")
        report.append(f"Throughput: {result.metrics.throughput:.2f} items/s")
        report.append("")

    return "\n".join(report)