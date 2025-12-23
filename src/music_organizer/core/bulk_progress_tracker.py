"""Enhanced progress tracking for bulk file operations.

This module provides specialized progress tracking for bulk operations,
including per-batch metrics, conflict resolution tracking, and detailed
performance statistics.
"""

import time
import math
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from ..progress_tracker import ProgressStage, IntelligentProgressTracker


class BulkProgressStage(Enum):
    """Stages specific to bulk operations."""
    PREPARING_OPERATIONS = "preparing_operations"
    CREATING_DIRECTORIES = "creating_directories"
    BATCH_PROCESSING = "batch_processing"
    CONFLICT_RESOLUTION = "conflict_resolution"
    VERIFICATION = "verification"
    FINALIZATION = "finalization"


@dataclass
class BatchMetrics:
    """Metrics for a single batch of operations."""
    batch_id: int
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_operations: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    conflicts_resolved: int = 0
    total_size_mb: float = 0.0
    throughput_mb_per_sec: float = 0.0

    @property
    def duration(self) -> float:
        """Duration of this batch in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def success_rate(self) -> float:
        """Success rate for this batch."""
        if self.total_operations == 0:
            return 0.0
        return (self.successful / self.total_operations) * 100

    def finalize(self):
        """Finalize batch metrics."""
        self.end_time = time.time()
        if self.duration > 0:
            self.throughput_mb_per_sec = self.total_size_mb / self.duration


@dataclass
class ConflictMetrics:
    """Metrics for conflict resolution."""
    total_conflicts: int = 0
    skipped_conflicts: int = 0
    renamed_conflicts: int = 0
    replaced_conflicts: int = 0
    conflict_resolution_time: float = 0.0

    @property
    def conflict_rate(self) -> float:
        """Percentage of files that had conflicts."""
        if self.total_conflicts == 0:
            return 100.0  # No conflicts means 100% success rate
        # Calculate rate based on resolved conflicts vs total
        resolved = self.skipped_conflicts + self.renamed_conflicts + self.replaced_conflicts
        return (resolved / max(1, self.total_conflicts)) * 100


class BulkProgressTracker:
    """Enhanced progress tracker for bulk file operations."""

    def __init__(self,
                 update_interval: float = 0.5,
                 batch_size: int = 100):
        """Initialize bulk progress tracker.

        Args:
            update_interval: Minimum time between progress updates (seconds)
            batch_size: Expected size of each processing batch
        """
        self.update_interval = update_interval
        self.base_tracker = IntelligentProgressTracker(update_interval)
        self.batch_size = batch_size

        # Bulk-specific metrics
        self.current_batch: int = 0
        self.total_batches: int = 0
        self.batch_metrics: List[BatchMetrics] = []
        self.current_batch_metrics: Optional[BatchMetrics] = None

        # Conflict tracking
        self.conflict_metrics = ConflictMetrics()

        # Performance tracking
        self.peak_throughput_mb_per_sec: float = 0.0
        self.average_batch_duration: float = 0.0
        self.directory_creation_time: float = 0.0
        self.verification_time: float = 0.0

        # Stage tracking
        self.bulk_stages: Dict[BulkProgressStage, float] = {}

        # Callbacks
        self.batch_callbacks: List[Callable[[BatchMetrics], None]] = []

    def add_batch_callback(self, callback: Callable[[BatchMetrics], None]):
        """Add a callback for batch completion events."""
        self.batch_callbacks.append(callback)

    def start_bulk_operation(self, total_files: int, estimated_batches: Optional[int] = None):
        """Start tracking a bulk operation."""
        self.base_tracker.set_total_files(total_files)
        if estimated_batches is None:
            estimated_batches = (total_files + self.batch_size - 1) // self.batch_size
        self.total_batches = estimated_batches
        self.current_batch = 0

        # Initialize bulk stages
        self.base_tracker.start_stage(ProgressStage.MOVING, total_files)

    def start_batch(self, batch_id: int, operations_in_batch: int):
        """Start tracking a new batch."""
        self.current_batch = batch_id
        self.current_batch_metrics = BatchMetrics(
            batch_id=batch_id,
            total_operations=operations_in_batch
        )

    def update_batch_progress(self,
                            current: int,
                            total: int,
                            size_mb: float = 0.0,
                            conflict_resolved: bool = False):
        """Update progress within the current batch."""
        if self.current_batch_metrics:
            self.current_batch_metrics.total_size_mb += size_mb
            if conflict_resolved:
                self.current_batch_metrics.conflicts_resolved += 1
                self.conflict_metrics.total_conflicts += 1

        # Update base tracker
        overall_current = (self.current_batch - 1) * self.batch_size + current
        self.base_tracker.set_completed(overall_current)

    def complete_batch(self,
                      successful: int,
                      failed: int,
                      skipped: int,
                      conflicts_resolved: int = 0):
        """Complete the current batch with results."""
        if self.current_batch_metrics:
            self.current_batch_metrics.successful = successful
            self.current_batch_metrics.failed = failed
            self.current_batch_metrics.skipped = skipped
            self.current_batch_metrics.conflicts_resolved = conflicts_resolved
            self.current_batch_metrics.finalize()

            # Store metrics
            self.batch_metrics.append(self.current_batch_metrics)

            # Update peak throughput
            if self.current_batch_metrics.throughput_mb_per_sec > self.peak_throughput_mb_per_sec:
                self.peak_throughput_mb_per_sec = self.current_batch_metrics.throughput_mb_per_sec

            # Update average batch duration
            total_duration = sum(m.duration for m in self.batch_metrics)
            self.average_batch_duration = total_duration / len(self.batch_metrics)

            # Trigger callbacks
            for callback in self.batch_callbacks:
                try:
                    callback(self.current_batch_metrics)
                except Exception:
                    pass

        self.current_batch_metrics = None

    def start_bulk_stage(self, stage: BulkProgressStage):
        """Start a bulk-specific stage."""
        self.bulk_stages[stage] = time.time()

    def finish_bulk_stage(self, stage: BulkProgressStage):
        """Finish a bulk-specific stage."""
        if stage in self.bulk_stages:
            duration = time.time() - self.bulk_stages[stage]

            # Track specific stage durations
            if stage == BulkProgressStage.CREATING_DIRECTORIES:
                self.directory_creation_time = duration
            elif stage == BulkProgressStage.VERIFICATION:
                self.verification_time = duration

            del self.bulk_stages[stage]

    def update_conflict_resolution(self, strategy: str, resolution_time: float):
        """Update conflict resolution metrics."""
        self.conflict_metrics.total_conflicts += 1
        self.conflict_metrics.conflict_resolution_time += resolution_time

        if strategy == "skip":
            self.conflict_metrics.skipped_conflicts += 1
        elif strategy in ["rename", "keep_both"]:
            self.conflict_metrics.renamed_conflicts += 1
        elif strategy == "replace":
            self.conflict_metrics.replaced_conflicts += 1

    def get_bulk_summary(self) -> Dict[str, Any]:
        """Get comprehensive bulk operation summary."""
        base_summary = self.base_tracker.get_summary()

        # Calculate bulk-specific metrics
        total_successful = sum(m.successful for m in self.batch_metrics)
        total_failed = sum(m.failed for m in self.batch_metrics)
        total_skipped = sum(m.skipped for m in self.batch_metrics)
        total_size_mb = sum(m.total_size_mb for m in self.batch_metrics)

        # Calculate efficiency metrics
        if len(self.batch_metrics) > 0:
            avg_batch_efficiency = sum(m.success_rate for m in self.batch_metrics) / len(self.batch_metrics)
        else:
            avg_batch_efficiency = 0.0

        # Current stage progress
        current_bulk_stage = None
        if self.bulk_stages:
            current_bulk_stage = list(self.bulk_stages.keys())[0]

        bulk_summary = {
            # Basic stats
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'batch_completion_percentage': (self.current_batch / max(1, self.total_batches)) * 100,

            # Operation stats
            'total_successful': total_successful,
            'total_failed': total_failed,
            'total_skipped': total_skipped,
            'total_size_mb': total_size_mb,

            # Performance metrics
            'peak_throughput_mb_per_sec': self.peak_throughput_mb_per_sec,
            'average_batch_duration': self.average_batch_duration,
            'avg_batch_efficiency': avg_batch_efficiency,

            # Timing breakdown
            'directory_creation_time': self.directory_creation_time,
            'verification_time': self.verification_time,
            'conflict_resolution_time': self.conflict_metrics.conflict_resolution_time,

            # Conflict metrics
            'conflicts': {
                'total': self.conflict_metrics.total_conflicts,
                'skipped': self.conflict_metrics.skipped_conflicts,
                'renamed': self.conflict_metrics.renamed_conflicts,
                'replaced': self.conflict_metrics.replaced_conflicts,
                'conflict_rate': self.conflict_metrics.conflict_rate
            },

            # Current stage
            'current_bulk_stage': current_bulk_stage.value if current_bulk_stage else None,

            # Batch details
            'batch_count': len(self.batch_metrics),
            'recent_batches': [
                {
                    'batch_id': m.batch_id,
                    'success_rate': m.success_rate,
                    'throughput_mb_per_sec': m.throughput_mb_per_sec,
                    'duration': m.duration,
                    'conflicts_resolved': m.conflicts_resolved
                }
                for m in self.batch_metrics[-5:]  # Last 5 batches
            ]
        }

        # Merge with base summary
        base_summary.update(bulk_summary)
        return base_summary

    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance analysis."""
        if not self.batch_metrics:
            return {"error": "No batches completed yet"}

        # Calculate performance statistics
        durations = [m.duration for m in self.batch_metrics]
        throughputs = [m.throughput_mb_per_sec for m in self.batch_metrics]
        success_rates = [m.success_rate for m in self.batch_metrics]

        # Identify best and worst batches
        best_batch = max(self.batch_metrics, key=lambda m: m.throughput_mb_per_sec)
        worst_batch = min(self.batch_metrics, key=lambda m: m.throughput_mb_per_sec)

        report = {
            "performance_analysis": {
                "total_batches": len(self.batch_metrics),
                "total_time": sum(durations),
                "average_batch_duration": {
                    "mean": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "std_dev": self._std_dev(durations)
                },
                "throughput_analysis": {
                    "peak_mb_per_sec": max(throughputs),
                    "average_mb_per_sec": sum(throughputs) / len(throughputs),
                    "min_mb_per_sec": min(throughputs),
                    "std_dev": self._std_dev(throughputs)
                },
                "success_rate_analysis": {
                    "average": sum(success_rates) / len(success_rates),
                    "min": min(success_rates),
                    "max": max(success_rates)
                }
            },
            "best_batch": {
                "batch_id": best_batch.batch_id,
                "throughput_mb_per_sec": best_batch.throughput_mb_per_sec,
                "success_rate": best_batch.success_rate,
                "duration": best_batch.duration
            },
            "worst_batch": {
                "batch_id": worst_batch.batch_id,
                "throughput_mb_per_sec": worst_batch.throughput_mb_per_sec,
                "success_rate": worst_batch.success_rate,
                "duration": worst_batch.duration
            },
            "optimization_suggestions": self._generate_optimization_suggestions()
        }

        return report

    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def _generate_optimization_suggestions(self) -> List[str]:
        """Generate performance optimization suggestions."""
        suggestions = []

        if self.peak_throughput_mb_per_sec > 0:
            avg_throughput = sum(m.throughput_mb_per_sec for m in self.batch_metrics) / len(self.batch_metrics)
            if avg_throughput < self.peak_throughput_mb_per_sec * 0.7:
                suggestions.append(
                    "Consider increasing batch size or worker count for better consistency"
                )

        if self.conflict_metrics.total_conflicts > 0:
            conflict_rate = self.conflict_metrics.total_conflicts / max(1, sum(m.total_operations for m in self.batch_metrics))
            if conflict_rate > 0.1:  # More than 10% conflicts
                suggestions.append(
                    "High conflict rate detected. Consider using different naming strategy or pre-scanning target directory"
                )

        if self.average_batch_duration > 10:  # Batches taking more than 10 seconds
            suggestions.append(
                "Batches are taking long to process. Consider reducing batch size or investigating filesystem performance"
            )

        if self.directory_creation_time > 5:
            suggestions.append(
                "Directory creation is taking significant time. Consider batch directory creation optimization"
            )

        return suggestions

    def reset(self):
        """Reset all tracking metrics."""
        self.base_tracker = IntelligentProgressTracker(self.base_tracker.update_interval)
        self.current_batch = 0
        self.total_batches = 0
        self.batch_metrics.clear()
        self.current_batch_metrics = None
        self.conflict_metrics = ConflictMetrics()
        self.peak_throughput_mb_per_sec = 0.0
        self.average_batch_duration = 0.0
        self.directory_creation_time = 0.0
        self.verification_time = 0.0
        self.bulk_stages.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Finalize any ongoing batch
        if self.current_batch_metrics:
            self.current_batch_metrics.finalize()
            self.batch_metrics.append(self.current_batch_metrics)