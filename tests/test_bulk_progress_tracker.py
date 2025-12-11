"""Tests for bulk operation progress tracking."""

import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import MagicMock, AsyncMock
import pytest

from music_organizer.core.bulk_progress_tracker import (
    BulkProgressTracker,
    BulkProgressStage,
    BatchMetrics,
    ConflictMetrics
)
from music_organizer.core.bulk_operations import BatchMetrics as BulkBatchMetrics


class TestBatchMetrics:
    """Test the BatchMetrics dataclass."""

    def test_batch_metrics_creation(self):
        """Test creating BatchMetrics."""
        metrics = BatchMetrics(
            batch_id=1,
            total_operations=100,
            successful=90,
            failed=5,
            skipped=5,
            total_size_mb=1000.0
        )

        assert metrics.batch_id == 1
        assert metrics.total_operations == 100
        assert metrics.successful == 90
        assert metrics.failed == 5
        assert metrics.skipped == 5
        assert metrics.total_size_mb == 1000.0

    def test_batch_metrics_success_rate(self):
        """Test success rate calculation."""
        metrics = BatchMetrics(
            batch_id=1,
            total_operations=100,
            successful=90,
            failed=5,
            skipped=5
        )

        assert metrics.success_rate == 90.0

        # Test zero total
        metrics.total_operations = 0
        assert metrics.success_rate == 0.0

    def test_batch_metrics_duration(self):
        """Test duration calculation."""
        start_time = time.time()
        metrics = BatchMetrics(
            batch_id=1,
            start_time=start_time
        )

        # Duration should be positive
        time.sleep(0.1)
        duration = metrics.duration
        assert duration > 0.09  # At least 100ms

        # Test finalized metrics
        metrics.end_time = start_time + 5.0
        assert metrics.duration == 5.0

    def test_batch_metrics_finalize(self):
        """Test finalizing batch metrics."""
        metrics = BatchMetrics(
            batch_id=1,
            total_operations=100,
            successful=90,
            failed=5,
            skipped=5,
            total_size_mb=1000.0
        )

        # Simulate some processing time
        time.sleep(0.1)
        metrics.finalize()

        assert metrics.end_time is not None
        assert metrics.duration > 0
        assert metrics.throughput_mb_per_sec > 0
        # Throughput should be size_mb / duration
        expected_throughput = 1000.0 / metrics.duration
        assert abs(metrics.throughput_mb_per_sec - expected_throughput) < 0.1


class TestConflictMetrics:
    """Test the ConflictMetrics dataclass."""

    def test_conflict_metrics_creation(self):
        """Test creating ConflictMetrics."""
        metrics = ConflictMetrics(
            total_conflicts=100,
            skipped_conflicts=20,
            renamed_conflicts=60,
            replaced_conflicts=20
        )

        assert metrics.total_conflicts == 100
        assert metrics.skipped_conflicts == 20
        assert metrics.renamed_conflicts == 60
        assert metrics.replaced_conflicts == 20

    def test_conflict_rate(self):
        """Test conflict rate calculation."""
        metrics = ConflictMetrics(total_conflicts=100)

        # Mock total_operations
        metrics.total_conflicts = 50

        # Conflict rate should be percentage
        assert metrics.conflict_rate == 100.0  # 50/50 * 100

        # Test zero total
        metrics.total_conflicts = 0
        assert metrics.conflict_rate == 100.0  # Should handle division by zero


class TestBulkProgressTracker:
    """Test the BulkProgressTracker class."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = BulkProgressTracker(
            update_interval=0.5,
            batch_size=200
        )

        assert tracker.base_tracker is not None
        assert tracker.batch_size == 200
        assert tracker.update_interval == 0.5
        assert tracker.current_batch == 0
        assert tracker.total_batches == 0
        assert len(tracker.batch_metrics) == 0
        assert tracker.current_batch_metrics is None
        assert len(tracker.batch_callbacks) == 0

    def test_start_bulk_operation(self):
        """Test starting a bulk operation."""
        tracker = BulkProgressTracker(batch_size=100)

        tracker.start_bulk_operation(
            total_files=1000,
            estimated_batches=10
        )

        assert tracker.total_batches == 10
        assert tracker.current_batch == 0
        assert tracker.base_tracker.metrics.files_total == 1000

    def test_start_bulk_operation_without_estimates(self):
        """Test starting bulk operation without batch estimates."""
        tracker = BulkProgressTracker(batch_size=100)

        tracker.start_bulk_operation(total_files=1000)

        # Should estimate batches based on batch_size
        assert tracker.total_batches == 10  # 1000 / 100
        assert tracker.current_batch == 0

    def test_start_and_complete_batch(self):
        """Test starting and completing a batch."""
        tracker = BulkProgressTracker()
        tracker.start_bulk_operation(total_files=100, estimated_batches=2)

        # Start first batch
        tracker.start_batch(batch_id=1, operations_in_batch=50)
        assert tracker.current_batch == 1
        assert tracker.current_batch_metrics is not None
        assert tracker.current_batch_metrics.batch_id == 1
        assert tracker.current_batch_metrics.total_operations == 50

        # Update progress
        tracker.update_batch_progress(current=25, total=50, size_mb=10.0)
        assert tracker.base_tracker.metrics.files_processed == 25

        # Complete batch
        tracker.complete_batch(
            successful=45,
            failed=3,
            skipped=2,
            conflicts_resolved=5
        )

        assert tracker.current_batch_metrics is None
        assert len(tracker.batch_metrics) == 1
        assert tracker.batch_metrics[0].successful == 45
        assert tracker.batch_metrics[0].failed == 3
        assert tracker.batch_metrics[0].skipped == 2
        assert tracker.batch_metrics[0].conflicts_resolved == 5

    def test_multiple_batches(self):
        """Test tracking multiple batches."""
        tracker = BulkProgressTracker()
        tracker.start_bulk_operation(total_files=200, estimated_batches=4)

        # Complete multiple batches
        for batch_id in range(1, 4):
            tracker.start_batch(batch_id=batch_id, operations_in_batch=50)
            tracker.complete_batch(successful=48, failed=1, skipped=1)

        assert len(tracker.batch_metrics) == 3
        assert tracker.current_batch == 3  # Last completed batch

    def test_bulk_stage_tracking(self):
        """Test tracking bulk-specific stages."""
        tracker = BulkProgressTracker()

        # Start a stage
        tracker.start_bulk_stage(BulkProgressStage.CREATING_DIRECTORIES)
        assert BulkProgressStage.CREATING_DIRECTORIES in tracker.bulk_stages

        # Finish the stage
        time.sleep(0.1)  # Simulate some work
        tracker.finish_bulk_stage(BulkProgressStage.CREATING_DIRECTORIES)
        assert BulkProgressStage.CREATING_DIRECTORIES not in tracker.bulk_stages
        assert tracker.directory_creation_time > 0

    def test_conflict_resolution_tracking(self):
        """Test tracking conflict resolution."""
        tracker = BulkProgressTracker()

        # Update conflict resolution
        tracker.update_conflict_resolution("skip", 0.1)
        tracker.update_conflict_resolution("rename", 0.2)
        tracker.update_conflict_resolution("replace", 0.15)

        assert tracker.conflict_metrics.total_conflicts == 3
        assert tracker.conflict_metrics.skipped_conflicts == 1
        assert tracker.conflict_metrics.renamed_conflicts == 1
        assert tracker.conflict_metrics.replaced_conflicts == 1
        assert tracker.conflict_metrics.conflict_resolution_time == 0.45

    def test_batch_callbacks(self):
        """Test batch completion callbacks."""
        tracker = BulkProgressTracker()
        tracker.start_bulk_operation(total_files=100)

        # Track callback calls
        callback_calls = []

        def test_callback(batch_metrics):
            callback_calls.append(batch_metrics.batch_id)

        tracker.add_batch_callback(test_callback)

        # Complete a batch
        tracker.start_batch(batch_id=1, operations_in_batch=50)
        tracker.complete_batch(successful=50, failed=0, skipped=0)

        # Verify callback was called
        assert len(callback_calls) == 1
        assert callback_calls[0] == 1

    def test_get_bulk_summary(self):
        """Test getting bulk operation summary."""
        tracker = BulkProgressTracker()
        tracker.start_bulk_operation(total_files=100, estimated_batches=2)

        # Complete some batches
        tracker.start_batch(batch_id=1, operations_in_batch=50)
        tracker.complete_batch(successful=45, failed=3, skipped=2)

        tracker.start_batch(batch_id=2, operations_in_batch=50)
        tracker.complete_batch(successful=48, failed=1, skipped=1)

        # Add some conflicts
        tracker.update_conflict_resolution("rename", 0.1)
        tracker.update_conflict_resolution("skip", 0.05)

        summary = tracker.get_bulk_summary()

        # Verify summary contains expected data
        assert 'current_batch' in summary
        assert 'total_batches' in summary
        assert 'batch_completion_percentage' in summary
        assert summary['current_batch'] == 2
        assert summary['total_batches'] == 2
        assert summary['batch_completion_percentage'] == 100.0

        assert 'total_successful' in summary
        assert 'total_failed' in summary
        assert 'total_skipped' in summary
        assert summary['total_successful'] == 93  # 45 + 48
        assert summary['total_failed'] == 4  # 3 + 1
        assert summary['total_skipped'] == 3  # 2 + 1

        assert 'conflicts' in summary
        assert summary['conflicts']['total'] == 2
        assert summary['conflicts']['renamed'] == 1
        assert summary['conflicts']['skipped'] == 1

        assert 'batch_count' in summary
        assert summary['batch_count'] == 2
        assert 'recent_batches' in summary
        assert len(summary['recent_batches']) == 2

    def test_get_performance_report(self):
        """Test getting performance analysis report."""
        tracker = BulkProgressTracker()
        tracker.start_bulk_operation(total_files=100, estimated_batches=3)

        # Create batches with different performance characteristics
        # Fast batch
        tracker.start_batch(batch_id=1, operations_in_batch=33)
        time.sleep(0.05)  # Short duration
        tracker.complete_batch(successful=33, failed=0, skipped=0)

        # Slow batch
        tracker.start_batch(batch_id=2, operations_in_batch=33)
        time.sleep(0.15)  # Longer duration
        tracker.complete_batch(successful=30, failed=2, skipped=1)

        # Medium batch
        tracker.start_batch(batch_id=3, operations_in_batch=34)
        time.sleep(0.1)  # Medium duration
        tracker.complete_batch(successful=32, failed=1, skipped=1)

        report = tracker.get_performance_report()

        # Verify report structure
        assert 'performance_analysis' in report
        assert 'best_batch' in report
        assert 'worst_batch' in report
        assert 'optimization_suggestions' in report

        # Verify performance analysis
        analysis = report['performance_analysis']
        assert analysis['total_batches'] == 3
        assert 'average_batch_duration' in analysis
        assert 'throughput_analysis' in analysis
        assert 'success_rate_analysis' in analysis

        # Verify best/worst batch identification
        best = report['best_batch']
        worst = report['worst_batch']
        assert best['batch_id'] == 1  # Fastest batch
        assert worst['batch_id'] == 2  # Slowest batch
        assert best['throughput_mb_per_sec'] >= worst['throughput_mb_per_sec']

    def test_performance_report_no_batches(self):
        """Test performance report with no completed batches."""
        tracker = BulkProgressTracker()

        report = tracker.get_performance_report()
        assert 'error' in report
        assert 'No batches completed yet' in report['error']

    def test_optimization_suggestions(self):
        """Test optimization suggestion generation."""
        tracker = BulkProgressTracker()

        # Create scenarios that should trigger suggestions
        tracker.start_bulk_operation(total_files=100, estimated_batches=2)

        # High conflict rate scenario
        tracker.start_batch(batch_id=1, operations_in_batch=50)
        tracker.update_batch_progress(current=25, total=50, conflict_resolved=True)  # 50% conflict rate
        tracker.complete_batch(successful=25, failed=0, skipped=25)  # 25 conflicts

        # Slow batch scenario
        tracker.start_bulk(batch_id=2, operations_in_batch=50)
        tracker.directory_creation_time = 10.0  # High directory creation time
        tracker.complete_batch(successful=50, failed=0, skipped=0)

        report = tracker.get_performance_report()
        suggestions = report['optimization_suggestions']

        assert len(suggestions) > 0
        # Should include suggestions for high conflict rate and slow operations
        suggestions_text = ' '.join(suggestions).lower()
        assert 'conflict' in suggestions_text or 'directory' in suggestions_text

    def test_reset_functionality(self):
        """Test resetting the tracker."""
        tracker = BulkProgressTracker()
        tracker.start_bulk_operation(total_files=100, estimated_batches=2)

        # Add some data
        tracker.start_batch(batch_id=1, operations_in_batch=50)
        tracker.complete_batch(successful=45, failed=3, skipped=2)
        tracker.update_conflict_resolution("rename", 0.1)

        # Reset
        tracker.reset()

        # Verify everything is reset
        assert tracker.current_batch == 0
        assert tracker.total_batches == 0
        assert len(tracker.batch_metrics) == 0
        assert tracker.current_batch_metrics is None
        assert tracker.conflict_metrics.total_conflicts == 0
        assert tracker.peak_throughput_mb_per_sec == 0.0
        assert tracker.directory_creation_time == 0.0

    def test_context_manager(self):
        """Test using tracker as context manager."""
        with BulkProgressTracker() as tracker:
            tracker.start_bulk_operation(total_files=100, estimated_batches=2)
            tracker.start_batch(batch_id=1, operations_in_batch=50)
            tracker.complete_batch(successful=50, failed=0, skipped=0)

        # After context exit, any ongoing batch should be finalized
        assert len(tracker.batch_metrics) == 1

    def test_peak_throughput_tracking(self):
        """Test peak throughput calculation."""
        tracker = BulkProgressTracker()
        tracker.start_bulk_operation(total_files=100, estimated_batches=3)

        # Create batches with increasing throughput
        for i in range(3):
            tracker.start_batch(batch_id=i+1, operations_in_batch=33)
            # Simulate different processing times
            time.sleep(0.1 / (i + 1))  # Faster each batch
            tracker.complete_batch(successful=33, failed=0, skipped=0)

        # The last batch should have the highest throughput
        assert tracker.peak_throughput_mb_per_sec > 0
        assert tracker.peak_throughput_mb_per_sec >= tracker.batch_metrics[-1].throughput_mb_per_sec

    def test_average_batch_duration(self):
        """Test average batch duration calculation."""
        tracker = BulkProgressTracker()
        tracker.start_bulk_operation(total_files=100, estimated_batches=3)

        # Create batches with known durations
        durations = [0.1, 0.2, 0.15]
        for i, duration in enumerate(durations):
            tracker.start_batch(batch_id=i+1, operations_in_batch=33)
            time.sleep(duration)
            tracker.complete_batch(successful=33, failed=0, skipped=0)

        expected_average = sum(durations) / len(durations)
        assert abs(tracker.average_batch_duration - expected_average) < 0.05

    def test_error_handling_in_callbacks(self):
        """Test that errors in callbacks don't break processing."""
        tracker = BulkProgressTracker()
        tracker.start_bulk_operation(total_files=100)

        # Add callbacks that raise exceptions
        def failing_callback(batch_metrics):
            raise Exception("Callback error")

        def working_callback(batch_metrics):
            working_callback.called = True

        working_callback.called = False

        tracker.add_batch_callback(failing_callback)
        tracker.add_batch_callback(working_callback)

        # Complete a batch - should not raise exceptions
        tracker.start_batch(batch_id=1, operations_in_batch=50)
        tracker.complete_batch(successful=50, failed=0, skipped=0)

        # Working callback should still be called despite failing callback
        assert working_callback.called
        assert len(tracker.batch_metrics) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])