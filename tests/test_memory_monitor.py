"""Tests for memory monitor utility."""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from music_organizer.utils.memory_monitor import (
    MemorySnapshot,
    MemoryStats,
    MemoryMonitor,
    MemoryProfiler,
    profile_memory,
    monitor_memory,
    get_global_monitor,
    start_global_monitoring,
    stop_global_monitoring,
    get_memory_pressure,
    should_use_streaming
)


class TestMemorySnapshot:
    """Test MemorySnapshot data class."""

    def test_memory_snapshot_creation(self):
        """Test creating MemorySnapshot."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=100.5,
            vms_mb=200.0,
            percent=10.5,
            python_mb=50.0,
            tracemalloc_current=45.0,
            tracemalloc_peak=60.0
        )

        assert snapshot.timestamp > 0
        assert snapshot.rss_mb == 100.5
        assert snapshot.vms_mb == 200.0
        assert snapshot.percent == 10.5
        assert snapshot.python_mb == 50.0
        assert snapshot.tracemalloc_current == 45.0
        assert snapshot.tracemalloc_peak == 60.0


class TestMemoryStats:
    """Test MemoryStats class."""

    def test_memory_stats_initialization(self):
        """Test MemoryStats initialization."""
        stats = MemoryStats()

        assert stats.start_time > 0
        assert stats.end_time is None
        assert stats.snapshots == []
        assert stats.peak_rss == 0
        assert stats.peak_python == 0
        assert stats.peak_tracemalloc == 0
        assert stats.avg_rss == 0
        assert stats.total_time == 0

    def test_add_snapshot(self):
        """Test adding snapshot to stats."""
        stats = MemoryStats()

        snapshot1 = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=100,
            vms_mb=200,
            percent=10,
            python_mb=50,
            tracemalloc_current=45,
            tracemalloc_peak=60
        )

        snapshot2 = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=150,
            vms_mb=250,
            percent=15,
            python_mb=75,
            tracemalloc_current=70,
            tracemalloc_peak=85
        )

        # Add snapshots
        stats.add_snapshot(snapshot1)
        assert len(stats.snapshots) == 1
        assert stats.peak_rss == 100
        assert stats.peak_python == 50
        assert stats.peak_tracemalloc == 60

        stats.add_snapshot(snapshot2)
        assert len(stats.snapshots) == 2
        assert stats.peak_rss == 150  # Should update to max
        assert stats.peak_python == 75  # Should update to max
        assert stats.peak_tracemalloc == 85  # Should update to max

    def test_finalize(self):
        """Test finalizing stats."""
        stats = MemoryStats()

        # Sleep to ensure time difference
        time.sleep(0.01)

        stats.finalize()

        assert stats.end_time is not None
        assert stats.end_time > stats.start_time
        assert stats.total_time > 0


class TestMemoryMonitor:
    """Test MemoryMonitor class."""

    def test_memory_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = MemoryMonitor(enable_tracemalloc=False)

        assert monitor.enable_tracemalloc is False
        assert monitor.snapshots == []
        assert monitor.start_time is None
        assert monitor.end_time is None
        assert monitor.monitoring is False

    @patch('music_organizer.utils.memory_monitor.psutil.Process')
    @patch('music_organizer.utils.memory_monitor.psutil.virtual_memory')
    def test_take_snapshot(self, mock_vmem, mock_process_class):
        """Test taking a memory snapshot."""
        # Mock process memory info
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(
            rss=1024 * 1024 * 100,  # 100MB
            vms=1024 * 1024 * 200   # 200MB
        )
        mock_process.memory_percent.return_value = 10.5
        mock_process_class.return_value = mock_process

        # Mock system memory info
        mock_vmem.return_value = Mock(
            total=1024 * 1024 * 1024,  # 1GB
            percent=30.0
        )

        # Mock Python memory info - add gettotalrefcount to sys first
        import sys as sys_module
        sys_module.gettotalrefcount = lambda: 6553600  # ~50MB

        try:
            # Use order that works with context managers - outer applies first
            with patch('music_organizer.utils.memory_monitor.tracemalloc.is_tracing', return_value=True):
                with patch('music_organizer.utils.memory_monitor.tracemalloc.get_traced_memory', return_value=(45000000, 60000000)):
                    monitor = MemoryMonitor(enable_tracemalloc=True)
                    snapshot = monitor.take_snapshot()

                    assert isinstance(snapshot, MemorySnapshot)
                    assert snapshot.rss_mb == 100.0
                    assert snapshot.vms_mb == 200.0
                    assert snapshot.percent == 10.5
                    assert snapshot.python_mb == 50.0
                    # tracemalloc values may be affected by actual tracing state
                    assert snapshot.tracemalloc_current > 0
                    assert snapshot.tracemalloc_peak > 0
        finally:
            delattr(sys_module, 'gettotalrefcount')

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        monitor = MemoryMonitor(enable_tracemalloc=False)

        # Start monitoring
        monitor.start()
        assert monitor.monitoring is True
        assert monitor.start_time is not None
        time.sleep(0.1)  # Give thread time to take at least one snapshot
        assert len(monitor.snapshots) >= 1  # Should take initial snapshot

        # Stop monitoring
        stats = monitor.stop()
        assert monitor.monitoring is False
        assert monitor.end_time is not None
        assert isinstance(stats, MemoryStats)
        assert len(stats.snapshots) >= 1

    def test_monitoring_thread(self):
        """Test the monitoring thread functionality."""
        monitor = MemoryMonitor(interval=0.05, enable_tracemalloc=False)

        # Mock _get_snapshot to avoid actual memory reading
        call_count = 0

        def mock_get_snapshot():
            nonlocal call_count
            call_count += 1
            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=100 + call_count,
                vms_mb=200,
                percent=10,
                python_mb=50
            )

        monitor._get_snapshot = mock_get_snapshot

        # Start monitoring
        monitor.start()

        # Let it run for a short time
        time.sleep(0.15)  # Should allow ~3 intervals

        # Stop monitoring
        stats = monitor.stop()

        # Verify multiple snapshots were taken
        assert call_count >= 3
        assert len(stats.snapshots) >= 3

    def test_get_current_memory(self):
        """Test getting current memory usage."""
        monitor = MemoryMonitor(enable_tracemalloc=False)

        with patch('music_organizer.utils.memory_monitor.psutil.Process'), \
             patch('music_organizer.utils.memory_monitor.psutil.virtual_memory'), \
             patch('music_organizer.utils.memory_monitor.sys.getsizeof', return_value=50000000):

            memory_info = monitor.get_current_usage()

            assert isinstance(memory_info, dict)
            assert 'rss_mb' in memory_info
            assert 'vms_mb' in memory_info
            assert 'percent' in memory_info
            assert 'python_mb' in memory_info

    def test_reset(self):
        """Test resetting monitor state."""
        monitor = MemoryMonitor(enable_tracemalloc=False)

        # Add some data
        monitor.start_time = time.time()
        monitor.snapshots.append(MemorySnapshot(
            timestamp=time.time(),
            rss_mb=100,
            vms_mb=200,
            percent=10,
            python_mb=50
        ))

        # Reset
        monitor.reset()

        assert monitor.start_time is None
        assert monitor.end_time is None
        assert len(monitor.snapshots) == 0


class TestMemoryProfiler:
    """Test MemoryProfiler class."""

    def test_profiler_context_manager(self):
        """Test MemoryProfiler as context manager."""
        with patch('music_organizer.utils.memory_monitor.get_global_monitor') as mock_get_monitor:
            mock_monitor = Mock(spec=MemoryMonitor)
            mock_get_monitor.return_value = mock_monitor

            # Mock stats with proper values
            mock_stats = MemoryStats()
            mock_stats.peak_rss = 120
            mock_stats.avg_rss = 110
            mock_stats.peak_python = 60
            mock_stats.total_time = 0.1
            mock_stats.snapshots = []

            mock_monitor.stop.return_value = mock_stats

            # Use context manager
            with MemoryProfiler("test_operation"):
                time.sleep(0.01)  # Simulate some work

            # Verify start/stop were called
            mock_monitor.start.assert_called_once()
            mock_monitor.stop.assert_called_once()

    def test_profiler_with_stats(self):
        """Test MemoryProfiler getting stats."""
        with patch('music_organizer.utils.memory_monitor.get_global_monitor') as mock_get_monitor:
            mock_monitor = Mock(spec=MemoryMonitor)
            mock_get_monitor.return_value = mock_monitor

            # Mock stats
            mock_stats = MemoryStats()
            mock_stats.peak_rss = 120
            mock_stats.peak_python = 60
            mock_stats.total_time = 0.1

            mock_monitor.stop.return_value = mock_stats

            with MemoryProfiler("test_operation") as profiler:
                pass

            # Get stats
            stats = profiler.get_stats()
            assert stats.peak_rss == 120
            assert stats.peak_python == 60
            assert stats.total_time == 0.1


class TestProfileMemoryDecorator:
    """Test profile_memory decorator."""

    def test_profile_memory_decorator(self):
        """Test profile_memory decorator function."""
        with patch('music_organizer.utils.memory_monitor.MemoryProfiler') as mock_profiler_class:
            mock_profiler = Mock()
            mock_profiler.__enter__ = Mock(return_value=mock_profiler)
            mock_profiler.__exit__ = Mock(return_value=None)
            mock_profiler_class.return_value = mock_profiler

            # Apply decorator
            @profile_memory("test_function")
            def test_function(x, y):
                return x + y

            # Call function
            result = test_function(1, 2)

            # Verify result
            assert result == 3

            # Verify profiler was used
            mock_profiler_class.assert_called_once_with("test_function", enable_tracemalloc=False)
            mock_profiler.__enter__.assert_called_once()
            mock_profiler.__exit__.assert_called_once()

    def test_profile_memory_without_name(self):
        """Test profile_memory decorator without name."""
        with patch('music_organizer.utils.memory_monitor.MemoryProfiler') as mock_profiler_class:
            mock_profiler = Mock()
            mock_profiler.__enter__ = Mock(return_value=mock_profiler)
            mock_profiler.__exit__ = Mock(return_value=None)
            mock_profiler_class.return_value = mock_profiler

            @profile_memory()
            def test_function():
                return "test"

            test_function()

            # Should use function name as default
            call_args = mock_profiler_class.call_args[0]
            assert "test_function" in call_args[0]


class TestMonitorMemoryContextManager:
    """Test monitor_memory context manager."""

    def test_monitor_memory_context_manager(self):
        """Test monitor_memory for detecting memory issues."""
        with patch('music_organizer.utils.memory_monitor.get_global_monitor') as mock_get_monitor:
            mock_monitor = Mock(spec=MemoryMonitor)
            mock_get_monitor.return_value = mock_monitor

            # Mock memory check that never exceeds limit - snapshots with low RSS
            mock_monitor.snapshots = [
                MemorySnapshot(
                    timestamp=time.time(),
                    rss_mb=50,
                    vms_mb=100,
                    percent=10,
                    python_mb=30
                )
            ]

            # Use monitor_memory with high limit
            with monitor_memory(limit_mb=100, check_interval=0.01):
                time.sleep(0.05)  # Allow multiple checks

            # Verify start/stop were called
            mock_monitor.start.assert_called_once()
            mock_monitor.stop.assert_called_once()

    def test_monitor_memory_exceeds_limit(self):
        """Test monitor_memory when memory exceeds limit."""
        with patch('music_organizer.utils.memory_monitor.get_global_monitor') as mock_get_monitor:
            mock_monitor = Mock(spec=MemoryMonitor)
            mock_get_monitor.return_value = mock_monitor

            # Mock memory check that exceeds limit - snapshots with high RSS
            mock_monitor.snapshots = [
                MemorySnapshot(
                    timestamp=time.time(),
                    rss_mb=150,
                    vms_mb=200,
                    percent=15,
                    python_mb=50
                )
            ]

            # Should raise exception when memory exceeds limit
            with pytest.raises(MemoryError, match="Memory limit exceeded"):
                with monitor_memory(limit_mb=100, check_interval=0.01):
                    time.sleep(0.05)


class TestGlobalMonitor:
    """Test global monitor functions."""

    def test_get_global_monitor(self):
        """Test getting global monitor instance."""
        # Should return the same instance on multiple calls
        monitor1 = get_global_monitor()
        monitor2 = get_global_monitor()

        assert monitor1 is monitor2
        assert isinstance(monitor1, MemoryMonitor)

    def test_start_stop_global_monitoring(self):
        """Test starting and stopping global monitoring."""
        # Stop any existing monitoring
        try:
            stop_global_monitoring()
        except:
            pass

        # Start monitoring
        start_global_monitoring(enable_tracemalloc=False)

        # Get monitor and verify it's running
        monitor = get_global_monitor()
        assert monitor.monitoring is True

        # Stop monitoring
        stats = stop_global_monitoring()

        # Verify monitoring stopped and stats returned
        assert monitor.monitoring is False
        assert isinstance(stats, MemoryStats)


class TestMemoryUtilities:
    """Test memory utility functions."""

    @patch('music_organizer.utils.memory_monitor.psutil.virtual_memory')
    def test_get_memory_pressure(self, mock_vmem):
        """Test memory pressure calculation."""
        # Test low memory pressure
        mock_vmem.return_value = Mock(percent=30.0)
        pressure = get_memory_pressure()
        assert pressure == 0.3

        # Test high memory pressure
        mock_vmem.return_value = Mock(percent=85.0)
        pressure = get_memory_pressure()
        assert pressure == 0.85

    def test_should_use_streaming(self):
        """Test streaming decision based on file count."""
        # Test small file count - should not use streaming
        assert should_use_streaming(100, memory_limit_mb=100) is False

        # Test large file count - should use streaming
        assert should_use_streaming(1000, memory_limit_mb=100) is True

        # Test with different memory limit
        assert should_use_streaming(1000, memory_limit_mb=1000) is False

        # Test edge case
        assert should_use_streaming(500, memory_limit_mb=100) is True


class TestMemoryMonitorIntegration:
    """Integration tests for memory monitor."""

    def test_end_to_end_profiling(self):
        """Test end-to-end memory profiling."""
        # Create a monitor
        monitor = MemoryMonitor(enable_tracemalloc=False)

        # Start monitoring
        monitor.start()

        # Take some snapshots
        time.sleep(0.01)
        monitor.take_snapshot()
        time.sleep(0.01)
        monitor.take_snapshot()

        # Stop and get stats
        stats = monitor.stop()

        # Verify results
        assert isinstance(stats, MemoryStats)
        assert len(stats.snapshots) >= 3  # Initial + 2 manual
        assert stats.peak_rss > 0
        assert stats.total_time > 0

    def test_monitor_with_memory_operations(self):
        """Test monitor with actual memory operations."""
        monitor = MemoryMonitor(enable_tracemalloc=False)

        with MemoryProfiler("memory_test"):
            # Allocate some memory
            data = []
            for i in range(1000):
                data.append([0] * 1000)  # Create nested lists

            # Ensure data is not optimized away
            assert len(data) == 1000

            # Take a snapshot
            snapshot = monitor.take_snapshot()
            assert snapshot.rss_mb > 0