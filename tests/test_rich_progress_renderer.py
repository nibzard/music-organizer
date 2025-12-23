"""Tests for Rich progress renderer."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rich.console import Console
from rich.progress import Progress
from rich.live import Live

from music_organizer.progress_tracker import ProgressMetrics, ProgressStage, StageProgress
from music_organizer.rich_progress_renderer import RichProgressRenderer


class TestRichProgressRenderer:
    """Test cases for RichProgressRenderer."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock console to avoid actual console output during tests
        self.mock_console = Mock(spec=Console)
        self.renderer = RichProgressRenderer(console=self.mock_console)

    def test_initialization(self):
        """Test renderer initialization."""
        renderer = RichProgressRenderer()
        assert renderer.console is not None
        assert renderer.progress is None
        assert renderer.main_task is None
        assert renderer.stage_tasks == {}
        assert renderer.live is None
        assert renderer.metrics is None

    def test_initialization_with_custom_console(self):
        """Test renderer initialization with custom console."""
        assert self.renderer.console == self.mock_console

    @patch('music_organizer.rich_progress_renderer.Progress')
    @patch('music_organizer.rich_progress_renderer.Live')
    def test_render_first_call(self, mock_live_class, mock_progress_class):
        """Test first render call initializes progress and live display."""
        # Setup mocks
        mock_progress = Mock(spec=Progress)
        mock_progress_class.return_value = mock_progress
        mock_task = "task_id"
        mock_progress.add_task.return_value = mock_task

        mock_live = Mock(spec=Live)
        mock_live_class.return_value = mock_live

        # Create metrics
        metrics = ProgressMetrics(
            files_total=100,
            files_processed=0,
            current_stage=ProgressStage.SCANNING
        )
        metrics.stages[ProgressStage.SCANNING] = StageProgress(stage=ProgressStage.SCANNING, total=10, completed=0)

        # Render
        self.renderer.render(metrics)

        # Verify Progress was created with correct columns
        mock_progress_class.assert_called_once()
        args, kwargs = mock_progress_class.call_args
        assert 'console' in kwargs
        assert kwargs['console'] == self.mock_console
        assert kwargs['transient'] is True

        # Verify main task was created (first call)
        assert mock_progress.add_task.call_count >= 1
        first_call = mock_progress.add_task.call_args_list[0]
        assert first_call[0][0] == "Organizing music files..."
        assert first_call[1]['total'] == 100

        # Verify Live was started
        mock_live_class.assert_called_once_with(mock_progress, console=self.mock_console, refresh_per_second=10)
        mock_live.start.assert_called_once()

        # Verify internal state
        assert self.renderer.progress == mock_progress
        assert self.renderer.main_task == mock_task
        assert self.renderer.live == mock_live

    def test_render_update_progress(self):
        """Test updating progress on subsequent renders."""
        # Setup: simulate first render
        self.renderer.progress = Mock(spec=Progress)
        self.renderer.main_task = Mock()
        self.renderer.live = Mock(spec=Live)
        self.renderer.stage_tasks = {}

        # Create metrics with progress
        metrics = ProgressMetrics(
            files_total=100,
            files_processed=50,
            errors=2
        )

        # Render
        self.renderer.render(metrics)

        # Verify progress update was called
        assert self.renderer.progress.update.called
        # Check that update was called with completed and total
        call_args_list = self.renderer.progress.update.call_args_list
        # First call should update completed/total
        first_call = call_args_list[0]
        assert first_call[1]['completed'] == 50
        assert first_call[1]['total'] == 100

    def test_render_with_slow_rate(self):
        """Test rendering with slow rate (< 1 file/sec)."""
        # Setup
        self.renderer.progress = Mock(spec=Progress)
        self.renderer.main_task = Mock()

        # Create metrics with slow rate
        import time
        now = time.time()
        metrics = ProgressMetrics(
            files_total=100,
            files_processed=10
        )
        # Set start_time to create a slow processing rate (0.5 files/sec = 20s elapsed for 10 files)
        metrics.start_time = now - 20

        # Render
        self.renderer.render(metrics)

        # Verify update was called
        assert self.renderer.progress.update.called

    def test_render_with_errors(self):
        """Test rendering with errors included."""
        # Setup
        self.renderer.progress = Mock(spec=Progress)
        self.renderer.main_task = Mock()

        # Create metrics with errors
        metrics = ProgressMetrics(
            files_total=100,
            files_processed=30,
            errors=5
        )

        # Render
        self.renderer.render(metrics)

        # Verify update was called (errors shown in description)
        assert self.renderer.progress.update.called

    def test_render_stage_progress(self):
        """Test rendering stage-specific progress."""
        # Setup
        self.renderer.progress = Mock(spec=Progress)
        self.renderer.main_task = Mock()
        self.renderer.stage_tasks = {}
        mock_stage_task = "stage_task_id"
        self.renderer.progress.add_task.return_value = mock_stage_task

        # Create metrics with active stage
        metrics = ProgressMetrics(
            files_total=100,
            files_processed=10,
            current_stage=ProgressStage.METADATA_EXTRACTION
        )
        metrics.stages[ProgressStage.METADATA_EXTRACTION] = StageProgress(
            stage=ProgressStage.METADATA_EXTRACTION,
            total=20,
            completed=5
        )

        # Render
        self.renderer.render(metrics)

        # Verify stage task was created
        self.renderer.progress.add_task.assert_called()
        call_args = self.renderer.progress.add_task.call_args_list
        # Find the call for the stage task (second call after main task)
        stage_call = [c for c in call_args if "Metadata Extraction" in str(c[0][0])]
        assert len(stage_call) > 0 or any("  Metadata Extraction" in str(c) for c in call_args)
        assert "stage_metadata_extraction" in self.renderer.stage_tasks

    def test_render_update_existing_stage(self):
        """Test updating existing stage progress."""
        # Setup with existing stage task
        self.renderer.progress = Mock(spec=Progress)
        self.renderer.main_task = Mock()
        existing_task = "existing_task"
        self.renderer.stage_tasks = {"stage_classification": existing_task}

        # Create metrics with existing stage
        metrics = ProgressMetrics(
            files_total=100,
            files_processed=10,
            current_stage=ProgressStage.CLASSIFICATION
        )
        metrics.stages[ProgressStage.CLASSIFICATION] = StageProgress(
            stage=ProgressStage.CLASSIFICATION,
            total=20,
            completed=15
        )

        # Render
        self.renderer.render(metrics)

        # Verify existing stage task was updated (not created)
        # Check that update was called for the stage task
        call_args_list = self.renderer.progress.update.call_args_list
        stage_updates = [c for c in call_args_list if c[0][0] == existing_task]
        assert len(stage_updates) > 0

    def test_render_complete_finished_stages(self):
        """Test that finished stages are marked complete."""
        # Setup with finished stage
        self.renderer.progress = Mock(spec=Progress)
        self.renderer.main_task = Mock()
        finished_task = "finished_task"
        self.renderer.stage_tasks = {"stage_scanning": finished_task}

        # Create metrics with finished stage
        metrics = ProgressMetrics(
            files_total=100,
            files_processed=10,
            current_stage=ProgressStage.CLASSIFICATION
        )
        metrics.stages[ProgressStage.SCANNING] = StageProgress(
            stage=ProgressStage.SCANNING,
            total=10,
            completed=10  # Complete
        )
        metrics.stages[ProgressStage.CLASSIFICATION] = StageProgress(
            stage=ProgressStage.CLASSIFICATION,
            total=20,
            completed=5
        )

        # Render
        self.renderer.render(metrics)

        # Verify update was called for finished stage
        assert self.renderer.progress.update.called
        # Check for update with the finished task
        call_args_list = self.renderer.progress.update.call_args_list
        finished_updates = [c for c in call_args_list if len(c[0]) > 0 and c[0][0] == finished_task]
        assert len(finished_updates) > 0

    def test_clear(self):
        """Test clearing the progress display."""
        # Setup with live display
        mock_live = Mock(spec=Live)
        self.renderer.live = mock_live
        self.renderer.progress = Mock()
        self.renderer.main_task = Mock()
        self.renderer.stage_tasks = {"task1": Mock()}

        # Clear
        self.renderer.clear()

        # Verify live was stopped
        mock_live.stop.assert_called_once()

        # Verify state was reset
        assert self.renderer.live is None
        assert self.renderer.progress is None
        assert self.renderer.main_task is None
        assert self.renderer.stage_tasks == {}

    def test_clear_without_live(self):
        """Test clearing when no live display exists."""
        # Should not raise any errors
        self.renderer.clear()
        assert self.renderer.live is None

    @patch('music_organizer.rich_progress_renderer.Panel')
    def test_finish_with_stats(self, mock_panel_class):
        """Test finishing with complete statistics."""
        # Setup mock panel
        mock_panel = Mock()
        mock_panel_class.return_value = mock_panel

        # Create metrics with full stats
        import time
        now = time.time()
        metrics = ProgressMetrics(
            files_total=100,
            files_processed=95,
            bytes_processed=1024 * 1024 * 50,  # 50MB
            errors=1
        )
        # Set start_time to get elapsed time of ~123.45s and rate of ~0.77 files/sec
        metrics.start_time = now - 123.45

        # Finish
        self.renderer.finish(metrics)

        # Verify clear was called
        # Panel should be created with summary
        mock_panel_class.assert_called_once()
        args, kwargs = mock_panel_class.call_args

        # Verify panel parameters
        assert kwargs['title'] == "[bold green]✨ Organization Complete![/bold green]"
        assert kwargs['border_style'] == "green"
        assert kwargs['padding'] == (1, 2)

        # Verify summary contains key info
        summary = args[0]
        assert "95" in summary  # Files Processed
        assert "2m" in summary  # Duration
        assert "MB" in summary  # Size
        assert "1" in summary  # Errors

        # Verify panel was printed
        self.mock_console.print.assert_called_once_with(mock_panel)

    @patch('music_organizer.rich_progress_renderer.Panel')
    def test_finish_without_rate(self, mock_panel_class):
        """Test finishing without rate information."""
        mock_panel = Mock()
        mock_panel_class.return_value = mock_panel

        metrics = ProgressMetrics(
            files_total=100,
            files_processed=0,  # No rate when no files processed
            errors=0
        )

        self.renderer.finish(metrics)

        args, _ = mock_panel_class.call_args
        summary = args[0]
        # Should not contain rate info when no processing happened
        # or rate should be 0
        assert "0" in summary

    @patch('music_organizer.rich_progress_renderer.Panel')
    def test_finish_without_size(self, mock_panel_class):
        """Test finishing without size information."""
        mock_panel = Mock()
        mock_panel_class.return_value = mock_panel

        metrics = ProgressMetrics(
            files_total=100,
            files_processed=100,
            bytes_processed=0  # No size
        )

        self.renderer.finish(metrics)

        args, _ = mock_panel_class.call_args
        summary = args[0]
        # Should not contain size info when no bytes processed
        assert "0 B" in summary or "MB" not in summary

    def test_format_duration(self):
        """Test duration formatting."""
        # Test various durations
        assert self.renderer._format_duration(30) == "30s"
        assert self.renderer._format_duration(90) == "01m 30s"
        assert self.renderer._format_duration(3661) == "1h 01m 01s"
        assert self.renderer._format_duration(7200) == "2h 00m 00s"

    def test_format_bytes(self):
        """Test byte formatting."""
        # Test various byte sizes
        assert self.renderer._format_bytes(512) == "512.0 B"
        assert self.renderer._format_bytes(1536) == "1.5 KB"
        assert self.renderer._format_bytes(1048576) == "1.0 MB"
        assert self.renderer._format_bytes(1073741824) == "1.0 GB"
        assert self.renderer._format_bytes(1099511627776) == "1.0 TB"

    def test_format_bytes_petabytes(self):
        """Test formatting very large byte sizes."""
        # Test petabyte conversion
        pb_size = 1024 * 1024 * 1024 * 1024 * 1024 * 1024  # 1 PB in bytes
        result = self.renderer._format_bytes(pb_size)
        assert result == "1024.0 PB"

    @patch('music_organizer.rich_progress_renderer.Progress')
    @patch('music_organizer.rich_progress_renderer.Live')
    def test_render_with_none_total(self, mock_live_class, mock_progress_class):
        """Test rendering when files_total is 0 (unknown total)."""
        mock_progress = Mock(spec=Progress)
        mock_progress_class.return_value = mock_progress
        mock_task = Mock()
        mock_progress.add_task.return_value = mock_task
        mock_live = Mock(spec=Live)
        mock_live_class.return_value = mock_live

        # Create metrics with 0 total (unknown)
        metrics = ProgressMetrics(
            files_total=0,
            files_processed=10
        )

        # Render
        self.renderer.render(metrics)

        # Verify main task was created
        mock_progress.add_task.assert_called_once()