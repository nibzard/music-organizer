"""
Intelligent progress tracking with real-time metrics.
Provides file/sec processing rate, ETA calculations, and detailed stage progress.
"""
import time
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class ProgressStage(Enum):
    """Processing stages for detailed progress tracking."""
    SCANNING = "scanning"
    METADATA_EXTRACTION = "metadata_extraction"
    CLASSIFICATION = "classification"
    MOVING = "moving"
    CLEANUP = "cleanup"


@dataclass
class StageProgress:
    """Progress information for a specific stage."""
    stage: ProgressStage
    completed: int = 0
    total: int = 0
    start_time: float = field(default_factory=time.time)
    errors: int = 0

    @property
    def percentage(self) -> float:
        """Calculate completion percentage for this stage."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100

    @property
    def elapsed(self) -> float:
        """Calculate elapsed time for this stage."""
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        """Calculate processing rate for this stage (items/sec)."""
        elapsed = self.elapsed
        if elapsed == 0:
            return 0.0
        return self.completed / elapsed


@dataclass
class ProgressMetrics:
    """Real-time processing metrics."""
    files_processed: int = 0
    files_total: int = 0
    bytes_processed: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)

    # Rolling window for rate calculation (last 10 seconds)
    rate_window: deque = field(default_factory=lambda: deque(maxlen=20))

    # Stage-specific progress
    stages: Dict[ProgressStage, StageProgress] = field(default_factory=dict)
    current_stage: Optional[ProgressStage] = None

    @property
    def elapsed(self) -> float:
        """Total elapsed time since start."""
        return time.time() - self.start_time

    @property
    def overall_rate(self) -> float:
        """Overall processing rate (files/sec)."""
        elapsed = self.elapsed
        if elapsed == 0:
            return 0.0
        return self.files_processed / elapsed

    @property
    def instantaneous_rate(self) -> float:
        """Instantaneous processing rate from rolling window."""
        if len(self.rate_window) < 2:
            return self.overall_rate

        # Calculate rate from most recent samples
        recent_samples = list(self.rate_window)[-10:]
        if len(recent_samples) < 2:
            return self.overall_rate

        time_diff = recent_samples[-1][0] - recent_samples[0][0]
        count_diff = recent_samples[-1][1] - recent_samples[0][1]

        if time_diff == 0:
            return 0.0
        return count_diff / time_diff

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time to completion in seconds."""
        if self.files_total == 0 or self.files_processed == 0:
            return None

        rate = self.instantaneous_rate or self.overall_rate
        if rate == 0:
            return None

        remaining = self.files_total - self.files_processed
        return remaining / rate

    @property
    def percentage(self) -> float:
        """Overall completion percentage."""
        if self.files_total == 0:
            return 0.0
        return (self.files_processed / self.files_total) * 100

    def update(self, files_processed: int, bytes_processed: int = 0, error: bool = False):
        """Update progress metrics."""
        now = time.time()

        # Update counters
        self.files_processed = files_processed
        self.bytes_processed += bytes_processed
        if error:
            self.errors += 1

        # Add to rate window (timestamp, count)
        self.rate_window.append((now, files_processed))
        self.last_update_time = now

        # Update current stage if set
        if self.current_stage and self.current_stage in self.stages:
            self.stages[self.current_stage].completed = files_processed
            if error:
                self.stages[self.current_stage].errors += 1

    def start_stage(self, stage: ProgressStage, total: int = 0):
        """Start a new processing stage."""
        self.current_stage = stage
        if stage not in self.stages:
            self.stages[stage] = StageProgress(stage=stage, total=total)
        else:
            self.stages[stage].start_time = time.time()
            self.stages[stage].total = total

    def finish_stage(self, stage: ProgressStage):
        """Mark a stage as finished."""
        if stage in self.stages:
            self.stages[stage].completed = self.stages[stage].total or self.files_processed
        if self.current_stage == stage:
            self.current_stage = None

    def set_total(self, total: int):
        """Set the total number of files to process."""
        self.files_total = total


class IntelligentProgressTracker:
    """
    Intelligent progress tracker with real-time metrics and stage tracking.
    """

    def __init__(self, update_interval: float = 0.1):
        """Initialize the progress tracker.

        Args:
            update_interval: Minimum time between progress updates (seconds)
        """
        self.metrics = ProgressMetrics()
        self.update_interval = update_interval
        self.last_render_time = 0.0
        self.render_callbacks = []

    def add_render_callback(self, callback):
        """Add a callback for rendering progress updates."""
        self.render_callbacks.append(callback)

    def set_total_files(self, total: int):
        """Set the total number of files to process."""
        self.metrics.set_total(total)

    def start_stage(self, stage: ProgressStage, total: Optional[int] = None):
        """Start a new processing stage."""
        stage_total = total or self.metrics.files_total
        self.metrics.start_stage(stage, stage_total)
        self._try_render()

    def finish_stage(self, stage: ProgressStage):
        """Finish a processing stage."""
        self.metrics.finish_stage(stage)
        self._try_render()

    def update(self, increment: int = 1, bytes_processed: int = 0, error: bool = False):
        """Update progress by the specified increment."""
        new_total = self.metrics.files_processed + increment
        self.metrics.update(new_total, bytes_processed, error)
        self._try_render()

    def set_completed(self, completed: int, bytes_processed: int = 0, error: bool = False):
        """Set the exact number of completed files."""
        self.metrics.update(completed, bytes_processed, error)
        self._try_render()

    def _try_render(self):
        """Render progress if enough time has passed."""
        now = time.time()
        if now - self.last_render_time >= self.update_interval:
            self._render()
            self.last_render_time = now

    def _render(self):
        """Render progress using all registered callbacks."""
        for callback in self.render_callbacks:
            try:
                callback(self.metrics)
            except Exception:
                # Don't let rendering errors break processing
                pass

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of processing metrics."""
        return {
            "files_processed": self.metrics.files_processed,
            "files_total": self.metrics.files_total,
            "percentage": self.metrics.percentage,
            "overall_rate": self.metrics.overall_rate,
            "instantaneous_rate": self.metrics.instantaneous_rate,
            "eta_seconds": self.metrics.eta_seconds,
            "elapsed": self.metrics.elapsed,
            "errors": self.metrics.errors,
            "bytes_processed": self.metrics.bytes_processed,
            "stages": {
                stage.value: {
                    "completed": progress.completed,
                    "total": progress.total,
                    "percentage": progress.percentage,
                    "rate": progress.rate,
                    "errors": progress.errors
                }
                for stage, progress in self.metrics.stages.items()
            }
        }

    def format_eta(self, eta_seconds: Optional[float]) -> str:
        """Format ETA as human-readable string."""
        if eta_seconds is None or eta_seconds == math.inf:
            return "--:--"

        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        elif minutes > 0:
            return f"{minutes:02d}m {seconds:02d}s"
        else:
            return f"{seconds:02d}s"

    def format_rate(self, rate: float) -> str:
        """Format processing rate as human-readable string."""
        if rate == 0:
            return "0 files/sec"
        elif rate < 1:
            return f"{1/rate:.1f} sec/file"
        else:
            return f"{rate:.1f} files/sec"

    def format_bytes(self, bytes_count: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.1f} PB"