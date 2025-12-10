"""Rich progress renderer for the sync CLI."""

from typing import Optional
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn, \
    TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .progress_tracker import ProgressMetrics, ProgressStage


class RichProgressRenderer:
    """Renders progress updates using Rich library."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the Rich progress renderer."""
        self.console = console or Console()
        self.progress = None
        self.main_task = None
        self.stage_tasks = {}
        self.live = None
        self.metrics = None

    def render(self, metrics: ProgressMetrics):
        """Render progress metrics using Rich."""
        self.metrics = metrics

        if not self.progress:
            # Initialize progress display
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self.console,
                transient=True
            )

            # Create main task
            self.main_task = self.progress.add_task(
                "Organizing music files...",
                total=metrics.files_total or None
            )

            # Start live display
            self.live = Live(self.progress, console=self.console, refresh_per_second=10)
            self.live.start()

        # Update main progress
        self.progress.update(
            self.main_task,
            completed=metrics.files_processed,
            total=metrics.files_total
        )

        # Update task description with rate info
        rate = metrics.instantaneous_rate or metrics.overall_rate
        if rate > 0:
            if rate < 1:
                rate_str = f"{1/rate:.1f}s/file"
            else:
                rate_str = f"{rate:.1f} files/s"

            description = f"Organizing music files... ({rate_str})"
            if metrics.errors > 0:
                description += f" [red]{metrics.errors} errors[/red]"
            self.progress.update(self.main_task, description=description)

        # Manage stage-specific tasks
        if metrics.current_stage:
            stage = metrics.stages.get(metrics.current_stage)
            if stage:
                stage_name = stage.stage.value.replace("_", " ").title()
                task_id = f"stage_{stage.stage.value}"

                if task_id not in self.stage_tasks:
                    # Create new stage task
                    task = self.progress.add_task(
                        f"  {stage_name}",
                        total=stage.total or None
                    )
                    self.stage_tasks[task_id] = task
                else:
                    # Update existing stage task
                    self.progress.update(
                        self.stage_tasks[task_id],
                        completed=stage.completed,
                        total=stage.total
                    )

        # Complete finished stages
        for stage_name, stage in metrics.stages.items():
            if stage.completed >= stage.total > 0:
                task_id = f"stage_{stage_name.value}"
                if task_id in self.stage_tasks:
                    task = self.stage_tasks[task_id]
                    # Mark as complete
                    self.progress.update(task, completed=stage.total)

    def clear(self):
        """Clear the progress display."""
        if self.live:
            self.live.stop()
            self.live = None
            self.progress = None
            self.main_task = None
            self.stage_tasks = {}

    def finish(self, metrics: ProgressMetrics):
        """Display final completion message."""
        self.clear()

        # Create summary panel
        duration_text = f"[bold]Duration:[/bold] {self._format_duration(metrics.elapsed)}"

        rate = metrics.overall_rate
        if rate > 0:
            if rate < 1:
                rate_str = f"{1/rate:.1f} seconds/file"
            else:
                rate_str = f"{rate:.1f} files/second"
            rate_text = f"[bold]Average Rate:[/bold] {rate_str}"
        else:
            rate_text = ""

        files_text = f"[bold]Files Processed:[/bold] {metrics.files_processed:,}"

        if metrics.bytes_processed > 0:
            size_text = f"[bold]Total Size:[/bold] {self._format_bytes(metrics.bytes_processed)}"
        else:
            size_text = ""

        # Build summary
        summary_lines = [files_text, duration_text]
        if rate_text:
            summary_lines.append(rate_text)
        if size_text:
            summary_lines.append(size_text)

        if metrics.errors > 0:
            summary_lines.append(f"[bold red]Errors:[/bold red] {metrics.errors}")

        summary = "\n".join(summary_lines)

        # Display completion panel
        panel = Panel(
            summary,
            title="[bold green]✨ Organization Complete![/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)

    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        elif minutes > 0:
            return f"{minutes:02d}m {secs:02d}s"
        else:
            return f"{secs:02d}s"

    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.1f} PB"