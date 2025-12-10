"""Progress renderer for async CLI using only standard library."""

import sys
import os
from typing import Dict, Any
from .progress_tracker import ProgressMetrics, ProgressStage


class AsyncProgressRenderer:
    """Renders progress updates for the async CLI using terminal control codes."""

    def __init__(self):
        self.last_lines = 0
        self.use_colors = self._supports_color()

    def _supports_color(self) -> bool:
        """Check if terminal supports colors."""
        return (
            hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and
            os.getenv("TERM") != "dumb" and
            os.getenv("NO_COLOR") is None
        )

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if supported."""
        if not self.use_colors:
            return text

        colors = {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "green": "\033[92m",
            "red": "\033[91m",
            "yellow": "\033[93m",
            "cyan": "\033[96m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def render(self, metrics: ProgressMetrics):
        """Render progress metrics to terminal."""
        lines = []

        # Clear previous output
        if self.last_lines > 0:
            sys.stdout.write(f"\033[{self.last_lines}F")  # Move up
            sys.stdout.write("\033[J")  # Clear from cursor down

        # Title
        title = f"🎵 Music Organizer - {self._color('Real-time Progress', 'cyan')}"
        lines.append(title)
        lines.append("─" * 80)

        # Main progress bar
        percentage = metrics.percentage
        bar_length = 60
        filled = int((percentage / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        # Main progress line
        progress_line = (
            f"Progress: [{self._color(bar, 'blue')}] "
            f"{percentage:5.1f}% "
            f"({metrics.files_processed:,}/{metrics.files_total:,} files)"
        )
        lines.append(progress_line)

        # Rate and ETA
        rate = metrics.instantaneous_rate or metrics.overall_rate
        if rate > 0:
            if rate < 1:
                rate_str = f"{1/rate:.1f} sec/file"
            else:
                rate_str = f"{rate:.1f} files/sec"
        else:
            rate_str = "calculating..."

        eta = metrics.eta_seconds
        if eta and eta > 0:
            eta_str = self._format_eta(eta)
        else:
            eta_str = "calculating..."

        metrics_line = f"Rate: {self._color(rate_str, 'green')} | ETA: {self._color(eta_str, 'yellow')}"
        lines.append(metrics_line)

        # Current stage
        if metrics.current_stage:
            stage = metrics.stages.get(metrics.current_stage)
            if stage:
                stage_percent = stage.percentage
                stage_bar_length = 30
                stage_filled = int((stage_percent / 100) * stage_bar_length)
                stage_bar = "█" * stage_filled + "░" * (stage_bar_length - stage_filled)

                stage_name = stage.stage.value.replace("_", " ").title()
                stage_line = (
                    f"Stage: {self._color(stage_name, 'magenta')} "
                    f"[{stage_bar}] {stage_percent:5.1f}%"
                )
                lines.append(stage_line)

        # Detailed metrics (if we have stages)
        if metrics.stages and len(metrics.stages) > 1:
            lines.append("")
            lines.append(self._color("Stage Details:", "bold"))
            for stage_name, stage in sorted(metrics.stages.items()):
                if stage.stage != metrics.current_stage:
                    status = "✓" if stage.completed >= stage.total > 0 else "○"
                    line = f"  {status} {stage.stage.value.replace('_', ' ').title()}: "
                    if stage.total > 0:
                        line += f"{stage.completed}/{stage.total}"
                    else:
                        line += f"{stage.completed} files"
                    lines.append(line)

        # Error tracking
        if metrics.errors > 0:
            error_line = f"Errors: {self._color(str(metrics.errors), 'red')}"
            lines.append("")
            lines.append(error_line)

        # Bytes processed (if significant)
        if metrics.bytes_processed > 0:
            bytes_str = self._format_bytes(metrics.bytes_processed)
            bytes_line = f"Processed: {self._color(bytes_str, 'cyan')}"
            lines.append(bytes_line)

        # Time elapsed
        elapsed_line = f"Elapsed: {self._format_duration(metrics.elapsed)}"
        lines.append("")
        lines.append(elapsed_line)

        # Render all lines
        output = "\n".join(lines)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()

        self.last_lines = len(lines)

    def _format_eta(self, seconds: float) -> str:
        """Format ETA as human-readable string."""
        if seconds == 0:
            return "0s"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        elif minutes > 0:
            return f"{minutes:02d}m {secs:02d}s"
        else:
            return f"{secs:02d}s"

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

    def clear(self):
        """Clear the progress display."""
        if self.last_lines > 0:
            sys.stdout.write(f"\033[{self.last_lines}F")  # Move up
            sys.stdout.write("\033[J")  # Clear from cursor down
            sys.stdout.flush()
            self.last_lines = 0

    def finish(self, metrics: ProgressMetrics):
        """Display final completion message."""
        self.clear()

        duration = self._format_duration(metrics.elapsed)
        avg_rate = metrics.overall_rate

        print(f"\n{self._color('✨ Organization Complete!', 'green')}")
        print(f"   Processed: {self._color(f'{metrics.files_processed:,} files', 'cyan')}")
        print(f"   Duration: {self._color(duration, 'yellow')}")

        if avg_rate > 0:
            if avg_rate < 1:
                rate_str = f"{1/avg_rate:.1f} sec/file"
            else:
                rate_str = f"{avg_rate:.1f} files/sec"
            print(f"   Average Rate: {self._color(rate_str, 'green')}")

        if metrics.bytes_processed > 0:
            bytes_str = self._format_bytes(metrics.bytes_processed)
            print(f"   Total Size: {self._color(bytes_str, 'cyan')}")

        if metrics.errors > 0:
            print(f"   Errors: {self._color(f'{metrics.errors}', 'red')}")

        print()