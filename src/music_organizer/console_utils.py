"""Simple console utilities without external dependencies."""

import sys
from typing import Optional, List, Any
from datetime import datetime


class SimpleProgress:
    """Simple progress bar implementation without external dependencies."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        self.last_update = 0

    def update(self, advance: int = 1, force: bool = False):
        """Update progress bar."""
        self.current += advance
        # Only redraw every 1% to reduce flicker
        if force or (self.current - self.last_update) >= max(1, self.total // 100):
            self._draw()
            self.last_update = self.current

    def set_description(self, description: str):
        """Update progress description."""
        self.description = description
        self._draw()

    def _draw(self):
        """Draw the progress bar."""
        if self.total == 0:
            return

        percent = min(100, (self.current * 100) // self.total)
        bar_length = 50
        filled = (percent * bar_length) // 100
        bar = '█' * filled + '░' * (bar_length - filled)

        elapsed = datetime.now() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f"{int(eta.total_seconds())}s"
        else:
            eta_str = "?:??"

        print(f"\r{self.description}: [{bar}] {percent}% ({self.current}/{self.total}) ETA: {eta_str}", end='', flush=True)

        if self.current >= self.total:
            print()  # New line when complete

    def finish(self):
        """Mark progress as finished."""
        self.current = self.total
        self._draw()


class SimpleConsole:
    """Simple console output without rich dependency."""

    COLORS = {
        'bold': '\033[1m',
        'dim': '\033[2m',
        'italic': '\033[3m',
        'underline': '\033[4m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m',
    }

    @staticmethod
    def print(text: str, style: Optional[str] = None):
        """Print text with optional style."""
        if style and style in SimpleConsole.COLORS:
            print(f"{SimpleConsole.COLORS[style]}{text}{SimpleConsole.COLORS['reset']}")
        else:
            print(text)

    @staticmethod
    def rule(title: str = "", character: str = "-"):
        """Print a horizontal rule with optional title."""
        width = 80
        if title:
            title = f" {title} "
            padding = (width - len(title)) // 2
            line = character * padding + title + character * padding
            if len(line) < width:
                line += character * (width - len(line))
        else:
            line = character * width
        print(line)

    @staticmethod
    def table(rows: List[List[str]], headers: List[str], title: Optional[str] = None):
        """Print a simple table."""
        if not rows:
            return

        if title:
            SimpleConsole.rule(title)

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))

        # Print rows
        for row in rows:
            # Pad row if shorter than headers
            padded_row = row + [""] * (len(headers) - len(row))
            row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(padded_row))
            print(row_line)

        print()  # Add spacing after table

    @staticmethod
    def panel(content: str, title: Optional[str] = None, border: str = "═"):
        """Print content in a panel."""
        lines = content.split('\n')
        max_width = max(len(line) for line in lines)

        if title:
            title_section = f" {title} "
            if len(title_section) > max_width:
                max_width = len(title_section)

        border_line = border * (max_width + 4)

        print(border_line)

        if title:
            padding = (max_width - len(title)) // 2
            print(f"║ {title.center(max_width)} ║")
            print(f"║{'╡' + '═' * (max_width) + '╞'}║")

        for line in lines:
            print(f"║ {line.ljust(max_width)} ║")

        print(border_line)

    @staticmethod
    def prompt(message: str, default: Optional[Any] = None) -> str:
        """Prompt for user input."""
        if default:
            prompt_msg = f"{message} [{default}]: "
        else:
            prompt_msg = f"{message}: "

        result = input(prompt_msg)
        return result if result else str(default) if default else result

    @staticmethod
    def confirm(message: str, default: bool = False) -> bool:
        """Prompt for yes/no confirmation."""
        suffix = " [Y/n]" if default else " [y/N]"
        response = input(f"{message}{suffix}: ").strip().lower()

        if not response:
            return default

        return response.startswith('y')

    @staticmethod
    def clear_line():
        """Clear the current line."""
        print('\r' + ' ' * 80 + '\r', end='')

    @staticmethod
    def newline():
        """Print a blank line."""
        print()


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"