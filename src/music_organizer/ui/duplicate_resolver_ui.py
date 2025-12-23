"""Terminal UI for Interactive Duplicate Resolution."""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from ..core.interactive_duplicate_resolver import (
    DuplicateGroup, DuplicateAction, ResolutionDecision
)
from ..models.audio_file import AudioFile


class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'


class DuplicateResolverUI:
    """Terminal UI for interactive duplicate resolution."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.terminal_width = self._get_terminal_width()

    def _get_terminal_width(self) -> int:
        """Get the terminal width."""
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 80  # Default width

    def clear_screen(self):
        """Clear the terminal screen."""
        import subprocess
        try:
            if os.name == 'nt':
                subprocess.run(['cls'], shell=True, check=False)
            else:
                subprocess.run(['clear'], check=False)
        except (OSError, subprocess.SubprocessError):
            # Silently fail if clearing screen doesn't work
            pass

    def _print_header(self, title: str):
        """Print a formatted header."""
        width = min(self.terminal_width - 4, 100)
        border = f"{'='*width}"
        print(f"\n{border}")
        print(f"{title:^{width}}")
        print(f"{border}\n")

    def _print_file_info(self, audio_file: AudioFile, index: int, is_best: bool = False):
        """Print formatted information about an audio file."""
        # File number and indicator
        indicator = "★ BEST" if is_best else f"[{index + 1}]"
        color = Colors.GREEN if is_best else Colors.CYAN

        # Format file size
        try:
            size_mb = audio_file.path.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB"
        except OSError:
            size_str = "Unknown size"

        # Format bitrate and sample rate
        metadata = audio_file.metadata if isinstance(audio_file.metadata, dict) else {}
        bitrate = f"{metadata.get('bitrate')} kbps" if metadata.get('bitrate') else "Unknown"
        sample_rate = f"{metadata.get('sample_rate')} Hz" if metadata.get('sample_rate') else "Unknown"

        # Print header
        print(f"\n{color}{Colors.BOLD}{indicator}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * (len(indicator) - 4)}{'─' * (self.terminal_width - len(indicator) - 4)}{Colors.RESET}")
        print(f"{Colors.BOLD}Path:{Colors.RESET} {audio_file.path}")
        print(f"{Colors.BOLD}Size:{Colors.RESET} {size_str} | {Colors.BOLD}Format:{Colors.RESET} {audio_file.file_type.upper()}")
        print(f"{Colors.BOLD}Quality:{Colors.RESET} {bitrate} | {sample_rate}")

        # Print metadata
        print(f"\n{Colors.BOLD}Metadata:{Colors.RESET}")
        if audio_file.title:
            print(f"  Title: {audio_file.title}")
        if audio_file.artists:
            print(f"  Artist: {', '.join(str(a) for a in audio_file.artists)}")
        if audio_file.album:
            print(f"  Album: {audio_file.album}")
        if audio_file.year:
            print(f"  Year: {audio_file.year}")
        if audio_file.genre:
            print(f"  Genre: {audio_file.genre}")
        if audio_file.track_number:
            print(f"  Track: {audio_file.track_number}")

    def _print_comparison(self, files: List[AudioFile], scores: List[float]):
        """Print a side-by-side comparison of files."""
        if len(files) < 2:
            return

        print(f"\n{Colors.YELLOW}{Colors.BOLD}Side-by-Side Comparison:{Colors.RESET}")
        print(f"{'─' * self.terminal_width}")

        # Create comparison table
        file1, file2 = files[0], files[1]

        # Determine which has better quality
        better_idx = 0 if scores[0] > scores[1] else 1

        # Compare key attributes
        comparisons = [
            ("Format", file1.file_type.upper(), file2.file_type.upper(), better_idx == 0),
            ("Size", f"{file1.path.stat().st_size / (1024*1024):.1f} MB",
                   f"{file2.path.stat().st_size / (1024*1024):.1f} MB", better_idx == 0),
            ("Bitrate", f"{file1.metadata.get('bitrate')} kbps" if isinstance(file1.metadata, dict) and file1.metadata.get('bitrate') else "Unknown",
                       f"{file2.metadata.get('bitrate')} kbps" if isinstance(file2.metadata, dict) and file2.metadata.get('bitrate') else "Unknown", better_idx == 0),
            ("Sample Rate", f"{file1.metadata.get('sample_rate')} Hz" if isinstance(file1.metadata, dict) and file1.metadata.get('sample_rate') else "Unknown",
                          f"{file2.metadata.get('sample_rate')} Hz" if isinstance(file2.metadata, dict) and file2.metadata.get('sample_rate') else "Unknown", better_idx == 0),
        ]

        # Print comparison table
        for attr, val1, val2, first_is_better in comparisons:
            marker1 = Colors.GREEN + "✓" if first_is_better else Colors.RED + "✗"
            marker2 = Colors.GREEN + "✓" if not first_is_better else Colors.RED + "✗"
            print(f"{attr:12} | {marker1} {val1:<20} | {marker2} {val2:<20}")

        print(f"\n{Colors.DIM}Quality Scores: File 1: {scores[0]:.1f} | File 2: {scores[1]:.1f}{Colors.RESET}")

    def _get_user_action(self, num_files: int) -> DuplicateAction:
        """Get action choice from user."""
        actions = [
            ("1", f"Keep first file", DuplicateAction.KEEP_FIRST),
            ("2", f"Keep second file", DuplicateAction.KEEP_SECOND),
            ("3", f"Keep best quality", DuplicateAction.KEEP_BEST),
            ("4", f"Keep both files", DuplicateAction.KEEP_BOTH),
        ]

        if self.dry_run:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}DRY RUN MODE - No files will be modified{Colors.RESET}")
        else:
            actions.extend([
                ("5", f"Move second file", DuplicateAction.MOVE_DUPLICATE),
                ("6", f"Delete second file", DuplicateAction.DELETE_DUPLICATE),
            ])

        print(f"\n{Colors.BOLD}Choose action:{Colors.RESET}")
        for key, desc, _ in actions:
            print(f"  {Colors.CYAN}{key}{Colors.RESET}) {desc}")

        while True:
            try:
                choice = input(f"\n{Colors.BOLD}Enter choice [1-{len(actions)}]:{Colors.RESET} ").strip()

                for key, _, action in actions:
                    if choice == key:
                        return action

                print(f"{Colors.RED}Invalid choice. Please try again.{Colors.RESET}")
            except (KeyboardInterrupt, EOFError):
                print(f"\n\n{Colors.YELLOW}Operation cancelled by user.{Colors.RESET}")
                raise

    async def show_duplicate_group(self, group: DuplicateGroup) -> Optional[ResolutionDecision]:
        """Show a duplicate group and get user decision."""
        if len(group.files) < 2:
            return None

        # Sort files by quality score
        from ..core.interactive_duplicate_resolver import DuplicateQualityScorer
        scorer = DuplicateQualityScorer()
        scored_files = [(f, scorer.score_file(f)) for f in group.files]
        scored_files.sort(key=lambda x: x[1], reverse=True)

        files = [f for f, _ in scored_files]
        scores = [s for _, s in scored_files]

        # Show header
        confidence_color = Colors.GREEN if group.confidence > 0.9 else Colors.YELLOW if group.confidence > 0.7 else Colors.RED
        self._print_header(f"DUPLICATE GROUP: {group.duplicate_type.upper()}")
        print(f"{Colors.BOLD}Confidence:{Colors.RESET} {confidence_color}{group.confidence:.1%}{Colors.RESET}")
        print(f"{Colors.BOLD}Reason:{Colors.RESET} {group.reason}")
        print(f"{Colors.BOLD}Files in group:{Colors.RESET} {len(group.files)}")

        # Show file details
        for i, file in enumerate(files):
            self._print_file_info(file, i, is_best=(i == 0))

        # Show comparison
        if len(files) >= 2:
            self._print_comparison(files[:2], scores[:2])

        # Get user action
        try:
            action = self._get_user_action(len(files))
        except (KeyboardInterrupt, EOFError):
            return None

        # Create decision
        target_file = None
        reason = None

        if action == DuplicateAction.KEEP_FIRST:
            target_file = files[0].path
            reason = "User chose to keep first file"
        elif action == DuplicateAction.KEEP_SECOND and len(files) > 1:
            target_file = files[1].path
            reason = "User chose to keep second file"
        elif action == DuplicateAction.KEEP_BEST:
            target_file = files[0].path
            reason = f"User chose to keep best quality file (score: {scores[0]:.1f})"
        elif action == DuplicateAction.KEEP_BOTH:
            reason = "User chose to keep both files"
        elif action in [DuplicateAction.MOVE_DUPLICATE, DuplicateAction.DELETE_DUPLICATE]:
            if len(files) > 1:
                target_file = files[1].path
                reason = f"User chose to {action.value} second file"

        return ResolutionDecision(
            action=action,
            target_file=target_file,
            reason=reason
        )

    def show_summary(self, summary: Dict[str, Any]):
        """Show the resolution summary."""
        self._print_header("RESOLUTION SUMMARY")

        print(f"{Colors.BOLD}Groups Processed:{Colors.RESET}")
        print(f"  Total groups: {summary['total_groups']}")
        print(f"  Resolved groups: {summary['resolved_groups']}")
        print(f"  Skipped groups: {summary['total_groups'] - summary['resolved_groups']}")

        print(f"\n{Colors.BOLD}File Actions:{Colors.RESET}")
        print(f"  Files kept: {Colors.GREEN}{summary['kept_files']}{Colors.RESET}")
        print(f"  Files moved: {Colors.BLUE}{summary['moved_files']}{Colors.RESET}")
        print(f"  Files deleted: {Colors.RED}{summary['deleted_files']}{Colors.RESET}")
        print(f"  Files skipped: {Colors.YELLOW}{summary['skipped_files']}{Colors.RESET}")

        if summary.get('space_saved_mb', 0) > 0:
            print(f"\n{Colors.BOLD}Space Saved:{Colors.RESET} {Colors.GREEN}{summary['space_saved_mb']:.2f} MB{Colors.RESET}")

    def show_progress(self, current: int, total: int):
        """Show progress indicator."""
        progress = current / total
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"\r{Colors.CYAN}Progress: [{bar}] {progress:.1%} ({current}/{total}){Colors.RESET}", end="")

        if current == total:
            print()  # New line when complete

    def confirm_action(self, action: str, details: str = "") -> bool:
        """Ask user to confirm an action."""
        print(f"\n{Colors.YELLOW}{Colors.BOLD}Confirm Action:{Colors.RESET}")
        print(f"  Action: {action}")
        if details:
            print(f"  Details: {details}")

        while True:
            response = input(f"\n{Colors.BOLD}Proceed? [y/N]:{Colors.RESET} ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['', 'n', 'no']:
                return False
            print(f"{Colors.RED}Please enter 'y' or 'n'.{Colors.RESET}")