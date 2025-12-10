"""Async command line interface for music organizer."""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

from .core.async_organizer import AsyncMusicOrganizer, organize_files_async
from .core.metadata import MetadataHandler
from .core.classifier import ContentClassifier
from .models.config import Config, load_config
from .exceptions import MusicOrganizerError


class SimpleProgress:
    """Simple progress bar implementation without external dependencies."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()

    def update(self, advance: int = 1):
        """Update progress bar."""
        self.current += advance
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

        print(f"\r{self.description}: [{bar}] {percent}% ({self.current}/{self.total}) ETA: {eta_str}", end='')

        if self.current >= self.total:
            print()  # New line when complete


class SimpleConsole:
    """Simple console output without rich dependency."""

    @staticmethod
    def print(text: str, style: Optional[str] = None):
        """Print text with optional style."""
        if style == 'bold':
            print(f"\033[1m{text}\033[0m")
        elif style == 'green':
            print(f"\033[92m{text}\033[0m")
        elif style == 'red':
            print(f"\033[91m{text}\033[0m")
        elif style == 'yellow':
            print(f"\033[93m{text}\033[0m")
        elif style == 'cyan':
            print(f"\033[96m{text}\033[0m")
        else:
            print(text)

    @staticmethod
    def rule(title: str):
        """Print a horizontal rule with title."""
        width = 80
        padding = (width - len(title) - 2) // 2
        print(f"{'-' * padding} {title} {'-' * padding}")

    @staticmethod
    def table(rows: List[List[str]], headers: List[str]):
        """Print a simple table."""
        if not rows:
            return

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))

        # Print rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            print(row_line)


class AsyncMusicCLI:
    """Async CLI implementation for music organizer."""

    def __init__(self):
        self.console = SimpleConsole()

    async def organize(self,
                      source: Path,
                      target: Path,
                      config_path: Optional[Path] = None,
                      dry_run: bool = False,
                      interactive: bool = False,
                      backup: bool = True,
                      max_workers: int = 4) -> int:
        """Organize music files asynchronously."""
        try:
            # Load configuration
            if config_path and config_path.exists():
                config = load_config(config_path)
            else:
                config = Config(
                    source_directory=source,
                    target_directory=target,
                    file_operations=type('FileOps', (), {'backup': backup})()
                )

            # Validate directories
            if not source.exists():
                self.console.print(f"Error: Source directory does not exist: {source}", 'red')
                return 1

            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)

            # Initialize organizer
            async with AsyncMusicOrganizer(
                config,
                dry_run=dry_run,
                interactive=interactive,
                max_workers=max_workers
            ) as organizer:

                self.console.print(f"\n🎵 Music Organizer (Async Mode)")
                self.console.print(f"Source: {source}")
                self.console.print(f"Target: {target}")
                self.console.print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
                self.console.print(f"Workers: {max_workers}")

                # Count total files first
                self.console.print("\n🔍 Scanning files...")
                file_count = 0
                async for _ in organizer.scan_directory(source):
                    file_count += 1

                if file_count == 0:
                    self.console.print("No audio files found!", 'yellow')
                    return 0

                self.console.print(f"Found {file_count} audio files")

                # Show progress
                progress = SimpleProgress(file_count, "Organizing files")

                # Process files in streaming mode for memory efficiency
                results = {
                    'processed': 0,
                    'moved': 0,
                    'skipped': 0,
                    'by_category': {
                        'Albums': 0,
                        'Live': 0,
                        'Collaborations': 0,
                        'Compilations': 0,
                        'Rarities': 0,
                        'Unknown': 0
                    },
                    'errors': []
                }

                async for file_path, success, error in organizer.organize_files_streaming(
                    organizer.scan_directory(source),
                    batch_size=50
                ):
                    results['processed'] += 1
                    if success:
                        results['moved'] += 1
                    else:
                        results['skipped'] += 1
                        if error:
                            results['errors'].append(f"{file_path.name}: {error}")

                    progress.update()

                # Show results
                self.console.rule("\n📊 Results")

                # Summary table
                summary_data = [
                    ["Processed", str(results['processed'])],
                    ["Moved", str(results['moved'])],
                    ["Skipped", str(results['skipped'])],
                    ["Errors", str(len(results['errors']))]
                ]
                self.console.table(summary_data, ["Metric", "Count"])

                # Category breakdown
                if results['moved'] > 0:
                    self.console.print("\n📂 Files by category:")
                    for category, count in results['by_category'].items():
                        if count > 0:
                            self.console.print(f"  {category}: {count}", 'green')

                # Show errors if any
                if results['errors']:
                    self.console.print("\n❌ Errors encountered:", 'red')
                    for error in results['errors'][:10]:  # Show first 10 errors
                        self.console.print(f"  • {error}", 'red')
                    if len(results['errors']) > 10:
                        self.console.print(f"  ... and {len(results['errors']) - 10} more errors", 'red')

                # Get operation summary
                if not dry_run:
                    summary = await organizer.get_operation_summary()
                    self.console.print(f"\n💾 Directories created: {summary.get('directories_created', 0)}")

                return 0

        except MusicOrganizerError as e:
            self.console.print(f"Error: {e}", 'red')
            return 1
        except KeyboardInterrupt:
            self.console.print("\nOperation cancelled by user", 'yellow')
            return 130
        except Exception as e:
            self.console.print(f"Unexpected error: {e}", 'red')
            if '--debug' in sys.argv:
                import traceback
                traceback.print_exc()
            return 1

    async def scan(self,
                  directory: Path,
                  config_path: Optional[Path] = None,
                  detailed: bool = False) -> int:
        """Scan and analyze music library asynchronously."""
        try:
            if not directory.exists():
                self.console.print(f"Error: Directory does not exist: {directory}", 'red')
                return 1

            self.console.print(f"🔍 Scanning: {directory}")

            # Initialize components
            metadata_handler = MetadataHandler()
            classifier = ContentClassifier()

            # Scan files
            file_count = 0
            total_size = 0
            formats = {}
            categories = {
                'Albums': 0,
                'Live': 0,
                'Collaborations': 0,
                'Compilations': 0,
                'Rarities': 0,
                'Unknown': 0
            }
            errors = []

            progress = SimpleProgress(0, "Scanning files")

            async with AsyncMusicOrganizer(Config(
                source_directory=directory,
                target_directory=directory.parent / "organized"
            )) as organizer:
                async for file_path in organizer.scan_directory(directory):
                    try:
                        # Get file size
                        stat = file_path.stat()
                        file_size = stat.st_size
                        total_size += file_size
                        file_count += 1

                        # Track format
                        ext = file_path.suffix.lower()
                        formats[ext] = formats.get(ext, 0) + 1

                        if detailed:
                            # Extract metadata and classify
                            audio_file = await asyncio.get_event_loop().run_in_executor(
                                None, metadata_handler.extract_metadata, file_path
                            )

                            content_type, _ = await asyncio.get_event_loop().run_in_executor(
                                None, classifier.classify, audio_file
                            )

                            # Update category count
                            category_name = organizer._get_category_name(content_type)
                            categories[category_name] += 1

                        progress.total = file_count + 1
                        progress._draw()

                    except Exception as e:
                        errors.append(f"{file_path.name}: {e}")

                progress.current = progress.total
                progress._draw()
                print()

            # Display results
            self.console.rule("📊 Scan Results")

            # Basic stats
            stats_data = [
                ["Total Files", f"{file_count:,}"],
                ["Total Size", self._format_size(total_size)],
                ["Formats", str(len(formats))],
                ["Errors", str(len(errors))]
            ]
            self.console.table(stats_data, ["Metric", "Value"])

            # File formats
            if formats:
                self.console.print("\n📄 File Formats:")
                for ext, count in sorted(formats.items(), key=lambda x: x[1], reverse=True):
                    percent = (count * 100) // file_count
                    self.console.print(f"  {ext}: {count} files ({percent}%)")

            # Categories (if detailed)
            if detailed and categories:
                self.console.print("\n📂 Content Categories:")
                for category, count in categories.items():
                    if count > 0:
                        percent = (count * 100) // sum(categories.values())
                        self.console.print(f"  {category}: {count} files ({percent}%)")

            # Errors
            if errors:
                self.console.print(f"\n❌ {len(errors)} errors encountered:", 'red')
                for error in errors[:5]:
                    self.console.print(f"  • {error}", 'red')
                if len(errors) > 5:
                    self.console.print(f"  ... and {len(errors) - 5} more", 'red')

            return 0

        except Exception as e:
            self.console.print(f"Error: {e}", 'red')
            return 1

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"


def create_async_cli():
    """Create the async CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize your music library with smart metadata-based categorization (Async Mode)",
        epilog="Example: music-organize-async /path/to/music /path/to/organized --dry-run"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Organize command
    org_parser = subparsers.add_parser('organize', help='Organize music files')
    org_parser.add_argument('source', type=Path, help='Source directory containing music files')
    org_parser.add_argument('target', type=Path, help='Target directory for organized files')
    org_parser.add_argument('--config', type=Path, help='Configuration file path')
    org_parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    org_parser.add_argument('--interactive', action='store_true', help='Prompt for ambiguous categorizations')
    org_parser.add_argument('--no-backup', action='store_true', help='Disable backup creation')
    org_parser.add_argument('--workers', type=int, default=4, help='Number of worker threads (default: 4)')
    org_parser.add_argument('--debug', action='store_true', help='Enable debug output')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Analyze music library')
    scan_parser.add_argument('directory', type=Path, help='Directory to scan')
    scan_parser.add_argument('--config', type=Path, help='Configuration file path')
    scan_parser.add_argument('--detailed', action='store_true', help='Include detailed analysis')
    scan_parser.add_argument('--debug', action='store_true', help='Enable debug output')

    # Add version
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Create CLI instance
    cli = AsyncMusicCLI()

    # Run appropriate command
    if args.command == 'organize':
        return asyncio.run(cli.organize(
            source=args.source,
            target=args.target,
            config_path=args.config,
            dry_run=args.dry_run,
            interactive=args.interactive,
            backup=not args.no_backup,
            max_workers=args.workers
        ))
    elif args.command == 'scan':
        return asyncio.run(cli.scan(
            directory=args.directory,
            config_path=args.config,
            detailed=args.detailed
        ))

    return 0


def main():
    """Entry point for async CLI."""
    import sys
    sys.exit(create_async_cli())