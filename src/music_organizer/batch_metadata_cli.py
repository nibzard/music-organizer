"""CLI for batch metadata operations."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from music_organizer.core.batch_metadata import (
        BatchMetadataProcessor,
        BatchMetadataConfig,
        MetadataOperation,
        OperationType,
        ConflictStrategy,
        MetadataOperationBuilder
    )
    from music_organizer.console_utils import SimpleConsole
    from music_organizer.progress_tracker import IntelligentProgressTracker
except ImportError:
    # For direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from music_organizer.core.batch_metadata import (
        BatchMetadataProcessor,
        BatchMetadataConfig,
        MetadataOperation,
        OperationType,
        ConflictStrategy,
        MetadataOperationBuilder
    )
    from music_organizer.console_utils import SimpleConsole
    from music_organizer.progress_tracker import IntelligentProgressTracker


class ProgressTracker:
    """Simple progress tracker for batch operations."""

    def __init__(self, total_files: int, quiet: bool = False):
        self.total = total_files
        self.current = 0
        self.quiet = quiet

    def update(self, completed: int, operations_count: int = 1):
        """Update progress."""
        self.current = completed
        if not self.quiet:
            percent = (self.current * 100) // self.total
            bar_length = 40
            filled = (percent * bar_length) // 100
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f'\rProgress: [{bar}] {percent}% ({self.current}/{self.total})', end='', flush=True)

    def complete(self):
        """Mark progress as complete."""
        if not self.quiet:
            print()  # New line


class BatchMetadataCLI:
    """Command-line interface for batch metadata operations."""

    def __init__(self):
        self.console = SimpleConsole()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            prog='music-batch-metadata',
            description='Batch metadata operations for music files'
        )

        parser.add_argument(
            'directory',
            type=Path,
            help='Directory containing music files to process'
        )

        parser.add_argument(
            '--operations',
            type=Path,
            help='JSON file containing metadata operations'
        )

        parser.add_argument(
            '--filter',
            type=str,
            help='Filter files by pattern (e.g., "*.flac")'
        )

        parser.add_argument(
            '--workers',
            type=int,
            default=4,
            help='Number of parallel workers (default: 4)'
        )

        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Files per batch (default: 100)'
        )

        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Preview changes without applying them'
        )

        parser.add_argument(
            '--no-backup',
            action='store_true',
            help='Skip creating backup before updates'
        )

        parser.add_argument(
            '--continue-on-error',
            action='store_true',
            help='Continue processing even if some files fail'
        )

        parser.add_argument(
            '--preserve-time',
            action='store_true',
            default=True,
            help='Preserve file modification timestamps'
        )

        parser.add_argument(
            '--quiet',
            action='store_true',
            help='Suppress progress output'
        )

        # Quick operation options
        parser.add_argument(
            '--set-genre',
            type=str,
            help='Set genre for all files'
        )

        parser.add_argument(
            '--set-year',
            type=int,
            help='Set year for all files'
        )

        parser.add_argument(
            '--add-artist',
            type=str,
            help='Add artist to all files'
        )

        parser.add_argument(
            '--standardize-genres',
            action='store_true',
            help='Standardize genre names (built-in mapping)'
        )

        parser.add_argument(
            '--capitalize-titles',
            action='store_true',
            help='Capitalize title and album names'
        )

        parser.add_argument(
            '--fix-track-numbers',
            action='store_true',
            help='Fix track number formatting'
        )

        parser.add_argument(
            '--remove-feat-artists',
            action='store_true',
            help='Remove featuring artists from title field'
        )

        return parser

    async def run(self, args: List[str] = None) -> int:
        """Run the CLI with given arguments."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        try:
            # Validate directory
            if not parsed_args.directory.exists():
                self.console.print(f"Directory does not exist: {parsed_args.directory}", 'red')
                return 1

            # Create operations
            operations = self._create_operations(parsed_args)
            if not operations:
                self.console.print("No operations specified", 'red')
                return 1

            # Find files
            files = self._find_files(parsed_args.directory, parsed_args.filter)
            if not files:
                self.console.print("No audio files found", 'red')
                return 1

            # Create config
            config = BatchMetadataConfig(
                max_workers=parsed_args.workers,
                batch_size=parsed_args.batch_size,
                dry_run=parsed_args.dry_run,
                backup_before_update=not parsed_args.no_backup,
                continue_on_error=parsed_args.continue_on_error,
                preserve_modified_time=parsed_args.preserve_time
            )

            # Process files
            result = await self._process_files(files, operations, config, parsed_args.quiet)

            # Display results
            self._display_results(result, parsed_args.dry_run)

            return 0 if result.failed == 0 else 1

        except KeyboardInterrupt:
            self.console.print("\nOperation cancelled by user", 'yellow')
            return 130
        except Exception as e:
            self.console.print(f"Error: {e}", 'red')
            return 1

    def _create_operations(self, args: argparse.Namespace) -> List[MetadataOperation]:
        """Create metadata operations from arguments."""
        operations = []

        # Load operations from file if specified
        if args.operations:
            operations.extend(self._load_operations_file(args.operations))

        # Quick operations from command line
        if args.set_genre:
            operations.append(MetadataOperationBuilder.set_genre(args.set_genre))

        if args.set_year:
            operations.append(MetadataOperationBuilder.set_year(args.set_year))

        if args.add_artist:
            operations.append(MetadataOperationBuilder.add_artist(args.add_artist))

        if args.standardize_genres:
            operations.append(MetadataOperationBuilder.standardize_genre({
                'rock': 'Rock',
                'pop': 'Pop',
                'jazz': 'Jazz',
                'classical': 'Classical',
                'electronic': 'Electronic',
                'hip-hop': 'Hip-Hop',
                'hip hop': 'Hip-Hop',
                'r&b': 'R&B',
                'rnb': 'R&B',
                'country': 'Country',
                'folk': 'Folk',
                'blues': 'Blues',
                'metal': 'Metal',
                'punk': 'Punk',
                'indie': 'Indie',
                'alternative': 'Alternative',
                'dance': 'Dance',
                'techno': 'Techno',
                'house': 'House',
                'ambient': 'Ambient',
                'soundtrack': 'Soundtrack',
                'ost': 'Soundtrack'
            }))

        if args.capitalize_titles:
            operations.extend(MetadataOperationBuilder.capitalize_fields(['title', 'album']))

        if args.fix_track_numbers:
            operations.append(MetadataOperationBuilder.fix_track_numbers())

        if args.remove_feat_artists:
            # Remove featuring artists from title
            operations.append(MetadataOperation(
                field='title',
                operation=OperationType.TRANSFORM,
                pattern='s/\\s*\\(feat\\..*?\\)//g'
            ))
            operations.append(MetadataOperation(
                field='title',
                operation=OperationType.TRANSFORM,
                pattern='s/\\s*\\[feat\\..*?\\]//g'
            ))

        return operations

    def _load_operations_file(self, file_path: Path) -> List[MetadataOperation]:
        """Load operations from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            operations = []
            for op_data in data.get('operations', []):
                operation = MetadataOperation(
                    field=op_data['field'],
                    operation=OperationType(op_data['operation']),
                    value=op_data.get('value'),
                    condition=op_data.get('condition', {}),
                    pattern=op_data.get('pattern'),
                    conflict_strategy=ConflictStrategy(
                        op_data.get('conflict_strategy', 'replace')
                    )
                )
                operations.append(operation)

            return operations

        except Exception as e:
            self.console.print(f"Failed to load operations file: {e}", 'red')
            return []

    def _find_files(self, directory: Path, pattern: Optional[str]) -> List[Path]:
        """Find audio files in directory."""
        audio_extensions = {'.mp3', '.flac', '.wav', '.m4a', '.mp4', '.aac', '.ogg', '.opus', '.wma', '.aiff', '.aif'}
        files = []

        for path in directory.rglob('*'):
            if path.is_file() and path.suffix.lower() in audio_extensions:
                if pattern is None or path.match(pattern):
                    files.append(path)

        return sorted(files)

    async def _process_files(self,
                           files: List[Path],
                           operations: List[MetadataOperation],
                           config: BatchMetadataConfig,
                           quiet: bool) -> 'BatchResult':
        """Process files with metadata operations."""
        processor = BatchMetadataProcessor(config)

        try:
            # Create progress tracker
            progress = ProgressTracker(
                total_files=len(files),
                quiet=quiet
            )

            # Progress callback
            async def progress_callback(completed: int, total: int):
                progress.update(completed, operations_count=len(operations))

            # Process files
            result = await processor.apply_operations(
                files,
                operations,
                progress_callback
            )

            # Complete progress
            progress.complete()

            return result

        finally:
            await processor.cleanup()

    def _display_results(self, result: 'BatchResult', dry_run: bool) -> None:
        """Display operation results."""
        self.console.print(f"\n{'='*60}")
        self.console.print(f"Batch Metadata {'Preview' if dry_run else 'Update'} Results")
        self.console.print(f"{'='*60}")

        # Summary
        self.console.print(f"\n{'Summary:':<20} {result.total_files} files")
        self.console.print(f"{'Successful:':<20} {result.successful}")
        self.console.print(f"{'Failed:':<20} {result.failed}")
        self.console.print(f"{'Skipped:':<20} {result.skipped}")
        self.console.print(f"{'Conflicts:':<20} {result.conflicts}")

        if result.duration_seconds > 0:
            self.console.print(f"{'Duration:':<20} {result.duration_seconds:.2f} seconds")
            self.console.print(f"{'Throughput:':<20} {result.throughput_files_per_sec:.2f} files/sec")

        self.console.print(f"{'Success Rate:':<20} {result.success_rate:.1f}%")

        # Errors
        if result.errors:
            self.console.print(f"\n{'Errors:':<20} {len(result.errors)}")
            for i, error in enumerate(result.errors[:10], 1):
                self.console.print(f"  {i}. {error.get('file', 'Unknown')}: {error.get('error', 'Unknown error')}", 'red')
            if len(result.errors) > 10:
                self.console.print(f"  ... and {len(result.errors) - 10} more errors", 'yellow')

        # Warnings
        if result.warnings:
            self.console.print(f"\n{'Warnings:':<20} {len(result.warnings)}")
            for warning in result.warnings:
                self.console.print(f"  - {warning}", 'yellow')

        # Operations performed
        if result.operations_performed and not dry_run:
            self.console.print(f"\n{'Operations Applied:':<20}")
            field_counts = {}
            for op in result.operations_performed:
                for field in op['operations']:
                    field_counts[field] = field_counts.get(field, 0) + 1

            for field, count in sorted(field_counts.items()):
                self.console.print(f"  - {field}: {count} files")

        self.console.print(f"\n{'='*60}")


def main():
    """Entry point for CLI."""
    cli = BatchMetadataCLI()
    return asyncio.run(cli.run())


if __name__ == '__main__':
    sys.exit(main())