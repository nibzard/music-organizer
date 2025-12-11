"""Command line interface for interactive duplicate resolution."""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional
from enum import Enum

from .core.interactive_duplicate_resolver import ResolutionStrategy
from .core.duplicate_resolver_organizer import DuplicateResolverOrganizer, quick_duplicate_resolution
from .models.config import Config, load_config
from .exceptions import MusicOrganizerError
from .console_utils import SimpleConsole


class DuplicateResolverCLI:
    """CLI for duplicate resolution."""

    def __init__(self):
        self.console = SimpleConsole()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog='music-organize-duplicates',
            description='Interactive duplicate resolution for music libraries',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Interactive duplicate resolution
  music-organize-duplicates resolve /music/library

  # Preview duplicates without resolving
  music-organize-duplicates preview /music/library

  # Auto-resolve by keeping best quality
  music-organize-duplicates resolve /music/library --strategy auto_best

  # Move duplicates to separate directory
  music-organize-duplicates resolve /music/library --move-duplicates-to /duplicates

  # Dry run to see what would happen
  music-organize-duplicates resolve /music/library --dry-run
            """
        )

        parser.add_argument(
            'command',
            choices=['resolve', 'preview', 'organize'],
            help='Command to execute'
        )

        parser.add_argument(
            'source',
            type=Path,
            help='Source directory containing music files'
        )

        parser.add_argument(
            'target',
            nargs='?',
            type=Path,
            help='Target directory for organized music (only for organize command)'
        )

        # Duplicate resolution options
        parser.add_argument(
            '--strategy', '-s',
            type=str,
            default='interactive',
            choices=['interactive', 'auto_best', 'auto_first', 'auto_smart'],
            help='Duplicate resolution strategy (default: interactive)'
        )

        parser.add_argument(
            '--move-duplicates-to',
            type=Path,
            metavar='DIR',
            help='Move duplicate files to this directory instead of deleting'
        )

        parser.add_argument(
            '--dry-run', '-n',
            action='store_true',
            help='Show what would be done without making changes'
        )

        parser.add_argument(
            '--no-duplicates',
            action='store_true',
            help='Skip duplicate detection and resolution'
        )

        # Organization options (for organize command)
        parser.add_argument(
            '--resolve-first',
            action='store_true',
            default=True,
            help='Resolve duplicates before organizing (default: True)'
        )

        parser.add_argument(
            '--config', '-c',
            type=Path,
            metavar='FILE',
            help='Configuration file path'
        )

        parser.add_argument(
            '--workers', '-w',
            type=int,
            default=4,
            metavar='N',
            help='Number of worker threads (default: 4)'
        )

        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )

        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug output'
        )

        return parser

    async def resolve_duplicates(self,
                                source_dir: Path,
                                strategy: str = 'interactive',
                                duplicate_dir: Optional[Path] = None,
                                dry_run: bool = False) -> int:
        """Resolve duplicates in the source directory."""
        try:
            if not source_dir.exists():
                self.console.print(f"Error: Directory does not exist: {source_dir}", 'red')
                return 1

            if not source_dir.is_dir():
                self.console.print(f"Error: Path is not a directory: {source_dir}", 'red')
                return 1

            self.console.print(f"🔍 Scanning for duplicates in: {source_dir}")
            self.console.print(f"Strategy: {strategy}")
            if dry_run:
                self.console.print("Mode: DRY RUN - No files will be modified", 'yellow')

            # Convert strategy string to enum
            strategy_map = {
                'interactive': ResolutionStrategy.INTERACTIVE,
                'auto_best': ResolutionStrategy.AUTO_KEEP_BEST,
                'auto_first': ResolutionStrategy.AUTO_FIRST,
                'auto_smart': ResolutionStrategy.AUTO_SMART
            }
            resolution_strategy = strategy_map.get(strategy, ResolutionStrategy.INTERACTIVE)

            # Resolve duplicates
            resolution_summary = await quick_duplicate_resolution(
                source_dir=source_dir,
                strategy=resolution_strategy,
                duplicate_dir=duplicate_dir,
                dry_run=dry_run
            )

            if resolution_summary is None:
                self.console.print("No duplicates found!", 'green')
                return 0

            # Show results
            self.console.print(f"\n✅ Duplicate resolution completed!")
            self.console.print(f"Groups resolved: {resolution_summary.resolved_groups}")
            self.console.print(f"Files kept: {resolution_summary.kept_files}")
            self.console.print(f"Files moved: {resolution_summary.moved_files}")
            self.console.print(f"Files deleted: {resolution_summary.deleted_files}")

            if resolution_summary.space_saved_mb > 0:
                self.console.print(f"Space saved: {resolution_summary.space_saved_mb:.2f} MB", 'green')

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

    async def preview_duplicates(self, source_dir: Path, limit: int = 10) -> int:
        """Preview duplicates without resolving them."""
        try:
            if not source_dir.exists():
                self.console.print(f"Error: Directory does not exist: {source_dir}", 'red')
                return 1

            config = Config.default()

            async with DuplicateResolverOrganizer(
                config=config,
                dry_run=True,
                enable_duplicate_resolution=True
            ) as resolver:
                preview = await resolver.get_duplicate_preview(source_dir, limit)

            self.console.print(f"\n📊 Duplicate Preview for: {source_dir}")
            self.console.print(f"Files scanned: {preview['files_scanned']}")

            summary = preview['duplicate_summary']
            self.console.print(f"\nDuplicate Summary:")
            self.console.print(f"  Total duplicate groups: {summary['total_duplicate_groups']}")
            self.console.print(f"  Total duplicate files: {summary['total_duplicate_files']}")

            if preview['sample_duplicates']:
                self.console.print(f"\nSample Duplicates (first {len(preview['sample_duplicates'])}):")
                for i, dup in enumerate(preview['sample_duplicates'], 1):
                    self.console.print(f"\n{i}. {Path(dup['file']).name}")
                    self.console.print(f"   Duplicates: {dup['duplicate_count']}")
                    if dup['duplicate_types']:
                        for dup_type in dup['duplicate_types']:
                            self.console.print(f"   - {dup_type['type']}: {dup_type['reason']}")

            return 0

        except MusicOrganizerError as e:
            self.console.print(f"Error: {e}", 'red')
            return 1
        except Exception as e:
            self.console.print(f"Unexpected error: {e}", 'red')
            if '--debug' in sys.argv:
                import traceback
                traceback.print_exc()
            return 1

    async def organize_with_duplicates(self,
                                     source_dir: Path,
                                     target_dir: Path,
                                     strategy: str = 'interactive',
                                     duplicate_dir: Optional[Path] = None,
                                     resolve_first: bool = True,
                                     config_path: Optional[Path] = None,
                                     dry_run: bool = False,
                                     workers: int = 4) -> int:
        """Organize music with duplicate resolution."""
        try:
            if not source_dir.exists():
                self.console.print(f"Error: Source directory does not exist: {source_dir}", 'red')
                return 1

            # Load configuration
            if config_path and config_path.exists():
                config = load_config(config_path)
            else:
                config = Config.default()
                # Update target directory in config
                config.target_directory = target_dir

            # Convert strategy
            strategy_map = {
                'interactive': ResolutionStrategy.INTERACTIVE,
                'auto_best': ResolutionStrategy.AUTO_KEEP_BEST,
                'auto_first': ResolutionStrategy.AUTO_FIRST,
                'auto_smart': ResolutionStrategy.AUTO_SMART
            }
            resolution_strategy = strategy_map.get(strategy, ResolutionStrategy.INTERACTIVE)

            self.console.print(f"🎵 Organizing music from {source_dir} to {target_dir}")
            self.console.print(f"Duplicate resolution: {'Enabled' if resolve_first else 'Disabled'}")
            self.console.print(f"Strategy: {strategy}")
            if dry_run:
                self.console.print("Mode: DRY RUN - No files will be modified", 'yellow')

            async with DuplicateResolverOrganizer(
                config=config,
                dry_run=dry_run,
                duplicate_strategy=resolution_strategy,
                duplicate_dir=duplicate_dir,
                enable_duplicate_resolution=resolve_first,
                max_workers=workers
            ) as resolver:
                org_result, dup_summary = await resolver.organize_with_duplicate_resolution(
                    source_dir=source_dir,
                    target_dir=target_dir,
                    resolve_duplicates_first=resolve_first
                )

            # Show organization results
            if org_result:
                self.console.print(f"\n✅ Organization completed!")
                if 'files_processed' in org_result:
                    self.console.print(f"Files processed: {org_result['files_processed']}")
                if 'errors' in org_result and org_result['errors']:
                    self.console.print(f"Errors: {len(org_result['errors'])}", 'yellow')

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

    async def run(self, args: argparse.Namespace) -> int:
        """Run the CLI with the given arguments."""
        if args.command == 'resolve':
            return await self.resolve_duplicates(
                source_dir=args.source,
                strategy=args.strategy,
                duplicate_dir=args.move_duplicates_to,
                dry_run=args.dry_run
            )
        elif args.command == 'preview':
            return await self.preview_duplicates(args.source)
        elif args.command == 'organize':
            if not args.target:
                self.console.print("Error: Target directory is required for organize command", 'red')
                return 1
            return await self.organize_with_duplicates(
                source_dir=args.source,
                target_dir=args.target,
                strategy=args.strategy,
                duplicate_dir=args.move_duplicates_to,
                resolve_first=args.resolve_first,
                config_path=args.config,
                dry_run=args.dry_run,
                workers=args.workers
            )
        else:
            self.console.print(f"Error: Unknown command: {args.command}", 'red')
            return 1


def main():
    """Main entry point for the duplicate resolver CLI."""
    cli = DuplicateResolverCLI()
    parser = cli.create_parser()
    args = parser.parse_args()

    # Run the async main function
    return asyncio.run(cli.run(args))


if __name__ == '__main__':
    sys.exit(main())