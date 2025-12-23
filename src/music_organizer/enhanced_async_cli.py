"""Enhanced async command line interface with rollback functionality."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

from .core.enhanced_async_organizer import EnhancedAsyncMusicOrganizer
from .utils.security import SecurityUtils, PathValidationError
from .core.async_organizer import AsyncMusicOrganizer, organize_files_async
from .core.metadata import MetadataHandler
from .core.classifier import ContentClassifier
from .core.bulk_operations import BulkOperationConfig, ConflictStrategy
from .core.bulk_organizer import BulkAsyncOrganizer
from .core.bulk_progress_tracker import BulkProgressTracker
from .core.operation_history import OperationHistoryTracker, OperationRollbackService
from .models.config import Config, load_config
from .exceptions import MusicOrganizerError
from .progress_tracker import IntelligentProgressTracker, ProgressStage
from .async_progress_renderer import AsyncProgressRenderer
from .rollback_cli import (
    setup_rollback_parser,
    setup_history_parser,
    setup_sessions_parser,
    cmd_rollback,
    cmd_history,
    cmd_sessions,
    restore_from_backup
)
from .regex_rules_cli import setup_rules_parser, handle_rules_command
from .console_utils import SimpleConsole


console = SimpleConsole()


def create_parser() -> argparse.ArgumentParser:
    """Create the enhanced command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced Music Organizer with Rollback Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard organization with operation history
  %(prog)s organize /music/unsorted /music/organized

  # Organization with explicit session ID
  %(prog)s organize /music/unsorted /music/organized --session-id "my_session"

  # Rollback a session (dry run)
  %(prog)s rollback SESSION_ID --dry-run

  # View operation history
  %(prog)s history --session-id SESSION_ID

  # List recent sessions
  %(prog)s sessions

  # Restore from backup
  %(prog)s restore /path/to/backup /music/organized
        """
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Organize command (main)
    organize_parser = subparsers.add_parser(
        'organize',
        help='Organize music files with operation tracking'
    )
    add_organize_arguments(organize_parser)

    # Legacy organize command (backward compatibility)
    legacy_organize_parser = subparsers.add_parser(
        'organize-legacy',
        help='Legacy organize command without operation tracking'
    )
    add_organize_arguments(legacy_organize_parser)

    # Rollback command
    setup_rollback_parser(subparsers)

    # History command
    setup_history_parser(subparsers)

    # Sessions command
    setup_sessions_parser(subparsers)

    # Rules command
    setup_rules_parser(subparsers)

    # Restore command
    restore_parser = subparsers.add_parser(
        'restore',
        help='Restore files from backup directory'
    )
    restore_parser.add_argument('backup_dir', help='Backup directory path')
    restore_parser.add_argument('target_dir', help='Target directory to restore to')
    restore_parser.add_argument('--dry-run', action='store_true', help='Show what would be restored')

    return parser


def add_organize_arguments(parser: argparse.ArgumentParser):
    """Add arguments to the organize command parser."""
    parser.add_argument('source', help='Source directory containing music files')
    parser.add_argument('target', help='Target directory for organized music')

    # Performance options
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker threads (default: 4)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--processes', action='store_true',
                       help='Use process pool instead of thread pool')

    # Caching options
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable metadata caching')
    parser.add_argument('--cache-ttl', type=int, default=30,
                       help='Cache TTL in days (default: 30)')
    parser.add_argument('--smart-cache', action='store_true',
                       help='Enable smart caching with adaptive TTL')

    # Incremental scanning
    parser.add_argument('--incremental', action='store_true',
                       help='Enable incremental scanning (only process new/modified files)')
    parser.add_argument('--force-full-scan', action='store_true',
                       help='Force full scan even with incremental enabled')

    # Bulk operations
    parser.add_argument('--bulk', action='store_true',
                       help='Enable bulk operations for better performance')
    parser.add_argument('--chunk-size', type=int, default=200,
                       help='Chunk size for bulk operations (default: 200)')
    parser.add_argument('--conflict-strategy', choices=['skip', 'rename', 'replace', 'keep_both'],
                       default='rename', help='Strategy for handling conflicts (default: rename)')
    parser.add_argument('--verify-copies', action='store_true',
                       help='Verify file integrity after copy operations')
    parser.add_argument('--preview-bulk', action='store_true',
                       help='Preview bulk operations before execution')
    parser.add_argument('--no-batch-dirs', action='store_true',
                       help='Disable batch directory creation')
    parser.add_argument('--bulk-memory-threshold', type=int, default=512,
                       help='Memory threshold in MB for bulk operations (default: 512)')

    # Rollback options
    parser.add_argument('--session-id', help='Custom session ID for operation tracking')
    parser.add_argument('--disable-history', action='store_true',
                       help='Disable operation history tracking')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Enable backup creation (default: enabled)')
    parser.add_argument('--no-backup', dest='backup', action='store_false',
                       help='Disable backup creation')

    # Organization options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive mode for ambiguous cases')
    parser.add_argument('--config', help='Path to configuration file')

    # Smart cache options (when enabled)
    parser.add_argument('--no-cache-warming', action='store_true',
                       help='Disable automatic cache warming')
    parser.add_argument('--no-cache-optimize', action='store_true',
                       help='Disable automatic cache optimization')
    parser.add_argument('--warm-cache-dir', help='Pre-warm cache for specific directory')
    parser.add_argument('--cache-health', action='store_true',
                       help='Show cache health report after organization')

    # Memory monitoring
    parser.add_argument('--memory-threshold', type=float, default=80.0,
                       help='Memory usage percentage threshold (default: 80.0)')

    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode (minimal output)')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')


async def main():
    """Main entry point for the enhanced async CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle non-organize commands
    if args.command == 'rollback':
        await cmd_rollback(args)
        return
    elif args.command == 'history':
        await cmd_history(args)
        return
    elif args.command == 'sessions':
        await cmd_sessions(args)
        return
    elif args.command == 'rules':
        exit_code = await handle_rules_command(args)
        sys.exit(exit_code)
        return
    elif args.command == 'restore':
        await restore_from_backup(Path(args.backup_dir), Path(args.target_dir), args.dry_run)
        return

    # Handle organize commands
    if args.command not in ['organize', 'organize-legacy']:
        console.error(f"Unknown command: {args.command}")
        sys.exit(1)

    try:
        # Convert and validate paths with security checks
        try:
            source_dir = SecurityUtils.sanitize_path(Path(args.source))
            target_dir = SecurityUtils.sanitize_path(Path(args.target))
        except PathValidationError as e:
            console.error(f"Path validation error: {e}")
            sys.exit(1)

        # Validate directories
        if not source_dir.exists():
            console.error(f"Source directory does not exist: {source_dir}")
            sys.exit(1)

        if not source_dir.is_dir():
            console.error(f"Source path is not a directory: {source_dir}")
            sys.exit(1)

        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        config = load_config(args.config) if args.config else Config()
        config.source_directory = source_dir
        config.target_directory = target_dir

        # Update config with CLI arguments
        if hasattr(args, 'backup'):
            config.file_operations.backup = args.backup
        if args.conflict_strategy:
            config.file_operations.conflict_strategy = ConflictStrategy(args.conflict_strategy)

        # Choose organizer based on command
        if args.command == 'organize-legacy' or args.disable_history:
            # Use legacy organizer without operation history
            await run_legacy_organizer(args, config, source_dir, target_dir)
        else:
            # Use enhanced organizer with operation history
            await run_enhanced_organizer(args, config, source_dir, target_dir)

    except KeyboardInterrupt:
        console.print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        console.error(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def run_enhanced_organizer(args, config: Config, source_dir: Path, target_dir: Path):
    """Run the enhanced organizer with operation history tracking."""
    # Create enhanced organizer
    async with EnhancedAsyncMusicOrganizer(
        config=config,
        dry_run=args.dry_run,
        interactive=args.interactive,
        max_workers=args.workers,
        use_cache=not args.no_cache,
        cache_ttl=args.cache_ttl,
        enable_parallel_extraction=not args.no_parallel,
        use_processes=args.processes,
        use_smart_cache=args.smart_cache,
        session_id=args.session_id,
        enable_operation_history=not args.disable_history
    ) as organizer:

        # Configure incremental scanning if enabled
        if args.incremental:
            organizer.enable_incremental = True
            if args.force_full_scan:
                await organizer.incremental_scanner.clear_history(source_dir)

        # Handle bulk operations
        if args.bulk:
            # Create bulk operation config
            bulk_config = BulkOperationConfig(
                max_workers=args.workers,
                chunk_size=args.chunk_size,
                conflict_strategy=ConflictStrategy(args.conflict_strategy),
                verify_copies=args.verify_copies,
                memory_threshold=args.bulk_memory_threshold
            )

            # Run bulk organization
            result = await organizer.organize_files_bulk(source_dir, target_dir, bulk_config)
        else:
            # Run standard organization
            result = await organizer.organize_files(source_dir, target_dir)

        # Handle result
        if result.is_success():
            stats = result.value()

            if args.json:
                import json
                console.print(json.dumps(stats, indent=2))
            else:
                console.print("\n[green]Organization completed successfully![/green]")
                console.print(f"Session ID: {stats.get('session_id', 'N/A')}")
                console.print(f"Total files: {stats['total_files']}")
                console.print(f"Organized files: {stats['organized_files']}")
                console.print(f"Failed files: {stats['failed_files']}")

                if stats.get('metadata_extraction_failures', 0) > 0:
                    console.print(f"Metadata extraction failures: {stats['metadata_extraction_failures']}")

                if stats.get('organization_failures', 0) > 0:
                    console.print(f"Organization failures: {stats['organization_failures']}")

                if not args.dry_run and not args.disable_history:
                    console.print("\n[cyan]Operation tracking enabled![/cyan]")
                    console.print(f"View history: music-organize-async history --session-id {stats.get('session_id')}")
                    console.print(f"Rollback: music-organize-async rollback {stats.get('session_id')}")

        else:
            console.error(f"Organization failed: {result.error()}")
            sys.exit(1)


async def run_legacy_organizer(args, config: Config, source_dir: Path, target_dir: Path):
    """Run the legacy organizer without operation history tracking."""
    # Create legacy organizer
    async with AsyncMusicOrganizer(
        config=config,
        dry_run=args.dry_run,
        interactive=args.interactive,
        max_workers=args.workers,
        use_cache=not args.no_cache,
        cache_ttl=args.cache_ttl,
        enable_parallel_extraction=not args.no_parallel,
        use_processes=args.processes
    ) as organizer:

        # Configure incremental scanning if enabled
        if args.incremental:
            organizer.enable_incremental = True
            if args.force_full_scan:
                await organizer.incremental_scanner.clear_history(source_dir)

        # Handle bulk operations
        if args.bulk:
            # Create bulk organizer
            bulk_organizer = BulkAsyncOrganizer(
                config=config,
                max_workers=args.workers,
                enable_parallel_extraction=not args.no_parallel
            )

            # Create bulk operation config
            bulk_config = BulkOperationConfig(
                max_workers=args.workers,
                chunk_size=args.chunk_size,
                conflict_strategy=ConflictStrategy(args.conflict_strategy),
                verify_copies=args.verify_copies,
                memory_threshold=args.bulk_memory_threshold
            )

            # Run bulk organization
            stats = await bulk_organizer.organize_bulk(source_dir, target_dir, bulk_config)
        else:
            # Run standard organization
            stats = await organize_files_async(organizer, source_dir, target_dir)

        # Display results
        if args.json:
            import json
            console.print(json.dumps(stats, indent=2))
        else:
            console.print("\n[green]Organization completed successfully![/green]")
            console.print(f"Total files: {stats['total_files']}")
            console.print(f"Organized files: {stats['organized_files']}")
            console.print(f"Failed files: {stats['failed_files']}")


if __name__ == '__main__':
    asyncio.run(main())