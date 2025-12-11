"""CLI commands for operation history and rollback functionality."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from .core.operation_history import (
    OperationHistoryTracker,
    OperationRollbackService,
    OperationStatus
)
from .core.enhanced_async_organizer import EnhancedAsyncMusicOrganizer
from .models.config import Config
from .console_utils import SimpleConsole, SimpleProgressBar


console = SimpleConsole()


def setup_rollback_parser(subparsers):
    """Setup rollback command arguments."""
    rollback_parser = subparsers.add_parser(
        'rollback',
        help='Rollback file operations',
        description='Rollback previous file organization operations'
    )

    rollback_parser.add_argument(
        'session_id',
        nargs='?',
        help='Session ID to rollback (if not provided, shows list of recent sessions)'
    )

    rollback_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be rolled back without making changes'
    )

    rollback_parser.add_argument(
        '--operation-ids',
        nargs='*',
        help='Specific operation IDs to rollback (partial rollback)'
    )

    rollback_parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompts'
    )

    return rollback_parser


def setup_history_parser(subparsers):
    """Setup history command arguments."""
    history_parser = subparsers.add_parser(
        'history',
        help='View operation history',
        description='View history of file organization operations'
    )

    history_parser.add_argument(
        '--session-id',
        help='Show operations for specific session'
    )

    history_parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Number of recent sessions to show (default: 20)'
    )

    history_parser.add_argument(
        '--status',
        choices=['pending', 'in_progress', 'completed', 'failed', 'rolled_back'],
        help='Filter operations by status'
    )

    history_parser.add_argument(
        '--format',
        choices=['table', 'json'],
        default='table',
        help='Output format (default: table)'
    )

    return history_parser


def setup_sessions_parser(subparsers):
    """Setup sessions command arguments."""
    sessions_parser = subparsers.add_parser(
        'sessions',
        help='List operation sessions',
        description='List recent operation sessions'
    )

    sessions_parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of sessions to show (default: 10)'
    )

    sessions_parser.add_argument(
        '--format',
        choices=['table', 'json'],
        default='table',
        help='Output format (default: table)'
    )

    return sessions_parser


async def cmd_rollback(args):
    """Handle rollback command."""
    # Initialize history tracker and rollback service
    history_tracker = OperationHistoryTracker()
    rollback_service = OperationRollbackService(history_tracker)

    # If no session ID provided, show recent sessions
    if not args.session_id:
        await list_recent_sessions(history_tracker, limit=10)
        return

    # Get session details
    session = await history_tracker.get_session(args.session_id)
    if not session:
        console.error(f"Session {args.session_id} not found")
        sys.exit(1)

    # Show session information
    console.print(f"\nSession ID: {session.session_id}")
    console.print(f"Status: {session.status}")
    console.print(f"Start Time: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if session.end_time:
        console.print(f"End Time: {session.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"Source: {session.source_root}")
    console.print(f"Target: {session.target_root}")
    console.print(f"Operations: {session.completed_operations} completed, {session.failed_operations} failed")

    # Get operations for this session
    operations = await history_tracker.get_session_operations(args.session_id, OperationStatus.COMPLETED)

    if not operations:
        console.print("\nNo completed operations to rollback")
        return

    console.print(f"\nFound {len(operations)} operations that can be rolled back")

    # Show operations to be rolled back
    if args.dry_run:
        console.print("\n[bold]DRY RUN MODE[/bold] - No files will be modified")

    if args.operation_ids:
        # Partial rollback
        operations = [op for op in operations if op.id in args.operation_ids]
        console.print(f"\nRolling back {len(operations)} specific operations:")
    else:
        console.print("\nRolling back all operations:")

    for i, op in enumerate(operations, 1):
        console.print(f"  {i}. {op.operation_type.value}: {op.source_path} -> {op.target_path}")

    # Confirm rollback
    if not args.force and not args.dry_run:
        console.print("\n[yellow]Warning: This will move files back to their original locations[/yellow]")
        response = input("Are you sure you want to continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            console.print("Rollback cancelled")
            return

    # Perform rollback
    console.print("\nRolling back operations...")

    with SimpleProgressBar() as progress:
        progress.set_total(len(operations))

        if args.operation_ids:
            result = await rollback_service.rollback_partial(args.session_id, args.operation_ids, args.dry_run)
        else:
            result = await rollback_service.rollback_session(args.session_id, args.dry_run)

        progress.set_progress(len(operations))

    if result.is_success():
        data = result.value()

        if args.dry_run:
            console.print("\n[green]Dry run completed successfully[/green]")
            console.print(f"Operations that would be rolled back: {data['total_operations']}")
        else:
            console.print("\n[green]Rollback completed successfully[/green]")
            console.print(f"Successful rollbacks: {data.get('successful_rollbacks', 0)}")
            console.print(f"Failed rollbacks: {data.get('failed_rollbacks', 0)}")
            console.print(f"Skipped rollbacks: {data.get('skipped_rollbacks', 0)}")

            if data.get('failed_operations'):
                console.print("\n[red]Failed operations:[/red]")
                for failed in data['failed_operations']:
                    console.print(f"  - {failed['operation_id']}: {failed['error']}")

            if data.get('skipped_operations'):
                console.print("\n[yellow]Skipped operations:[/yellow]")
                for skipped in data['skipped_operations']:
                    console.print(f"  - {skipped['operation_id']}: {skipped['reason']}")
    else:
        console.error(f"Rollback failed: {result.error()}")
        sys.exit(1)


async def cmd_history(args):
    """Handle history command."""
    history_tracker = OperationHistoryTracker()

    if args.session_id:
        # Show operations for specific session
        await show_session_operations(history_tracker, args.session_id, args.status, args.format)
    else:
        # Show recent sessions summary
        await list_recent_sessions(history_tracker, args.limit, args.format)


async def cmd_sessions(args):
    """Handle sessions command."""
    history_tracker = OperationHistoryTracker()
    await list_recent_sessions(history_tracker, args.limit, args.format)


async def show_session_operations(history_tracker, session_id: str,
                                 status_filter: Optional[str], format_type: str):
    """Show operations for a specific session."""
    # Get session
    session = await history_tracker.get_session(session_id)
    if not session:
        console.error(f"Session {session_id} not found")
        sys.exit(1)

    # Get operations
    status_enum = OperationStatus(status_filter) if status_filter else None
    operations = await history_tracker.get_session_operations(session_id, status_enum)

    # Display header
    console.print(f"\n[bold]Session: {session_id}[/bold]")
    console.print(f"Status: {session.status}")
    console.print(f"Start: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if session.end_time:
        console.print(f"End: {session.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"Source: {session.source_root}")
    console.print(f"Target: {session.target_root}")

    if status_filter:
        console.print(f"\n[bold]Operations ({status_filter.upper()}):[/bold]")
    else:
        console.print(f"\n[bold]All Operations:[/bold]")

    if not operations:
        console.print("No operations found")
        return

    # Display operations
    if format_type == 'json':
        ops_data = [op.to_dict() for op in operations]
        console.print(json.dumps(ops_data, indent=2))
    else:
        # Table format
        console.print(f"\n{'ID':<8} {'Type':<10} {'Status':<12} {'Source':<40} {'Target':<40}")
        console.print("-" * 120)

        for op in operations:
            source = str(op.source_path)[-37:] + "..." if op.source_path and len(str(op.source_path)) > 40 else str(op.source_path) or ""
            target = str(op.target_path)[-37:] + "..." if op.target_path and len(str(op.target_path)) > 40 else str(op.target_path) or ""

            console.print(f"{op.id[:7]:<8} {op.operation_type.value:<10} {op.status.value:<12} {source:<40} {target:<40}")

            if op.error_message:
                console.print(f"         [red]Error: {op.error_message}[/red]")


async def list_recent_sessions(history_tracker, limit: int, format_type: str = 'table'):
    """List recent operation sessions."""
    sessions = await history_tracker.list_sessions(limit)

    if not sessions:
        console.print("No sessions found")
        return

    console.print(f"\n[bold]Recent {len(sessions)} Sessions:[/bold]")

    if format_type == 'json':
        sessions_data = [session.to_dict() for session in sessions]
        console.print(json.dumps(sessions_data, indent=2))
    else:
        # Table format
        console.print(f"\n{'Session ID':<20} {'Status':<12} {'Start Time':<20} {'Duration':<10} {'Operations':<12}")
        console.print("-" * 80)

        for session in sessions:
            duration = "N/A"
            if session.end_time:
                duration = str(session.end_time - session.start_time).split('.')[0]

            total_ops = session.completed_operations + session.failed_operations

            console.print(
                f"{session.session_id[:19]:<20} "
                f"{session.status:<12} "
                f"{session.start_time.strftime('%Y-%m-%d %H:%M'):<20} "
                f"{duration:<10} "
                f"{total_ops} ({session.completed_operations} ✓, {session.failed_operations} ✗)"
            )


async def restore_from_backup(backup_dir: Path, target_dir: Path, dry_run: bool = False):
    """Restore files from a backup directory."""
    if not backup_dir.exists():
        console.error(f"Backup directory not found: {backup_dir}")
        sys.exit(1)

    # Check for manifest
    manifest_path = backup_dir / 'manifest.json'
    if not manifest_path.exists():
        console.error("No manifest.json found in backup directory")
        sys.exit(1)

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    console.print(f"\nBackup from: {manifest['timestamp']}")
    console.print(f"Source: {manifest['source_root']}")
    console.print(f"Files: {len(manifest['files'])}")

    if dry_run:
        console.print("\n[bold]DRY RUN MODE[/bold] - No files will be modified")

    # Restore files
    restored = 0
    failed = 0

    with SimpleProgressBar() as progress:
        progress.set_total(len(manifest['files']))

        for file_info in manifest['files']:
            backup_path = backup_dir / file_info['path']
            target_path = target_dir / file_info['path']

            if not backup_path.exists():
                console.warning(f"Backup file missing: {file_info['path']}")
                failed += 1
                progress.update_progress()
                continue

            if target_path.exists():
                console.warning(f"Target file already exists: {file_info['path']}")
                failed += 1
                progress.update_progress()
                continue

            if not dry_run:
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(backup_path, target_path)
                    restored += 1
                except Exception as e:
                    console.error(f"Failed to restore {file_info['path']}: {e}")
                    failed += 1
            else:
                restored += 1

            progress.update_progress()

    # Summary
    console.print(f"\n[green]Restore completed[/green]")
    console.print(f"Restored: {restored} files")
    console.print(f"Failed: {failed} files")