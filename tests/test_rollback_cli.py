"""Tests for Rollback CLI module."""

import pytest
import argparse
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

# Create a mock SimpleProgressBar class and add it to console_utils module
# before importing rollback_cli
from music_organizer import console_utils
if not hasattr(console_utils, 'SimpleProgressBar'):
    console_utils.SimpleProgressBar = Mock

from music_organizer.rollback_cli import (
    setup_rollback_parser,
    setup_history_parser,
    setup_sessions_parser,
    cmd_rollback,
    cmd_history,
    cmd_sessions,
    show_session_operations,
    list_recent_sessions,
    restore_from_backup
)
from music_organizer.core.operation_history import OperationStatus


class TestSetupParsers:
    """Test parser setup functions."""

    def test_setup_rollback_parser(self):
        """Test rollback parser setup."""
        import argparse
        subparsers = argparse.ArgumentParser().add_subparsers()
        parser = setup_rollback_parser(subparsers)

        # Test parsing basic rollback
        args = parser.parse_args(['test-session-id'])
        assert args.session_id == 'test-session-id'
        assert args.dry_run is False
        assert args.force is False

        # Test parsing with options
        args = parser.parse_args([
            'test-session-id',
            '--dry-run',
            '--force',
            '--operation-ids', 'op1', 'op2'
        ])
        assert args.dry_run is True
        assert args.force is True
        assert args.operation_ids == ['op1', 'op2']

    def test_setup_history_parser(self):
        """Test history parser setup."""
        import argparse
        subparsers = argparse.ArgumentParser().add_subparsers()
        parser = setup_history_parser(subparsers)

        # Test parsing basic history
        args = parser.parse_args([])
        assert args.limit == 20
        assert args.format == 'table'
        assert args.session_id is None
        assert args.status is None

        # Test parsing with options
        args = parser.parse_args([
            '--session-id', 'test-session',
            '--limit', '50',
            '--status', 'completed',
            '--format', 'json'
        ])
        assert args.session_id == 'test-session'
        assert args.limit == 50
        assert args.status == 'completed'
        assert args.format == 'json'

    def test_setup_sessions_parser(self):
        """Test sessions parser setup."""
        import argparse
        subparsers = argparse.ArgumentParser().add_subparsers()
        parser = setup_sessions_parser(subparsers)

        # Test parsing basic sessions
        args = parser.parse_args([])
        assert args.limit == 10
        assert args.format == 'table'

        # Test parsing with options
        args = parser.parse_args(['--limit', '20', '--format', 'json'])
        assert args.limit == 20
        assert args.format == 'json'


class TestCmdRollback:
    """Test cmd_rollback function."""

    @pytest.mark.asyncio
    async def test_rollback_no_session_id(self):
        """Test rollback without session ID (lists sessions)."""
        args = argparse.Namespace(
            session_id=None,
            dry_run=False,
            operation_ids=None,
            force=False
        )

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.list_recent_sessions') as mock_list:
                await cmd_rollback(args)
                mock_list.assert_called_once_with(mock_tracker, limit=10)

    @pytest.mark.asyncio
    async def test_rollback_session_not_found(self):
        """Test rollback with non-existent session."""
        args = argparse.Namespace(
            session_id='nonexistent-session',
            dry_run=False,
            operation_ids=None,
            force=False
        )

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker.get_session.return_value = None
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.console'):
                with patch('music_organizer.rollback_cli.sys.exit') as mock_exit:
                    mock_exit.side_effect = SystemExit(1)
                    with pytest.raises(SystemExit):
                        await cmd_rollback(args)
                    mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_rollback_no_completed_operations(self):
        """Test rollback with no completed operations."""
        args = argparse.Namespace(
            session_id='test-session',
            dry_run=False,
            operation_ids=None,
            force=False
        )

        mock_session = Mock()
        mock_session.session_id = 'test-session'
        mock_session.status = 'completed'
        mock_session.start_time = datetime.now()
        mock_session.end_time = datetime.now()
        mock_session.source_root = '/src'
        mock_session.target_root = '/tgt'
        mock_session.completed_operations = 0
        mock_session.failed_operations = 0

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker.get_session.return_value = mock_session
            mock_tracker.get_session_operations.return_value = []
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.console'):
                await cmd_rollback(args)
                # Should complete without error

    @pytest.mark.asyncio
    async def test_rollback_partial_success(self):
        """Test partial rollback success."""
        args = argparse.Namespace(
            session_id='test-session',
            dry_run=False,
            operation_ids=['op1', 'op2'],
            force=True
        )

        mock_session = Mock()
        mock_session.session_id = 'test-session'
        mock_session.status = 'completed'
        mock_session.start_time = datetime.now()
        mock_session.end_time = datetime.now()
        mock_session.source_root = '/src'
        mock_session.target_root = '/tgt'
        mock_session.completed_operations = 5
        mock_session.failed_operations = 0

        mock_op = Mock()
        mock_op.id = 'op1'
        mock_op.operation_type.value = 'move'
        mock_op.source_path = '/src/file.mp3'
        mock_op.target_path = '/tgt/file.mp3'

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker.get_session.return_value = mock_session
            mock_tracker.get_session_operations.return_value = [mock_op]
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.OperationRollbackService') as mock_rollback_class:
                mock_rollback = AsyncMock()
                mock_result = Mock()
                mock_result.is_success.return_value = True
                mock_result.value.return_value = {
                    'successful_rollbacks': 2,
                    'failed_rollbacks': 0,
                    'skipped_rollbacks': 0
                }
                mock_rollback.rollback_partial.return_value = mock_result
                mock_rollback_class.return_value = mock_rollback

                with patch('music_organizer.rollback_cli.SimpleProgressBar'):
                    with patch('music_organizer.rollback_cli.console'):
                        await cmd_rollback(args)
                        mock_rollback.rollback_partial.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_dry_run(self):
        """Test rollback with dry run."""
        args = argparse.Namespace(
            session_id='test-session',
            dry_run=True,
            operation_ids=None,
            force=False
        )

        mock_session = Mock()
        mock_session.session_id = 'test-session'
        mock_session.status = 'completed'
        mock_session.start_time = datetime.now()
        mock_session.end_time = datetime.now()
        mock_session.source_root = '/src'
        mock_session.target_root = '/tgt'
        mock_session.completed_operations = 5
        mock_session.failed_operations = 0

        mock_op = Mock()
        mock_op.id = 'op1'
        mock_op.operation_type.value = 'move'
        mock_op.source_path = '/src/file.mp3'
        mock_op.target_path = '/tgt/file.mp3'

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker.get_session.return_value = mock_session
            mock_tracker.get_session_operations.return_value = [mock_op]
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.OperationRollbackService') as mock_rollback_class:
                mock_rollback = AsyncMock()
                mock_result = Mock()
                mock_result.is_success.return_value = True
                mock_result.value.return_value = {
                    'total_operations': 5
                }
                mock_rollback.rollback_session.return_value = mock_result
                mock_rollback_class.return_value = mock_rollback

                with patch('music_organizer.rollback_cli.SimpleProgressBar'):
                    with patch('music_organizer.rollback_cli.console'):
                        await cmd_rollback(args)
                        mock_rollback.rollback_session.assert_called_once_with(
                            'test-session', True
                        )

    @pytest.mark.asyncio
    async def test_rollback_failed_result(self):
        """Test rollback with failed result."""
        args = argparse.Namespace(
            session_id='test-session',
            dry_run=False,
            operation_ids=None,
            force=True
        )

        mock_session = Mock()
        mock_session.session_id = 'test-session'
        mock_session.status = 'completed'
        mock_session.start_time = datetime.now()
        mock_session.end_time = datetime.now()
        mock_session.source_root = '/src'
        mock_session.target_root = '/tgt'
        mock_session.completed_operations = 5
        mock_session.failed_operations = 0

        mock_op = Mock()
        mock_op.id = 'op1'
        mock_op.operation_type.value = 'move'
        mock_op.source_path = '/src/file.mp3'
        mock_op.target_path = '/tgt/file.mp3'

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker.get_session.return_value = mock_session
            mock_tracker.get_session_operations.return_value = [mock_op]
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.OperationRollbackService') as mock_rollback_class:
                mock_rollback = AsyncMock()
                mock_result = Mock()
                mock_result.is_success.return_value = False
                mock_result.error.return_value = "Rollback failed"
                mock_rollback.rollback_session.return_value = mock_result
                mock_rollback_class.return_value = mock_rollback

                with patch('music_organizer.rollback_cli.SimpleProgressBar'):
                    with patch('music_organizer.rollback_cli.console'):
                        with patch('sys.exit') as mock_exit:
                            await cmd_rollback(args)
                            mock_exit.assert_called_once_with(1)


class TestCmdHistory:
    """Test cmd_history function."""

    @pytest.mark.asyncio
    async def test_history_with_session_id(self):
        """Test history with specific session ID."""
        args = argparse.Namespace(
            session_id='test-session',
            limit=20,
            status='completed',
            format='table'
        )

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.show_session_operations') as mock_show:
                await cmd_history(args)
                mock_show.assert_called_once_with(
                    mock_tracker, 'test-session', 'completed', 'table'
                )

    @pytest.mark.asyncio
    async def test_history_without_session_id(self):
        """Test history without session ID."""
        args = argparse.Namespace(
            session_id=None,
            limit=20,
            status=None,
            format='json'
        )

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.list_recent_sessions') as mock_list:
                await cmd_history(args)
                mock_list.assert_called_once_with(mock_tracker, 20, 'json')


class TestCmdSessions:
    """Test cmd_sessions function."""

    @pytest.mark.asyncio
    async def test_sessions_default(self):
        """Test sessions command with defaults."""
        args = argparse.Namespace(
            limit=10,
            format='table'
        )

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.list_recent_sessions') as mock_list:
                await cmd_sessions(args)
                mock_list.assert_called_once_with(mock_tracker, 10, 'table')


class TestShowSessionOperations:
    """Test show_session_operations function."""

    @pytest.mark.asyncio
    async def test_show_operations_session_not_found(self):
        """Test showing operations for non-existent session."""
        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker.get_session.return_value = None
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.console'):
                with patch('music_organizer.rollback_cli.sys.exit') as mock_exit:
                    mock_exit.side_effect = SystemExit(1)
                    with pytest.raises(SystemExit):
                        await show_session_operations(mock_tracker, 'test-session', None, 'table')
                    mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_show_operations_no_operations(self):
        """Test showing operations when none exist."""
        mock_session = Mock()
        mock_session.session_id = 'test-session'
        mock_session.status = 'completed'
        mock_session.start_time = datetime.now()
        mock_session.end_time = datetime.now()
        mock_session.source_root = '/src'
        mock_session.target_root = '/tgt'

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker.get_session.return_value = mock_session
            mock_tracker.get_session_operations.return_value = []
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.console'):
                await show_session_operations(mock_tracker, 'test-session', None, 'table')
                # Should not crash

    @pytest.mark.asyncio
    async def test_show_operations_json_format(self):
        """Test showing operations in JSON format."""
        mock_session = Mock()
        mock_session.session_id = 'test-session'
        mock_session.status = 'completed'
        mock_session.start_time = datetime.now()
        mock_session.end_time = datetime.now()
        mock_session.source_root = '/src'
        mock_session.target_root = '/tgt'

        mock_op = Mock()
        mock_op.to_dict.return_value = {'id': 'op1', 'type': 'move'}

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker.get_session.return_value = mock_session
            mock_tracker.get_session_operations.return_value = [mock_op]
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.console'):
                await show_session_operations(mock_tracker, 'test-session', None, 'json')
                # Should print JSON

    @pytest.mark.asyncio
    async def test_show_operations_with_error(self):
        """Test showing operations with error messages."""
        mock_session = Mock()
        mock_session.session_id = 'test-session'
        mock_session.status = 'completed'
        mock_session.start_time = datetime.now()
        mock_session.end_time = datetime.now()
        mock_session.source_root = '/src'
        mock_session.target_root = '/tgt'

        mock_op = Mock()
        mock_op.id = 'op1'
        mock_op.operation_type.value = 'move'
        mock_op.status.value = 'failed'
        mock_op.source_path = '/src/file.mp3'
        mock_op.target_path = '/tgt/file.mp3'
        mock_op.error_message = "Error moving file"

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker.get_session.return_value = mock_session
            mock_tracker.get_session_operations.return_value = [mock_op]
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.console'):
                await show_session_operations(mock_tracker, 'test-session', None, 'table')
                # Should show error


class TestListRecentSessions:
    """Test list_recent_sessions function."""

    @pytest.mark.asyncio
    async def test_list_sessions_no_sessions(self):
        """Test listing when no sessions exist."""
        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker.list_sessions.return_value = []
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.console'):
                await list_recent_sessions(mock_tracker, 10, 'table')
                # Should not crash

    @pytest.mark.asyncio
    async def test_list_sessions_json_format(self):
        """Test listing sessions in JSON format."""
        mock_session = Mock()
        mock_session.to_dict.return_value = {'id': 'session1'}

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker.list_sessions.return_value = [mock_session]
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.console'):
                await list_recent_sessions(mock_tracker, 10, 'json')
                # Should print JSON

    @pytest.mark.asyncio
    async def test_list_sessions_with_end_time(self):
        """Test listing sessions with end time."""
        mock_session = Mock()
        mock_session.session_id = 'test-session-1234567890123456789'
        mock_session.status = 'completed'
        mock_session.start_time = datetime.now()
        mock_session.end_time = datetime.now()
        mock_session.completed_operations = 10
        mock_session.failed_operations = 2

        with patch('music_organizer.rollback_cli.OperationHistoryTracker') as mock_tracker_class:
            mock_tracker = AsyncMock()
            mock_tracker.list_sessions.return_value = [mock_session]
            mock_tracker_class.return_value = mock_tracker

            with patch('music_organizer.rollback_cli.console'):
                await list_recent_sessions(mock_tracker, 10, 'table')
                # Should not crash


class TestRestoreFromBackup:
    """Test restore_from_backup function."""

    @pytest.mark.asyncio
    async def test_restore_nonexistent_backup_dir(self):
        """Test restoring from non-existent backup directory."""
        backup_dir = Mock(spec=Path)
        backup_dir.exists.return_value = False
        backup_dir.__str__ = lambda self: "/nonexistent/backup"
        target_dir = Path("/target")

        with patch('music_organizer.rollback_cli.console'):
            with patch('music_organizer.rollback_cli.sys.exit') as mock_exit:
                mock_exit.side_effect = SystemExit(1)
                with pytest.raises(SystemExit):
                    await restore_from_backup(backup_dir, target_dir)
                mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_restore_no_manifest(self):
        """Test restoring without manifest file."""
        backup_dir = Mock(spec=Path)
        backup_dir.exists.return_value = True
        backup_dir.__truediv__ = Mock(return_value=Mock(spec=Path, exists=Mock(return_value=False)))
        backup_dir.__str__ = lambda self: "/backup"
        target_dir = Path("/target")

        with patch('music_organizer.rollback_cli.console'):
            with patch('music_organizer.rollback_cli.sys.exit') as mock_exit:
                mock_exit.side_effect = SystemExit(1)
                with pytest.raises(SystemExit):
                    await restore_from_backup(backup_dir, target_dir)
                mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_restore_dry_run(self):
        """Test restore in dry run mode."""
        manifest_data = {
            'timestamp': '2024-01-01T12:00:00',
            'source_root': '/src',
            'files': [
                {'path': 'file1.mp3'}
            ]
        }

        backup_file = Mock(spec=Path)
        backup_file.exists.return_value = True

        backup_dir = Mock(spec=Path)
        backup_dir.exists.return_value = True
        backup_dir.__truediv__ = Mock(side_effect=lambda x: backup_file if x == 'file1.mp3' else Mock(spec=Path, exists=Mock(return_value=True)))
        backup_dir.__str__ = lambda self: "/backup"

        target_dir = Path("/target")
        target_file = target_dir / 'file1.mp3'

        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = json.dumps(manifest_data)
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=False)
            mock_open.return_value = mock_file

            with patch('music_organizer.rollback_cli.SimpleProgressBar'):
                with patch('music_organizer.rollback_cli.console'):
                    await restore_from_backup(backup_dir, target_dir, dry_run=True)

    @pytest.mark.asyncio
    async def test_restore_with_missing_backup_file(self):
        """Test restore with missing backup file."""
        manifest_data = {
            'timestamp': '2024-01-01T12:00:00',
            'source_root': '/src',
            'files': [
                {'path': 'missing.mp3'}
            ]
        }

        # Manifest exists (for reading)
        manifest_path = Mock(spec=Path)
        manifest_path.exists.return_value = True

        # But backup file doesn't exist
        backup_file = Mock(spec=Path)
        backup_file.exists.return_value = False

        backup_dir = Mock(spec=Path)
        backup_dir.exists.return_value = True
        # Return manifest_path for 'manifest.json', backup_file for actual file
        backup_dir.__truediv__ = Mock(side_effect=lambda x: manifest_path if x == 'manifest.json' else backup_file)
        backup_dir.__str__ = lambda self: "/backup"

        target_dir = Path("/target")

        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = json.dumps(manifest_data)
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=False)
            mock_open.return_value = mock_file

            with patch('music_organizer.rollback_cli.SimpleProgressBar'):
                with patch('music_organizer.rollback_cli.console'):
                    await restore_from_backup(backup_dir, target_dir, dry_run=False)

    @pytest.mark.asyncio
    async def test_restore_with_existing_target_file(self):
        """Test restore when target file already exists."""
        manifest_data = {
            'timestamp': '2024-01-01T12:00:00',
            'source_root': '/src',
            'files': [
                {'path': 'file1.mp3'}
            ]
        }

        backup_file = Mock(spec=Path)
        backup_file.exists.return_value = True

        backup_dir = Mock(spec=Path)
        backup_dir.exists.return_value = True
        backup_dir.__truediv__ = Mock(return_value=backup_file)
        backup_dir.__str__ = lambda self: "/backup"

        target_dir = Mock(spec=Path)
        target_file = Mock(spec=Path)
        target_file.exists.return_value = True
        target_dir.__truediv__ = Mock(return_value=target_file)
        target_dir.__str__ = lambda self: "/target"
        target_file.parent = Mock()
        target_file.parent.mkdir = Mock()

        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = json.dumps(manifest_data)
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=False)
            mock_open.return_value = mock_file

            with patch('music_organizer.rollback_cli.SimpleProgressBar'):
                with patch('music_organizer.rollback_cli.console'):
                    await restore_from_backup(backup_dir, target_dir, dry_run=False)


if __name__ == '__main__':
    pytest.main([__file__])
