"""Tests for Batch Metadata CLI module."""

import pytest
import asyncio
import json
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from music_organizer.batch_metadata_cli import (
    ProgressTracker,
    BatchMetadataCLI,
    main
)
from music_organizer.core.batch_metadata import (
    MetadataOperation,
    OperationType,
    ConflictStrategy,
    MetadataOperationBuilder,
    BatchMetadataConfig
)


class TestProgressTracker:
    """Test ProgressTracker class."""

    def test_initialization(self):
        """Test progress tracker initialization."""
        tracker = ProgressTracker(100, quiet=False)
        assert tracker.total == 100
        assert tracker.current == 0
        assert tracker.quiet is False

    def test_initialization_quiet(self):
        """Test quiet progress tracker initialization."""
        tracker = ProgressTracker(100, quiet=True)
        assert tracker.quiet is True

    def test_update(self):
        """Test updating progress."""
        tracker = ProgressTracker(100, quiet=True)
        tracker.update(50, operations_count=2)
        assert tracker.current == 50

    def test_update_not_quiet(self):
        """Test updating progress with display."""
        tracker = ProgressTracker(100, quiet=False)
        with patch('builtins.print'):
            tracker.update(50)
        assert tracker.current == 50

    def test_complete(self):
        """Test completing progress."""
        tracker = ProgressTracker(100, quiet=False)
        with patch('builtins.print'):
            tracker.complete()
        # Should not crash

    def test_complete_quiet(self):
        """Test completing quiet progress."""
        tracker = ProgressTracker(100, quiet=True)
        tracker.complete()
        # Should not crash


class TestBatchMetadataCLI:
    """Test BatchMetadataCLI class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = BatchMetadataCLI()
        self.temp_dir = Path("/tmp/test_music")

    def test_create_parser(self):
        """Test parser creation."""
        parser = self.cli.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

        # Test basic parsing
        args = parser.parse_args([str(self.temp_dir)])
        assert args.directory == self.temp_dir
        assert args.dry_run is False
        assert args.workers == 4
        assert args.batch_size == 100

    def test_parser_with_options(self):
        """Test parsing with various options."""
        parser = self.cli.create_parser()

        args = parser.parse_args([
            str(self.temp_dir),
            '--operations', 'ops.json',
            '--filter', '*.flac',
            '--workers', '8',
            '--batch-size', '200',
            '--dry-run',
            '--no-backup',
            '--continue-on-error',
            '--preserve-time',
            '--quiet'
        ])

        assert args.directory == self.temp_dir
        assert args.operations == Path('ops.json')
        assert args.filter == '*.flac'
        assert args.workers == 8
        assert args.batch_size == 200
        assert args.dry_run is True
        assert args.no_backup is True
        assert args.continue_on_error is True
        assert args.preserve_time is True
        assert args.quiet is True

    def test_parser_quick_operations(self):
        """Test parsing quick operation options."""
        parser = self.cli.create_parser()

        args = parser.parse_args([
            str(self.temp_dir),
            '--set-genre', 'Rock',
            '--set-year', '2023',
            '--add-artist', 'Various Artists',
            '--standardize-genres',
            '--capitalize-titles',
            '--fix-track-numbers',
            '--remove-feat-artists'
        ])

        assert args.set_genre == 'Rock'
        assert args.set_year == 2023
        assert args.add_artist == 'Various Artists'
        assert args.standardize_genres is True
        assert args.capitalize_titles is True
        assert args.fix_track_numbers is True
        assert args.remove_feat_artists is True

    @pytest.mark.asyncio
    async def test_run_nonexistent_directory(self):
        """Test running with non-existent directory."""
        args = argparse.Namespace(
            directory=self.temp_dir,
            operations=None,
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre=None,
            set_year=None,
            add_artist=None,
            standardize_genres=False,
            capitalize_titles=False,
            fix_track_numbers=False,
            remove_feat_artists=False
        )

        with patch('pathlib.Path.exists', return_value=False):
            result = await self.cli.run(args)
            assert result == 1

    @pytest.mark.asyncio
    async def test_run_no_operations(self):
        """Test running with no operations specified."""
        args = argparse.Namespace(
            directory=self.temp_dir,
            operations=None,
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre=None,
            set_year=None,
            add_artist=None,
            standardize_genres=False,
            capitalize_titles=False,
            fix_track_numbers=False,
            remove_feat_artists=False
        )

        with patch('pathlib.Path.exists', return_value=True):
            result = await self.cli.run(args)
            assert result == 1

    @pytest.mark.asyncio
    async def test_run_no_audio_files(self):
        """Test running with no audio files found."""
        args = argparse.Namespace(
            directory=self.temp_dir,
            operations=None,
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre='Rock',
            set_year=None,
            add_artist=None,
            standardize_genres=False,
            capitalize_titles=False,
            fix_track_numbers=False,
            remove_feat_artists=False
        )

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(self.cli, '_find_files', return_value=[]):
                result = await self.cli.run(args)
                assert result == 1

    @pytest.mark.asyncio
    async def test_run_keyboard_interrupt(self):
        """Test handling keyboard interrupt."""
        args = argparse.Namespace(
            directory=self.temp_dir,
            operations=None,
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre='Rock',
            set_year=None,
            add_artist=None,
            standardize_genres=False,
            capitalize_titles=False,
            fix_track_numbers=False,
            remove_feat_artists=False
        )

        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(self.cli, '_find_files', side_effect=KeyboardInterrupt()):
                result = await self.cli.run(args)
                assert result == 130

    @pytest.mark.asyncio
    async def test_run_exception(self):
        """Test handling exception during run."""
        args = argparse.Namespace(
            directory=self.temp_dir,
            operations=None,
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre='Rock',
            set_year=None,
            add_artist=None,
            standardize_genres=False,
            capitalize_titles=False,
            fix_track_numbers=False,
            remove_feat_artists=False
        )

        with patch('pathlib.Path.exists', side_effect=Exception("Test error")):
            result = await self.cli.run(args)
            assert result == 1

    def test_create_operations_from_file(self):
        """Test creating operations from file."""
        ops_data = {
            'operations': [
                {
                    'field': 'genre',
                    'operation': 'set',
                    'value': 'Rock'
                }
            ]
        }

        args = argparse.Namespace(
            operations=Path('/tmp/ops.json'),
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre=None,
            set_year=None,
            add_artist=None,
            standardize_genres=False,
            capitalize_titles=False,
            fix_track_numbers=False,
            remove_feat_artists=False
        )

        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(ops_data)
            operations = self.cli._create_operations(args)
            assert len(operations) == 1
            assert operations[0].field == 'genre'

    def test_create_operations_quick_set_genre(self):
        """Test creating quick operations for genre."""
        args = argparse.Namespace(
            operations=None,
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre='Rock',
            set_year=None,
            add_artist=None,
            standardize_genres=False,
            capitalize_titles=False,
            fix_track_numbers=False,
            remove_feat_artists=False
        )

        operations = self.cli._create_operations(args)
        assert len(operations) == 1
        assert operations[0].field == 'genre'

    def test_create_operations_quick_set_year(self):
        """Test creating quick operations for year."""
        args = argparse.Namespace(
            operations=None,
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre=None,
            set_year=2023,
            add_artist=None,
            standardize_genres=False,
            capitalize_titles=False,
            fix_track_numbers=False,
            remove_feat_artists=False
        )

        operations = self.cli._create_operations(args)
        assert len(operations) == 1
        assert operations[0].field == 'year'

    def test_create_operations_quick_add_artist(self):
        """Test creating quick operations for adding artist."""
        args = argparse.Namespace(
            operations=None,
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre=None,
            set_year=None,
            add_artist='New Artist',
            standardize_genres=False,
            capitalize_titles=False,
            fix_track_numbers=False,
            remove_feat_artists=False
        )

        operations = self.cli._create_operations(args)
        assert len(operations) == 1

    def test_create_operations_standardize_genres(self):
        """Test creating operations for standardizing genres."""
        args = argparse.Namespace(
            operations=None,
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre=None,
            set_year=None,
            add_artist=None,
            standardize_genres=True,
            capitalize_titles=False,
            fix_track_numbers=False,
            remove_feat_artists=False
        )

        operations = self.cli._create_operations(args)
        assert len(operations) == 1

    def test_create_operations_capitalize_titles(self):
        """Test creating operations for capitalizing titles."""
        args = argparse.Namespace(
            operations=None,
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre=None,
            set_year=None,
            add_artist=None,
            standardize_genres=False,
            capitalize_titles=True,
            fix_track_numbers=False,
            remove_feat_artists=False
        )

        operations = self.cli._create_operations(args)
        assert len(operations) == 2  # title and album

    def test_create_operations_fix_track_numbers(self):
        """Test creating operations for fixing track numbers."""
        args = argparse.Namespace(
            operations=None,
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre=None,
            set_year=None,
            add_artist=None,
            standardize_genres=False,
            capitalize_titles=False,
            fix_track_numbers=True,
            remove_feat_artists=False
        )

        operations = self.cli._create_operations(args)
        assert len(operations) == 1

    def test_create_operations_remove_feat_artists(self):
        """Test creating operations for removing featuring artists."""
        args = argparse.Namespace(
            operations=None,
            filter=None,
            workers=4,
            batch_size=100,
            dry_run=False,
            no_backup=False,
            continue_on_error=False,
            preserve_time=True,
            quiet=False,
            set_genre=None,
            set_year=None,
            add_artist=None,
            standardize_genres=False,
            capitalize_titles=False,
            fix_track_numbers=False,
            remove_feat_artists=True
        )

        operations = self.cli._create_operations(args)
        assert len(operations) == 2  # Two transform operations

    def test_load_operations_file_error(self):
        """Test loading operations file with error."""
        file_path = Path('/tmp/ops.json')

        with patch('builtins.open', side_effect=IOError("Read error")):
            operations = self.cli._load_operations_file(file_path)
            assert operations == []

    def test_find_files(self):
        """Test finding audio files."""
        with patch('pathlib.Path.rglob') as mock_rglob:
            mock_file1 = Mock()
            mock_file1.is_file.return_value = True
            mock_file1.suffix.lower = '.mp3'
            mock_file1.__str__ = lambda self: '/music/file1.mp3'

            mock_file2 = Mock()
            mock_file2.is_file.return_value = True
            mock_file2.suffix.lower = '.flac'
            mock_file2.__str__ = lambda self: '/music/file2.flac'

            mock_dir = Mock()
            mock_dir.is_file.return_value = False

            mock_rglob.return_value = [mock_file1, mock_file2, mock_dir]

            files = self.cli._find_files(self.temp_dir, None)
            # Should find audio files

    def test_find_files_with_filter(self):
        """Test finding files with pattern filter."""
        with patch('pathlib.Path.rglob') as mock_rglob:
            mock_file1 = Mock()
            mock_file1.is_file.return_value = True
            mock_file1.suffix.lower = '.mp3'
            mock_file1.match.return_value = True
            mock_file1.__str__ = lambda self: '/music/file1.mp3'

            mock_rglob.return_value = [mock_file1]

            files = self.cli._find_files(self.temp_dir, '*.mp3')
            # Should apply filter

    @pytest.mark.asyncio
    async def test_process_files_success(self):
        """Test processing files successfully."""
        files = [Path('/music/file1.mp3')]

        operations = [MetadataOperation(
            field='genre',
            operation=OperationType.SET,
            value='Rock'
        )]

        config = BatchMetadataConfig(
            max_workers=4,
            batch_size=100,
            dry_run=False,
            backup_before_update=True,
            continue_on_error=False,
            preserve_modified_time=True
        )

        mock_result = Mock()
        mock_result.total_files = 1
        mock_result.successful = 1
        mock_result.failed = 0
        mock_result.skipped = 0
        mock_result.conflicts = 0
        mock_result.duration_seconds = 1.5
        mock_result.throughput_files_per_sec = 0.67
        mock_result.success_rate = 100.0
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.operations_performed = []

        with patch('music_organizer.batch_metadata_cli.BatchMetadataProcessor') as mock_proc_class:
            mock_proc = AsyncMock()
            mock_proc.apply_operations.return_value = mock_result
            mock_proc.cleanup = AsyncMock()
            mock_proc_class.return_value = mock_proc

            result = await self.cli._process_files(files, operations, config, quiet=True)
            assert result == mock_result

    def test_display_results_dry_run(self):
        """Test displaying results for dry run."""
        mock_result = Mock()
        mock_result.total_files = 10
        mock_result.successful = 10
        mock_result.failed = 0
        mock_result.skipped = 0
        mock_result.conflicts = 0
        mock_result.duration_seconds = 5.0
        mock_result.throughput_files_per_sec = 2.0
        mock_result.success_rate = 100.0
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.operations_performed = []

        with patch.object(self.cli.console, 'print'):
            self.cli._display_results(mock_result, dry_run=True)
            # Should not crash

    def test_display_results_with_errors(self):
        """Test displaying results with errors."""
        mock_result = Mock()
        mock_result.total_files = 10
        mock_result.successful = 8
        mock_result.failed = 2
        mock_result.skipped = 0
        mock_result.conflicts = 0
        mock_result.duration_seconds = 5.0
        mock_result.throughput_files_per_sec = 2.0
        mock_result.success_rate = 80.0
        mock_result.errors = [
            {'file': 'file1.mp3', 'error': 'Read error'},
            {'file': 'file2.mp3', 'error': 'Write error'}
        ]
        mock_result.warnings = []
        mock_result.operations_performed = []

        with patch.object(self.cli.console, 'print'):
            self.cli._display_results(mock_result, dry_run=False)
            # Should show errors

    def test_display_results_with_warnings(self):
        """Test displaying results with warnings."""
        mock_result = Mock()
        mock_result.total_files = 10
        mock_result.successful = 10
        mock_result.failed = 0
        mock_result.skipped = 0
        mock_result.conflicts = 0
        mock_result.duration_seconds = 5.0
        mock_result.throughput_files_per_sec = 2.0
        mock_result.success_rate = 100.0
        mock_result.errors = []
        mock_result.warnings = ['Warning 1', 'Warning 2']
        mock_result.operations_performed = []

        with patch.object(self.cli.console, 'print'):
            self.cli._display_results(mock_result, dry_run=False)
            # Should show warnings

    def test_display_results_with_operations_performed(self):
        """Test displaying results with operations performed."""
        mock_result = Mock()
        mock_result.total_files = 10
        mock_result.successful = 10
        mock_result.failed = 0
        mock_result.skipped = 0
        mock_result.conflicts = 0
        mock_result.duration_seconds = 5.0
        mock_result.throughput_files_per_sec = 2.0
        mock_result.success_rate = 100.0
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.operations_performed = [
            {'operations': ['genre', 'year']},
            {'operations': ['genre']}
        ]

        with patch.object(self.cli.console, 'print'):
            self.cli._display_results(mock_result, dry_run=False)
            # Should show operations performed

    def test_display_results_many_errors(self):
        """Test displaying results with many errors (truncated)."""
        errors = [{'file': f'file{i}.mp3', 'error': f'Error {i}'} for i in range(15)]

        mock_result = Mock()
        mock_result.total_files = 15
        mock_result.successful = 0
        mock_result.failed = 15
        mock_result.skipped = 0
        mock_result.conflicts = 0
        mock_result.duration_seconds = 5.0
        mock_result.throughput_files_per_sec = 3.0
        mock_result.success_rate = 0.0
        mock_result.errors = errors
        mock_result.warnings = []
        mock_result.operations_performed = []

        with patch.object(self.cli.console, 'print'):
            self.cli._display_results(mock_result, dry_run=False)
            # Should truncate errors


class TestMainFunction:
    """Test main entry point."""

    @patch('sys.argv', ['music-batch-metadata', '/music', '--set-genre', 'Rock'])
    def test_main_entry_point(self):
        """Test main function entry point."""
        with patch('music_organizer.batch_metadata_cli.BatchMetadataCLI') as mock_cli_class:
            mock_cli = Mock()
            mock_cli.run = AsyncMock(return_value=0)
            mock_cli_class.return_value = mock_cli

            with patch('asyncio.run', return_value=0):
                result = main()
                assert result == 0

    @patch('sys.argv', ['music-batch-metadata', '/music'])
    def test_main_keyboard_interrupt(self):
        """Test main with keyboard interrupt."""
        with patch('music_organizer.batch_metadata_cli.BatchMetadataCLI') as mock_cli_class:
            mock_cli = Mock()
            mock_cli.run = AsyncMock(return_value=130)
            mock_cli_class.return_value = mock_cli

            with patch('asyncio.run', return_value=130):
                result = main()
                assert result == 130


if __name__ == '__main__':
    pytest.main([__file__])
