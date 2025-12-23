"""Tests for CLI module."""

import pytest
import asyncio
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from music_organizer.cli import (
    organize_command_async,
    organize_command,
    scan_command,
    inspect_command,
    validate_command,
    main,
    _organize_with_progress,
    _get_category_name
)
from music_organizer.models.config import Config
from music_organizer.exceptions import MusicOrganizerError


def async_generator(items):
    """Helper to create an async generator from a list."""
    async def gen():
        for item in items:
            yield item
    return gen()


class TestOrganizeCommandAsync:
    """Test async organize command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_source = Path("/tmp/test_source")
        self.temp_target = Path("/tmp/test_target")

    @pytest.mark.asyncio
    async def test_organize_with_config(self):
        """Test organizing with config file."""
        config_path = Path("/tmp/config.json")

        args = argparse.Namespace(
            source=self.temp_source,
            target=self.temp_target,
            config=config_path,
            dry_run=False,
            interactive=False,
            backup=True,
            verbose=False,
            incremental=False,
            force_full_scan=False,
            workers=4
        )

        with patch('music_organizer.cli.load_config') as mock_load:
            mock_load.return_value = Config(
                source_directory=self.temp_source,
                target_directory=self.temp_target
            )

            with patch('music_organizer.cli.AsyncMusicOrganizer') as mock_org_class:
                mock_org = Mock()
                mock_org.get_scan_info = Mock(return_value={'has_history': False})
                mock_org.scan_directory = Mock(return_value=async_generator([]))
                mock_org.organize_files = AsyncMock(return_value={
                    'processed': 0,
                    'moved': 0,
                    'skipped': 0,
                    'by_category': {},
                    'errors': []
                })
                mock_org_class.return_value = mock_org

                with patch('music_organizer.cli.console.confirm', return_value=True):
                    result = await organize_command_async(args)
                    assert result == 0

    @pytest.mark.asyncio
    async def test_organize_without_config(self):
        """Test organizing without config file."""
        args = argparse.Namespace(
            source=self.temp_source,
            target=self.temp_target,
            config=None,
            dry_run=True,
            interactive=False,
            backup=True,
            verbose=False,
            incremental=False,
            force_full_scan=False,
            workers=4
        )

        with patch('music_organizer.cli.AsyncMusicOrganizer') as mock_org_class:
            mock_org = Mock()
            mock_org.get_scan_info = Mock(return_value={'has_history': False})
            mock_org.scan_directory = Mock(return_value=async_generator([]))
            mock_org.organize_files = AsyncMock(return_value={
                'processed': 0,
                'moved': 0,
                'skipped': 0,
                'by_category': {},
                'errors': []
            })
            mock_org_class.return_value = mock_org

            with patch('music_organizer.cli.console.confirm', return_value=True):
                result = await organize_command_async(args)
                assert result == 0

    @pytest.mark.asyncio
    async def test_organize_incremental_scan_with_history(self):
        """Test incremental scan with history."""
        args = argparse.Namespace(
            source=self.temp_source,
            target=self.temp_target,
            config=None,
            dry_run=True,
            interactive=False,
            backup=True,
            verbose=False,
            incremental=True,
            force_full_scan=False,
            workers=4
        )

        with patch('music_organizer.cli.AsyncMusicOrganizer') as mock_org_class:
            mock_org = Mock()
            mock_org.get_scan_info = Mock(return_value={
                'has_history': True,
                'last_scan': '2024-01-01T12:00:00'
            })
            mock_org.scan_directory_incremental = Mock(return_value=async_generator([]))
            mock_org.organize_files = AsyncMock(return_value={
                'processed': 0,
                'moved': 0,
                'skipped': 0,
                'by_category': {},
                'errors': []
            })
            mock_org_class.return_value = mock_org

            with patch('music_organizer.cli.console.confirm', return_value=True):
                result = await organize_command_async(args)
                assert result == 0

    @pytest.mark.asyncio
    async def test_organize_force_full_scan(self):
        """Test forcing full scan."""
        args = argparse.Namespace(
            source=self.temp_source,
            target=self.temp_target,
            config=None,
            dry_run=True,
            interactive=False,
            backup=True,
            verbose=False,
            incremental=True,
            force_full_scan=True,
            workers=4
        )

        with patch('music_organizer.cli.AsyncMusicOrganizer') as mock_org_class:
            mock_org = Mock()
            mock_org.get_scan_info = Mock(return_value={'has_history': True})
            mock_org.scan_directory_incremental = Mock(return_value=async_generator([]))
            mock_org.organize_files = AsyncMock(return_value={
                'processed': 0,
                'moved': 0,
                'skipped': 0,
                'by_category': {},
                'errors': []
            })
            mock_org_class.return_value = mock_org

            with patch('music_organizer.cli.console.confirm', return_value=True):
                result = await organize_command_async(args)
                assert result == 0

    @pytest.mark.asyncio
    async def test_organize_with_files_found(self):
        """Test organizing with files found."""
        args = argparse.Namespace(
            source=self.temp_source,
            target=self.temp_target,
            config=None,
            dry_run=True,
            interactive=False,
            backup=True,
            verbose=False,
            incremental=False,
            force_full_scan=False,
            workers=4
        )

        with patch('music_organizer.cli.AsyncMusicOrganizer') as mock_org_class:
            mock_org = Mock()
            mock_org.get_scan_info = Mock(return_value={'has_history': False})
            mock_org.scan_directory = Mock(return_value=async_generator([
                Path("/tmp/file1.mp3"),
                Path("/tmp/file2.mp3")
            ]))
            mock_org.organize_files = AsyncMock(return_value={
                'processed': 2,
                'moved': 2,
                'skipped': 0,
                'by_category': {'Albums': 2},
                'errors': []
            })
            mock_org_class.return_value = mock_org

            with patch('music_organizer.cli.console.confirm', return_value=True):
                result = await organize_command_async(args)
                assert result == 0

    @pytest.mark.asyncio
    async def test_organize_music_organizer_error(self):
        """Test handling MusicOrganizerError."""
        args = argparse.Namespace(
            source=self.temp_source,
            target=self.temp_target,
            config=None,
            dry_run=False,
            interactive=False,
            backup=True,
            verbose=False,
            incremental=False,
            force_full_scan=False,
            workers=4
        )

        with patch('music_organizer.cli.AsyncMusicOrganizer') as mock_org_class:
            mock_org = Mock()
            mock_org.get_scan_info = Mock(side_effect=MusicOrganizerError("Test error"))
            mock_org_class.return_value = mock_org

            with patch('music_organizer.cli.console.confirm', return_value=True):
                result = await organize_command_async(args)
                assert result == 1

    @pytest.mark.asyncio
    async def test_organize_unexpected_error_with_verbose(self):
        """Test handling unexpected error with verbose output."""
        args = argparse.Namespace(
            source=self.temp_source,
            target=self.temp_target,
            config=None,
            dry_run=False,
            interactive=False,
            backup=True,
            verbose=True,
            incremental=False,
            force_full_scan=False,
            workers=4
        )

        with patch('music_organizer.cli.AsyncMusicOrganizer') as mock_org_class:
            mock_org = Mock()
            mock_org.get_scan_info = Mock(side_effect=Exception("Unexpected error"))
            mock_org_class.return_value = mock_org

            with patch('music_organizer.cli.console.confirm', return_value=True):
                with patch('traceback.format_exc', return_value="Traceback..."):
                    result = await organize_command_async(args)
                    assert result == 1


class TestOrganizeCommand:
    """Test sync wrapper for organize command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_source = Path("/tmp/test_source")
        self.temp_target = Path("/tmp/test_target")

    def test_organize_command_wrapper(self):
        """Test the sync wrapper calls async version."""
        args = argparse.Namespace(
            source=self.temp_source,
            target=self.temp_target,
            config=None,
            dry_run=True,
            interactive=False,
            backup=True,
            verbose=False,
            incremental=False,
            force_full_scan=False,
            workers=4
        )

        with patch('music_organizer.cli.asyncio.run') as mock_run:
            mock_run.return_value = 0
            result = organize_command(args)
            assert result == 0
            mock_run.assert_called_once()


class TestScanCommand:
    """Test scan command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path("/tmp/test_music")

    def test_scan_nonrecursive_no_files(self):
        """Test scanning non-recursively with no audio files."""
        args = argparse.Namespace(
            directory=self.temp_dir,
            recursive=False
        )

        with patch('pathlib.Path.rglob'):
            with patch('pathlib.Path.glob', return_value=[]):
                with patch('music_organizer.cli.SimpleConsole'):
                    result = scan_command(args)
                    assert result == 0

    def test_scan_recursive_with_files(self):
        """Test scanning recursively with files."""
        args = argparse.Namespace(
            directory=self.temp_dir,
            recursive=True
        )

        with patch('music_organizer.cli.MetadataHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_audio = Mock()
            mock_audio.album = "Test Album"
            mock_audio.title = "Test Title"
            mock_audio.artists = ["Test Artist"]
            mock_audio.size_mb = 5.0
            mock_handler.extract_metadata.return_value = mock_audio
            mock_handler_class.return_value = mock_handler

            with patch('music_organizer.cli.ContentClassifier') as mock_classifier_class:
                mock_classifier = Mock()
                mock_classifier.classify.return_value = (Mock(value="album"), 0.9)
                mock_classifier_class.return_value = mock_classifier

                with patch('builtins.print'):
                    result = scan_command(args)
                    # Should not crash

    def test_scan_with_metadata_extraction_error(self):
        """Test scanning with metadata extraction errors."""
        args = argparse.Namespace(
            directory=self.temp_dir,
            recursive=True
        )

        with patch('music_organizer.cli.MetadataHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.extract_metadata.side_effect = Exception("Extract error")
            mock_handler_class.return_value = mock_handler

            with patch('music_organizer.cli.ContentClassifier'):
                with patch('builtins.print'):
                    result = scan_command(args)
                    # Should handle errors gracefully


class TestInspectCommand:
    """Test inspect command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_file = Path("/tmp/test.mp3")

    def test_inspect_unsupported_file(self):
        """Test inspecting unsupported file type."""
        args = argparse.Namespace(
            file_path=Path("/tmp/test.txt")
        )

        with patch('music_organizer.cli.SimpleConsole'):
            result = inspect_command(args)
            assert result == 0

    def test_inspect_audio_file(self):
        """Test inspecting audio file."""
        args = argparse.Namespace(
            file_path=self.temp_file
        )

        with patch('music_organizer.cli.MetadataHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_audio = Mock()
            mock_audio.file_type = "MP3"
            mock_audio.size_mb = 5.0
            mock_audio.artists = ["Test Artist"]
            mock_audio.primary_artist = "Test Artist"
            mock_audio.album = "Test Album"
            mock_audio.title = "Test Title"
            mock_audio.year = 2023
            mock_audio.date = "2023"
            mock_audio.location = "US"
            mock_audio.track_number = 1
            mock_audio.genre = "Rock"
            mock_audio.has_cover_art = True
            mock_audio.metadata = {'title': 'Test Title'}
            mock_audio.get_target_path.return_value = Path("/target/Test.mp3")
            mock_handler.extract_metadata.return_value = mock_audio
            mock_handler_class.return_value = mock_handler

            with patch('music_organizer.cli.ContentClassifier') as mock_classifier_class:
                mock_classifier = Mock()
                mock_classifier.classify.return_value = (Mock(value="album"), 0.9)
                mock_classifier_class.return_value = mock_classifier

                with patch('music_organizer.cli.SimpleConsole'):
                    result = inspect_command(args)
                    assert result == 0

    def test_inspect_minimal_metadata(self):
        """Test inspecting file with minimal metadata."""
        args = argparse.Namespace(
            file_path=self.temp_file
        )

        with patch('music_organizer.cli.MetadataHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_audio = Mock()
            mock_audio.file_type = "FLAC"
            mock_audio.size_mb = 10.0
            mock_audio.artists = None
            mock_audio.primary_artist = None
            mock_audio.album = None
            mock_audio.title = None
            mock_audio.year = None
            mock_audio.date = None
            mock_audio.location = None
            mock_audio.track_number = None
            mock_audio.genre = None
            mock_audio.has_cover_art = False
            mock_audio.metadata = {}
            mock_audio.get_target_path.return_value = Path("/target/Test.flac")
            mock_handler.extract_metadata.return_value = mock_audio
            mock_handler_class.return_value = mock_handler

            with patch('music_organizer.cli.ContentClassifier') as mock_classifier_class:
                mock_classifier = Mock()
                mock_classifier.classify.return_value = (Mock(value="unknown"), 0.5)
                mock_classifier_class.return_value = mock_classifier

                with patch('music_organizer.cli.SimpleConsole'):
                    result = inspect_command(args)
                    assert result == 0


class TestValidateCommand:
    """Test validate command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path("/tmp/test_music")

    def test_validate_all_directories_exist(self):
        """Test validation with all directories existing."""
        args = argparse.Namespace(
            directory=self.temp_dir
        )

        with patch('music_organizer.cli.DirectoryOrganizer.validate_structure') as mock_validate:
            mock_validate.return_value = {
                'Albums': True,
                'Live': True,
                'Compilations': True
            }

            with patch('music_organizer.cli.DirectoryOrganizer.get_empty_directories') as mock_empty:
                mock_empty.return_value = []

                with patch('music_organizer.cli.SimpleConsole'):
                    result = validate_command(args)
                    assert result == 0

    def test_validate_with_empty_directories(self):
        """Test validation with empty directories."""
        args = argparse.Namespace(
            directory=self.temp_dir
        )

        with patch('music_organizer.cli.DirectoryOrganizer.validate_structure') as mock_validate:
            mock_validate.return_value = {
                'Albums': True,
                'Live': False
            }

            with patch('music_organizer.cli.DirectoryOrganizer.get_empty_directories') as mock_empty:
                mock_empty.return_value = [Path("/empty1")]

                with patch('music_organizer.cli.console.confirm', return_value=False):
                    result = validate_command(args)
                    assert result == 0

    def test_validate_with_misplaced_files(self):
        """Test validation with misplaced files."""
        args = argparse.Namespace(
            directory=self.temp_dir
        )

        with patch('music_organizer.cli.DirectoryOrganizer.validate_structure') as mock_validate:
            mock_validate.return_value = {}

            with patch('music_organizer.cli.DirectoryOrganizer.get_empty_directories') as mock_empty:
                mock_empty.return_value = []

                with patch('music_organizer.cli.console.confirm', return_value=False):
                    with patch('pathlib.Path.iterdir') as mock_iter:
                        mock_file = Mock()
                        mock_file.is_file.return_value = True
                        mock_file.suffix.lower.return_value = '.mp3'
                        mock_file.name = "test.mp3"
                        mock_iter.return_value = iter([mock_file])

                        result = validate_command(args)
                        assert result == 0


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_category_name(self):
        """Test getting category name from content type."""
        from music_organizer.models.audio_file import ContentType

        assert _get_category_name(ContentType.STUDIO) == 'Albums'
        assert _get_category_name(ContentType.LIVE) == 'Live'
        assert _get_category_name(ContentType.COLLABORATION) == 'Collaborations'
        assert _get_category_name(ContentType.COMPILATION) == 'Compilations'
        assert _get_category_name(None) == 'Unknown'

    def test_organize_with_progress(self):
        """Test organizing with progress tracking."""
        with patch('music_organizer.cli.Config'):
            mock_organizer = Mock()
            mock_organizer.dry_run = True
            mock_organizer.config = Mock()
            mock_organizer.config.source_directory = Path("/src")
            mock_organizer.file_mover = Mock()
            mock_organizer._process_file = Mock(return_value=True)
            mock_organizer._group_files = Mock(return_value={})

            files = [Path("/file1.mp3")]

            with patch('music_organizer.cli.SimpleProgress'):
                result = _organize_with_progress(mock_organizer, files, Mock())

                assert 'processed' in result
                assert 'moved' in result
                assert 'skipped' in result
                assert 'by_category' in result
                assert 'errors' in result


class TestMainFunction:
    """Test main entry point."""

    @patch('sys.argv', ['music-organize', 'scan', '/music'])
    def test_main_scan_command(self):
        """Test main with scan command."""
        with patch('music_organizer.cli.scan_command', return_value=0):
            result = main()
            assert result == 0

    @patch('sys.argv', ['music-organize', 'inspect', '/file.mp3'])
    def test_main_inspect_command(self):
        """Test main with inspect command."""
        with patch('music_organizer.cli.inspect_command', return_value=0):
            result = main()
            assert result == 0

    @patch('sys.argv', ['music-organize', 'validate', '/music'])
    def test_main_validate_command(self):
        """Test main with validate command."""
        with patch('music_organizer.cli.validate_command', return_value=0):
            result = main()
            assert result == 0

    @patch('sys.argv', ['music-organize'])
    def test_main_no_command(self):
        """Test main with no command."""
        with patch('argparse.ArgumentParser.print_help'):
            result = main()
            assert result == 1

    @patch('sys.argv', ['music-organize', 'organize', '/src', '/tgt'])
    def test_main_organize_command(self):
        """Test main with organize command."""
        with patch('music_organizer.cli.organize_command', return_value=0):
            result = main()
            assert result == 0


if __name__ == '__main__':
    pytest.main([__file__])
