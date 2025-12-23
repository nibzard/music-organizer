"""Tests for Async CLI module."""

import pytest
import asyncio
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from datetime import datetime

from music_organizer.async_cli import (
    AsyncMusicCLI,
    SimpleProgress,
    SimpleConsole,
    create_async_cli,
    main
)
from music_organizer.models.config import Config
from music_organizer.exceptions import MusicOrganizerError


class TestSimpleConsole:
    """Test SimpleConsole class."""

    def test_print_plain_text(self):
        """Test printing plain text."""
        console = SimpleConsole()
        with patch('builtins.print') as mock_print:
            console.print("Hello World")
            mock_print.assert_called_once_with("Hello World")

    def test_print_bold_text(self):
        """Test printing bold text."""
        console = SimpleConsole()
        with patch('builtins.print') as mock_print:
            console.print("Bold Text", 'bold')
            mock_print.assert_called_once_with("\033[1mBold Text\033[0m")

    def test_print_green_text(self):
        """Test printing green text."""
        console = SimpleConsole()
        with patch('builtins.print') as mock_print:
            console.print("Success", 'green')
            mock_print.assert_called_once_with("\033[92mSuccess\033[0m")

    def test_print_red_text(self):
        """Test printing red text."""
        console = SimpleConsole()
        with patch('builtins.print') as mock_print:
            console.print("Error", 'red')
            mock_print.assert_called_once_with("\033[91mError\033[0m")

    def test_print_yellow_text(self):
        """Test printing yellow text."""
        console = SimpleConsole()
        with patch('builtins.print') as mock_print:
            console.print("Warning", 'yellow')
            mock_print.assert_called_once_with("\033[93mWarning\033[0m")

    def test_print_cyan_text(self):
        """Test printing cyan text."""
        console = SimpleConsole()
        with patch('builtins.print') as mock_print:
            console.print("Info", 'cyan')
            mock_print.assert_called_once_with("\033[96mInfo\033[0m")

    def test_rule(self):
        """Test printing a horizontal rule."""
        console = SimpleConsole()
        with patch('builtins.print') as mock_print:
            console.rule("Title")
            # Check that print was called
            assert mock_print.called

    def test_table_empty(self):
        """Test table with no rows."""
        console = SimpleConsole()
        with patch('builtins.print'):
            console.table([], ["Header1", "Header2"])
            # Should not crash

    def test_table_with_data(self):
        """Test table with data."""
        console = SimpleConsole()
        with patch('builtins.print') as mock_print:
            console.table([["A", "B"], ["C", "D"]], ["Col1", "Col2"])
            # Should print header and rows
            assert mock_print.call_count >= 2


class TestSimpleProgress:
    """Test SimpleProgress class."""

    def test_initialization(self):
        """Test progress initialization."""
        progress = SimpleProgress(100, "Processing")
        assert progress.total == 100
        assert progress.current == 0
        assert progress.description == "Processing"

    def test_update(self):
        """Test updating progress."""
        progress = SimpleProgress(100, "Processing")
        with patch('builtins.print'):
            progress.update(10)
        assert progress.current == 10

    def test_update_zero_total(self):
        """Test update with zero total."""
        progress = SimpleProgress(0, "Processing")
        with patch('builtins.print'):
            progress.update(10)
        # Should not crash

    def test_update_complete(self):
        """Test updating to completion."""
        progress = SimpleProgress(100, "Processing")
        with patch('builtins.print'):
            progress.update(100)
        # Should complete and print newline


class TestAsyncMusicCLI:
    """Test AsyncMusicCLI class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = AsyncMusicCLI()
        self.temp_source = Path("/tmp/test_source")
        self.temp_target = Path("/tmp/test_target")

    @pytest.mark.asyncio
    async def test_organize_nonexistent_source(self):
        """Test organizing with non-existent source directory."""
        with patch('pathlib.Path.exists', return_value=False):
            result = await self.cli.organize(
                source=self.temp_source,
                target=self.temp_target
            )
            assert result == 1

    @pytest.mark.asyncio
    async def test_organize_with_config_file(self):
        """Test organizing with config file."""
        config_path = Path("/tmp/config.json")

        with patch('pathlib.Path.exists', return_value=True):
            with patch('music_organizer.async_cli.load_config') as mock_load:
                mock_load.return_value = Config(
                    source_directory=self.temp_source,
                    target_directory=self.temp_target
                )

                with patch('music_organizer.async_cli.AsyncMusicOrganizer') as mock_org_class:
                    mock_org = AsyncMock()
                    mock_org.scan_directory.return_value = []
                    mock_org.__aenter__.return_value = mock_org
                    mock_org.__aexit__.return_value = None
                    mock_org_class.return_value = mock_org

                    result = await self.cli.organize(
                        source=self.temp_source,
                        target=self.temp_target,
                        config_path=config_path
                    )

                    mock_load.assert_called_once_with(config_path)

    @pytest.mark.asyncio
    async def test_organize_magic_mode_analyze(self):
        """Test organizing with magic mode analyze."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(self.cli, '_handle_magic_mode') as mock_magic:
                mock_magic.return_value = 0

                result = await self.cli.organize(
                    source=self.temp_source,
                    target=self.temp_target,
                    magic_analyze=True
                )

                mock_magic.assert_called_once()
                assert result == 0

    @pytest.mark.asyncio
    async def test_organize_preview_mode(self):
        """Test organizing with preview mode."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(self.cli, '_handle_organization_preview') as mock_preview:
                mock_preview.return_value = 0

                result = await self.cli.organize(
                    source=self.temp_source,
                    target=self.temp_target,
                    preview=True
                )

                mock_preview.assert_called_once()
                assert result == 0

    @pytest.mark.asyncio
    async def test_organize_keyboard_interrupt(self):
        """Test handling keyboard interrupt during organization."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('music_organizer.async_cli.AsyncMusicOrganizer') as mock_org_class:
                mock_org = AsyncMock()
                mock_org.scan_directory.side_effect = KeyboardInterrupt()
                mock_org.__aenter__.return_value = mock_org
                mock_org.__aexit__.return_value = None
                mock_org_class.return_value = mock_org

                result = await self.cli.organize(
                    source=self.temp_source,
                    target=self.temp_target
                )

                assert result == 130

    @pytest.mark.asyncio
    async def test_organize_music_organizer_error(self):
        """Test handling MusicOrganizerError."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('music_organizer.async_cli.AsyncMusicOrganizer') as mock_org_class:
                mock_org = AsyncMock()
                mock_org.scan_directory.side_effect = MusicOrganizerError("Test error")
                mock_org.__aenter__.return_value = mock_org
                mock_org.__aexit__.return_value = None
                mock_org_class.return_value = mock_org

                result = await self.cli.organize(
                    source=self.temp_source,
                    target=self.temp_target
                )

                assert result == 1

    @pytest.mark.asyncio
    async def test_scan_nonexistent_directory(self):
        """Test scanning non-existent directory."""
        with patch('pathlib.Path.exists', return_value=False):
            result = await self.cli.scan(Path("/nonexistent"))
            assert result == 1

    @pytest.mark.asyncio
    async def test_scan_success(self):
        """Test successful scan."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('music_organizer.async_cli.AsyncMusicOrganizer') as mock_org_class:
                mock_org = AsyncMock()
                mock_org.scan_directory.return_value = []  # No files
                mock_org.__aenter__.return_value = mock_org
                mock_org.__aexit__.return_value = None
                mock_org_class.return_value = mock_org

                result = await self.cli.scan(self.temp_source)

                assert result == 0

    @pytest.mark.asyncio
    async def test_scan_detailed(self):
        """Test detailed scan."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('music_organizer.async_cli.AsyncMusicOrganizer') as mock_org_class:
                mock_org = AsyncMock()
                mock_org.scan_directory.return_value = []
                mock_org._get_category_name = Mock(return_value="Albums")
                mock_org.__aenter__.return_value = mock_org
                mock_org.__aexit__.return_value = None
                mock_org_class.return_value = mock_org

                with patch('music_organizer.async_cli.MetadataHandler'):
                    with patch('music_organizer.async_cli.ContentClassifier'):
                        result = await self.cli.scan(self.temp_source, detailed=True)
                        assert result == 0

    @pytest.mark.asyncio
    async def test_cache_stats_command(self):
        """Test cache stats command."""
        with patch('music_organizer.async_cli.get_cached_metadata_handler') as mock_get:
            mock_handler = Mock()
            mock_handler.get_cache_stats.return_value = {
                'total_entries': 100,
                'valid_entries': 90,
                'expired_entries': 10,
                'size_mb': 50.5,
                'cache_hits': 1000,
                'cache_misses': 100,
                'hit_rate': 0.9
            }
            mock_get.return_value = mock_handler

            result = await self.cli.handle_cache_command('stats')

            assert result == 0
            mock_handler.get_cache_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_cleanup_command(self):
        """Test cache cleanup command."""
        with patch('music_organizer.async_cli.get_cached_metadata_handler') as mock_get:
            mock_handler = Mock()
            mock_handler.cleanup_expired.return_value = 15
            mock_get.return_value = mock_handler

            result = await self.cli.handle_cache_command('cleanup')

            assert result == 0
            mock_handler.cleanup_expired.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_clear_without_confirm(self):
        """Test cache clear without confirmation."""
        with patch('music_organizer.async_cli.get_cached_metadata_handler') as mock_get:
            mock_handler = Mock()
            mock_get.return_value = mock_handler

            result = await self.cli.handle_cache_command('clear', confirm=False)

            assert result == 1
            mock_handler.clear_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_clear_with_confirm(self):
        """Test cache clear with confirmation."""
        with patch('music_organizer.async_cli.get_cached_metadata_handler') as mock_get:
            mock_handler = Mock()
            mock_get.return_value = mock_handler

            result = await self.cli.handle_cache_command('clear', confirm=True)

            assert result == 0
            mock_handler.clear_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_unknown_command(self):
        """Test unknown cache command."""
        with patch('music_organizer.async_cli.get_cached_metadata_handler') as mock_get:
            mock_handler = Mock()
            mock_get.return_value = mock_handler

            result = await self.cli.handle_cache_command('unknown')

            assert result == 1

    def test_format_size(self):
        """Test formatting file sizes."""
        assert "B" in self.cli._format_size(100)
        assert "KB" in self.cli._format_size(2000)
        assert "MB" in self.cli._format_size(2000000)
        assert "GB" in self.cli._format_size(2000000000)

    @pytest.mark.asyncio
    async def test_handle_magic_mode_analyze_only(self):
        """Test magic mode analyze only."""
        with patch('music_organizer.async_cli.MagicMusicOrganizer') as mock_magic_class:
            mock_magic = AsyncMock()
            mock_magic.analyze_library_for_magic_mode.return_value = {
                'recommended_strategy': 'bulk',
                'confidence': 0.9
            }
            mock_magic.initialize = AsyncMock()
            mock_magic._show_magic_analysis = AsyncMock()
            mock_magic_class.return_value = mock_magic

            result = await self.cli._handle_magic_mode(
                source=self.temp_source,
                target=self.temp_target,
                config=Config(
                    source_directory=self.temp_source,
                    target_directory=self.temp_target
                ),
                dry_run=False,
                magic_analyze=True,
                magic_auto=False,
                magic_sample=None,
                magic_preview=False,
                magic_save_config=None,
                magic_threshold=0.6,
                backup=True,
                max_workers=4,
                use_processes=False,
                enable_parallel_extraction=True,
                memory_threshold=80.0,
                use_cache=True,
                cache_ttl=None,
                incremental=False,
                force_full_scan=False,
                bulk_mode=False,
                chunk_size=200,
                conflict_strategy='rename',
                verify_copies=False,
                batch_dirs=True,
                preview_bulk=False,
                bulk_memory_threshold=512,
                smart_cache=None,
                cache_warming=None,
                cache_optimize=None,
                warm_cache_dir=None,
                cache_health=False
            )

            assert result == 0
            mock_magic.analyze_library_for_magic_mode.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_magic_mode_exception(self):
        """Test magic mode with exception."""
        with patch('music_organizer.async_cli.MagicMusicOrganizer') as mock_magic_class:
            mock_magic = AsyncMock()
            mock_magic.initialize.side_effect = Exception("Magic error")
            mock_magic_class.return_value = mock_magic

            result = await self.cli._handle_magic_mode(
                source=self.temp_source,
                target=self.temp_target,
                config=Config(
                    source_directory=self.temp_source,
                    target_directory=self.temp_target
                ),
                dry_run=False,
                magic_analyze=False,
                magic_auto=False,
                magic_sample=None,
                magic_preview=False,
                magic_save_config=None,
                magic_threshold=0.6,
                backup=True,
                max_workers=4,
                use_processes=False,
                enable_parallel_extraction=True,
                memory_threshold=80.0,
                use_cache=True,
                cache_ttl=None,
                incremental=False,
                force_full_scan=False,
                bulk_mode=False,
                chunk_size=200,
                conflict_strategy='rename',
                verify_copies=False,
                batch_dirs=True,
                preview_bulk=False,
                bulk_memory_threshold=512,
                smart_cache=None,
                cache_warming=None,
                cache_optimize=None,
                warm_cache_dir=None,
                cache_health=False
            )

            assert result == 1

    def test_show_magic_results(self):
        """Test displaying magic results."""
        result = {
            "stats": {
                "total_files": 100,
                "processed": 95,
                "errors": 5,
                "success_rate": 0.95,
                "strategy_used": "bulk",
                "confidence": 0.9
            },
            "organized_files": [
                {"source": "/src/file1.mp3", "target": "/tgt/file1.mp3"}
            ],
            "errors": []
        }

        with patch.object(self.cli.console, 'print'):
            self.cli._show_magic_results(result)
            # Should not crash

    @pytest.mark.asyncio
    async def test_handle_organization_preview(self):
        """Test organization preview."""
        with patch('music_organizer.async_cli.MetadataHandler'):
            with patch('music_organizer.async_cli.ContentClassifier'):
                with patch('os.walk') as mock_walk:
                    mock_walk.return_value = [
                        (str(self.temp_source), [], ["file1.mp3"])
                    ]

                    with patch('pathlib.Path.suffix', '.mp3'):
                        with patch('pathlib.Path.exists', return_value=True):
                            result = await self.cli._handle_organization_preview(
                                source=self.temp_source,
                                target=self.temp_target,
                                config=Config(
                                    source_directory=self.temp_source,
                                    target_directory=self.temp_target
                                ),
                                preview_detailed=False,
                                preview_interactive=False,
                                export_preview=None,
                                max_workers=4,
                                use_processes=False,
                                enable_parallel_extraction=True,
                                memory_threshold=80.0,
                                use_cache=True,
                                cache_ttl=None,
                                incremental=False,
                                force_full_scan=False,
                                bulk_mode=False,
                                chunk_size=200,
                                conflict_strategy='rename',
                                smart_cache=None,
                                cache_warming=None,
                                cache_optimize=None,
                                warm_cache_dir=None
                            )

                            assert result == 0


class TestCreateAsyncCLI:
    """Test create_async_cli function."""

    def test_parser_creation(self):
        """Test parser is created."""
        parser = create_async_cli()
        assert isinstance(parser, int) or isinstance(parser, argparse.ArgumentParser)

    def test_parser_organize_command(self):
        """Test parsing organize command."""
        with patch('sys.argv', ['music-organize-async', 'organize', '/src', '/tgt']):
            with patch('music_organizer.async_cli.AsyncMusicCLI') as mock_cli_class:
                mock_cli = AsyncMock()
                mock_cli.organize.return_value = 0
                mock_cli_class.return_value = mock_cli

                with patch('asyncio.run', return_value=0):
                    result = create_async_cli()
                    assert result == 0

    def test_parser_scan_command(self):
        """Test parsing scan command."""
        with patch('sys.argv', ['music-organize-async', 'scan', '/music']):
            with patch('music_organizer.async_cli.AsyncMusicCLI') as mock_cli_class:
                mock_cli = AsyncMock()
                mock_cli.scan.return_value = 0
                mock_cli_class.return_value = mock_cli

                with patch('asyncio.run', return_value=0):
                    result = create_async_cli()
                    assert result == 0

    def test_parser_cache_command(self):
        """Test parsing cache command."""
        with patch('sys.argv', ['music-organize-async', 'cache', 'stats']):
            with patch('music_organizer.async_cli.AsyncMusicCLI') as mock_cli_class:
                mock_cli = AsyncMock()
                mock_cli.handle_cache_command.return_value = 0
                mock_cli_class.return_value = mock_cli

                with patch('asyncio.run', return_value=0):
                    result = create_async_cli()
                    assert result == 0


class TestMainFunction:
    """Test main entry point."""

    @patch('sys.argv', ['music-organize-async', 'organize', '/src', '/tgt'])
    def test_main_entry_point(self):
        """Test main function entry point."""
        with patch('music_organizer.async_cli.create_async_cli', return_value=0):
            result = main()
            assert result == 0


if __name__ == '__main__':
    pytest.main([__file__])
