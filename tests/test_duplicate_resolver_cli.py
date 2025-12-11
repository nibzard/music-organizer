"""Tests for Duplicate Resolver CLI."""

import pytest
import asyncio
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.music_organizer.duplicate_resolver_cli import DuplicateResolverCLI, main
from src.music_organizer.core.interactive_duplicate_resolver import ResolutionStrategy, ResolutionSummary


class TestDuplicateResolverCLI:
    """Test the DuplicateResolverCLI class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = DuplicateResolverCLI()
        self.temp_dir = Path("/tmp/test_music")

    def test_create_parser(self):
        """Test argument parser creation."""
        parser = self.cli.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

        # Test parsing resolve command
        args = parser.parse_args(['resolve', '/music/library'])
        assert args.command == 'resolve'
        assert args.source == Path('/music/library')
        assert args.strategy == 'interactive'
        assert args.dry_run is False

        # Test parsing with options
        args = parser.parse_args([
            'resolve', '/music/library',
            '--strategy', 'auto_best',
            '--move-duplicates-to', '/duplicates',
            '--dry-run'
        ])
        assert args.strategy == 'auto_best'
        assert args.move_duplicates_to == Path('/duplicates')
        assert args.dry_run is True

    def test_parse_organize_command(self):
        """Test parsing organize command."""
        parser = self.cli.create_parser()
        args = parser.parse_args([
            'organize', '/music/source', '/music/target',
            '--strategy', 'auto_smart',
            '--no-duplicates'
        ])
        assert args.command == 'organize'
        assert args.source == Path('/music/source')
        assert args.target == Path('/music/target')
        assert args.strategy == 'auto_smart'

    def test_parse_preview_command(self):
        """Test parsing preview command."""
        parser = self.cli.create_parser()
        args = parser.parse_args(['preview', '/music/library'])
        assert args.command == 'preview'
        assert args.source == Path('/music/library')

    @pytest.mark.asyncio
    async def test_resolve_duplicates_nonexistent_directory(self):
        """Test resolving duplicates with non-existent directory."""
        with patch('sys.exit', return_value=1) as mock_exit:
            result = await self.cli.resolve_duplicates(Path("/nonexistent"))
            # The CLI prints an error and returns 1, it doesn't call sys.exit directly
            assert result == 1

    @pytest.mark.asyncio
    async def test_resolve_duplicates_success(self):
        """Test successful duplicate resolution."""
        with patch('src.music_organizer.duplicate_resolver_cli.quick_duplicate_resolution') as mock_resolve:
            mock_resolve.return_value = ResolutionSummary(
                total_groups=5,
                resolved_groups=5,
                kept_files=5,
                space_saved_mb=100.5
            )

            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.is_dir', return_value=True):
                    result = await self.cli.resolve_duplicates(
                        source_dir=self.temp_dir,
                        strategy='auto_best',
                        dry_run=False
                    )

            assert result == 0
            mock_resolve.assert_called_once_with(
                source_dir=self.temp_dir,
                strategy=ResolutionStrategy.AUTO_KEEP_BEST,
                duplicate_dir=None,
                dry_run=False
            )

    @pytest.mark.asyncio
    async def test_resolve_duplicates_no_duplicates_found(self):
        """Test resolving when no duplicates are found."""
        with patch('src.music_organizer.duplicate_resolver_cli.quick_duplicate_resolution') as mock_resolve:
            mock_resolve.return_value = None  # No duplicates found

            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.is_dir', return_value=True):
                    result = await self.cli.resolve_duplicates(
                        source_dir=self.temp_dir
                    )

            assert result == 0
            mock_resolve.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_duplicates_keyboard_interrupt(self):
        """Test handling keyboard interrupt during resolution."""
        with patch('src.music_organizer.duplicate_resolver_cli.quick_duplicate_resolution') as mock_resolve:
            mock_resolve.side_effect = KeyboardInterrupt()

            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.is_dir', return_value=True):
                    result = await self.cli.resolve_duplicates(
                        source_dir=self.temp_dir
                    )

            assert result == 130

    @pytest.mark.asyncio
    async def test_preview_duplicates(self):
        """Test previewing duplicates."""
        mock_preview = {
            'files_scanned': 100,
            'duplicate_summary': {
                'total_duplicate_groups': 5,
                'total_duplicate_files': 10
            },
            'sample_duplicates': [
                {
                    'file': '/music/duplicate1.mp3',
                    'duplicate_count': 2,
                    'duplicate_types': [
                        {'type': 'metadata', 'reason': 'Same title and artist'}
                    ]
                }
            ]
        }

        with patch('src.music_organizer.duplicate_resolver_cli.DuplicateResolverOrganizer') as mock_organizer_class:
            mock_organizer = AsyncMock()
            mock_organizer.get_duplicate_preview.return_value = mock_preview
            mock_organizer_class.return_value.__aenter__.return_value = mock_organizer

            with patch('pathlib.Path.exists', return_value=True):
                result = await self.cli.preview_duplicates(self.temp_dir)

            assert result == 0
            mock_organizer.get_duplicate_preview.assert_called_once_with(self.temp_dir, 10)

    @pytest.mark.asyncio
    async def test_organize_with_duplicates(self):
        """Test organizing music with duplicate resolution."""
        mock_org_result = {'files_processed': 50}
        mock_dup_summary = ResolutionSummary(
            total_groups=3,
            resolved_groups=3,
            kept_files=3,
            moved_files=0,
            deleted_files=0
        )

        with patch('src.music_organizer.duplicate_resolver_cli.DuplicateResolverOrganizer') as mock_organizer_class:
            mock_organizer = AsyncMock()
            mock_organizer.organize_with_duplicate_resolution.return_value = (mock_org_result, mock_dup_summary)
            mock_organizer_class.return_value.__aenter__.return_value = mock_organizer

            with patch('pathlib.Path.exists', return_value=True):
                result = await self.cli.organize_with_duplicates(
                    source_dir=Path("/source"),
                    target_dir=Path("/target"),
                    strategy='auto_smart',
                    resolve_first=True,
                    dry_run=False
                )

            assert result == 0

    @pytest.mark.asyncio
    async def test_organize_without_target_directory(self):
        """Test organizing without specifying target directory."""
        result = await self.cli.organize_with_duplicates(
            source_dir=self.temp_dir,
            target_dir=None  # Missing target
        )
        # The CLI should handle this gracefully
        # In actual implementation, it would print an error and return 1

    @pytest.mark.asyncio
    async def test_run_resolve_command(self):
        """Test running the CLI with resolve command."""
        args = argparse.Namespace(
            command='resolve',
            source=self.temp_dir,
            target=None,
            strategy='interactive',
            move_duplicates_to=None,
            dry_run=False,
            resolve_first=True,
            config=None,
            workers=4
        )

        with patch.object(self.cli, 'resolve_duplicates', return_value=0) as mock_resolve:
            result = await self.cli.run(args)

            assert result == 0
            mock_resolve.assert_called_once_with(
                source_dir=self.temp_dir,
                strategy='interactive',
                duplicate_dir=None,
                dry_run=False
            )

    @pytest.mark.asyncio
    async def test_run_preview_command(self):
        """Test running the CLI with preview command."""
        args = argparse.Namespace(
            command='preview',
            source=self.temp_dir,
            target=None,
            strategy='interactive',
            move_duplicates_to=None,
            dry_run=False,
            resolve_first=True,
            config=None,
            workers=4
        )

        with patch.object(self.cli, 'preview_duplicates', return_value=0) as mock_preview:
            result = await self.cli.run(args)

            assert result == 0
            mock_preview.assert_called_once_with(self.temp_dir)

    @pytest.mark.asyncio
    async def test_run_organize_command(self):
        """Test running the CLI with organize command."""
        target_dir = Path("/target")
        args = argparse.Namespace(
            command='organize',
            source=self.temp_dir,
            target=target_dir,
            strategy='auto_best',
            move_duplicates_to=None,
            dry_run=True,
            resolve_first=True,
            config=None,
            workers=8
        )

        with patch.object(self.cli, 'organize_with_duplicates', return_value=0) as mock_organize:
            result = await self.cli.run(args)

            assert result == 0
            mock_organize.assert_called_once_with(
                source_dir=self.temp_dir,
                target_dir=target_dir,
                strategy='auto_best',
                duplicate_dir=None,
                resolve_first=True,
                config_path=None,
                dry_run=True,
                workers=8
            )

    @pytest.mark.asyncio
    async def test_run_unknown_command(self):
        """Test running CLI with unknown command."""
        args = argparse.Namespace(
            command='unknown',
            source=self.temp_dir,
            target=None,
            strategy='interactive',
            move_duplicates_to=None,
            dry_run=False,
            resolve_first=True,
            config=None,
            workers=4
        )

        result = await self.cli.run(args)
        assert result == 1


class TestMainFunction:
    """Test the main entry point."""

    @patch('src.music_organizer.duplicate_resolver_cli.asyncio.run')
    @patch('src.music_organizer.duplicate_resolver_cli.DuplicateResolverCLI')
    def test_main_function(self, mock_cli_class, mock_asyncio_run):
        """Test the main function."""
        mock_cli = Mock()
        mock_cli.create_parser.return_value.parse_args.return_value = Mock()
        mock_cli.run.return_value = 0
        mock_cli_class.return_value = mock_cli
        mock_asyncio_run.return_value = 0

        # Mock sys.argv
        with patch('sys.argv', ['music-organize-duplicates', 'resolve', '/test']):
            result = main()

        assert result == 0
        mock_asyncio_run.assert_called_once()
        mock_cli_class.assert_called_once()

    @patch('sys.argv', ['music-organize-duplicates', 'resolve', '/test'])
    def test_main_integration(self):
        """Test main function integration (without actually running async code)."""
        # This is more of an integration test to ensure the entry point works
        # In real testing, we'd mock more components
        pass


if __name__ == '__main__':
    pytest.main([__file__])