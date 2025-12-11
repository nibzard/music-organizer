"""Tests for the dashboard CLI module."""

import pytest
import argparse
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from music_organizer.dashboard_cli import create_parser, run_dashboard, main


class TestDashboardCLI:
    """Test dashboard CLI functionality."""

    @pytest.fixture
    def temp_library(self):
        """Create a temporary directory with mock music files."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create mock directory structure
        artists = ["The Beatles", "Led Zeppelin", "Pink Floyd"]
        for artist in artists:
            artist_dir = temp_dir / artist
            artist_dir.mkdir()

            albums = ["Album 1", "Album 2"]
            for album in albums:
                album_dir = artist_dir / album
                album_dir.mkdir()

                # Create mock audio files
                for track in range(1, 6):
                    track_file = album_dir / f"Track {track:02d}.mp3"
                    track_file.write_bytes(b"mock audio data")

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_create_parser(self):
        """Test argument parser creation."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

        # Test parsing required argument
        args = parser.parse_args(["/path/to/music"])
        assert args.library_path == Path("/path/to/music")

    def test_parser_with_options(self):
        """Test parsing with various options."""
        parser = create_parser()

        # Test with overview option
        args = parser.parse_args(["/path/to/music", "--overview"])
        assert args.overview is True

        # Test with artist option
        args = parser.parse_args(["/path/to/music", "--artist", "The Beatles"])
        assert args.artist == "The Beatles"

        # Test with export option
        args = parser.parse_args(["/path/to/music", "--export", "stats.json"])
        assert args.export == "stats.json"

        # Test with format option
        args = parser.parse_args(["/path/to/music", "--export", "stats.csv", "--format", "csv"])
        assert args.format == "csv"

        # Test with max items
        args = parser.parse_args(["/path/to/music", "--max-items", "20"])
        assert args.max_items == 20

        # Test with charts
        args = parser.parse_args(["/path/to/music", "--charts"])
        assert args.charts is True

        # Test with no details
        args = parser.parse_args(["/path/to/music", "--no-details"])
        assert args.no_details is True

        # Test with interactive
        args = parser.parse_args(["/path/to/music", "--interactive"])
        assert args.interactive is True

    @pytest.mark.asyncio
    async def test_run_dashboard_invalid_path(self):
        """Test dashboard with invalid library path."""
        parser = create_parser()
        args = parser.parse_args(["/nonexistent/path"])

        result = await run_dashboard(args)
        assert result == 1

    @pytest.mark.asyncio
    async def test_run_dashboard_not_directory(self, temp_library):
        """Test dashboard with non-directory path."""
        parser = create_parser()
        args = parser.parse_args([str(temp_library / "The Beatles")])

        result = await run_dashboard(args)
        assert result == 1

    @pytest.mark.asyncio
    async def test_run_dashboard_overview(self, temp_library):
        """Test dashboard overview mode."""
        parser = create_parser()
        args = parser.parse_args([str(temp_library), "--overview"])

        with patch('music_organizer.dashboard.StatisticsDashboard') as mock_dashboard_class:
            mock_dashboard = MagicMock()
            mock_dashboard_class.return_value = mock_dashboard

            result = await run_dashboard(args)

            # Verify dashboard was created and initialized
            mock_dashboard_class.assert_called_once()
            mock_dashboard.initialize.assert_called_once_with(temp_library)
            mock_dashboard.show_library_overview.assert_called_once()

            assert result == 0

    @pytest.mark.asyncio
    async def test_run_dashboard_artist_analysis(self, temp_library):
        """Test dashboard artist analysis mode."""
        parser = create_parser()
        args = parser.parse_args([str(temp_library), "--artist", "The Beatles"])

        with patch('music_organizer.dashboard.StatisticsDashboard') as mock_dashboard_class:
            mock_dashboard = MagicMock()
            mock_dashboard_class.return_value = mock_dashboard

            result = await run_dashboard(args)

            # Verify artist analysis was called
            mock_dashboard.show_artist_details.assert_called_once_with("The Beatles")

            assert result == 0

    @pytest.mark.asyncio
    async def test_run_dashboard_export_json(self, temp_library):
        """Test dashboard export to JSON."""
        parser = create_parser()
        export_path = temp_library / "stats.json"
        args = parser.parse_args([str(temp_library), "--export", str(export_path)])

        with patch('music_organizer.dashboard.StatisticsDashboard') as mock_dashboard_class:
            mock_dashboard = MagicMock()
            mock_dashboard_class.return_value = mock_dashboard

            result = await run_dashboard(args)

            # Verify export was called
            mock_dashboard.export_statistics.assert_called_once_with(export_path)

            assert result == 0

    @pytest.mark.asyncio
    async def test_run_dashboard_export_csv(self, temp_library):
        """Test dashboard export to CSV."""
        parser = create_parser()
        export_path = temp_library / "stats.csv"
        args = parser.parse_args([str(temp_library), "--export", str(export_path), "--format", "csv"])

        with patch('music_organizer.dashboard.StatisticsDashboard') as mock_dashboard_class:
            mock_dashboard = MagicMock()
            mock_config = MagicMock()
            mock_dashboard_class.return_value = mock_dashboard
            mock_dashboard.config = mock_config

            result = await run_dashboard(args)

            # Verify config was updated and export was called
            assert mock_config.export_format == "csv"
            mock_dashboard.export_statistics.assert_called_once_with(export_path)

            assert result == 0

    @pytest.mark.asyncio
    async def test_run_dashboard_interactive(self, temp_library):
        """Test dashboard interactive mode."""
        parser = create_parser()
        args = parser.parse_args([str(temp_library), "--interactive"])

        with patch('music_organizer.dashboard.StatisticsDashboard') as mock_dashboard_class:
            mock_dashboard = MagicMock()
            mock_dashboard_class.return_value = mock_dashboard

            result = await run_dashboard(args)

            # Verify interactive mode was called
            mock_dashboard.interactive_dashboard.assert_called_once_with(temp_library)

            assert result == 0

    @pytest.mark.asyncio
    async def test_run_dashboard_default_behavior(self, temp_library):
        """Test dashboard default behavior (interactive mode)."""
        parser = create_parser()
        args = parser.parse_args([str(temp_library)])  # No specific options

        with patch('music_organizer.dashboard.StatisticsDashboard') as mock_dashboard_class:
            mock_dashboard = MagicMock()
            mock_dashboard_class.return_value = mock_dashboard

            result = await run_dashboard(args)

            # Should default to interactive mode
            mock_dashboard.interactive_dashboard.assert_called_once_with(temp_library)

            assert result == 0

    @pytest.mark.asyncio
    async def test_run_dashboard_keyboard_interrupt(self, temp_library):
        """Test dashboard with keyboard interrupt."""
        parser = create_parser()
        args = parser.parse_args([str(temp_library), "--overview"])

        with patch('music_organizer.dashboard.StatisticsDashboard') as mock_dashboard_class:
            mock_dashboard = MagicMock()
            mock_dashboard.initialize.side_effect = KeyboardInterrupt()

            result = await run_dashboard(args)
            assert result == 1

    @pytest.mark.asyncio
    async def test_run_dashboard_exception(self, temp_library):
        """Test dashboard with exception."""
        parser = create_parser()
        args = parser.parse_args([str(temp_library), "--overview"])

        with patch('music_organizer.dashboard.StatisticsDashboard') as mock_dashboard_class:
            mock_dashboard = MagicMock()
            mock_dashboard.initialize.side_effect = Exception("Test error")

            result = await run_dashboard(args)
            assert result == 1

    @pytest.mark.asyncio
    async def test_run_dashboard_quality_analysis(self, temp_library):
        """Test dashboard quality analysis mode."""
        parser = create_parser()
        args = parser.parse_args([str(temp_library), "--quality-only"])

        with patch('music_organizer.dashboard.StatisticsDashboard') as mock_dashboard_class:
            mock_dashboard = MagicMock()
            mock_dashboard_class.return_value = mock_dashboard

            with patch('music_organizer.dashboard_cli.SimpleConsole') as mock_console:
                result = await run_dashboard(args)

                # Verify dashboard was initialized
                mock_dashboard.initialize.assert_called_once()
                assert result == 0

    @pytest.mark.asyncio
    async def test_run_dashboard_genre_analysis(self, temp_library):
        """Test dashboard genre analysis mode."""
        parser = create_parser()
        args = parser.parse_args([str(temp_library), "--genre-analysis"])

        with patch('music_organizer.dashboard.StatisticsDashboard') as mock_dashboard_class:
            mock_dashboard = MagicMock()
            mock_dashboard_class.return_value = mock_dashboard

            with patch('music_organizer.dashboard_cli.SimpleConsole') as mock_console:
                result = await run_dashboard(args)

                # Verify dashboard was initialized
                mock_dashboard.initialize.assert_called_once()
                assert result == 0

    @pytest.mark.asyncio
    async def test_run_dashboard_format_analysis(self, temp_library):
        """Test dashboard format analysis mode."""
        parser = create_parser()
        args = parser.parse_args([str(temp_library), "--format-only"])

        with patch('music_organizer.dashboard.StatisticsDashboard') as mock_dashboard_class:
            mock_dashboard = MagicMock()
            mock_dashboard_class.return_value = mock_dashboard

            with patch('music_organizer.dashboard_cli.SimpleConsole') as mock_console:
                result = await run_dashboard(args)

                # Verify dashboard was initialized
                mock_dashboard.initialize.assert_called_once()
                assert result == 0

    def test_parser_help_message(self):
        """Test parser help message includes examples."""
        parser = create_parser()
        help_text = parser.format_help()

        # Check that help contains examples
        assert "Examples:" in help_text
        assert "--overview" in help_text
        assert "--artist" in help_text
        assert "--export" in help_text
        assert "--interactive" in help_text

    def test_main_function(self):
        """Test main function entry point."""
        test_args = ["music-organize-dashboard", "/path/to/music", "--overview"]

        with patch('sys.argv', test_args):
            with patch('music_organizer.dashboard_cli.asyncio.run') as mock_run:
                mock_run.return_value = 0

                with patch('sys.exit') as mock_exit:
                    main()

                    mock_exit.assert_called_once_with(0)
                    mock_run.assert_called_once()


class TestDashboardCLIIntegration:
    """Integration tests for dashboard CLI."""

    @pytest.fixture
    def real_temp_library(self):
        """Create a real temporary library with actual files."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create a realistic music library structure
        artists_data = {
            "The Beatles": {
                "Abbey Road (1969)": ["Come Together", "Something", "Here Comes the Sun"],
                "Sgt. Pepper's (1967)": ["Lucy in the Sky", "With a Little Help", "Day in the Life"]
            },
            "Led Zeppelin": {
                "Led Zeppelin IV (1971)": ["Black Dog", "Rock and Roll", "Stairway to Heaven"],
                "Physical Graffiti (1975)": ["Kashmir", "Trampled Under Foot", "Houses of the Holy"]
            }
        }

        for artist, albums in artists_data.items():
            artist_dir = temp_dir / artist
            artist_dir.mkdir()

            for album, tracks in albums.items():
                album_dir = artist_dir / album
                album_dir.mkdir()

                for track in tracks:
                    track_file = album_dir / f"{track}.mp3"
                    track_file.write_bytes(b"mock audio data for testing")

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_full_cli_integration(self, real_temp_library):
        """Test full CLI integration with real files."""
        # This test verifies the CLI can handle real files
        parser = create_parser()
        args = parser.parse_args([str(real_temp_library), "--overview"])

        # We'll mock the expensive parts but verify the flow works
        with patch('music_organizer.dashboard.AsyncMusicOrganizer') as mock_organizer:
            mock_organizer.return_value.extract_metadata_async.return_value = MagicMock()
            mock_organizer.return_value.extract_metadata_async.return_value.metadata = {
                'title': 'Test Track',
                'artist': 'Test Artist',
                'album': 'Test Album',
                'year': 2023,
                'genre': 'Rock'
            }

            # Mock the async function
            import asyncio
            result = asyncio.run(run_dashboard(args))

            # Should complete without error
            assert result == 0