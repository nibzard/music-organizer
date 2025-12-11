"""Tests for the statistics dashboard module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from music_organizer.dashboard import (
    StatisticsDashboard,
    DashboardConfig,
    SimpleProgress
)
from music_organizer.application.queries.catalog.statistics_queries import (
    LibraryStatistics,
    ArtistStatistics
)


class TestDashboardConfig:
    """Test DashboardConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DashboardConfig()
        assert config.include_charts is False
        assert config.max_top_items == 10
        assert config.show_file_details is True
        assert config.show_quality_analysis is True
        assert config.export_format == "json"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DashboardConfig(
            include_charts=True,
            max_top_items=20,
            show_file_details=False,
            show_quality_analysis=False,
            export_format="csv"
        )
        assert config.include_charts is True
        assert config.max_top_items == 20
        assert config.show_file_details is False
        assert config.show_quality_analysis is False
        assert config.export_format == "csv"


@pytest.fixture
def mock_library_stats():
    """Create mock library statistics for testing."""
    return LibraryStatistics(
        total_recordings=1000,
        total_artists=150,
        total_releases=250,
        total_size_gb=50.5,
        total_duration_hours=60.0,
        format_distribution={"FLAC": 400, "MP3": 350, "WAV": 150, "OGG": 100},
        genre_distribution={"Rock": 300, "Jazz": 200, "Classical": 150, "Pop": 250, "Electronic": 100},
        decade_distribution={"1960s": 100, "1970s": 150, "1980s": 200, "1990s": 250, "2000s": 200, "2010s": 100},
        top_artists=[("The Beatles", 50),("Led Zeppelin", 45),("Pink Floyd", 40),("Queen", 35)],
        top_genres=[("Rock", 300),("Pop", 250),("Jazz", 200),("Classical", 150),("Electronic", 100)],
        recently_added=50,
        duplicates_count=25,
        average_bitrate=256.5,
        quality_distribution={
            "High (320+ kbps)": 200,
            "Good (256-319 kbps)": 300,
            "Standard (192-255 kbps)": 350,
            "Low (128-191 kbps)": 100,
            "Very Low (<128 kbps)": 50
        }
    )


@pytest.fixture
def mock_artist_stats():
    """Create mock artist statistics for testing."""
    return ArtistStatistics(
        artist_name="The Beatles",
        total_recordings=50,
        total_releases=12,
        total_duration_hours=3.5,
        genres=["Rock", "Pop", "Psychedelic Rock"],
        decade_distribution={"1960s": 40, "1970s": 10},
        average_year=1967.5,
        top_releases=[("Abbey Road", 17),("Sgt. Pepper's", 13),("Revolver", 14)],
        collaborations=["Billy Preston", "Eric Clapton", "Tony Sheridan"]
    )


class TestStatisticsDashboard:
    """Test StatisticsDashboard class."""

    @pytest.fixture
    def dashboard(self):
        """Create a dashboard instance for testing."""
        config = DashboardConfig(
            include_charts=True,
            max_top_items=5,
            show_file_details=True,
            show_quality_analysis=True,
            export_format="json"
        )
        return StatisticsDashboard(config)

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
                for track in range(1, 11):
                    track_file = album_dir / f"Track {track:02d}.mp3"
                    track_file.write_bytes(b"mock audio data")

        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_dashboard_initialization(self, dashboard, temp_library):
        """Test dashboard initialization."""
        await dashboard.initialize(temp_library)

        assert dashboard.query_bus is not None
        assert dashboard.recording_repo is not None
        assert dashboard.catalog_repo is not None

    @pytest.mark.asyncio
    async def test_library_overview_display(self, dashboard, mock_library_stats):
        """Test library overview display."""
        # Mock the query bus
        dashboard.query_bus = AsyncMock()
        dashboard.query_bus.dispatch.return_value = mock_library_stats

        # Capture print output
        with patch('builtins.print') as mock_print:
            await dashboard.show_library_overview()

            # Verify that print was called multiple times (sections were printed)
            assert mock_print.call_count > 10

    def test_print_overview_section(self, dashboard, mock_library_stats):
        """Test overview section printing."""
        with patch('music_organizer.dashboard.SimpleConsole') as mock_console:
            dashboard._print_overview_section(mock_library_stats)

            # Verify console methods were called
            mock_console.return_value.print.assert_called()
            mock_console.return_value.table.assert_called()

    def test_print_format_section(self, dashboard, mock_library_stats):
        """Test format section printing."""
        with patch('music_organizer.dashboard.SimpleConsole') as mock_console:
            dashboard._print_format_section(mock_library_stats)

            # Verify table was called with format data
            mock_console.return_value.table.assert_called()
            call_args = mock_console.return_value.table.call_args[0]
            assert "Format" in call_args[1]
            assert "Count" in call_args[1]

    def test_print_genre_section(self, dashboard, mock_library_stats):
        """Test genre section printing."""
        with patch('music_organizer.dashboard.SimpleConsole') as mock_console:
            dashboard._print_genre_section(mock_library_stats)

            # Verify table was called with genre data
            mock_console.return_value.table.assert_called()

    def test_print_quality_section(self, dashboard, mock_library_stats):
        """Test quality section printing."""
        with patch('music_organizer.dashboard.SimpleConsole') as mock_console:
            dashboard._print_quality_section(mock_library_stats)

            # Verify table was called with quality data
            mock_console.return_value.table.assert_called()
            call_args = mock_console.return_value.table.call_args[0]
            assert "Quality" in call_args[1]

    def test_print_artists_section(self, dashboard, mock_library_stats):
        """Test artists section printing."""
        with patch('music_organizer.dashboard.SimpleConsole') as mock_console:
            dashboard._print_artists_section(mock_library_stats)

            # Verify table was called with artist data
            mock_console.return_value.table.assert_called()
            call_args = mock_console.return_value.table.call_args[0]
            assert "Artist" in call_args[1]

    def test_print_temporal_section(self, dashboard, mock_library_stats):
        """Test temporal section printing."""
        with patch('music_organizer.dashboard.SimpleConsole') as mock_console:
            dashboard._print_temporal_section(mock_library_stats)

            # Verify table was called with decade data
            mock_console.return_value.table.assert_called()

    def test_print_file_details_section(self, dashboard, mock_library_stats):
        """Test file details section printing."""
        with patch('music_organizer.dashboard.SimpleConsole') as mock_console:
            dashboard._print_file_details_section(mock_library_stats)

            # Verify table was called with file data
            mock_console.return_value.table.assert_called()
            call_args = mock_console.return_value.table.call_args[0]
            assert "Metric" in call_args[1]

    @pytest.mark.asyncio
    async def test_show_artist_details(self, dashboard, mock_artist_stats):
        """Test artist details display."""
        dashboard.query_bus = AsyncMock()
        dashboard.query_bus.dispatch.return_value = mock_artist_stats

        with patch('music_organizer.dashboard.SimpleConsole') as mock_console:
            await dashboard.show_artist_details("The Beatles")

            # Verify console methods were called
            mock_console.return_value.rule.assert_called()
            mock_console.return_value.table.assert_called()

    @pytest.mark.asyncio
    async def test_show_artist_details_not_found(self, dashboard):
        """Test artist details when artist not found."""
        empty_stats = ArtistStatistics(
            artist_name="Unknown Artist",
            total_recordings=0,
            total_releases=0,
            total_duration_hours=0.0,
            genres=[],
            decade_distribution={},
            average_year=None,
            top_releases=[],
            collaborations=[]
        )

        dashboard.query_bus = AsyncMock()
        dashboard.query_bus.dispatch.return_value = empty_stats

        with patch('music_organizer.dashboard.SimpleConsole') as mock_console:
            await dashboard.show_artist_details("Unknown Artist")

            # Verify warning message was printed
            mock_console.return_value.print.assert_called_with(
                "Artist 'Unknown Artist' not found in library",
                style="yellow"
            )

    @pytest.mark.asyncio
    async def test_export_statistics_json(self, dashboard, mock_library_stats):
        """Test exporting statistics to JSON."""
        import json

        dashboard.query_bus = AsyncMock()
        dashboard.query_bus.dispatch.return_value = mock_library_stats

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)

        try:
            await dashboard.export_statistics(output_path)

            # Verify file was created and contains valid JSON
            assert output_path.exists()
            with open(output_path, 'r') as f:
                data = json.load(f)

            assert "generated_at" in data
            assert "library_overview" in data
            assert "format_distribution" in data
            assert data["library_overview"]["total_recordings"] == 1000

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_export_statistics_csv(self, dashboard, mock_library_stats):
        """Test exporting statistics to CSV."""
        import csv

        dashboard.config.export_format = "csv"
        dashboard.query_bus = AsyncMock()
        dashboard.query_bus.dispatch.return_value = mock_library_stats

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = Path(f.name)

        try:
            await dashboard.export_statistics(output_path)

            # Verify file was created and contains CSV data
            assert output_path.exists()
            with open(output_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) > 0  # Should have header and data rows

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_export_statistics_txt(self, dashboard, mock_library_stats):
        """Test exporting statistics to text."""
        dashboard.config.export_format = "txt"
        dashboard.query_bus = AsyncMock()
        dashboard.query_bus.dispatch.return_value = mock_library_stats

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = Path(f.name)

        try:
            await dashboard.export_statistics(output_path)

            # Verify file was created and contains text data
            assert output_path.exists()
            with open(output_path, 'r') as f:
                content = f.read()

            assert "MUSIC LIBRARY STATISTICS REPORT" in content
            assert "library_overview" in content

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_not_initialized_error(self, dashboard):
        """Test error when dashboard not initialized."""
        with patch('music_organizer.dashboard.SimpleConsole') as mock_console:
            await dashboard.show_library_overview()

            # Verify error message was printed
            mock_console.return_value.print.assert_called_with(
                "Dashboard not initialized. Call initialize() first.",
                style="red"
            )


class TestSimpleProgress:
    """Test SimpleProgress class."""

    def test_progress_initialization(self):
        """Test progress bar initialization."""
        progress = SimpleProgress(total=100, description="Test")
        assert progress.total == 100
        assert progress.current == 0
        assert progress.description == "Test"

    def test_progress_update(self):
        """Test progress bar update."""
        progress = SimpleProgress(total=100, description="Test")

        with patch('builtins.print') as mock_print:
            progress.update(10)
            assert progress.current == 10
            mock_print.assert_called()

    def test_progress_completion(self):
        """Test progress bar completion."""
        progress = SimpleProgress(total=100, description="Test")

        with patch('builtins.print') as mock_print:
            progress.update(100)
            assert progress.current == 100

            # Should have printed newline at the end
            final_call = mock_print.call_args_list[-1]
            assert final_call[0][0] == '\n'


@pytest.mark.asyncio
async def test_dashboard_integration():
    """Integration test for the complete dashboard workflow."""
    # Create a temporary library with known structure
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create test files
        test_files = [
            "Artist1/Album1/track1.mp3",
            "Artist1/Album1/track2.mp3",
            "Artist2/Album1/track1.flac",
            "Artist3/Album1/track1.wav",
        ]

        for file_path in test_files:
            full_path = temp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_bytes(b"mock audio data")

        # Test dashboard initialization and basic operations
        dashboard = StatisticsDashboard()
        await dashboard.initialize(temp_dir)

        # Verify query bus is initialized
        assert dashboard.query_bus is not None

        # Test library statistics query
        stats = await dashboard.query_bus.dispatch(
            dashboard.GetLibraryStatisticsQuery(include_detailed_breakdown=True)
        )

        # Verify basic statistics
        assert stats.total_recordings == 4
        assert stats.total_artists == 3
        assert stats.total_releases == 3

    finally:
        shutil.rmtree(temp_dir)