"""Tests for metadata extraction."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from music_organizer.core.metadata import MetadataHandler
from music_organizer.models.audio_file import AudioFile, ContentType
from music_organizer.exceptions import MetadataError


class TestMetadataHandler:
    """Test metadata extraction functionality."""

    def test_extract_flac_metadata(self):
        """Test FLAC metadata extraction."""
        # Create a mock FLAC file
        mock_flac = Mock()
        mock_flac.tags = {
            'ARTIST': ['Test Artist'],
            'ALBUM': ['Test Album'],
            'TITLE': ['Test Track'],
            'DATE': ['2023'],
            'TRACKNUMBER': ['01'],
            'GENRE': ['Rock']
        }
        mock_flac.pictures = [Mock()]  # Has cover art

        with patch('music_organizer.core.metadata.FLAC') as mock_flac_class, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1000000
            mock_flac_class.return_value = mock_flac

            handler = MetadataHandler()
            audio_file = AudioFile.from_path(Path("/test/file.flac"))

            result = handler._extract_flac_metadata(audio_file, mock_flac)

            assert result.artists == ['Test Artist']
            assert result.album == ['Test Album']  # _get_single_field returns a list
            assert result.title == ['Test Track']
            assert result.year == 2023
            assert result.track_number == 1
            assert result.genre == ['Rock']
            assert result.has_cover_art

    def test_extract_mp3_metadata(self):
        """Test MP3 metadata extraction."""
        # Create mock ID3 frames
        mock_frame = Mock()
        mock_frame.text = ['Test Artist']

        mock_id3 = Mock()
        mock_id3.tags = {
            'TPE1': mock_frame,  # Artist
            'TALB': Mock(text=['Test Album']),  # Album
            'TIT2': Mock(text=['Test Track']),  # Title
            'TRCK': Mock(text=['5']),  # Track
            'TDRC': Mock(text=['2023']),  # Year
            'TCON': Mock(text=['Rock']),  # Genre
        }

        with patch('music_organizer.core.metadata.ID3') as mock_id3_class, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 5000000
            mock_id3_class.return_value = mock_id3

            handler = MetadataHandler()
            audio_file = AudioFile.from_path(Path("/test/file.mp3"))

            result = handler._extract_id3_metadata(audio_file, mock_id3)

            assert result.artists == ['Test Artist']
            assert result.album == 'Test Album'
            assert result.title == 'Test Track'
            assert result.track_number == 5

    def test_parse_artists(self):
        """Test artist string parsing."""
        test_cases = [
            ("Artist1 feat. Artist2", (["Artist1", "Artist2"], "Artist1")),
            ("Artist1 & Artist2", (["Artist1", "Artist2"], "Artist1")),
            ("Artist1 x Artist2", (["Artist1", "Artist2"], "Artist1")),
            ("Single Artist", (["Single Artist"], "Single Artist")),
            ("", ([], None)),
        ]

        handler = MetadataHandler()
        for artist_string, expected in test_cases:
            result = handler.parse_artists(artist_string)
            assert result == expected

    def test_clean_artist_name(self):
        """Test artist name cleaning."""
        test_cases = [
            ("The Beatles", "Beatles"),
            ("  Artist Name  ", "Artist Name"),
            ("Artist1 feat. Artist2", "Artist1 feat. Artist2"),
            ("ARTIST NAME", "ARTIST NAME"),  # Preserves case for standardization elsewhere
        ]

        for input_name, expected in test_cases:
            result = MetadataHandler._clean_artist_name(input_name)
            assert result == expected

    def test_standardize_genre(self):
        """Test genre standardization."""
        test_cases = [
            ("rock & roll", "Rock"),
            ("R&B", "R&B"),
            ("hip-hop", "Hip Hop"),
            ("ELECTRONICA", "Electronic"),
            ("Jazz", "Jazz"),  # Already standard
            ("", ""),
        ]

        for input_genre, expected in test_cases:
            result = MetadataHandler._standardize_genre(input_genre)
            assert result == expected

    def test_find_cover_art(self, tmp_path):
        """Test cover art detection."""
        # Create test files
        (tmp_path / "folder.jpg").touch()
        (tmp_path / "front.png").touch()
        (tmp_path / "album.jpg").touch()
        (tmp_path / "song.mp3").touch()  # Not cover art

        cover_files = MetadataHandler.find_cover_art(tmp_path)

        # Should find 3 cover art files
        assert len(cover_files) == 3
        assert any("folder.jpg" in str(f) for f in cover_files)
        assert any("front.png" in str(f) for f in cover_files)
        assert any("album.jpg" in str(f) for f in cover_files)

    def test_extract_metadata_unsupported_file(self):
        """Test handling of unsupported file types."""
        handler = MetadataHandler()

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch('music_organizer.core.metadata.MutagenFile', return_value=None):
            mock_stat.return_value.st_size = 1000

            with pytest.raises(MetadataError, match="Unsupported file format"):
                handler.extract_metadata(Path("/test/file.xyz"))

    def test_extract_metadata_missing_file(self):
        """Test handling of missing files."""
        handler = MetadataHandler()

        with pytest.raises(MetadataError, match="File does not exist"):
            handler.extract_metadata(Path("/nonexistent/file.flac"))