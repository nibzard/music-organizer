"""Tests for new audio format support (OGG, OPUS, WMA, AIFF)."""

import pytest
from pathlib import Path
from src.music_organizer.models.audio_file import AudioFile
from src.music_organizer.core.metadata import MetadataHandler
from src.music_organizer.exceptions import MetadataError


class TestNewAudioFormats:
    """Test support for additional audio formats."""

    def test_audio_file_format_recognition(self):
        """Test that AudioFile correctly identifies new formats."""
        formats_to_test = [
            ('.ogg', 'OGG'),
            ('.opus', 'OPUS'),
            ('.wma', 'WMA'),
            ('.aiff', 'AIFF'),
            ('.aif', 'AIFF'),
        ]

        for ext, expected_type in formats_to_test:
            # Create a temporary file with the extension
            test_file = Path(f"/fake/path/test{ext}")

            # Mock the file existence check
            original_exists = Path.exists
            Path.exists = lambda self: True if self == test_file else original_exists(self)

            try:
                original_is_file = Path.is_file
                Path.is_file = lambda self: True if self == test_file else original_is_file(self)

                # Mock file size
                original_stat = Path.stat
                def mock_stat(self):
                    if self == test_file:
                        import os
                        result = original_stat(self)
                        # Create a mock stat result with required attributes
                        class MockStat:
                            def __init__(self):
                                self.st_size = 1000
                                self.st_mtime = 1234567890
                        return MockStat()
                    return original_stat(self)

                Path.stat = mock_stat

                # Test AudioFile.from_path
                audio_file = AudioFile.from_path(test_file)
                assert audio_file.file_type == expected_type, f"Expected {expected_type}, got {audio_file.file_type}"

            finally:
                # Restore original methods
                Path.exists = original_exists
                Path.is_file = original_is_file
                Path.stat = original_stat

    def test_metadata_extraction_methods_exist(self):
        """Test that metadata extraction methods are available for new formats."""
        # Check that methods exist
        assert hasattr(MetadataHandler, '_extract_ogg_metadata')
        assert hasattr(MetadataHandler, '_extract_wma_metadata')
        assert hasattr(MetadataHandler, '_get_wma_field')

    def test_extract_ogg_metadata_method(self):
        """Test OGG metadata extraction method signature."""
        # Create a mock AudioFile and ogg_file
        audio_file = AudioFile(path=Path("/test.ogg"), file_type="OGG")

        # Mock OGG file with tags
        class MockOggFile:
            def __init__(self):
                self.tags = {
                    'ARTIST': ['Test Artist'],
                    'ALBUM': ['Test Album'],
                    'TITLE': ['Test Title'],
                    'GENRE': ['Rock'],
                    'DATE': ['2023'],
                    'TRACKNUMBER': ['5']
                }

        ogg_file = MockOggFile()

        # Test extraction
        result = MetadataHandler._extract_ogg_metadata(audio_file, ogg_file)

        assert result.artists == ['Test Artist']
        assert result.album == 'Test Album'
        assert result.title == 'Test Title'
        assert result.genre == 'Rock'
        assert result.year == 2023
        assert result.track_number == 5

    def test_extract_wma_metadata_method(self):
        """Test WMA metadata extraction method signature."""
        # Create a mock AudioFile and wma_file
        audio_file = AudioFile(path=Path("/test.wma"), file_type="WMA")

        # Mock WMA file with tags
        class MockWmaFile:
            def __init__(self):
                self.tags = {
                    'Author': ['Test Artist'],
                    'AlbumTitle': ['Test Album'],
                    'Title': ['Test Title'],
                    'Genre': ['Rock'],
                    'Year': ['2023'],
                    'TrackNumber': ['5']
                }

        wma_file = MockWmaFile()

        # Test extraction
        result = MetadataHandler._extract_wma_metadata(audio_file, wma_file)

        assert result.artists == ['Test Artist']
        assert result.album == 'Test Album'
        assert result.title == 'Test Title'
        assert result.genre == 'Rock'
        assert result.year == 2023
        assert result.track_number == 5

    def test_mutagen_import_handling(self):
        """Test that mutagen format modules are handled gracefully."""
        # This test verifies that the code doesn't crash if mutagen modules are missing
        # We can't easily test this without manipulating imports, but we can verify
        # the imports are wrapped in try/except blocks

        from src.music_organizer.core import metadata

        # Check that the module imports exist and are either available or None
        assert hasattr(metadata, 'OggVorbis')
        assert hasattr(metadata, 'OggOpus')
        assert hasattr(metadata, 'WMA')

        # These might be None if the modules aren't available, which is fine
        # The important thing is that the code handles it gracefully