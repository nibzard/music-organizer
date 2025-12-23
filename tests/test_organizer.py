"""Tests for core music organizer functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any

from music_organizer.core.organizer import MusicOrganizer
from music_organizer.models.audio_file import AudioFile, ContentType, CoverArt
from music_organizer.models.config import Config
from music_organizer.domain.value_objects import FileFormat, ArtistName, Metadata
from music_organizer.exceptions import MusicOrganizerError


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample configuration for testing."""
    return Config(
        source_directory=tmp_path / "source",
        target_directory=tmp_path / "target"
    )


@pytest.fixture
def sample_audio_files(tmp_path) -> List[AudioFile]:
    """Create sample audio files for testing."""
    files = []

    # Create source directory
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    # Create a few audio files with proper metadata
    for i in range(3):
        file_path = source_dir / f"song{i+1}.flac"
        file_path.write_bytes(b"fake audio data")

        audio_file = AudioFile(
            path=file_path,
            file_type="FLAC",
            metadata={},
            content_type=ContentType.STUDIO,
            artists=["Test Artist"],
            primary_artist="Test Artist",
            album="Test Album",
            title=f"Test Song {i+1}",
            year=2020,
            genre="Rock",
            track_number=i+1
        )
        files.append(audio_file)

    return files


class TestMusicOrganizer:
    """Test MusicOrganizer class."""

    @pytest.fixture
    def organizer(self, sample_config):
        """Create a MusicOrganizer instance."""
        return MusicOrganizer(config=sample_config)

    def test_initialization(self, organizer, sample_config):
        """Test organizer initialization."""
        assert organizer.config == sample_config
        assert organizer.dry_run is False
        assert organizer.interactive is False
        assert organizer.user_decisions == {}
        assert organizer.metadata_handler is not None
        assert organizer.classifier is not None
        assert organizer.file_mover is not None

    def test_initialization_with_options(self, sample_config):
        """Test initialization with dry_run and interactive options."""
        organizer = MusicOrganizer(
            config=sample_config,
            dry_run=True,
            interactive=True
        )

        assert organizer.dry_run is True
        assert organizer.interactive is True

    def test_scan_directory_non_existent(self, organizer):
        """Test scanning a non-existent directory."""
        with pytest.raises(MusicOrganizerError, match="Directory does not exist"):
            organizer.scan_directory(Path("/non/existent/path"))

    def test_scan_directory_not_a_directory(self, organizer, tmp_path):
        """Test scanning a path that is not a directory."""
        # Create a file instead of a directory
        file_path = tmp_path / "not_a_directory.txt"
        file_path.write_text("test")

        with pytest.raises(MusicOrganizerError, match="Path is not a directory"):
            organizer.scan_directory(file_path)

    def test_scan_directory_success(self, organizer, sample_config, tmp_path):
        """Test successful directory scanning."""
        # Create audio files in source directory
        source_dir = sample_config.source_directory
        source_dir.mkdir(parents=True, exist_ok=True)

        # Create test audio files
        audio_extensions = ['.flac', '.mp3', '.wav']
        for i, ext in enumerate(audio_extensions):
            test_file = source_dir / f"test{i}{ext}"
            test_file.write_bytes(b"fake audio data")

        # Also create a non-audio file that should be ignored
        (source_dir / "readme.txt").write_text("This should be ignored")

        # Scan the directory
        files = organizer.scan_directory(source_dir)

        # Should find only audio files
        assert len(files) == 3
        assert all(f.suffix.lower() in audio_extensions for f in files)

    def test_scan_directory_recursive(self, organizer, sample_config):
        """Test recursive directory scanning."""
        source_dir = sample_config.source_directory
        source_dir.mkdir(parents=True, exist_ok=True)

        # Create nested directory structure
        subdir1 = source_dir / "artist1" / "album1"
        subdir1.mkdir(parents=True)
        subdir2 = source_dir / "artist2" / "album2"
        subdir2.mkdir(parents=True)

        # Create files in subdirectories
        (subdir1 / "song1.flac").write_bytes(b"data1")
        (subdir1 / "song2.flac").write_bytes(b"data2")
        (subdir2 / "song3.mp3").write_bytes(b"data3")

        # Scan should find all audio files recursively
        files = organizer.scan_directory(source_dir)

        assert len(files) == 3
        assert all(f.is_absolute() for f in files)

    @patch.object(MusicOrganizer, '_process_file')
    def test_organize_files_basic(self, mock_process_file, organizer, sample_audio_files):
        """Test basic file organization."""
        # Mock process_file to return success
        mock_audio_file = Mock(spec=AudioFile)
        mock_audio_file.content_type = ContentType.STUDIO
        mock_process_file.return_value = mock_audio_file

        # Create a list of file paths
        file_paths = [f.path for f in sample_audio_files]

        results = organizer.organize_files(file_paths)

        # Check results
        assert results['processed'] == 3
        assert results['moved'] == 3
        assert results['skipped'] == 0
        assert len(results['errors']) == 0
        assert results['by_category']['Albums'] == 3

        # Verify process_file was called for each file
        assert mock_process_file.call_count == 3

    @patch.object(MusicOrganizer, '_process_file')
    def test_organize_files_with_errors(self, mock_process_file, organizer, sample_audio_files):
        """Test file organization with some errors."""
        # Mock process_file to fail on second file
        mock_process_file.side_effect = [
            Mock(content_type=ContentType.STUDIO),
            Exception("Test error"),
            Mock(content_type=ContentType.LIVE)
        ]

        file_paths = [f.path for f in sample_audio_files]

        results = organizer.organize_files(file_paths)

        # Should process all files but record errors
        assert results['processed'] == 3
        assert results['moved'] == 2  # Two succeeded
        assert results['skipped'] == 0
        assert len(results['errors']) == 1
        assert "Test error" in results['errors'][0]

    @patch.object(MusicOrganizer, '_process_file')
    def test_organize_files_dry_run(self, mock_process_file, sample_config):
        """Test file organization in dry run mode."""
        organizer = MusicOrganizer(config=sample_config, dry_run=True)

        # Mock to return audio file
        mock_audio_file = Mock(spec=AudioFile)
        mock_audio_file.content_type = ContentType.STUDIO
        mock_audio_file.path = Path("/source/song.flac")
        mock_audio_file.get_target_path = Mock(return_value=Path("/target/Artist/Album"))
        mock_audio_file.get_target_filename = Mock(return_value="01 Song.flac")
        mock_process_file.return_value = mock_audio_file

        file_paths = [Path("/source/song.flac")]

        # Just verify it doesn't error in dry run mode
        results = organizer.organize_files(file_paths)

        # In dry run, should still process files
        assert results['processed'] == 1

    def test_group_files(self, organizer, sample_audio_files):
        """Test file grouping by directory."""
        file_paths = [f.path for f in sample_audio_files]

        # All files are in same directory, so should have one group
        groups = organizer._group_files(file_paths)

        assert len(groups) == 1
        assert sample_audio_files[0].path.parent in groups
        assert len(groups[sample_audio_files[0].path.parent]) == 3

    def test_group_files_multiple_directories(self, organizer, tmp_path):
        """Test grouping files from multiple directories."""
        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True)

        # Create files in different directories
        dir1 = source_dir / "artist1"
        dir2 = source_dir / "artist2"
        dir1.mkdir()
        dir2.mkdir()

        file1 = dir1 / "song1.flac"
        file2 = dir1 / "song2.flac"
        file3 = dir2 / "song3.flac"

        for f in [file1, file2, file3]:
            f.write_bytes(b"data")

        groups = organizer._group_files([file1, file2, file3])

        assert len(groups) == 2
        assert len(groups[dir1]) == 2
        assert len(groups[dir2]) == 1

    def test_process_file(self, organizer, tmp_path):
        """Test processing a single file."""
        # Create a real file
        file_path = tmp_path / "song.flac"
        file_path.write_bytes(b"fake audio data")

        # Create mock audio file
        mock_audio_file = AudioFile(
            path=file_path,
            file_type="FLAC",
            metadata={},
            content_type=ContentType.STUDIO,
            artists=["Test Artist"],
            primary_artist="Test Artist",
            album="Test Album",
            title="Test Song"
        )

        # Mock the metadata handler instance method
        organizer.metadata_handler.extract_metadata = Mock(return_value=mock_audio_file)

        # Mock classifier
        mock_classifier_instance = Mock()
        mock_classifier_instance.classify.return_value = (ContentType.STUDIO, 0.9)
        mock_classifier_instance.is_ambiguous.return_value = False
        organizer.classifier = mock_classifier_instance

        # Mock file_mover
        organizer.file_mover.move_file = Mock()
        organizer.file_mover.move_cover_art = Mock()

        result = organizer._process_file(file_path)

        assert result == mock_audio_file
        organizer.metadata_handler.extract_metadata.assert_called_once_with(file_path)
        mock_classifier_instance.classify.assert_called_once()

    def test_process_file_dry_run(self, sample_config, tmp_path):
        """Test processing a file in dry run mode."""
        organizer = MusicOrganizer(config=sample_config, dry_run=True)

        # Create a real file
        file_path = tmp_path / "song.flac"
        file_path.write_bytes(b"fake audio data")

        # Create a mock audio file using Mock to allow setting attributes
        mock_audio_file = Mock(spec=AudioFile)
        mock_audio_file.content_type = ContentType.STUDIO
        mock_audio_file.path = file_path
        # Return a target path that's actually within the config target directory
        target_dir = sample_config.target_directory / "Artist" / "Album"
        mock_audio_file.get_target_path = Mock(return_value=target_dir)
        mock_audio_file.get_target_filename = Mock(return_value="01 Song.flac")

        organizer.metadata_handler.extract_metadata = Mock(return_value=mock_audio_file)

        mock_classifier_instance = Mock()
        mock_classifier_instance.classify.return_value = (ContentType.STUDIO, 0.9)
        mock_classifier_instance.is_ambiguous.return_value = False
        organizer.classifier = mock_classifier_instance

        result = organizer._process_file(file_path)

        assert result == mock_audio_file

    def test_process_cover_art(self, organizer, tmp_path):
        """Test cover art processing."""
        # Create a directory with cover art
        source_dir = tmp_path / "source" / "album"
        source_dir.mkdir(parents=True)

        cover_file = source_dir / "cover.jpg"
        cover_file.write_bytes(b"fake image data")

        organizer.metadata_handler = Mock()
        organizer.metadata_handler.find_cover_art.return_value = [cover_file]

        organizer.file_mover = Mock()
        organizer.file_mover.move_cover_art = Mock()

        target_dir = tmp_path / "target" / "album"

        organizer._process_cover_art(Path("/source/album/song.flac"), target_dir)

        # Should find and move cover art
        organizer.metadata_handler.find_cover_art.assert_called_once()
        organizer.file_mover.move_cover_art.assert_called_once()

    def test_get_cache_key(self, organizer):
        """Test cache key generation for user decisions."""
        # Create mock audio file
        audio_file = Mock(spec=AudioFile)
        audio_file.artists = ["Artist One", "Artist Two"]
        audio_file.album = "Test Album"

        cache_key = organizer._get_cache_key(audio_file)

        # Should combine artists and album
        assert "Artist One" in cache_key
        assert "Test Album" in cache_key

        # Same metadata should produce same key
        audio_file2 = Mock(spec=AudioFile)
        audio_file2.artists = ["Artist One", "Artist Two"]
        audio_file2.album = "Test Album"

        cache_key2 = organizer._get_cache_key(audio_file2)
        assert cache_key == cache_key2

    def test_get_cache_key_no_artists(self, organizer):
        """Test cache key with no artists."""
        audio_file = Mock(spec=AudioFile)
        audio_file.artists = []
        audio_file.album = "Test Album"

        cache_key = organizer._get_cache_key(audio_file)

        assert "unknown" in cache_key
        assert "Test Album" in cache_key

    def test_get_cache_key_no_album(self, organizer):
        """Test cache key with no album."""
        audio_file = Mock(spec=AudioFile)
        audio_file.artists = ["Test Artist"]
        audio_file.album = None

        cache_key = organizer._get_cache_key(audio_file)

        assert "Test Artist" in cache_key
        assert "unknown" in cache_key

    def test_get_category_name(self, organizer):
        """Test mapping content type to category name."""
        assert organizer._get_category_name(ContentType.STUDIO) == "Albums"
        assert organizer._get_category_name(ContentType.LIVE) == "Live"
        assert organizer._get_category_name(ContentType.COLLABORATION) == "Collaborations"
        assert organizer._get_category_name(ContentType.COMPILATION) == "Compilations"
        assert organizer._get_category_name(ContentType.RARITY) == "Rarities"
        assert organizer._get_category_name(ContentType.UNKNOWN) == "Unknown"

    def test_rollback(self, organizer):
        """Test rollback functionality."""
        organizer.file_mover = Mock()

        organizer.rollback()

        organizer.file_mover.rollback.assert_called_once()

    def test_get_operation_summary(self, organizer):
        """Test getting operation summary."""
        mock_summary = {'total_files': 5}
        organizer.file_mover = Mock()
        organizer.file_mover.get_operation_summary.return_value = mock_summary

        summary = organizer.get_operation_summary()

        assert summary == mock_summary
        organizer.file_mover.get_operation_summary.assert_called_once()


class TestMusicOrganizerIntegration:
    """Integration tests for MusicOrganizer."""

    @pytest.fixture
    def organizer(self, sample_config):
        """Create organizer with mocked dependencies."""
        return MusicOrganizer(config=sample_config)

    def test_full_workflow_with_mocks(self, organizer, sample_config, tmp_path):
        """Test full organization workflow with mocks."""
        # Create source directory with files
        source_dir = sample_config.source_directory
        source_dir.mkdir(parents=True)

        for i in range(3):
            (source_dir / f"song{i}.flac").write_bytes(b"data")

        # Mock metadata handler and classifier
        with patch.object(organizer, '_process_file') as mock_process:
            mock_file = Mock()
            mock_file.content_type = ContentType.STUDIO
            mock_process.return_value = mock_file

            files = organizer.scan_directory(source_dir)
            results = organizer.organize_files(files)

        assert results['processed'] == 3
        assert results['moved'] == 3


if __name__ == "__main__":
    pytest.main([__file__])
