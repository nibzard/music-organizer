"""Integration tests for Result pattern usage in domain services.

This module tests how the Result pattern integrates with the actual domain services.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from music_organizer.domain.catalog.services import MetadataService, CatalogService
from music_organizer.domain.catalog.entities import Recording, Release, Artist, Catalog
from music_organizer.domain.catalog.value_objects import ArtistName, Metadata, AudioPath
from music_organizer.domain.result import Result, success, failure, MetadataError, DuplicateError
from music_organizer.domain.organization.services import OrganizationService
from music_organizer.domain.organization.entities import OrganizationRule
from music_organizer.domain.organization.value_objects import OrganizationPattern, ConflictStrategy
from music_organizer.domain.classification.services import ClassificationService
from music_organizer.domain.classification.entities import Classifier
from music_organizer.domain.value_objects import TrackNumber


class TestMetadataServiceResultPattern:
    """Test Result pattern usage in MetadataService."""

    @pytest.fixture
    def mock_recording_repo(self):
        """Mock recording repository."""
        repo = AsyncMock()
        repo.save = AsyncMock(return_value=None)
        repo.find_by_path = AsyncMock(return_value=None)
        return repo

    @pytest.fixture
    def recording(self):
        """Create a test recording."""
        audio_path = AudioPath("/music/test.flac")
        metadata = Metadata(
            title="Test Song",
            artists=frozenset([ArtistName("Test Artist")]),
            album="Test Album"
        )
        return Recording(path=audio_path, metadata=metadata)

    @pytest.fixture
    def enhanced_metadata(self):
        """Create enhanced metadata."""
        return Metadata(
            title="Enhanced Song",
            year=2023,
            genre="Rock"
        )

    @pytest.mark.asyncio
    async def test_enhance_metadata_success(self, mock_recording_repo, recording, enhanced_metadata):
        """Test successful metadata enhancement."""
        service = MetadataService(mock_recording_repo)

        result = await service.enhance_metadata(recording, enhanced_metadata)

        assert isinstance(result, Result)
        assert result.is_success()
        assert result.value().metadata.title == "Test Song"  # Original title preserved
        assert result.value().metadata.year == 2023  # Enhanced year added
        assert result.value().metadata.genre == "Rock"  # Enhanced genre added

    @pytest.mark.asyncio
    async def test_enhance_metadata_failure(self, mock_recording_repo, recording, enhanced_metadata):
        """Test metadata enhancement failure."""
        mock_recording_repo.save.side_effect = Exception("Database error")

        service = MetadataService(mock_recording_repo)

        result = await service.enhance_metadata(recording, enhanced_metadata)

        assert isinstance(result, Result)
        assert result.is_failure()
        assert isinstance(result.error(), MetadataError)
        assert "Failed to enhance metadata" in str(result.error())

    @pytest.mark.asyncio
    async def test_batch_enhance_metadata_success(self, mock_recording_repo):
        """Test successful batch metadata enhancement."""
        recordings = [
            Recording(
                path=AudioPath(f"/music/test{i}.flac"),
                metadata=Metadata(title=f"Song {i}", artists=frozenset([ArtistName("Artist")]))
            )
            for i in range(3)
        ]
        enhancements = [
            Metadata(year=2023 + i, genre=f"Genre{i}")
            for i in range(3)
        ]

        service = MetadataService(mock_recording_repo)

        result = await service.batch_enhance_metadata(recordings, enhancements)

        assert isinstance(result, Result)
        assert result.is_success()
        enhanced_recordings = result.value()
        assert len(enhanced_recordings) == 3
        assert enhanced_recordings[0].metadata.year == 2023
        assert enhanced_recordings[1].metadata.year == 2024
        assert enhanced_recordings[2].metadata.year == 2025

    @pytest.mark.asyncio
    async def test_batch_enhance_metadata_length_mismatch(self, mock_recording_repo):
        """Test batch enhancement with length mismatch."""
        recordings = [Recording(
            path=AudioPath("/music/test.flac"),
            metadata=Metadata(title="Song", artists=frozenset([ArtistName("Artist")]))
        )]
        enhancements = []  # Empty list

        service = MetadataService(mock_recording_repo)

        result = await service.batch_enhance_metadata(recordings, enhancements)

        assert isinstance(result, Result)
        assert result.is_failure()
        assert isinstance(result.error(), MetadataError)
        assert "must have same length" in str(result.error())


class TestCatalogServiceResultPattern:
    """Test Result pattern usage in CatalogService."""

    @pytest.fixture
    def mock_repos(self):
        """Mock repositories."""
        recording_repo = AsyncMock()
        recording_repo.find_by_path = AsyncMock(return_value=None)
        recording_repo.save = AsyncMock(return_value=None)
        recording_repo.delete = AsyncMock(return_value=None)

        release_repo = AsyncMock()
        release_repo.find_by_title_and_artist = AsyncMock(return_value=None)
        release_repo.save = AsyncMock(return_value=None)
        release_repo.delete = AsyncMock(return_value=None)

        artist_repo = AsyncMock()
        artist_repo.find_by_name = AsyncMock(return_value=None)
        artist_repo.save = AsyncMock(return_value=None)

        return {
            "recording": recording_repo,
            "release": release_repo,
            "artist": artist_repo
        }

    @pytest.fixture
    def catalog(self):
        """Create a test catalog."""
        return Catalog(name="Test Catalog")

    @pytest.fixture
    def recording(self):
        """Create a test recording."""
        audio_path = AudioPath("/music/test.flac")
        metadata = Metadata(
            title="Test Song",
            artists=frozenset([ArtistName("Test Artist")]),
            album="Test Album",
            year=2023
        )
        return Recording(path=audio_path, metadata=metadata)

    @pytest.mark.asyncio
    async def test_add_recording_success(self, mock_repos, catalog, recording):
        """Test successfully adding a recording to catalog."""
        service = CatalogService(
            mock_repos["recording"],
            mock_repos["release"],
            mock_repos["artist"]
        )

        result = await service.add_recording_to_catalog(catalog, recording)

        assert isinstance(result, Result)
        assert result.is_success()
        mock_repos["recording"].save.assert_called_once()
        mock_repos["artist"].save.assert_called_once()
        mock_repos["release"].save.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_recording_duplicate(self, mock_repos, catalog, recording):
        """Test adding duplicate recording to catalog."""
        mock_repos["recording"].find_by_path.return_value = recording  # Simulate duplicate

        service = CatalogService(
            mock_repos["recording"],
            mock_repos["release"],
            mock_repos["artist"]
        )

        result = await service.add_recording_to_catalog(catalog, recording)

        assert isinstance(result, Result)
        assert result.is_failure()
        assert isinstance(result.error(), DuplicateError)
        assert "already exists" in str(result.error())

    @pytest.mark.asyncio
    async def test_add_recording_repository_error(self, mock_repos, catalog, recording):
        """Test adding recording when repository raises error."""
        mock_repos["recording"].save.side_effect = Exception("Database connection failed")

        service = CatalogService(
            mock_repos["recording"],
            mock_repos["release"],
            mock_repos["artist"]
        )

        result = await service.add_recording_to_catalog(catalog, recording)

        assert isinstance(result, Result)
        assert result.is_failure()
        assert isinstance(result.error(), DuplicateError)  # Errors wrapped in DuplicateError
        assert "Failed to add recording" in str(result.error())


class TestOrganizationServiceResultPattern:
    """Test Result pattern usage in OrganizationService."""

    @pytest.fixture
    def temp_files(self, tmp_path):
        """Create temporary files for testing."""
        source_file = tmp_path / "source" / "test.flac"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_bytes(b"fake audio data")

        target_dir = tmp_path / "target"
        target_dir.mkdir(exist_ok=True)

        return {
            "source": source_file,
            "target_dir": target_dir
        }

    @pytest.fixture
    def target_path(self, temp_files):
        """Create a target path."""
        from music_organizer.domain.organization.value_objects import TargetPath, OrganizationPattern
        target_file = temp_files["target_dir"] / "organized" / "test.flac"

        return TargetPath(
            path=target_file,
            pattern_used=OrganizationPattern("pattern", "filename"),
            original_path=temp_files["source"]
        )

    @pytest.mark.asyncio
    async def test_organize_file_success(self, temp_files, target_path):
        """Test successful file organization."""
        service = OrganizationService()

        result = await service.organize_file(
            temp_files["source"],
            target_path,
            ConflictStrategy.SKIP,
            dry_run=False
        )

        assert isinstance(result, Result)
        assert result.is_success()
        moved_file = result.value()
        assert moved_file.status.value == "completed"
        assert target_path.path.exists()

    @pytest.mark.asyncio
    async def test_organize_file_not_found(self, temp_files, target_path):
        """Test organizing non-existent file."""
        non_existent = temp_files["source"].parent / "non_existent.flac"
        service = OrganizationService()

        result = await service.organize_file(
            non_existent,
            target_path,
            ConflictStrategy.SKIP,
            dry_run=False
        )

        assert isinstance(result, Result)
        assert result.is_success()  # Returns success with error in object
        moved_file = result.value()
        assert moved_file.status.value == "failed"
        assert "does not exist" in moved_file.error_message

    @pytest.mark.asyncio
    async def test_organize_file_dry_run(self, temp_files, target_path):
        """Test file organization in dry run mode."""
        service = OrganizationService()

        result = await service.organize_file(
            temp_files["source"],
            target_path,
            ConflictStrategy.SKIP,
            dry_run=True
        )

        assert isinstance(result, Result)
        assert result.is_success()
        moved_file = result.value()
        assert moved_file.status.value == "completed"
        assert temp_files["source"].exists()  # Source still exists
        assert not target_path.path.exists()  # Target not created

    @pytest.mark.asyncio
    async def test_organize_file_with_error(self, temp_files, target_path):
        """Test organization when an error occurs."""
        # Make target path invalid (e.g., parent is a file)
        target_path.path.parent.write_bytes(b"this is a file, not a directory")

        service = OrganizationService()

        result = await service.organize_file(
            temp_files["source"],
            target_path,
            ConflictStrategy.SKIP,
            dry_run=False
        )

        assert isinstance(result, Result)
        assert result.is_failure()
        assert isinstance(result.error(), OrganizationError)


class TestClassificationServiceResultPattern:
    """Test Result pattern usage in ClassificationService."""

    @pytest.fixture
    def mock_classifier(self):
        """Mock classifier."""
        classifier = MagicMock(spec=Classifier)
        classifier.classify_content_type.return_value = ("Studio", 0.9)
        classifier.classify_genre.return_value = MagicMock(
            get_all_genres=lambda: ["Rock"],
            confidence_score=0.85
        )
        return classifier

    @pytest.fixture
    def recording(self):
        """Create a test recording."""
        metadata = Metadata(
            title="Test Song",
            artists=frozenset([ArtistName("Test Artist")]),
            album="Test Album",
            year=2023,
            genre="Rock"
        )

        # Create recording mock with metadata
        recording = MagicMock()
        recording.metadata = metadata
        return recording

    @pytest.mark.asyncio
    async def test_classify_recording_success(self, mock_classifier, recording):
        """Test successful recording classification."""
        service = ClassificationService(mock_classifier)

        result = await service.classify_recording(recording)

        assert isinstance(result, Result)
        assert result.is_success()
        classification = result.value()
        assert classification["content_type"] == "Studio"
        assert classification["content_type_confidence"] == 0.9
        assert classification["genres"] == ["Rock"]
        assert classification["genre_confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_classify_recording_with_error(self, mock_classifier, recording):
        """Test classification when an error occurs."""
        mock_classifier.classify_content_type.side_effect = Exception("Classification failed")

        service = ClassificationService(mock_classifier)

        result = await service.classify_recording(recording)

        assert isinstance(result, Result)
        assert result.is_failure()
        assert isinstance(result.error(), Exception)

    @pytest.mark.asyncio
    async def test_batch_classify_all_success(self, mock_classifier):
        """Test batch classification with all successes."""
        service = ClassificationService(mock_classifier)

        recordings = [MagicMock(metadata=Metadata(title=f"Song {i}")) for i in range(3)]

        result = await service.batch_classify(recordings)

        assert isinstance(result, Result)
        assert result.is_success()
        classifications = result.value()
        assert len(classifications) == 3
        for classification in classifications:
            assert classification["content_type"] == "Studio"

    @pytest.mark.asyncio
    async def test_batch_classify_with_failures(self, mock_classifier):
        """Test batch classification with some failures."""
        # Make classifier fail on second call
        call_count = 0
        def classify_content_type(metadata, features):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Classification failed")
            return ("Studio", 0.9)

        mock_classifier.classify_content_type.side_effect = classify_content_type

        service = ClassificationService(mock_classifier)

        recordings = [MagicMock(metadata=Metadata(title=f"Song {i}")) for i in range(3)]

        result = await service.batch_classify(recordings)

        assert isinstance(result, Result)
        assert result.is_failure()
        assert len(result.error()) == 1  # One failure
        assert isinstance(result.error()[0], Exception)