"""Tests for organization preview functionality."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from collections import Counter
import json
import tempfile
import shutil

from music_organizer.core.organization_preview import (
    OrganizationPreview, PreviewOperation, DirectoryPreview,
    PreviewStatistics, InteractivePreview
)
from music_organizer.models.audio_file import AudioFile, ContentType, FileFormat
from music_organizer.models.config import Config
from music_organizer.domain.value_objects import ArtistName, Metadata


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return Config(
        source_directory=Path("/test/source"),
        target_directory=Path("/test/target")
    )


@pytest.fixture
def sample_audio_files():
    """Create sample audio files for testing."""
    files = []

    # Studio album
    metadata1 = Metadata(
        title="Test Song 1",
        artists=[ArtistName("Test Artist 1")],
        album="Test Album 1",
        year=2020,
        genre="Rock",
        track_number=1,
        content_type=ContentType.STUDIO
    )
    files.append(AudioFile(
        path=Path("/test/source/artist1/album1/song1.flac"),
        size_mb=25.0,
        format=FileFormat.FLAC,
        metadata=metadata1
    ))

    # Live recording
    metadata2 = Metadata(
        title="Live Song 2",
        artists=[ArtistName("Test Artist 2")],
        album="Live Album",
        year=2021,
        genre="Rock",
        track_number=1,
        content_type=ContentType.LIVE
    )
    files.append(AudioFile(
        path=Path("/test/source/artist2/live/song2.mp3"),
        size_mb=8.0,
        format=FileFormat.MP3,
        metadata=metadata2
    ))

    # Compilation
    metadata3 = Metadata(
        title="Compilation Track",
        artists=[ArtistName("Various Artists")],
        album="Best of 2020",
        year=2020,
        genre="Pop",
        track_number=5,
        content_type=ContentType.COMPILATION
    )
    files.append(AudioFile(
        path=Path("/test/source/compilations/best2020/track5.wav"),
        size_mb=45.0,
        format=FileFormat.WAV,
        metadata=metadata3
    ))

    return files


@pytest.fixture
def target_mapping():
    """Create sample target path mapping."""
    return {
        Path("/test/source/artist1/album1/song1.flac"):
        Path("/test/target/Albums/Test Artist 1/Test Album 1 (2020)/01 Test Song 1.flac"),

        Path("/test/source/artist2/live/song2.mp3"):
        Path("/test/target/Live/Test Artist 2/2021 - Live Album/01 Live Song 2.mp3"),

        Path("/test/source/compilations/best2020/track5.wav"):
        Path("/test/target/Compilations/Various Artists/Best of 2020 (2020)/05 Compilation Track.wav")
    }


class TestPreviewOperation:
    """Test PreviewOperation dataclass."""

    def test_preview_operation_creation(self):
        """Test creating a preview operation."""
        op = PreviewOperation(
            operation_type='move',
            source_path=Path("/source/test.flac"),
            target_path=Path("/target/test.flac"),
            file_size=1024 * 1024 * 25  # 25MB
        )

        assert op.operation_type == 'move'
        assert op.source_path == Path("/source/test.flac")
        assert op.target_path == Path("/target/test.flac")
        assert op.file_size == 1024 * 1024 * 25
        assert op.confidence == 1.0

    def test_conflict_operation(self):
        """Test creating a conflict operation."""
        op = PreviewOperation(
            operation_type='conflict',
            source_path=Path("/source/test.flac"),
            target_path=Path("/target/test.flac"),
            file_size=1024 * 1024 * 25,
            conflict_reason="Target file already exists",
            resolution_strategy="rename"
        )

        assert op.operation_type == 'conflict'
        assert op.conflict_reason == "Target file already exists"
        assert op.resolution_strategy == "rename"


class TestDirectoryPreview:
    """Test DirectoryPreview dataclass."""

    def test_empty_directory_preview(self):
        """Test creating an empty directory preview."""
        directory = DirectoryPreview(
            path=Path("/test/target"),
            file_count=0,
            total_size_mb=0.0,
            is_new=True,
            is_empty=True
        )

        assert directory.path == Path("/test/target")
        assert directory.file_count == 0
        assert directory.total_size_mb == 0.0
        assert directory.is_new
        assert directory.is_empty
        assert directory.subdirectories == []

    def test_directory_with_files(self):
        """Test creating a directory preview with files."""
        subdir = DirectoryPreview(
            path=Path("/test/target/Artist"),
            file_count=5,
            total_size_mb=125.0,
            content_types={"studio": 3, "live": 2},
            formats={"FLAC": 3, "MP3": 2}
        )

        root = DirectoryPreview(
            path=Path("/test/target"),
            file_count=15,
            total_size_mb=350.0,
            subdirectories=[subdir]
        )

        assert len(root.subdirectories) == 1
        assert root.subdirectories[0].file_count == 5
        assert root.subdirectories[0].total_size_mb == 125.0


class TestOrganizationPreview:
    """Test OrganizationPreview class."""

    @pytest.fixture
    def preview(self, sample_config):
        """Create an organization preview instance."""
        return OrganizationPreview(sample_config)

    def test_preview_initialization(self, preview):
        """Test preview initialization."""
        assert preview.config == sample_config
        assert preview.operations == []
        assert preview.statistics == PreviewStatistics()
        assert preview.file_operations == {}
        assert preview.conflicts == []

    @pytest.mark.asyncio
    async def test_collect_operations(self, preview, sample_audio_files, target_mapping):
        """Test collecting operations for preview."""
        await preview.collect_operations(sample_audio_files, target_mapping)

        # Should have operations for each file plus directory creation
        assert len(preview.operations) > 3  # File operations + directory operations

        # Should have file operations for each audio file
        assert len(preview.file_operations) == 3

        # Should have statistics calculated
        assert preview.statistics.total_files == 3
        assert preview.statistics.total_operations >= 3
        assert preview.statistics.total_size_mb > 0

        # Content types should be detected
        assert len(preview.statistics.content_types) > 0
        assert preview.statistics.content_types.get('studio', 0) >= 1
        assert preview.statistics.content_types.get('live', 0) >= 1
        assert preview.statistics.content_types.get('compilation', 0) >= 1

    @pytest.mark.asyncio
    async def test_collect_operations_with_conflicts(self, preview, sample_audio_files):
        """Test collecting operations with conflicts."""
        # Create conflicting target mapping (same target for multiple sources)
        conflicting_mapping = {
            Path("/test/source/artist1/album1/song1.flac"):
            Path("/test/target/Albums/Test Artist/Test Album/01 Song.flac"),

            Path("/test/source/artist2/live/song2.mp3"):
            Path("/test/target/Albums/Test Artist/Test Album/01 Song.flac")  # Same target!
        }

        await preview.collect_operations(
            sample_audio_files[:2],  # Only use first two files
            conflicting_mapping
        )

        # Should detect conflicts
        assert len(preview.conflicts) > 0
        assert preview.statistics.conflicts_detected > 0

        # Conflicts should have proper details
        conflict = preview.conflicts[0]
        assert conflict.operation_type == 'conflict'
        assert conflict.conflict_reason
        assert conflict.resolution_strategy

    def test_get_directories_to_create(self, preview):
        """Test getting directories that need to be created."""
        # Add some operations with target paths
        preview.operations = [
            PreviewOperation(
                operation_type='move',
                source_path=Path("/source/1.flac"),
                target_path=Path("/target/new/deep/1.flac"),
                file_size=1024
            ),
            PreviewOperation(
                operation_type='copy',
                source_path=Path("/source/2.flac"),
                target_path=Path("/target/existing/2.flac"),
                file_size=1024
            )
        ]

        directories = preview._get_directories_to_create()

        # Should include all parent directories that don't exist
        assert Path("/target/new") in directories
        assert Path("/target/new/deep") in directories

    def test_calculate_statistics(self, preview, sample_audio_files):
        """Test statistics calculation."""
        # Add operations manually
        for audio_file in sample_audio_files:
            preview.operations.append(PreviewOperation(
                operation_type='move',
                source_path=audio_file.path,
                target_path=Path(f"/target/{audio_file.path.name}"),
                file_size=int(audio_file.size_mb * 1024 * 1024),
                metadata=audio_file.metadata
            ))

        preview._calculate_statistics(sample_audio_files)

        stats = preview.statistics

        # Basic counts
        assert stats.total_files == 3
        assert stats.operations_by_type.get('move', 0) == 3

        # File sizes
        assert stats.total_size_mb == 25.0 + 8.0 + 45.0  # Sum of all file sizes

        # Content types
        assert len(stats.content_types) == 3
        assert stats.content_types['studio'] == 1
        assert stats.content_types['live'] == 1
        assert stats.content_types['compilation'] == 1

        # Formats
        assert len(stats.formats) == 3
        assert stats.formats['FLAC'] == 1
        assert stats.formats['MP3'] == 1
        assert stats.formats['WAV'] == 1

        # Artists
        assert len(stats.artists) == 3
        assert 'Test Artist 1' in stats.artists
        assert 'Test Artist 2' in stats.artists
        assert 'Various Artists' in stats.artists

        # Organization score should be reasonable
        assert 0 <= stats.organization_score <= 100

        # Time estimation should be positive
        assert stats.estimated_time_minutes > 0

    def test_calculate_organization_score(self, preview):
        """Test organization score calculation."""
        # Perfect organization
        preview.statistics.metadata_completeness = 1.0
        preview.statistics.conflicts_detected = 0
        preview.statistics.total_files = 100

        # Create a directory structure with optimal depth
        preview.directory_structure = DirectoryPreview(
            path=Path("/target"),
            subdirectories=[
                DirectoryPreview(
                    path=Path("/target/Artist1"),
                    subdirectories=[
                        DirectoryPreview(
                            path=Path("/target/Artist1/Album1")
                        )
                    ]
                )
            ]
        )

        preview._calculate_organization_score()
        assert preview.statistics.organization_score > 70  # Should be high

    def test_get_max_depth(self, preview):
        """Test getting maximum directory depth."""
        # Create a nested structure
        level3 = DirectoryPreview(path=Path("/target/A/B/C"))
        level2 = DirectoryPreview(path=Path("/target/A/B"), subdirectories=[level3])
        level1 = DirectoryPreview(path=Path("/target/A"), subdirectories=[level2])
        root = DirectoryPreview(path=Path("/target"), subdirectories=[level1])

        depth = preview._get_max_depth(root)
        assert depth == 3  # Should be 3 levels deep

    @pytest.mark.asyncio
    async def test_build_directory_preview(self, preview, sample_audio_files, target_mapping):
        """Test building directory structure preview."""
        await preview.collect_operations(sample_audio_files, target_mapping)

        # Directory structure should be built
        assert preview.directory_structure is not None
        assert preview.directory_structure.path == sample_config.target_directory

        # Should contain files
        assert preview.directory_structure.file_count > 0
        assert preview.directory_structure.total_size_mb > 0

    def test_export_preview(self, preview, sample_audio_files, target_mapping, tmp_path):
        """Test exporting preview to JSON."""
        # Set up some operations
        for audio_file in sample_audio_files:
            preview.operations.append(PreviewOperation(
                operation_type='move',
                source_path=audio_file.path,
                target_path=target_mapping[audio_file.path],
                file_size=int(audio_file.size_mb * 1024 * 1024),
                metadata=audio_file.metadata
            ))

        # Calculate stats
        preview._calculate_statistics(sample_audio_files)

        # Export to file
        export_path = tmp_path / "preview.json"
        preview.export_preview(export_path)

        # Verify file was created and contains valid JSON
        assert export_path.exists()

        with open(export_path) as f:
            data = json.load(f)

        # Check structure
        assert 'timestamp' in data
        assert 'statistics' in data
        assert 'operations' in data
        assert 'conflicts' in data

        # Check statistics
        stats = data['statistics']
        assert stats['total_files'] == 3
        assert stats['total_size_mb'] > 0
        assert len(stats['content_types']) > 0
        assert len(stats['formats']) > 0

        # Check operations
        assert len(data['operations']) == 3
        for op in data['operations']:
            assert 'operation_type' in op
            assert 'source_path' in op
            assert 'target_path' in op
            assert 'file_size' in op


class TestInteractivePreview:
    """Test InteractivePreview class."""

    @pytest.fixture
    def base_preview(self, sample_config, sample_audio_files, target_mapping):
        """Create a base preview with operations."""
        preview = OrganizationPreview(sample_config)

        # Add some operations synchronously for testing
        for audio_file in sample_audio_files:
            preview.operations.append(PreviewOperation(
                operation_type='move',
                source_path=audio_file.path,
                target_path=target_mapping[audio_file.path],
                file_size=int(audio_file.size_mb * 1024 * 1024),
                metadata=audio_file.metadata
            ))

        preview._calculate_statistics(sample_audio_files)
        return preview

    def test_interactive_preview_initialization(self, base_preview):
        """Test interactive preview initialization."""
        interactive = InteractivePreview(base_preview)

        assert interactive.preview == base_preview
        assert interactive.filters['operation_type'] is None
        assert interactive.filters['content_type'] is None
        assert interactive.filters['artist'] is None
        assert not interactive.filters['show_only_conflicts']

    @patch('builtins.input')
    def test_display_menu(self, mock_input, base_preview):
        """Test displaying the interactive menu."""
        interactive = InteractivePreview(base_preview)

        # Mock user input to quit
        mock_input.side_effect = ['q']

        # Should return False (user wants to quit)
        result = asyncio.run(interactive.run_interactive_preview())
        assert result is False

        # Verify menu was shown
        assert mock_input.called

    @patch('builtins.input')
    def test_interactive_preview_proceed(self, mock_input, base_preview):
        """Test interactive preview with proceed choice."""
        interactive = InteractivePreview(base_preview)

        # Mock user input to proceed
        mock_input.side_effect = ['4']  # Option 4: Proceed

        # Should return True (user wants to proceed)
        result = asyncio.run(interactive.run_interactive_preview())
        assert result is True

    @patch('builtins.input')
    def test_filter_operations(self, mock_input, base_preview):
        """Test filtering operations."""
        interactive = InteractivePreview(base_preview)

        # Mock filtering by operation type
        mock_input.side_effect = ['1', 'move', '5']  # Filter by move, then clear

        # Call filter method
        interactive._filter_operations()

        # Check filter was applied
        assert interactive.filters['operation_type'] == 'move'

        # Clear filters
        interactive._filter_operations()
        assert interactive.filters['operation_type'] is None

    def test_display_filtered_operations(self, base_preview):
        """Test displaying filtered operations."""
        interactive = InteractivePreview(base_preview)

        # Set a filter
        interactive.filters['operation_type'] = 'move'

        # Should display operations without error
        with patch('builtins.print'):  # Suppress print output
            interactive._display_filtered_operations()

    def test_export_preview_interactive(self, base_preview, tmp_path):
        """Test exporting preview from interactive mode."""
        interactive = InteractivePreview(base_preview)

        # Mock user input and test export
        with patch('builtins.input', return_value=str(tmp_path / "interactive_export.json")):
            interactive._export_preview()

        # Verify file was created
        export_path = tmp_path / "interactive_export.json"
        assert export_path.exists()


class TestPreviewIntegration:
    """Integration tests for organization preview."""

    @pytest.mark.asyncio
    async def test_full_preview_workflow(self, sample_config, sample_audio_files, tmp_path):
        """Test complete preview workflow."""
        # Create preview
        preview = OrganizationPreview(sample_config)

        # Create target mapping
        target_mapping = {
            audio_file.path: tmp_path / "target" / audio_file.path.name
            for audio_file in sample_audio_files
        }

        # Collect operations
        await preview.collect_operations(sample_audio_files, target_mapping)

        # Verify statistics
        assert preview.statistics.total_files == 3
        assert preview.statistics.total_size_mb > 0
        assert len(preview.statistics.content_types) == 3

        # Verify directory structure
        assert preview.directory_structure is not None

        # Export preview
        export_path = tmp_path / "workflow_preview.json"
        preview.export_preview(export_path)
        assert export_path.exists()

        # Create interactive preview
        interactive = InteractivePreview(preview)
        assert interactive.preview == preview

    @pytest.mark.asyncio
    async def test_preview_with_large_library(self, sample_config):
        """Test preview performance with larger library."""
        # Create many audio files
        audio_files = []
        target_mapping = {}

        for i in range(100):  # 100 files
            metadata = Metadata(
                title=f"Song {i}",
                artists=[ArtistName(f"Artist {i % 10}")],  # 10 unique artists
                album=f"Album {i // 10}",  # 10 albums
                year=2020 + (i % 3),
                genre="Rock",
                track_number=i % 12 + 1,
                content_type=ContentType.STUDIO
            )

            audio_file = AudioFile(
                path=Path(f"/source/artist{i % 10}/album{i // 10}/song{i}.flac"),
                size_mb=25.0,
                format=FileFormat.FLAC,
                metadata=metadata
            )

            audio_files.append(audio_file)
            target_mapping[audio_file.path] = Path(f"/target/Artist {i % 10}/Album {i // 10} ({2020 + (i % 3)})/{i+1:02d} Song {i}.flac")

        # Create preview and collect operations
        preview = OrganizationPreview(sample_config)
        await preview.collect_operations(audio_files, target_mapping)

        # Verify it handled the large library
        assert preview.statistics.total_files == 100
        assert preview.statistics.total_size_mb == 2500.0  # 100 * 25MB
        assert len(preview.statistics.artists) == 10  # 10 unique artists
        assert preview.statistics.operations_by_type['move'] == 100

        # Organization score should be reasonable for large library
        assert 0 <= preview.statistics.organization_score <= 100

        # Time estimation should scale with library size
        assert preview.statistics.estimated_time_minutes > 0

    def test_preview_statistics_edge_cases(self, sample_config):
        """Test preview statistics with edge cases."""
        preview = OrganizationPreview(sample_config)

        # Test with empty library
        preview._calculate_statistics([])
        assert preview.statistics.total_files == 0
        assert preview.statistics.total_size_mb == 0.0
        assert preview.statistics.organization_score == 0  # No organization needed

        # Test with perfect metadata
        perfect_metadata = Metadata(
            title="Perfect Song",
            artists=[ArtistName("Perfect Artist")],
            album="Perfect Album",
            year=2023,
            genre="Perfect Genre",
            track_number=1,
            content_type=ContentType.STUDIO
        )

        audio_file = AudioFile(
            path=Path("/perfect/song.flac"),
            size_mb=10.0,
            format=FileFormat.FLAC,
            metadata=perfect_metadata
        )

        preview.operations = [PreviewOperation(
            operation_type='move',
            source_path=audio_file.path,
            target_path=Path("/target/perfect/song.flac"),
            file_size=10 * 1024 * 1024,
            metadata=perfect_metadata
        )]

        preview._calculate_statistics([audio_file])
        assert preview.statistics.metadata_completeness == 1.0
        assert preview.statistics.organization_score > 50  # Should be high


if __name__ == '__main__':
    pytest.main([__file__])