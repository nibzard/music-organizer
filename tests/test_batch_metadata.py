"""Tests for batch metadata operations."""

import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

import pytest

from music_organizer.core.batch_metadata import (
    BatchMetadataProcessor,
    BatchMetadataConfig,
    MetadataOperation,
    OperationType,
    ConflictStrategy,
    MetadataOperationBuilder,
    BatchResult
)
from music_organizer.infrastructure.adapters.mutagen_adapter import MutagenMetadataAdapter
from music_organizer.domain.value_objects import Metadata, ArtistName, TrackNumber


class TestMetadataOperation:
    """Test metadata operation creation and validation."""

    def test_operation_creation(self):
        """Test creating metadata operations."""
        op = MetadataOperation(
            field='genre',
            operation=OperationType.SET,
            value='Rock',
            conflict_strategy=ConflictStrategy.MERGE
        )

        assert op.field == 'genre'
        assert op.operation == OperationType.SET
        assert op.value == 'Rock'
        assert op.conflict_strategy == ConflictStrategy.MERGE

    def test_operation_builder(self):
        """Test metadata operation builder."""
        # Test set genre
        op = MetadataOperationBuilder.set_genre('Jazz')
        assert op.field == 'genre'
        assert op.operation == OperationType.SET
        assert op.value == 'Jazz'

        # Test add artist
        op = MetadataOperationBuilder.add_artist('John Doe')
        assert op.field == 'artists'
        assert op.operation == OperationType.ADD
        assert op.value == 'John Doe'

        # Test capitalize fields
        ops = MetadataOperationBuilder.capitalize_fields(['title', 'album'])
        assert len(ops) == 2
        assert all(op.operation == OperationType.TRANSFORM for op in ops)


class TestBatchMetadataProcessor:
    """Test batch metadata processor."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter."""
        adapter = MagicMock(spec=MutagenMetadataAdapter)
        adapter.read_metadata = AsyncMock()
        adapter.write_metadata = AsyncMock(return_value=True)
        return adapter

    @pytest.fixture
    def processor(self, mock_adapter):
        """Create a test processor."""
        config = BatchMetadataConfig(
            max_workers=2,
            batch_size=10,
            dry_run=False,
            backup_before_update=False,
            continue_on_error=True
        )
        return BatchMetadataProcessor(config, adapter=mock_adapter)

    @pytest.fixture
    def mock_metadata(self):
        """Create mock metadata."""
        return Metadata(
            title='test song',
            artists=[ArtistName('Test Artist')],
            album='Test Album',
            year=2023,
            genre='Rock',
            track_number=TrackNumber('5')
        )

    @pytest.fixture
    def temp_audio_files(self):
        """Create temporary audio files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(5):
                file_path = Path(tmpdir) / f'test_{i}.mp3'
                file_path.write_bytes(b'fake audio data')
                files.append(file_path)
            yield files

    @pytest.mark.asyncio
    async def test_apply_set_operation(self, processor, mock_metadata):
        """Test applying a set operation."""
        operations = [
            MetadataOperation(
                field='genre',
                operation=OperationType.SET,
                value='Jazz'
            )
        ]

        updated_metadata, applied = await processor._apply_operations(
            mock_metadata, operations
        )

        assert updated_metadata.genre == 'Jazz'
        assert len(applied) == 1
        assert applied[0].field == 'genre'

    @pytest.mark.asyncio
    async def test_apply_add_operation(self, processor, mock_metadata):
        """Test applying an add operation."""
        operations = [
            MetadataOperation(
                field='artists',
                operation=OperationType.ADD,
                value='Featured Artist'
            )
        ]

        updated_metadata, applied = await processor._apply_operations(
            mock_metadata, operations
        )

        assert len(updated_metadata.artists) == 2
        assert ArtistName('Featured Artist') in updated_metadata.artists
        assert len(applied) == 1

    @pytest.mark.asyncio
    async def test_apply_transform_operation(self, processor, mock_metadata):
        """Test applying a transform operation."""
        operations = [
            MetadataOperation(
                field='title',
                operation=OperationType.TRANSFORM,
                pattern=r's/test/Test/g'  # Simple replace
            )
        ]

        # Test with lowercase title - use with_field to create new instance
        mock_metadata = mock_metadata.with_field(title='test song')
        updated_metadata, applied = await processor._apply_operations(
            mock_metadata, operations
        )

        assert updated_metadata.title == 'Test song'
        assert len(applied) == 1

    @pytest.mark.asyncio
    async def test_condition_filtering(self, processor, mock_metadata):
        """Test that operations are filtered by conditions."""
        operations = [
            MetadataOperation(
                field='genre',
                operation=OperationType.SET,
                value='Classical',
                condition={'genre': 'Rock'}
            ),
            MetadataOperation(
                field='year',
                operation=OperationType.SET,
                value=2024,
                condition={'year': {'equals': 2023}}
            )
        ]

        updated_metadata, applied = await processor._apply_operations(
            mock_metadata, operations
        )

        # First operation should apply (genre is Rock)
        # Second operation should apply (year is 2023)
        assert updated_metadata.genre == 'Classical'
        assert updated_metadata.year == 2024
        assert len(applied) == 2

    @pytest.mark.asyncio
    async def test_conflict_strategies(self, processor, mock_metadata):
        """Test different conflict strategies."""
        # Test SKIP strategy
        operations = [
            MetadataOperation(
                field='genre',
                operation=OperationType.SET,
                value='Pop',
                conflict_strategy=ConflictStrategy.SKIP
            )
        ]

        updated_metadata, applied = await processor._apply_operations(
            mock_metadata, operations
        )

        # Should skip because genre already has value
        assert updated_metadata.genre == 'Rock'
        assert len(applied) == 0

        # Test REPLACE strategy
        operations[0].conflict_strategy = ConflictStrategy.REPLACE
        updated_metadata, applied = await processor._apply_operations(
            mock_metadata, operations
        )

        assert updated_metadata.genre == 'Pop'
        assert len(applied) == 1

    @pytest.mark.asyncio
    async def test_batch_processing(self, processor, temp_audio_files):
        """Test processing multiple files in batch."""
        operations = [
            MetadataOperation(
                field='genre',
                operation=OperationType.SET,
                value='Test Genre'
            )
        ]

        # Mock the adapter methods
        with patch.object(processor.adapter, 'read_metadata') as mock_read, \
             patch.object(processor.adapter, 'write_metadata') as mock_write:

            # Mock metadata for each file
            mock_read.return_value = Metadata(
                title='Test Song',
                artists=[ArtistName('Test Artist')]
            )
            mock_write.return_value = True

            result = await processor.apply_operations(
                temp_audio_files,
                operations
            )

            assert result.total_files == 5
            assert result.successful == 5
            assert result.failed == 0
            assert mock_write.call_count == 5

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, processor, temp_audio_files):
        """Test dry run mode doesn't write files."""
        processor.config.dry_run = True
        operations = [
            MetadataOperation(
                field='genre',
                operation=OperationType.SET,
                value='Test Genre'
            )
        ]

        with patch.object(processor.adapter, 'read_metadata') as mock_read, \
             patch.object(processor.adapter, 'write_metadata') as mock_write:

            mock_read.return_value = Metadata(
                title='Test Song',
                artists=[ArtistName('Test Artist')]
            )

            result = await processor.apply_operations(
                temp_audio_files,
                operations
            )

            assert result.total_files == 5
            assert result.successful == 5
            # write_metadata should not be called in dry run
            assert mock_write.call_count == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, processor, temp_audio_files):
        """Test error handling in batch processing."""
        processor.config.continue_on_error = True
        # Disable undo log to avoid backup issues in this test
        processor.config.create_undo_log = False
        operations = [
            MetadataOperation(
                field='genre',
                operation=OperationType.SET,
                value='Test Genre'
            )
        ]

        # Track call count for side_effect
        call_count = [0]

        async def mock_read_with_errors(path):
            call_count[0] += 1
            if call_count[0] <= 3:
                return Metadata(title='Test', artists=[ArtistName('Artist')])
            else:
                raise Exception("Read error")

        async def mock_write(path, metadata):
            return True

        with patch.object(processor.adapter, 'read_metadata', new=mock_read_with_errors), \
             patch.object(processor.adapter, 'write_metadata', new=mock_write):

            result = await processor.apply_operations(
                temp_audio_files,
                operations
            )

            assert result.total_files == 5
            assert result.successful == 3
            assert result.failed == 2
            assert len(result.errors) == 2

    def test_validation(self, processor):
        """Test metadata validation."""
        # Valid metadata
        valid_metadata = Metadata(
            title='Test Song',
            artists=[ArtistName('Test Artist')],
            year=2023
        )
        errors = processor._validate_metadata(valid_metadata)
        assert len(errors) == 0

        # Invalid metadata
        invalid_metadata = Metadata(
            title='',  # Empty title
            artists=[],  # No artists
            year=1800  # Invalid year
        )
        errors = processor._validate_metadata(invalid_metadata)
        assert len(errors) == 3
        assert 'Title is required' in errors
        assert 'At least one artist is required' in errors
        assert 'Invalid year: 1800' in errors

    @pytest.mark.asyncio
    async def test_cleanup(self, processor):
        """Test resource cleanup."""
        # Verify executor exists
        assert processor.executor is not None

        # Call cleanup
        await processor.cleanup()

        # Executor should be shut down
        assert processor.executor._shutdown is True


class TestBatchResult:
    """Test batch result calculations."""

    def test_success_rate(self):
        """Test success rate calculation."""
        result = BatchResult(
            total_files=100,
            successful=85,
            failed=10,
            skipped=5
        )
        assert result.success_rate == 85.0

        # Test zero division
        result.total_files = 0
        assert result.success_rate == 0.0

    def test_throughput(self):
        """Test throughput calculation."""
        result = BatchResult(
            total_files=100,
            duration_seconds=10.0
        )
        assert result.throughput_files_per_sec == 10.0

        # Test zero division
        result.duration_seconds = 0.0
        assert result.throughput_files_per_sec == 0.0


class TestIntegration:
    """Integration tests for batch metadata operations."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete batch metadata workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            files = []
            for i in range(3):
                file_path = Path(tmpdir) / f'test_{i}.flac'
                file_path.write_bytes(b'fake flac data')
                files.append(file_path)

            # Create mock adapter
            mock_adapter = MagicMock(spec=MutagenMetadataAdapter)

            async def mock_read(path):
                return Metadata(
                    title='Test Song',
                    artists=[ArtistName('Original Artist')],
                    album='Test Album',
                    genre='Original Genre',
                    year=2023
                )

            async def mock_write(path, metadata):
                return True

            mock_adapter.read_metadata = mock_read
            mock_adapter.write_metadata = mock_write

            # Create processor with mock adapter
            config = BatchMetadataConfig(
                max_workers=2,
                batch_size=2,
                dry_run=False,
                create_undo_log=False  # Disable undo log for simpler test
            )
            processor = BatchMetadataProcessor(config, adapter=mock_adapter)

            # Define operations
            operations = [
                MetadataOperationBuilder.set_genre('Test Genre'),
                MetadataOperationBuilder.set_year(2024),
                MetadataOperationBuilder.add_artist('Additional Artist')
            ]

            try:
                # Process files
                result = await processor.apply_operations(files, operations)

                # Verify results
                assert result.total_files == 3
                assert result.successful == 3
                assert result.failed == 0

            finally:
                await processor.cleanup()


if __name__ == '__main__':
    pytest.main([__file__])