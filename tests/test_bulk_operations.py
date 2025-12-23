"""Comprehensive tests for bulk file operations."""

import asyncio
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from music_organizer.core.bulk_operations import (
    BulkFileOperator,
    BulkMoveOperator,
    BulkCopyOperator,
    BulkOperationConfig,
    ConflictStrategy,
    OperationType,
    FileOperation,
    BulkOperationResult
)
from music_organizer.exceptions import FileOperationError
from music_organizer.models.audio_file import AudioFile, CoverArt


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    source_dir = Path(tempfile.mkdtemp(prefix="bulk_test_source_"))
    target_dir = Path(tempfile.mkdtemp(prefix="bulk_test_target_"))
    backup_dir = Path(tempfile.mkdtemp(prefix="bulk_test_backup_"))

    yield source_dir, target_dir, backup_dir

    # Cleanup
    shutil.rmtree(source_dir, ignore_errors=True)
    shutil.rmtree(target_dir, ignore_errors=True)
    shutil.rmtree(backup_dir, ignore_errors=True)


@pytest.fixture
def sample_files(temp_dirs):
    """Create sample audio files for testing."""
    source_dir, _, _ = temp_dirs
    files = []

    # Create test audio files
    test_data = b"fake audio data" * 1000  # 16KB files

    for i in range(10):
        file_path = source_dir / f"track_{i:02d}.flac"
        with open(file_path, "wb") as f:
            f.write(test_data + str(i).encode())  # Make each file slightly different
        files.append(file_path)

    # Create some subdirectories with files
    subdir1 = source_dir / "album1"
    subdir1.mkdir()
    for i in range(5):
        file_path = subdir1 / f"album1_track_{i:02d}.mp3"
        with open(file_path, "wb") as f:
            f.write(test_data + f"album1_{i}".encode())
        files.append(file_path)

    subdir2 = source_dir / "album2"
    subdir2.mkdir()
    for i in range(3):
        file_path = subdir2 / f"album2_track_{i:02d}.wav"
        with open(file_path, "wb") as f:
            f.write(test_data + f"album2_{i}".encode())
        files.append(file_path)

    return files


class TestFileOperation:
    """Test the FileOperation dataclass."""

    def test_file_operation_creation(self, temp_dirs):
        """Test creating a FileOperation."""
        source_dir, _, _ = temp_dirs
        test_file = source_dir / "test.flac"
        test_file.write_bytes(b"test data")

        target_path = source_dir / "target" / "test.flac"
        operation = FileOperation(
            source=test_file,
            target=target_path,
            operation_type=OperationType.MOVE
        )

        assert operation.source == test_file
        assert operation.target == target_path
        assert operation.operation_type == OperationType.MOVE
        assert operation.size > 0
        assert operation.checksum is not None

    def test_file_operation_checksum_large_file(self, temp_dirs):
        """Test that large files don't calculate checksums."""
        source_dir, _, _ = temp_dirs
        # Create a file larger than 100MB threshold
        large_file = source_dir / "large.flac"
        large_data = b"x" * (101 * 1024 * 1024)  # 101MB
        large_file.write_bytes(large_data)

        target_path = source_dir / "target" / "large.flac"
        operation = FileOperation(
            source=large_file,
            target=target_path,
            operation_type=OperationType.MOVE
        )

        # Large files should not have checksums calculated
        assert operation.checksum is None


class TestBulkOperationConfig:
    """Test the BulkOperationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BulkOperationConfig()

        assert config.max_workers == 4
        assert config.chunk_size == 100
        assert config.conflict_strategy == ConflictStrategy.RENAME
        assert config.verify_copies is True
        assert config.create_dirs_batch is True
        assert config.preserve_timestamps is True
        assert config.skip_identical is True
        assert config.memory_threshold_mb == 512
        assert config.progress_report_interval == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BulkOperationConfig(
            max_workers=8,
            chunk_size=200,
            conflict_strategy=ConflictStrategy.SKIP,
            verify_copies=False,
            create_dirs_batch=False
        )

        assert config.max_workers == 8
        assert config.chunk_size == 200
        assert config.conflict_strategy == ConflictStrategy.SKIP
        assert config.verify_copies is False
        assert config.create_dirs_batch is False


class TestBulkOperationResult:
    """Test the BulkOperationResult class."""

    def test_result_calculations(self):
        """Test result metric calculations."""
        result = BulkOperationResult(
            total_files=100,
            successful=90,
            skipped=5,
            failed=5,
            total_size_mb=1000.0,
            duration_seconds=10.0
        )

        assert result.success_rate == 90.0
        assert result.throughput_mb_per_sec == 100.0

    def test_empty_result(self):
        """Test empty result calculations."""
        result = BulkOperationResult(
            total_files=0,
            successful=0,
            skipped=0,
            failed=0,
            total_size_mb=0.0,
            duration_seconds=0.0
        )

        assert result.success_rate == 0.0
        assert result.throughput_mb_per_sec == 0.0


class TestBulkFileOperator:
    """Test the BulkFileOperator class."""

    @pytest.mark.asyncio
    async def test_add_operations(self, sample_files, temp_dirs):
        """Test adding operations to the bulk operator."""
        _, target_dir, _ = temp_dirs
        config = BulkOperationConfig(max_workers=2)
        operator = BulkFileOperator(config)

        # Add some operations
        for i, file_path in enumerate(sample_files[:5]):
            target_path = target_dir / f"moved_{i:02d}.flac"
            await operator.add_operation(file_path, target_path, OperationType.MOVE)

        assert len(operator._operations) == 5

        # Cleanup
        await operator.cleanup()

    @pytest.mark.asyncio
    async def test_add_operations_batch(self, sample_files, temp_dirs):
        """Test adding operations in batch."""
        _, target_dir, _ = temp_dirs
        config = BulkOperationConfig(max_workers=2)
        operator = BulkFileOperator(config)

        # Prepare batch operations
        operations = []
        for i, file_path in enumerate(sample_files[:5]):
            target_path = target_dir / f"moved_{i:02d}.flac"
            operations.append((file_path, target_path, OperationType.MOVE))

        await operator.add_operations_batch(operations)
        assert len(operator._operations) == 5

        # Cleanup
        await operator.cleanup()

    @pytest.mark.asyncio
    async def test_execute_bulk_move(self, sample_files, temp_dirs):
        """Test executing bulk move operations."""
        _, target_dir, _ = temp_dirs
        config = BulkOperationConfig(
            max_workers=2,
            chunk_size=3,
            conflict_strategy=ConflictStrategy.RENAME
        )
        operator = BulkFileOperator(config)

        # Add move operations
        for i, file_path in enumerate(sample_files[:6]):
            target_path = target_dir / f"track_{i:02d}.flac"
            await operator.add_operation(file_path, target_path, OperationType.MOVE)

        # Execute bulk operation
        result = await operator.execute_bulk_operation()

        # Verify results
        assert result.total_files == 6
        assert result.successful == 6
        assert result.failed == 0
        assert result.total_size_mb > 0

        # Verify files were moved
        for i in range(6):
            original_file = sample_files[i]
            target_file = target_dir / f"track_{i:02d}.flac"
            assert not original_file.exists()
            assert target_file.exists()

        # Cleanup
        await operator.cleanup()

    @pytest.mark.asyncio
    async def test_execute_bulk_copy(self, sample_files, temp_dirs):
        """Test executing bulk copy operations."""
        _, target_dir, _ = temp_dirs
        config = BulkOperationConfig(
            max_workers=2,
            chunk_size=3,
            conflict_strategy=ConflictStrategy.RENAME,
            verify_copies=True
        )
        operator = BulkFileOperator(config)

        # Add copy operations
        for i, file_path in enumerate(sample_files[:6]):
            target_path = target_dir / f"copy_track_{i:02d}.flac"
            await operator.add_operation(file_path, target_path, OperationType.COPY)

        # Execute bulk operation
        result = await operator.execute_bulk_operation()

        # Verify results
        assert result.total_files == 6
        assert result.successful == 6
        assert result.failed == 0

        # Verify files were copied (originals should still exist)
        for i in range(6):
            original_file = sample_files[i]
            target_file = target_dir / f"copy_track_{i:02d}.flac"
            assert original_file.exists()  # Original should still exist
            assert target_file.exists()

        # Cleanup
        await operator.cleanup()

    @pytest.mark.asyncio
    async def test_conflict_resolution_skip(self, sample_files, temp_dirs):
        """Test conflict resolution with skip strategy."""
        _, target_dir, _ = temp_dirs
        config = BulkOperationConfig(
            max_workers=1,
            conflict_strategy=ConflictStrategy.SKIP
        )
        operator = BulkFileOperator(config)

        # Create a target file that will cause conflict
        target_file = target_dir / "track_00.flac"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_bytes(b"existing file")

        # Add operation that will conflict
        await operator.add_operation(sample_files[0], target_file, OperationType.MOVE)

        # Execute bulk operation
        result = await operator.execute_bulk_operation()

        # Verify conflict was skipped
        assert result.total_files == 1
        assert result.successful == 0  # Should be skipped, not successful
        assert result.skipped == 1

        # Cleanup
        await operator.cleanup()

    @pytest.mark.asyncio
    async def test_conflict_resolution_rename(self, sample_files, temp_dirs):
        """Test conflict resolution with rename strategy."""
        _, target_dir, _ = temp_dirs
        config = BulkOperationConfig(
            max_workers=1,
            conflict_strategy=ConflictStrategy.RENAME
        )
        operator = BulkFileOperator(config)

        # Create a target file that will cause conflict
        target_file = target_dir / "track_00.flac"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_bytes(b"existing file")

        # Add operation that will conflict
        await operator.add_operation(sample_files[0], target_file, OperationType.MOVE)

        # Execute bulk operation
        result = await operator.execute_bulk_operation()

        # Verify conflict was resolved by renaming
        assert result.total_files == 1
        assert result.successful == 1
        assert result.skipped == 0

        # Check that renamed file exists
        renamed_file = target_dir / "track_00 (1).flac"
        assert renamed_file.exists()

        # Cleanup
        await operator.cleanup()

    @pytest.mark.asyncio
    async def test_batch_directory_creation(self, temp_dirs):
        """Test that directories are created in batch."""
        source_dir, target_dir, _ = temp_dirs
        config = BulkOperationConfig(
            max_workers=2,
            create_dirs_batch=True
        )
        operator = BulkFileOperator(config)

        # Create test files
        test_files = []
        for i in range(4):
            subdir = source_dir / f"subdir_{i}"
            subdir.mkdir(parents=True, exist_ok=True)
            test_file = subdir / f"file_{i}.flac"
            test_file.write_bytes(b"test data")
            test_files.append(test_file)

        # Add operations that require creating multiple target directories
        for i, test_file in enumerate(test_files):
            target_subdir = target_dir / f"target_subdir_{i}"
            target_path = target_subdir / f"moved_file_{i}.flac"
            await operator.add_operation(test_file, target_path, OperationType.MOVE)

        # Execute bulk operation
        result = await operator.execute_bulk_operation()

        # Verify all directories were created and files moved
        assert result.successful == 4
        for i in range(4):
            target_subdir = target_dir / f"target_subdir_{i}"
            assert target_subdir.exists()
            target_file = target_subdir / f"moved_file_{i}.flac"
            assert target_file.exists()

        # Cleanup
        await operator.cleanup()

    @pytest.mark.asyncio
    async def test_empty_operations_error(self):
        """Test error when executing with no operations."""
        config = BulkOperationConfig()
        operator = BulkFileOperator(config)

        # Should raise error when trying to execute with no operations
        with pytest.raises(FileOperationError, match="No operations to execute"):
            await operator.execute_bulk_operation()

        # Cleanup
        await operator.cleanup()

    @pytest.mark.asyncio
    async def test_progress_callback(self, sample_files, temp_dirs):
        """Test progress callback functionality."""
        _, target_dir, _ = temp_dirs
        config = BulkOperationConfig(
            max_workers=2,
            chunk_size=2
        )
        operator = BulkFileOperator(config)

        # Track progress updates
        progress_updates = []

        async def progress_callback(current, total):
            progress_updates.append((current, total))

        # Add operations
        for i, file_path in enumerate(sample_files[:4]):
            target_path = target_dir / f"track_{i:02d}.flac"
            await operator.add_operation(file_path, target_path, OperationType.MOVE)

        # Execute with progress callback
        result = await operator.execute_bulk_operation(progress_callback)

        # Verify progress was tracked
        assert len(progress_updates) > 0
        assert result.successful == 4

        # Cleanup
        await operator.cleanup()


class TestBulkMoveOperator:
    """Test the BulkMoveOperator specialized class."""

    @pytest.mark.asyncio
    async def test_bulk_move_operator_initialization(self):
        """Test BulkMoveOperator initialization with default config."""
        mover = BulkMoveOperator()

        assert mover.config.conflict_strategy == ConflictStrategy.RENAME
        assert mover.config.verify_copies is False
        assert mover.config.preserve_timestamps is True

        await mover.cleanup()

    @pytest.mark.asyncio
    async def test_add_audio_move(self, temp_dirs):
        """Test adding audio file move operations."""
        source_dir, target_dir, _ = temp_dirs

        # Create test audio file
        test_file = source_dir / "test.flac"
        test_file.write_bytes(b"audio data")

        # Create AudioFile object
        audio_file = AudioFile(
            path=test_file,
            file_type="FLAC",
            artists=["Test Artist"],
            title="Test Track",
            album="Test Album"
        )

        mover = BulkMoveOperator()
        target_path = target_dir / "moved.flac"
        await mover.add_audio_move(audio_file, target_path)

        assert len(mover._operations) == 1
        assert mover._operations[0].source == test_file
        assert mover._operations[0].target == target_path
        assert mover._operations[0].operation_type == OperationType.MOVE

        await mover.cleanup()

    @pytest.mark.asyncio
    async def test_add_cover_art_move(self, temp_dirs):
        """Test adding cover art move operations."""
        source_dir, target_dir, _ = temp_dirs

        # Create test cover art
        cover_file = source_dir / "folder.jpg"
        cover_file.write_bytes(b"image data")

        # Create CoverArt object
        cover_art = CoverArt(
            path=cover_file,
            type="front",
            format="jpg",
            size=len(b"image data")
        )

        mover = BulkMoveOperator()
        await mover.add_cover_art_move(cover_art, target_dir)

        assert len(mover._operations) == 1
        assert mover._operations[0].source == cover_file
        # Check that target filename is standardized
        assert mover._operations[0].target.name == "folder.jpg"

        await mover.cleanup()


class TestBulkCopyOperator:
    """Test the BulkCopyOperator specialized class."""

    @pytest.mark.asyncio
    async def test_bulk_copy_operator_initialization(self):
        """Test BulkCopyOperator initialization with default config."""
        copier = BulkCopyOperator()

        assert copier.config.conflict_strategy == ConflictStrategy.RENAME
        assert copier.config.verify_copies is True
        assert copier.config.preserve_timestamps is True

        await copier.cleanup()


class TestIntegration:
    """Integration tests for bulk operations."""

    @pytest.mark.asyncio
    async def test_large_scale_bulk_operation(self, temp_dirs):
        """Test bulk operations with many files."""
        source_dir, target_dir, _ = temp_dirs
        config = BulkOperationConfig(
            max_workers=4,
            chunk_size=50,
            conflict_strategy=ConflictStrategy.RENAME
        )
        operator = BulkFileOperator(config)

        # Create many test files
        file_count = 100
        test_data = b"test data for large scale test"

        test_files = []
        for i in range(file_count):
            file_path = source_dir / f"large_test_{i:04d}.flac"
            file_path.write_bytes(test_data + str(i).encode())
            test_files.append(file_path)

        # Add operations
        for i, file_path in enumerate(test_files):
            target_path = target_dir / f"moved_{i:04d}.flac"
            await operator.add_operation(file_path, target_path, OperationType.MOVE)

        # Execute bulk operation
        result = await operator.execute_bulk_operation()

        # Verify results
        assert result.total_files == file_count
        assert result.successful == file_count
        assert result.failed == 0
        assert result.throughput_mb_per_sec > 0

        # Verify all files were moved
        for i in range(file_count):
            assert not test_files[i].exists()
            moved_file = target_dir / f"moved_{i:04d}.flac"
            assert moved_file.exists()

        # Cleanup
        await operator.cleanup()

    @pytest.mark.asyncio
    async def test_mixed_operation_types(self, temp_dirs):
        """Test bulk operations with mixed move and copy operations."""
        source_dir, target_dir, _ = temp_dirs
        config = BulkOperationConfig(
            max_workers=2,
            chunk_size=5
        )
        operator = BulkFileOperator(config)

        # Create test files
        test_files = []
        for i in range(10):
            file_path = source_dir / f"mixed_test_{i:02d}.flac"
            file_path.write_bytes(f"test data {i}".encode())
            test_files.append(file_path)

        # Add mixed operations
        for i, file_path in enumerate(test_files):
            if i % 2 == 0:
                # Even files: move
                target_path = target_dir / f"moved_{i:02d}.flac"
                await operator.add_operation(file_path, target_path, OperationType.MOVE)
            else:
                # Odd files: copy
                target_path = target_dir / f"copied_{i:02d}.flac"
                await operator.add_operation(file_path, target_path, OperationType.COPY)

        # Execute bulk operation
        result = await operator.execute_bulk_operation()

        # Verify results
        assert result.total_files == 10
        assert result.successful == 10
        assert result.failed == 0

        # Verify file states
        for i, file_path in enumerate(test_files):
            if i % 2 == 0:
                # Moved files should not exist in source
                assert not file_path.exists()
                moved_file = target_dir / f"moved_{i:02d}.flac"
                assert moved_file.exists()
            else:
                # Copied files should exist in both places
                assert file_path.exists()
                copied_file = target_dir / f"copied_{i:02d}.flac"
                assert copied_file.exists()

        # Cleanup
        await operator.cleanup()

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, temp_dirs):
        """Test error handling during bulk operations."""
        source_dir, target_dir, _ = temp_dirs
        config = BulkOperationConfig(max_workers=2)
        operator = BulkFileOperator(config)

        # Create some test files
        test_files = []
        for i in range(5):
            file_path = source_dir / f"error_test_{i:02d}.flac"
            file_path.write_bytes(f"test data {i}".encode())
            test_files.append(file_path)

        # Add operations, including one that will fail
        # Use a non-existent source directory for one operation
        fake_source = Path("/nonexistent/path/fake.flac")
        fake_target = target_dir / "fake.flac"
        await operator.add_operation(fake_source, fake_target, OperationType.MOVE)

        # Add valid operations
        for i, file_path in enumerate(test_files):
            target_path = target_dir / f"valid_{i:02d}.flac"
            await operator.add_operation(file_path, target_path, OperationType.MOVE)

        # Execute bulk operation
        result = await operator.execute_bulk_operation()

        # Verify error handling
        assert result.total_files == 6  # 1 fake + 5 real
        assert result.successful == 5  # Only real files should succeed
        assert result.failed == 1  # Fake file should fail

        # Check that the failed operation is in the operations list
        failed_ops = [op for op in result.operations if op.get('status') == 'failed']
        assert len(failed_ops) == 1
        assert 'nonexistent' in failed_ops[0]['source']

        # Verify valid files were still processed
        for i in range(5):
            moved_file = target_dir / f"valid_{i:02d}.flac"
            assert moved_file.exists()

        # Cleanup
        await operator.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])