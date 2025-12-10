"""Tests for async file mover."""

import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from music_organizer.core.async_mover import AsyncFileMover, AsyncDirectoryOrganizer
from music_organizer.models.audio_file import AudioFile, CoverArt
from music_organizer.exceptions import FileOperationError


@pytest.fixture
async def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "source"
        target = Path(tmpdir) / "target"
        backup = Path(tmpdir) / "backup"

        source.mkdir()
        target.mkdir()
        backup.mkdir()

        yield source, target, backup


@pytest.fixture
async def sample_audio_file(temp_dirs):
    """Create a sample audio file for testing."""
    source, _, _ = temp_dirs
    audio_file = source / "test.mp3"
    audio_file.write_bytes(b"fake audio data")

    audio = AudioFile.from_path(audio_file)
    audio.artists = ["Test Artist"]
    audio.album = "Test Album"
    audio.title = "Test Track"

    return audio


class TestAsyncFileMover:
    """Test cases for AsyncFileMover."""

    @pytest.mark.asyncio
    async def test_start_and_finish_operation(self, temp_dirs):
        """Test starting and finishing operations."""
        source, _, backup = temp_dirs
        mover = AsyncFileMover(backup_enabled=True, backup_dir=backup)

        await mover.start_operation(source)
        assert mover.started == True

        await mover.finish_operation()
        assert mover.started == False

    @pytest.mark.asyncio
    async def test_move_file(self, temp_dirs, sample_audio_file):
        """Test moving a file asynchronously."""
        source, target, _ = temp_dirs
        mover = AsyncFileMover(backup_enabled=False)

        await mover.start_operation(source)

        # Move file
        target_path = target / "moved.mp3"
        result = await mover.move_file(sample_audio_file, target_path)

        assert result == target_path
        assert not sample_audio_file.path.exists()
        assert target_path.exists()
        assert sample_audio_file.path == target_path

        await mover.finish_operation()

    @pytest.mark.asyncio
    async def test_move_files_batch(self, temp_dirs):
        """Test moving multiple files in batch."""
        source, target, _ = temp_dirs
        mover = AsyncFileMover(backup_enabled=False)

        # Create multiple files
        files = []
        moves = []
        for i in range(5):
            audio_path = source / f"test{i}.mp3"
            audio_path.write_bytes(b"fake audio data")
            audio = AudioFile.from_path(audio_path)
            audio.artists = [f"Artist {i}"]
            files.append(audio)

            target_path = target / f"moved{i}.mp3"
            moves.append((audio, target_path))

        await mover.start_operation(source)

        # Move files in batch
        results = await mover.move_files_batch(moves)

        assert len(results) == 5
        for success, path, error in results:
            assert success == True
            assert path is not None
            assert error is None
            assert path.exists()

        await mover.finish_operation()

    @pytest.mark.asyncio
    async def test_duplicate_resolution(self, temp_dirs, sample_audio_file):
        """Test resolving duplicate filenames."""
        source, target, _ = temp_dirs
        mover = AsyncFileMover(backup_enabled=False)

        # Create target file
        target_file = target / "test.mp3"
        target_file.write_bytes(b"existing data")

        await mover.start_operation(source)

        # Move file should resolve duplicate
        result = await mover.move_file(sample_audio_file, target_file)

        # Should have created a duplicate with number
        assert result.name == "test (1).mp3"
        assert result.exists()
        assert target_file.exists()  # Original should still exist

        await mover.finish_operation()

    @pytest.mark.asyncio
    async def test_backup_creation(self, temp_dirs, sample_audio_file):
        """Test that backup files are created when enabled."""
        source, target, backup = temp_dirs
        mover = AsyncFileMover(backup_enabled=True, backup_dir=backup)

        await mover.start_operation(source)

        # Move file
        target_path = target / "moved.mp3"
        await mover.move_file(sample_audio_file, target_path)

        # Check backup was created
        backup_file = backup / "test.mp3"
        assert backup_file.exists()
        assert backup_file.read_bytes() == b"fake audio data"

        await mover.finish_operation()

    @pytest.mark.asyncio
    async def test_rollback(self, temp_dirs, sample_audio_file):
        """Test rolling back operations."""
        source, target, _ = temp_dirs
        mover = AsyncFileMover(backup_enabled=False)

        await mover.start_operation(source)

        # Move file
        original_path = sample_audio_file.path
        target_path = target / "moved.mp3"
        await mover.move_file(sample_audio_file, target_path)

        # Verify file moved
        assert not original_path.exists()
        assert target_path.exists()

        # Rollback
        await mover.rollback()

        # Verify file restored
        assert original_path.exists()
        assert not target_path.exists()

        await mover.finish_operation()

    @pytest.mark.asyncio
    async def test_move_cover_art(self, temp_dirs):
        """Test moving cover art files."""
        source, target, _ = temp_dirs
        mover = AsyncFileMover(backup_enabled=False)

        # Create cover art file
        cover_path = source / "cover.jpg"
        cover_path.write_bytes(b"fake image data")
        cover_art = CoverArt.from_file(cover_path)
        cover_art.type = "front"

        await mover.start_operation(source)

        # Move cover art
        result = await mover.move_cover_art(cover_art, target)

        assert result == target / "folder.jpg"
        assert not cover_path.exists()
        assert result.exists()

        await mover.finish_operation()

    @pytest.mark.asyncio
    async def test_context_manager(self, temp_dirs):
        """Test using mover as async context manager."""
        source, _, _ = temp_dirs

        async with AsyncFileMover(backup_enabled=False) as mover:
            await mover.start_operation(source)
            assert mover.started == True

        # Should automatically close executor
        assert mover.executor._shutdown == True

    @pytest.mark.asyncio
    async def test_error_handling(self, temp_dirs):
        """Test error handling for invalid operations."""
        source, _, _ = temp_dirs
        mover = AsyncFileMover(backup_enabled=False)

        # Test moving without starting operation
        audio = AudioFile.from_path(source / "nonexistent.mp3")
        with pytest.raises(FileOperationError, match="Must start operation"):
            await mover.move_file(audio, Path("/target/test.mp3"))

        # Test starting twice
        await mover.start_operation(source)
        with pytest.raises(FileOperationError, match="Operation already in progress"):
            await mover.start_operation(source)


class TestAsyncDirectoryOrganizer:
    """Test cases for AsyncDirectoryOrganizer."""

    @pytest.mark.asyncio
    async def test_create_directory_structure(self, temp_dirs):
        """Test creating directory structure asynchronously."""
        _, target, _ = temp_dirs

        await AsyncDirectoryOrganizer.create_directory_structure(target)

        # Check all directories were created
        expected_dirs = ["Albums", "Live", "Collaborations", "Compilations", "Rarities"]
        for dir_name in expected_dirs:
            assert (target / dir_name).exists()

    @pytest.mark.asyncio
    async def test_validate_structure(self, temp_dirs):
        """Test validating directory structure."""
        _, target, _ = temp_dirs

        # Create some directories
        (target / "Albums").mkdir()
        (target / "Live").mkdir()

        validation = await AsyncDirectoryOrganizer.validate_structure(target)

        assert validation["Albums"] == True
        assert validation["Live"] == True
        assert validation["Collaborations"] == False
        assert validation["Compilations"] == False
        assert validation["Rarities"] == False

    @pytest.mark.asyncio
    async def test_get_empty_directories(self, temp_dirs):
        """Test finding empty directories."""
        source, _, _ = temp_dirs

        # Create some directory structure
        (source / "empty1").mkdir()
        (source / "empty2").mkdir()
        (source / "has_audio").mkdir()
        (source / "has_audio" / "music.mp3").write_bytes(b"audio")

        empty_dirs = await AsyncDirectoryOrganizer.get_empty_directories(source)

        # Should find empty directories but not ones with audio
        empty_paths = [Path(d) for d in empty_dirs]
        assert (source / "empty1") in empty_paths
        assert (source / "empty2") in empty_paths
        assert (source / "has_audio") not in empty_paths
        assert (source / "has_audio" / "music.mp3") not in empty_paths


@pytest.mark.asyncio
async def test_concurrent_moves():
    """Test that concurrent moves don't interfere with each other."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "source"
        target = Path(tmpdir) / "target"
        source.mkdir()
        target.mkdir()

        # Create many files
        files = []
        moves = []
        for i in range(20):
            audio_path = source / f"test{i}.mp3"
            audio_path.write_bytes(f"audio data {i}".encode())
            audio = AudioFile.from_path(audio_path)
            files.append(audio)

            target_path = target / f"moved{i}.mp3"
            moves.append((audio, target_path))

        # Move with high concurrency
        mover = AsyncFileMover(backup_enabled=False, max_workers=8)
        await mover.start_operation(source)

        results = await mover.move_files_batch(moves)

        # All moves should succeed
        assert len(results) == 20
        for success, path, error in results:
            assert success == True
            assert error is None
            assert path.exists()

        await mover.finish_operation()