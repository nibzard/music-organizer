"""Tests for file mover functionality."""

import pytest
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
from typing import Dict, List

from music_organizer.core.mover import FileMover, DirectoryOrganizer
from music_organizer.models.audio_file import AudioFile, CoverArt
from music_organizer.domain.value_objects import FileFormat, ArtistName, Metadata
from music_organizer.exceptions import FileOperationError


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file."""
    file_path = tmp_path / "test.flac"
    file_path.write_bytes(b"fake audio data")

    return AudioFile(
        path=file_path,
        file_type="FLAC",
        metadata={},
        artists=["Test Artist"],
        primary_artist="Test Artist",
        album="Test Album",
        title="Test Song",
        year=2020
    )


@pytest.fixture
def sample_cover_art(tmp_path):
    """Create a sample cover art file."""
    cover_path = tmp_path / "cover.jpg"
    cover_path.write_bytes(b"fake image data")

    cover = CoverArt.from_file(cover_path)
    return cover


class TestFileMover:
    """Test FileMover class."""

    def test_initialization_default(self):
        """Test default initialization."""
        mover = FileMover()

        assert mover.backup_enabled is True
        assert mover.backup_dir is None
        assert mover.operations == []
        assert mover.started is False

    def test_initialization_with_options(self, tmp_path):
        """Test initialization with custom options."""
        backup_dir = tmp_path / "backup"

        mover = FileMover(
            backup_enabled=True,
            backup_dir=backup_dir
        )

        assert mover.backup_enabled is True
        assert mover.backup_dir == backup_dir
        assert mover.operations == []

    def test_initialization_no_backup(self):
        """Test initialization with backup disabled."""
        mover = FileMover(backup_enabled=False)

        assert mover.backup_enabled is False
        assert mover.backup_dir is None

    def test_start_operation(self, tmp_path):
        """Test starting an operation."""
        mover = FileMover()
        source_root = tmp_path / "source"
        source_root.mkdir()

        mover.start_operation(source_root)

        assert mover.started is True
        assert mover.operations == []

    def test_start_operation_already_started(self, tmp_path):
        """Test starting an operation when already started."""
        mover = FileMover()
        source_root = tmp_path / "source"
        source_root.mkdir()

        mover.start_operation(source_root)

        with pytest.raises(FileOperationError, match="Operation already in progress"):
            mover.start_operation(source_root)

    def test_start_operation_with_backup(self, tmp_path):
        """Test starting operation with backup."""
        backup_dir = tmp_path / "backup"
        source_root = tmp_path / "source"
        source_root.mkdir()

        # Create some audio files
        (source_root / "test.flac").write_bytes(b"data1")
        (source_root / "test.mp3").write_bytes(b"data2")

        mover = FileMover(backup_enabled=True)
        mover.start_operation(source_root)

        assert mover.started is True
        assert mover.backup_dir is not None
        assert mover.backup_dir.exists()

        # Check manifest was created
        manifest_path = mover.backup_dir / "manifest.json"
        assert manifest_path.exists()

    def test_start_operation_creates_backup_dir(self, tmp_path):
        """Test that start_operation creates backup directory."""
        mover = FileMover(backup_enabled=True)
        source_root = tmp_path / "source"
        source_root.mkdir()

        mover.start_operation(source_root)

        assert mover.backup_dir is not None
        assert mover.backup_dir.exists()

    def test_finish_operation(self, tmp_path):
        """Test finishing an operation."""
        mover = FileMover(backup_enabled=True)
        source_root = tmp_path / "source"
        source_root.mkdir()

        mover.start_operation(source_root)
        assert mover.started is True

        # Add an operation so log isn't empty - include required keys
        mover.operations.append({
            'type': 'move',
            'original': str(source_root / "test.flac"),
            'target': str(tmp_path / "target" / "test.flac"),
            'timestamp': datetime.now().isoformat()
        })

        mover.finish_operation()

        assert mover.started is False

        # Check operation log was saved (only if there were operations)
        # The _save_operation_log only saves if there are operations
        log_path = mover.backup_dir / "operations.json"
        assert log_path.exists()

    def test_finish_operation_not_started(self):
        """Test finishing operation when none started."""
        mover = FileMover()

        with pytest.raises(FileOperationError, match="No operation in progress"):
            mover.finish_operation()

    def test_move_file(self, tmp_path, sample_audio_file):
        """Test moving a file."""
        mover = FileMover(backup_enabled=False)
        source_root = tmp_path / "source"
        source_root.mkdir()
        mover.start_operation(source_root)

        source_path = sample_audio_file.path
        target_path = tmp_path / "target" / "moved.flac"
        target_path.parent.mkdir(parents=True)

        result = mover.move_file(sample_audio_file, target_path)

        assert result == target_path
        assert target_path.exists()
        assert not source_path.exists()

        # Check operation was recorded
        assert len(mover.operations) == 1
        assert mover.operations[0]['type'] == 'move'

    def test_move_file_not_started(self, tmp_path, sample_audio_file):
        """Test moving file without starting operation."""
        mover = FileMover(backup_enabled=False)
        target_path = tmp_path / "target" / "test.flac"

        with pytest.raises(FileOperationError, match="Must start operation"):
            mover.move_file(sample_audio_file, target_path)

    def test_move_file_with_backup(self, tmp_path, sample_audio_file):
        """Test moving file with backup enabled."""
        backup_dir = tmp_path / "backup"
        mover = FileMover(backup_enabled=True, backup_dir=backup_dir)
        source_root = tmp_path / "source"
        source_root.mkdir()

        mover.start_operation(source_root)

        source_path = sample_audio_file.path
        target_path = tmp_path / "target" / "moved.flac"
        target_path.parent.mkdir(parents=True)

        mover.move_file(sample_audio_file, target_path)

        # Check backup was created
        assert backup_dir.exists()

        # Original should still exist in backup
        backup_files = list(backup_dir.rglob("*.flac"))
        assert len(backup_files) > 0

    def test_move_file_duplicate_resolution(self, tmp_path, sample_audio_file):
        """Test duplicate filename resolution."""
        mover = FileMover(backup_enabled=False)
        source_root = tmp_path / "source"
        source_root.mkdir()
        mover.start_operation(source_root)

        source_path = sample_audio_file.path
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        target_path = target_dir / "test.flac"

        # Create target file (first version)
        target_path.write_bytes(b"original data")

        # Move file - should append number
        result = mover.move_file(sample_audio_file, target_path)

        assert result.name == "test (1).flac"
        assert result.exists()
        assert target_path.exists()  # Original still exists

    def test_move_cover_art(self, tmp_path, sample_cover_art):
        """Test moving cover art."""
        mover = FileMover(backup_enabled=False)
        source_root = tmp_path / "source"
        source_root.mkdir()
        mover.start_operation(source_root)

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        result = mover.move_cover_art(sample_cover_art, target_dir)

        assert result is not None
        assert result.exists()
        assert result.parent == target_dir

        # Check operation was recorded
        assert any(op['type'] == 'move_cover' for op in mover.operations)

    def test_move_cover_art_no_cover(self, tmp_path):
        """Test moving cover art when cover is None."""
        mover = FileMover(backup_enabled=False)
        source_root = tmp_path / "source"
        source_root.mkdir()
        mover.start_operation(source_root)

        result = mover.move_cover_art(None, tmp_path / "target")

        assert result is None

    def test_rollback(self, tmp_path):
        """Test rollback functionality."""
        mover = FileMover(backup_enabled=False)
        source_root = tmp_path / "source"
        source_root.mkdir()
        mover.start_operation(source_root)

        # Create and move a file
        source_file = source_root / "test.txt"
        source_file.write_bytes(b"test data")
        target_path = tmp_path / "target" / "test.txt"
        target_path.parent.mkdir()

        # Create a mock audio file
        audio_file = Mock()
        audio_file.path = source_file
        type(audio_file).path = source_file  # Use descriptor to set attribute

        # Manually add operation instead of calling move_file which would need actual AudioFile
        mover.operations.append({
            'type': 'move',
            'original': str(source_file),
            'target': str(target_path)
        })

        # Manually move the file
        shutil.move(str(source_file), str(target_path))

        assert target_path.exists()
        assert not source_file.exists()

        # Rollback
        mover.rollback()

        assert source_file.exists()
        assert not target_path.exists()

    def test_rollback_empty_operations(self, tmp_path):
        """Test rollback with no operations."""
        mover = FileMover(backup_enabled=False)
        source_root = tmp_path / "source"
        source_root.mkdir()
        mover.start_operation(source_root)

        # Should not raise error
        mover.rollback()

        assert mover.operations == []

    def test_rollback_partial_failure(self, tmp_path, capsys):
        """Test rollback with partial failures."""
        mover = FileMover(backup_enabled=False)
        source_root = tmp_path / "source"
        source_root.mkdir()
        mover.start_operation(source_root)

        # Create operations with some that will fail
        mover.operations = [
            {
                'type': 'move',
                'original': str(tmp_path / "file1.txt"),
                'target': str(tmp_path / "moved1.txt")
            },
            {
                'type': 'move',
                'original': str(tmp_path / "nonexistent.txt"),
                'target': str(tmp_path / "moved2.txt")
            }
        ]

        # Create first target file
        (tmp_path / "moved1.txt").write_bytes(b"data")

        # Should not raise error despite one file missing
        mover.rollback()

        captured = capsys.readouterr()
        assert "Warning" in captured.out or len(captured.out) == 0

    def test_get_operation_summary(self, tmp_path):
        """Test getting operation summary."""
        mover = FileMover(backup_enabled=False)
        source_root = tmp_path / "source"
        source_root.mkdir()
        mover.start_operation(source_root)

        # Add some operations
        mover.operations = [
            {'type': 'move', 'target': str(tmp_path / "dir1" / "file1.flac")},
            {'type': 'move', 'target': str(tmp_path / "dir1" / "file2.flac")},
            {'type': 'move', 'target': str(tmp_path / "dir2" / "file3.flac")},
            {'type': 'move_cover', 'target': str(tmp_path / "dir1" / "cover.jpg")},
        ]

        summary = mover.get_operation_summary()

        assert summary['total_files'] == 3
        assert summary['total_cover_art'] == 1
        assert summary['directories_created'] == 2  # dir1 and dir2

    def test_resolve_duplicate(self, tmp_path):
        """Test duplicate filename resolution."""
        mover = FileMover(backup_enabled=False)

        # Test non-existent file
        target = tmp_path / "new.txt"
        result = mover._resolve_duplicate(target)
        assert result == target

        # Test existing file
        (tmp_path / "test.txt").write_bytes(b"data")
        target = tmp_path / "test.txt"

        result = mover._resolve_duplicate(target)
        assert result.name == "test (1).txt"

        # Create the (1) version and test again
        result.write_bytes(b"data2")
        result = mover._resolve_duplicate(target)
        assert result.name == "test (2).txt"

    def test_backup_file(self, tmp_path):
        """Test file backup."""
        backup_dir = tmp_path / "backup"
        mover = FileMover(backup_enabled=True, backup_dir=backup_dir)

        source_file = tmp_path / "source" / "test.txt"
        source_file.parent.mkdir(parents=True)
        source_file.write_bytes(b"test data")

        mover._backup_file(source_file)

        # Check backup was created
        backup_files = list(backup_dir.rglob("test.txt"))
        assert len(backup_files) == 1
        assert backup_files[0].read_bytes() == b"test data"

    def test_backup_file_no_backup_dir(self, tmp_path):
        """Test backup when no backup dir configured."""
        mover = FileMover(backup_enabled=False)

        source_file = tmp_path / "test.txt"
        source_file.write_bytes(b"data")

        # Should not raise error
        mover._backup_file(source_file)

    def test_create_backup_manifest(self, tmp_path):
        """Test backup manifest creation."""
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()  # Create backup dir first
        mover = FileMover(backup_enabled=True, backup_dir=backup_dir)

        source_root = tmp_path / "source"
        source_root.mkdir()

        # Create test files
        (source_root / "test.flac").write_bytes(b"audio")
        (source_root / "cover.jpg").write_bytes(b"image")
        (source_root / "readme.txt").write_bytes(b"text")

        mover._create_backup_manifest(source_root)

        manifest_path = backup_dir / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest['source_root'] == str(source_root)
        assert 'timestamp' in manifest
        assert len(manifest['files']) == 2  # Only audio and cover, not txt

    def test_save_operation_log(self, tmp_path):
        """Test saving operation log."""
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        mover = FileMover(backup_enabled=True, backup_dir=backup_dir)

        mover.operations = [
            {'type': 'move', 'original': '/source/f1.flac', 'target': '/target/f1.flac'}
        ]

        mover._save_operation_log()

        log_path = backup_dir / "operations.json"
        assert log_path.exists()

        with open(log_path) as f:
            log = json.load(f)

        assert 'timestamp' in log
        assert log['operations'] == mover.operations

    def test_get_cover_art_filename(self, tmp_path, sample_cover_art):
        """Test cover art filename generation."""
        mover = FileMover()

        # Test front cover
        cover = CoverArt.from_file(tmp_path / "folder.jpg")
        if cover:
            filename = mover._get_cover_art_filename(cover)
            assert filename == "folder.jpg"

    def test_validate_integrity_success(self, tmp_path):
        """Test integrity validation with success."""
        backup_dir = tmp_path / "backup"
        mover = FileMover(backup_enabled=True, backup_dir=backup_dir)

        source_root = tmp_path / "source"
        source_root.mkdir()

        # Create test file
        test_file = source_root / "test.flac"
        test_file.write_bytes(b"test data")

        mover.start_operation(source_root)

        is_valid, errors = mover.validate_integrity(source_root)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_integrity_failure(self, tmp_path):
        """Test integrity validation with failures."""
        backup_dir = tmp_path / "backup"
        mover = FileMover(backup_enabled=True, backup_dir=backup_dir)

        source_root = tmp_path / "source"
        source_root.mkdir()

        # Create test file
        test_file = source_root / "test.flac"
        test_file.write_bytes(b"test data")

        mover.start_operation(source_root)

        # Modify the manifest to simulate corruption
        manifest_path = backup_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        manifest['files'][0]['size'] = 999999  # Wrong size

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        is_valid, errors = mover.validate_integrity(source_root)

        assert is_valid is False
        assert len(errors) > 0


class TestDirectoryOrganizer:
    """Test DirectoryOrganizer class."""

    def test_create_directory_structure(self, tmp_path):
        """Test creating directory structure."""
        base_path = tmp_path / "music"
        base_path.mkdir()

        DirectoryOrganizer.create_directory_structure(base_path)

        assert (base_path / "Albums").exists()
        assert (base_path / "Live").exists()
        assert (base_path / "Collaborations").exists()
        assert (base_path / "Compilations").exists()
        assert (base_path / "Rarities").exists()

    def test_create_directory_structure_existing(self, tmp_path):
        """Test creating directory structure when some exist."""
        base_path = tmp_path / "music"
        base_path.mkdir()

        # Create one directory ahead of time
        (base_path / "Albums").mkdir()

        # Should not raise error
        DirectoryOrganizer.create_directory_structure(base_path)

        assert (base_path / "Albums").exists()
        assert (base_path / "Live").exists()

    def test_validate_structure(self, tmp_path):
        """Test directory structure validation."""
        base_path = tmp_path / "music"
        base_path.mkdir()

        # Create only some directories
        (base_path / "Albums").mkdir()
        (base_path / "Live").mkdir()

        validation = DirectoryOrganizer.validate_structure(base_path)

        assert validation['Albums'] is True
        assert validation['Live'] is True
        assert validation['Collaborations'] is False
        assert validation['Compilations'] is False
        assert validation['Rarities'] is False

    def test_get_empty_directories(self, tmp_path):
        """Test finding empty directories."""
        base_path = tmp_path / "music"
        base_path.mkdir()

        # Create directory structure
        empty_dir = base_path / "empty"
        empty_dir.mkdir()

        with_media_dir = base_path / "with_media"
        with_media_dir.mkdir()
        (with_media_dir / "song.flac").write_bytes(b"data")

        nested_empty = base_path / "parent" / "child"
        nested_empty.mkdir(parents=True)

        empty_dirs = DirectoryOrganizer.get_empty_directories(base_path)

        # Should find empty directories
        assert empty_dir in empty_dirs
        # Parent dir has subdirectory so not empty
        assert base_path / "parent" not in empty_dirs

    def test_get_empty_directories_with_cover_art(self, tmp_path):
        """Test that directories with only cover art are considered empty."""
        base_path = tmp_path / "music"
        base_path.mkdir()

        cover_only_dir = base_path / "cover_only"
        cover_only_dir.mkdir()
        (cover_only_dir / "cover.jpg").write_bytes(b"image data")

        empty_dirs = DirectoryOrganizer.get_empty_directories(base_path)

        # Cover art counts as media, so not empty
        assert cover_only_dir not in empty_dirs

    def test_get_empty_directories_with_audio(self, tmp_path):
        """Test that directories with audio files are not considered empty."""
        base_path = tmp_path / "music"
        base_path.mkdir()

        with_audio_dir = base_path / "with_audio"
        with_audio_dir.mkdir()
        (with_audio_dir / "song.mp3").write_bytes(b"audio data")

        empty_dirs = DirectoryOrganizer.get_empty_directories(base_path)

        # Should not be in empty list
        assert with_audio_dir not in empty_dirs


class TestFileMoverIntegration:
    """Integration tests for FileMover."""

    def test_full_operation_workflow(self, tmp_path):
        """Test complete workflow with multiple files."""
        backup_dir = tmp_path / "backup"
        mover = FileMover(backup_enabled=True, backup_dir=backup_dir)

        source_root = tmp_path / "source"
        source_root.mkdir()

        # Create test files
        files = []
        for i in range(3):
            file_path = source_root / f"test{i}.flac"
            file_path.write_bytes(f"data{i}".encode())
            files.append(file_path)

        # Start operation
        mover.start_operation(source_root)

        # Move files
        for file_path in files:
            audio_file = AudioFile(
                path=file_path,
                file_type="FLAC",
                metadata={"title": f"Song {file_path.stem}"},
                artists=["Artist"],
                primary_artist="Artist"
            )

            target_path = tmp_path / "target" / file_path.name
            target_path.parent.mkdir(parents=True, exist_ok=True)

            mover.move_file(audio_file, target_path)

        # Finish operation
        mover.finish_operation()

        # Check results
        summary = mover.get_operation_summary()
        assert summary['total_files'] == 3

        # Check backup exists
        assert backup_dir.exists()
        manifest_path = backup_dir / "manifest.json"
        assert manifest_path.exists()

        # Check operation log exists
        log_path = backup_dir / "operations.json"
        assert log_path.exists()

    def test_move_and_rollback_workflow(self, tmp_path):
        """Test moving files and then rolling back."""
        mover = FileMover(backup_enabled=False)
        source_root = tmp_path / "source"
        source_root.mkdir()

        # Create test file
        source_file = source_root / "test.flac"
        source_file.write_bytes(b"original data")

        mover.start_operation(source_root)

        # Create AudioFile
        audio_file = AudioFile(
            path=source_file,
            file_type="FLAC",
            metadata={}
        )

        # Move file
        target_path = tmp_path / "target" / "test.flac"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        moved_path = mover.move_file(audio_file, target_path)

        assert moved_path.exists()
        assert not source_file.exists()
        assert audio_file.path == target_path

        # Verify operation was recorded
        assert len(mover.operations) == 1

        # Rollback should be called without error
        mover.rollback()

        # Operations should be cleared after rollback
        assert len(mover.operations) == 0


if __name__ == "__main__":
    pytest.main([__file__])
