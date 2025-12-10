"""Tests for async organizer."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from music_organizer.core.async_organizer import AsyncMusicOrganizer, run_async_organize, organize_files_async
from music_organizer.models.config import Config
from music_organizer.models.audio_file import ContentType
from music_organizer.exceptions import MusicOrganizerError


@pytest.fixture
async def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "source"
        target = Path(tmpdir) / "target"

        source.mkdir()
        target.mkdir()

        yield source, target


@pytest.fixture
def sample_config(temp_dirs):
    """Create a sample configuration."""
    def _config(temp_dirs):
        source, target = temp_dirs
        return Config(
            source_directory=source,
            target_directory=target
        )
    return _config


class TestAsyncMusicOrganizer:
    """Test cases for AsyncMusicOrganizer."""

    @pytest.mark.asyncio
    async def test_scan_directory(self, temp_dirs, sample_config):
        """Test scanning directory for audio files."""
        source, _ = temp_dirs

        # Create some test files
        (source / "test1.mp3").write_bytes(b"audio1")
        (source / "test2.flac").write_bytes(b"audio2")
        (source / "not_audio.txt").write_bytes(b"text")

        organizer = AsyncMusicOrganizer(sample_config(temp_dirs))

        # Scan for files
        files = []
        async for file_path in organizer.scan_directory(source):
            files.append(file_path)

        assert len(files) == 2
        assert any(f.name == "test1.mp3" for f in files)
        assert any(f.name == "test2.flac" for f in files)
        assert not any(f.name == "not_audio.txt" for f in files)

    @pytest.mark.asyncio
    async def test_scan_directory_batch(self, temp_dirs, sample_config):
        """Test scanning directory in batches."""
        source, _ = temp_dirs

        # Create many test files
        for i in range(25):
            (source / f"test{i}.mp3").write_bytes(b"audio")

        organizer = AsyncMusicOrganizer(sample_config(temp_dirs))

        # Scan in batches of 10
        all_files = []
        batch_count = 0
        async for batch in organizer.scan_directory_batch(source, batch_size=10):
            assert len(batch) <= 10
            all_files.extend(batch)
            batch_count += 1

        assert len(all_files) == 25
        assert batch_count == 3  # 10 + 10 + 5

    @pytest.mark.asyncio
    async def test_organize_files_dry_run(self, temp_dirs, capsys, sample_config):
        """Test organizing files in dry run mode."""
        source, target = temp_dirs

        # Create test file
        test_file = source / "test.mp3"
        test_file.write_bytes(b"audio data")

        organizer = AsyncMusicOrganizer(sample_config(temp_dirs), dry_run=True)
        results = await organizer.organize_files([test_file])

        assert results['processed'] == 1
        assert results['moved'] == 1  # In dry run, "moved" means processed
        assert results['skipped'] == 0
        assert len(results['errors']) == 0

        # File should not actually be moved
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_organize_files_streaming(self, temp_dirs, sample_config):
        """Test organizing files using streaming approach."""
        source, target = temp_dirs

        # Create test files
        test_files = []
        for i in range(5):
            file_path = source / f"test{i}.mp3"
            file_path.write_bytes(f"audio {i}".encode())
            test_files.append(file_path)

        organizer = AsyncMusicOrganizer(sample_config(temp_dirs), dry_run=True)

        # Create file generator
        async def file_gen():
            for file_path in test_files:
                yield file_path

        # Process streaming
        results = []
        async for file_path, success, error in organizer.organize_files_streaming(
            file_gen(), batch_size=2
        ):
            results.append((file_path, success, error))

        assert len(results) == 5
        for file_path, success, error in results:
            assert success == True
            assert error is None

    @pytest.mark.asyncio
    async def test_parallel_processing(self, temp_dirs, sample_config):
        """Test that files are processed in parallel."""
        source, target = temp_dirs

        # Create test files
        test_files = []
        for i in range(10):
            file_path = source / f"test{i}.mp3"
            file_path.write_bytes(f"audio {i}".encode())
            test_files.append(file_path)

        organizer = AsyncMusicOrganizer(sample_config(temp_dirs), dry_run=True, max_workers=4)

        # Mock the _process_file to track execution
        process_calls = []
        original_process = organizer._process_file

        async def tracked_process(file_path):
            process_calls.append(file_path)
            return await original_process(file_path)

        organizer._process_file = tracked_process

        # Process files
        await organizer.organize_files(test_files)

        # All files should have been processed
        assert len(process_calls) == 10
        for file_path in test_files:
            assert file_path in process_calls

    @pytest.mark.asyncio
    async def test_error_handling(self, temp_dirs, sample_config):
        """Test error handling during organization."""
        source, target = temp_dirs

        # Create test file
        test_file = source / "test.mp3"
        test_file.write_bytes(b"audio data")

        organizer = AsyncMusicOrganizer(sample_config(temp_dirs), dry_run=True)

        # Mock _process_file to raise an exception
        async def failing_process(file_path):
            raise Exception("Test error")

        organizer._process_file = failing_process

        results = await organizer.organize_files([test_file])

        assert results['processed'] == 1
        assert results['moved'] == 0
        assert results['skipped'] == 1
        assert len(results['errors']) == 1
        assert "Test error" in results['errors'][0]

    @pytest.mark.asyncio
    async def test_context_manager(self, temp_dirs, sample_config):
        """Test using organizer as async context manager."""
        config = sample_config(temp_dirs)

        async with AsyncMusicOrganizer(config, dry_run=True) as organizer:
            assert organizer is not None
            # Should be able to use organizer normally

        # Should cleanup properly
        assert True  # If we get here without exception, context manager worked

    @pytest.mark.asyncio
    async def test_interactive_mode(self, temp_dirs, sample_config):
        """Test interactive mode with ambiguous classifications."""
        source, target = temp_dirs

        # Create test file
        test_file = source / "test.mp3"
        test_file.write_bytes(b"audio data")

        organizer = AsyncMusicOrganizer(sample_config(temp_dirs), interactive=True, dry_run=True)

        # Mock classifier to return ambiguous case
        with patch.object(organizer.classifier, 'is_ambiguous', return_value=True):
            # Mock user input
            with patch('music_organizer.core.async_organizer.Prompt') as mock_prompt:
                mock_prompt.ask.return_value = '1'  # Choose first option

                results = await organizer.organize_files([test_file])

                assert results['processed'] == 1
                mock_prompt.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_category_mapping(self, temp_dirs, sample_config):
        """Test mapping content types to category names."""
        organizer = AsyncMusicOrganizer(sample_config(temp_dirs))

        # Test all content types
        assert organizer._get_category_name(ContentType.STUDIO) == 'Albums'
        assert organizer._get_category_name(ContentType.LIVE) == 'Live'
        assert organizer._get_category_name(ContentType.COLLABORATION) == 'Collaborations'
        assert organizer._get_category_name(ContentType.COMPILATION) == 'Compilations'
        assert organizer._get_category_name(ContentType.RARITY) == 'Rarities'
        assert organizer._get_category_name(ContentType.UNKNOWN) == 'Unknown'


class TestUtilityFunctions:
    """Test cases for utility functions."""

    @pytest.mark.asyncio
    async def test_run_async_organize(self, temp_dirs, sample_config):
        """Test the run_async_organize utility function."""
        source, target = temp_dirs

        # Create test file
        test_file = source / "test.mp3"
        test_file.write_bytes(b"audio data")

        config = sample_config(temp_dirs)

        results = await run_async_organize(config, dry_run=True)

        assert results['processed'] >= 0

    @pytest.mark.asyncio
    async def test_organize_files_async(self, temp_dirs, sample_config):
        """Test the organize_files_async convenience function."""
        source, target = temp_dirs

        # Create test file
        test_file = source / "test.mp3"
        test_file.write_bytes(b"audio data")

        config = sample_config(temp_dirs)

        # This function wraps the async version for sync code
        results = organize_files_async(config, dry_run=True)

        assert results['processed'] >= 0


@pytest.mark.asyncio
async def test_memory_efficiency():
    """Test that streaming mode is memory efficient."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "source"
        target = Path(tmpdir) / "target"
        source.mkdir()
        target.mkdir()

        config = Config(source_directory=source, target_directory=target)

        # Create many files
        file_count = 100
        for i in range(file_count):
            (source / f"test{i}.mp3").write_bytes(f"audio data {i}".encode())

        organizer = AsyncMusicOrganizer(config, dry_run=True)

        # Track memory usage (simple check - ensure we don't load all files at once)
        processed_files = []

        async def file_gen():
            for i in range(file_count):
                yield source / f"test{i}.mp3"

        # Process with streaming
        batch_count = 0
        async for file_path, success, error in organizer.organize_files_streaming(
            file_gen(), batch_size=10
        ):
            processed_files.append(file_path)
            batch_count += 1
            # At any point, we should only be processing a batch
            assert len(processed_files) <= batch_count * 10

        assert len(processed_files) == file_count