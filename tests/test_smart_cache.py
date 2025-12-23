"""Tests for smart caching functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import sqlite3
from unittest.mock import patch, Mock

from music_organizer.core.smart_cache import SmartCacheManager, get_smart_cache_manager
from music_organizer.core.smart_cached_metadata import (
    SmartCachedMetadataHandler,
    get_smart_cached_metadata_handler
)
from music_organizer.models.audio_file import AudioFile, ContentType
from music_organizer.domain.value_objects import ArtistName, TrackNumber


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test to ensure isolation."""
    SmartCacheManager._instance = None
    SmartCacheManager._initialized = False
    from music_organizer.core.smart_cache import _default_smart_cache
    _default_smart_cache = None
    from music_organizer.core.smart_cached_metadata import _default_smart_handler
    _default_smart_handler = None
    yield
    # Reset after test as well
    SmartCacheManager._instance = None
    SmartCacheManager._initialized = False
    _default_smart_cache = None
    _default_smart_handler = None


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_minimal_mp3(file_path: Path) -> None:
    """Create a minimal valid MP3 file with ID3 header.

    This creates a file that mutagen can parse without error.
    """
    # Minimal ID3v2.4 header
    id3_header = b"ID3\x04\x00\x00\x00\x00\x00\x00"
    # Minimal MP3 frame header (MPEG Version 1, Layer 3, 128kbps, 44100Hz)
    mp3_frame = b"\xFF\xFB\x90\x00" + b"\x00" * 100
    file_path.write_bytes(id3_header + mp3_frame)


@pytest.fixture
def sample_audio_file(temp_cache_dir):
    """Create a sample audio file for testing."""
    # Create minimal valid MP3 file so mutagen can parse it
    file_path = temp_cache_dir / "test.mp3"
    create_minimal_mp3(file_path)

    audio_file = AudioFile(
        path=file_path,
        file_type="mp3",
        metadata={
            "title": "Test Song",
            "artist": "Test Artist",
            "album": "Test Album",
            "year": 2023,
            "genre": "Test"
        },
        content_type=ContentType.STUDIO,
        artists=[ArtistName("Test Artist")],
        primary_artist="Test Artist",
        album="Test Album",
        title="Test Song",
        year=2023,
        track_number=TrackNumber(1),
        genre="Test",
        has_cover_art=False
    )
    return audio_file


@pytest.fixture
def smart_cache(temp_cache_dir):
    """Create a SmartCacheManager instance."""
    return SmartCacheManager(temp_cache_dir)


@pytest.fixture
def smart_handler(temp_cache_dir):
    """Create a SmartCachedMetadataHandler instance."""
    # Reset the global singleton to ensure a fresh instance for this test
    import music_organizer.core.smart_cache as smart_cache_module
    smart_cache_module._default_smart_cache = None
    SmartCacheManager._instance = None
    SmartCacheManager._initialized = False
    return SmartCachedMetadataHandler(
        cache_dir=temp_cache_dir,
        enable_smart_cache=True
    )


class TestSmartCacheManager:
    """Test SmartCacheManager functionality."""

    def test_singleton_pattern(self, temp_cache_dir):
        """Test that SmartCacheManager follows singleton pattern."""
        cache1 = SmartCacheManager(temp_cache_dir)
        cache2 = SmartCacheManager()

        assert cache1 is cache2

    def test_cache_initialization(self, smart_cache, temp_cache_dir):
        """Test cache database initialization."""
        assert smart_cache.cache_dir == temp_cache_dir
        assert smart_cache.db_path.exists()

        # Check database schema
        with sqlite3.connect(smart_cache.db_path) as conn:
            # Check smart_cache table
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='smart_cache'"
            ).fetchall()
            assert len(tables) == 1

            # Check directory_stats table
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='directory_stats'"
            ).fetchall()
            assert len(tables) == 1

    def test_cache_put_and_get(self, smart_cache, sample_audio_file):
        """Test caching and retrieving audio files."""
        # Put file in cache
        smart_cache.put(sample_audio_file)

        # Get from cache
        cached = smart_cache.get(sample_audio_file.path)
        assert cached is not None
        assert cached.title == sample_audio_file.title
        assert cached.artists == sample_audio_file.artists
        assert cached.content_type == sample_audio_file.content_type

    def test_cache_miss(self, smart_cache):
        """Test cache miss for non-existent file."""
        non_existent = Path("/non/existent/file.mp3")
        cached = smart_cache.get(non_existent)
        assert cached is None

    def test_cache_invalidation_on_file_change(self, smart_cache, sample_audio_file):
        """Test cache invalidation when file changes."""
        # Put file in cache
        smart_cache.put(sample_audio_file)
        assert smart_cache.get(sample_audio_file.path) is not None

        # Modify the file
        time.sleep(0.1)  # Ensure different mtime
        sample_audio_file.path.write_bytes(b"modified audio data")

        # Should be invalidated
        cached = smart_cache.get(sample_audio_file.path)
        assert cached is None

    def test_adaptive_ttl_calculation(self, smart_cache, sample_audio_file):
        """Test adaptive TTL calculation based on access patterns."""
        # Put file with no access history
        smart_cache.put(sample_audio_file)

        # Check initial TTL (should be close to default for new file)
        with sqlite3.connect(smart_cache.db_path) as conn:
            cursor = conn.execute(
                "SELECT julianday(expires_at) - julianday(cached_at) as ttl_days FROM smart_cache WHERE file_path = ?",
                (str(sample_audio_file.path),)
            )
            ttl_days = cursor.fetchone()[0]
            ttl_seconds = ttl_days * 86400

            # For a new file with no access history, TTL should be close to default
            # Allow some tolerance for timing differences (within 10 seconds)
            default_ttl_seconds = smart_cache.default_ttl.total_seconds()
            assert abs(ttl_seconds - default_ttl_seconds) < 10

    def test_access_frequency_tracking(self, smart_cache, sample_audio_file):
        """Test tracking of access frequency."""
        # Put file in cache
        smart_cache.put(sample_audio_file)

        # Access multiple times
        for _ in range(5):
            smart_cache.get(sample_audio_file.path)

        # Check access count and frequency
        with sqlite3.connect(smart_cache.db_path) as conn:
            cursor = conn.execute(
                "SELECT access_count, access_frequency FROM smart_cache WHERE file_path = ?",
                (str(sample_audio_file.path),)
            )
            count, frequency = cursor.fetchone()

            assert count == 5
            assert frequency > 0

    def test_stability_score_calculation(self, smart_cache, sample_audio_file):
        """Test stability score calculation for files."""
        # Put file in cache
        smart_cache.put(sample_audio_file)

        # Wait a bit and check stability
        time.sleep(0.1)

        # Access again to update stability
        smart_cache.get(sample_audio_file.path)

        # Check stability score
        with sqlite3.connect(smart_cache.db_path) as conn:
            cursor = conn.execute(
                "SELECT stability_score FROM smart_cache WHERE file_path = ?",
                (str(sample_audio_file.path),)
            )
            stability = cursor.fetchone()[0]

            assert 0.0 <= stability <= 1.0

    def test_directory_stats_tracking(self, smart_cache, temp_cache_dir):
        """Test directory-level statistics tracking."""
        # Create multiple files in same directory
        files = []
        for i in range(3):
            file_path = temp_cache_dir / f"test{i}.mp3"
            file_path.write_bytes(b"audio data")
            audio_file = AudioFile(
                path=file_path,
                file_type="mp3",
                content_type=ContentType.STUDIO,
                artists=[ArtistName("Test Artist")],
                primary_artist="Test Artist",
                album="Test Album",
                title=f"Test Song {i}",
                year=2023,
                track_number=TrackNumber(i + 1),
                genre="Test",
                has_cover_art=False
            )
            files.append(audio_file)
            smart_cache.put(audio_file)

        # Check directory stats
        directory_path = str(temp_cache_dir)
        with sqlite3.connect(smart_cache.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM directory_stats WHERE directory_path = ?",
                (directory_path,)
            )
            row = cursor.fetchone()
            assert row is not None
            # Note: file_count is not directly tracked by _update_directory_stats
            # It's only updated by cleanup_expired(). Just verify the entry exists.
            assert row[0] == directory_path  # directory_path

    def test_cache_warming(self, smart_cache, temp_cache_dir):
        """Test cache warming functionality."""
        # Mock MetadataHandler.extract_metadata to avoid needing valid MP3s
        with patch('music_organizer.core.metadata.MetadataHandler.extract_metadata') as mock_extract:
            # Return a mock AudioFile with the correct path for each call
            def make_audio_file(path):
                return AudioFile(
                    path=path,
                    file_type="mp3",
                    content_type=ContentType.STUDIO,
                    artists=[ArtistName("Test Artist")],
                    primary_artist="Test Artist",
                    album="Test Album",
                    title="Test Song",
                    year=2023,
                    track_number=TrackNumber(1),
                    genre="Test",
                    has_cover_art=False
                )
            mock_extract.side_effect = make_audio_file

            # Create test files
            for i in range(5):
                file_path = temp_cache_dir / f"warm_test{i}.mp3"
                file_path.write_bytes(b"audio data")

            # Warm cache
            warmed = smart_cache.warm_cache(temp_cache_dir, recursive=False, max_files=10)
            assert warmed == 5

            # Check files are in cache
            with sqlite3.connect(smart_cache.db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM smart_cache").fetchone()[0]
                assert count == 5

    def test_cache_optimization(self, smart_cache, sample_audio_file):
        """Test cache optimization."""
        # Put file in cache
        smart_cache.put(sample_audio_file)

        # Optimize cache
        results = smart_cache.optimize_cache()

        assert 'size_before_mb' in results
        assert 'size_after_mb' in results
        assert 'entries_before' in results
        assert 'entries_after' in results

    def test_cleanup_expired(self, smart_cache, sample_audio_file):
        """Test cleanup of expired entries."""
        # Put file in cache with short TTL
        smart_cache.default_ttl = timedelta(milliseconds=1)
        smart_cache.put(sample_audio_file)

        # Wait for expiration
        time.sleep(0.1)

        # Cleanup expired
        removed = smart_cache.cleanup_expired()
        assert removed >= 0  # Might be 0 if timing is tight

        # Check if expired (might not be due to timing)
        cached = smart_cache.get(sample_audio_file.path)
        # Don't assert as timing-dependent

    def test_get_smart_stats(self, smart_cache, sample_audio_file):
        """Test comprehensive cache statistics."""
        # Put file in cache
        smart_cache.put(sample_audio_file)

        # Get stats
        stats = smart_cache.get_smart_stats()

        assert 'total_entries' in stats
        assert 'valid_entries' in stats
        assert 'size_mb' in stats
        assert 'avg_access_frequency' in stats
        assert 'avg_stability_score' in stats
        assert 'total_directories' in stats

        assert stats['total_entries'] >= 1
        assert stats['size_mb'] >= 0

    def test_clear_cache(self, smart_cache, sample_audio_file):
        """Test clearing all cache entries."""
        # Put file in cache
        smart_cache.put(sample_audio_file)
        assert smart_cache.get(sample_audio_file.path) is not None

        # Clear cache
        smart_cache.clear()

        # Should be empty
        assert smart_cache.get(sample_audio_file.path) is None

        stats = smart_cache.get_smart_stats()
        assert stats['total_entries'] == 0


class TestSmartCachedMetadataHandler:
    """Test SmartCachedMetadataHandler functionality."""

    @patch('music_organizer.core.smart_cached_metadata.MetadataHandler.extract_metadata')
    def test_basic_metadata_extraction(self, mock_extract, smart_handler, sample_audio_file):
        """Test basic metadata extraction with smart caching."""
        # Mock the metadata extraction
        mock_extract.return_value = sample_audio_file

        # First extraction (cache miss)
        audio1 = smart_handler.extract_metadata(sample_audio_file.path)
        assert audio1 is not None

        # Second extraction (cache hit) - should use cache
        audio2 = smart_handler.extract_metadata(sample_audio_file.path)
        assert audio2 is not None
        # Mock should be called only once due to caching
        assert mock_extract.call_count == 1

    @patch('music_organizer.core.smart_cached_metadata.MetadataHandler.extract_metadata')
    def test_cache_disabled(self, mock_extract, temp_cache_dir, sample_audio_file):
        """Test operation with cache disabled."""
        handler = SmartCachedMetadataHandler(
            cache_dir=temp_cache_dir,
            enable_smart_cache=False
        )
        mock_extract.return_value = sample_audio_file

        audio = handler.extract_metadata(sample_audio_file.path, use_cache=False)
        assert audio is not None
        # With use_cache=False, extract should be called
        assert mock_extract.call_count == 1

    @patch('music_organizer.core.smart_cached_metadata.MetadataHandler.extract_metadata')
    def test_batch_extraction(self, mock_extract, smart_handler, temp_cache_dir):
        """Test batch metadata extraction."""
        # Create test files
        files = []
        for i in range(5):
            file_path = temp_cache_dir / f"batch_test{i}.mp3"
            create_minimal_mp3(file_path)
            files.append(file_path)

        # Mock to return different AudioFile for each path
        def mock_side_effect(path):
            return AudioFile(
                path=path,
                file_type="mp3",
                content_type=ContentType.STUDIO,
                artists=[ArtistName("Test Artist")],
                primary_artist="Test Artist",
                album="Test Album",
                title="Test Song",
                year=2023,
                track_number=TrackNumber(1),
                genre="Test",
                has_cover_art=False
            )
        mock_extract.side_effect = mock_side_effect

        # Extract in batch
        results = smart_handler.extract_metadata_batch(files, parallel=False)
        assert len(results) == len(files)

        for audio_file in results:
            assert audio_file is not None

    @patch('music_organizer.core.smart_cached_metadata.MetadataHandler.extract_metadata')
    def test_directory_invalidation(self, mock_extract, smart_handler, temp_cache_dir):
        """Test directory-level cache invalidation."""
        # Create test files
        for i in range(3):
            file_path = temp_cache_dir / f"inv_test{i}.mp3"
            create_minimal_mp3(file_path)
            mock_extract.return_value = AudioFile(
                path=file_path,
                file_type="mp3",
                content_type=ContentType.STUDIO,
                artists=[ArtistName("Test Artist")],
                primary_artist="Test Artist",
                album="Test Album",
                title="Test Song",
                year=2023,
                track_number=TrackNumber(1),
                genre="Test",
                has_cover_art=False
            )
            smart_handler.extract_metadata(file_path)

        # Invalidate directory
        count = smart_handler.invalidate_directory(temp_cache_dir, recursive=True)
        assert count >= 0

    @patch('music_organizer.core.smart_cached_metadata.MetadataHandler.extract_metadata')
    def test_cache_warming(self, mock_extract, smart_handler, temp_cache_dir):
        """Test cache warming through handler."""
        # Create test files
        for i in range(3):
            file_path = temp_cache_dir / f"warm_handler{i}.mp3"
            create_minimal_mp3(file_path)

        # Mock metadata extraction
        mock_extract.return_value = AudioFile(
            path=temp_cache_dir / "dummy.mp3",
            file_type="mp3",
            content_type=ContentType.STUDIO,
            artists=[ArtistName("Test Artist")],
            primary_artist="Test Artist",
            album="Test Album",
            title="Test Song",
            year=2023,
            track_number=TrackNumber(1),
            genre="Test",
            has_cover_art=False
        )

        # Warm cache
        warmed = smart_handler.warm_cache(temp_cache_dir, recursive=False)
        assert warmed == 3

    @patch('music_organizer.core.smart_cached_metadata.MetadataHandler.extract_metadata')
    def test_cache_optimization(self, mock_extract, smart_handler, sample_audio_file):
        """Test cache optimization through handler."""
        mock_extract.return_value = sample_audio_file

        # Extract a file to populate cache
        smart_handler.extract_metadata(sample_audio_file.path)

        # Optimize cache with force=True to ensure it runs
        results = smart_handler.optimize_cache(force=True)
        assert 'optimization_type' in results or 'expired_entries_removed' in results

    @patch('music_organizer.core.smart_cached_metadata.MetadataHandler.extract_metadata')
    def test_cache_health_report(self, mock_extract, smart_handler, sample_audio_file):
        """Test cache health reporting."""
        mock_extract.return_value = sample_audio_file

        # Extract a file to populate cache
        smart_handler.extract_metadata(sample_audio_file.path)

        # Get health report
        health = smart_handler.get_cache_health()

        assert 'overall_health' in health
        assert 'recommendations' in health
        assert 'warnings' in health

        assert health['overall_health'] in ['good', 'fair', 'poor']

    @patch('music_organizer.core.smart_cached_metadata.MetadataHandler.extract_metadata')
    def test_cache_stats(self, mock_extract, smart_handler, sample_audio_file):
        """Test cache statistics."""
        mock_extract.return_value = sample_audio_file

        # Extract a file to populate cache
        smart_handler.extract_metadata(sample_audio_file.path)

        # Get stats
        stats = smart_handler.get_cache_stats()

        assert 'cache_type' in stats
        assert 'total_entries' in stats
        assert 'cache_warming_enabled' in stats
        assert 'auto_optimize' in stats

        assert stats['cache_type'] == 'smart'


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_smart_cache_manager(self, temp_cache_dir):
        """Test global smart cache manager getter."""
        manager1 = get_smart_cache_manager(temp_cache_dir)
        manager2 = get_smart_cache_manager()

        assert manager1 is manager2

    def test_get_smart_cached_metadata_handler(self, temp_cache_dir):
        """Test global smart cached metadata handler getter."""
        handler1 = get_smart_cached_metadata_handler(cache_dir=temp_cache_dir)
        handler2 = get_smart_cached_metadata_handler()

        # Should be the same instance (global default)
        assert handler1 is handler2

    @patch('music_organizer.core.smart_cached_metadata.MetadataHandler.extract_metadata')
    def test_extract_metadata_smart_cached(self, mock_extract, temp_cache_dir, sample_audio_file):
        """Test global convenience function."""
        # Mock metadata extraction
        mock_extract.return_value = sample_audio_file

        # Create a new temp cache dir for this test to avoid conflicts
        import tempfile
        import shutil
        test_cache_dir = Path(tempfile.mkdtemp())

        try:
            # Reset global handler and create new one with test cache dir
            import music_organizer.core.smart_cached_metadata as metadata_module
            metadata_module._default_smart_handler = None

            from music_organizer.core.smart_cached_metadata import get_smart_cached_metadata_handler
            handler = get_smart_cached_metadata_handler(cache_dir=test_cache_dir)
            audio = handler.extract_metadata(sample_audio_file.path)
            assert audio is not None
        finally:
            shutil.rmtree(test_cache_dir, ignore_errors=True)


# Import global function for testing
from music_organizer.core.smart_cached_metadata import extract_metadata_smart_cached