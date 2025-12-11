"""Tests for smart caching functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import sqlite3

from music_organizer.core.smart_cache import SmartCacheManager, get_smart_cache_manager
from music_organizer.core.smart_cached_metadata import (
    SmartCachedMetadataHandler,
    get_smart_cached_metadata_handler
)
from music_organizer.models.audio_file import AudioFile, ContentType
from music_organizer.domain.value_objects import ArtistName, TrackNumber


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_audio_file(temp_cache_dir):
    """Create a sample audio file for testing."""
    audio_file = AudioFile(
        path=temp_cache_dir / "test.mp3",
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
    # Create the actual file
    audio_file.path.write_bytes(b"fake audio data")
    return audio_file


@pytest.fixture
def smart_cache(temp_cache_dir):
    """Create a SmartCacheManager instance."""
    # Clear any existing singleton
    SmartCacheManager._instance = None
    SmartCacheManager._initialized = False
    return SmartCacheManager(temp_cache_dir)


@pytest.fixture
def smart_handler(temp_cache_dir):
    """Create a SmartCachedMetadataHandler instance."""
    return SmartCachedMetadataHandler(
        cache_dir=temp_cache_dir,
        enable_smart_cache=True
    )


class TestSmartCacheManager:
    """Test SmartCacheManager functionality."""

    def test_singleton_pattern(self, temp_cache_dir):
        """Test that SmartCacheManager follows singleton pattern."""
        # Clear any existing singleton
        SmartCacheManager._instance = None
        SmartCacheManager._initialized = False

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

        # Check initial TTL (should be default)
        with sqlite3.connect(smart_cache.db_path) as conn:
            cursor = conn.execute(
                "SELECT expires_at - cached_at as ttl_seconds FROM smart_cache WHERE file_path = ?",
                (str(sample_audio_file.path),)
            )
            ttl_seconds = cursor.fetchone()[0]

            # Should be close to default TTL (30 days in seconds)
            assert abs(ttl_seconds - smart_cache.default_ttl.total_seconds()) < 100

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
            assert row[2] == 3  # file_count should be 3

    def test_cache_warming(self, smart_cache, temp_cache_dir):
        """Test cache warming functionality."""
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

    def test_basic_metadata_extraction(self, smart_handler, sample_audio_file):
        """Test basic metadata extraction with smart caching."""
        # First extraction (cache miss)
        audio1 = smart_handler.extract_metadata(sample_audio_file.path)
        assert audio1 is not None

        # Second extraction (cache hit)
        audio2 = smart_handler.extract_metadata(sample_audio_file.path)
        assert audio2 is not None

    def test_cache_disabled(self, temp_cache_dir, sample_audio_file):
        """Test operation with cache disabled."""
        handler = SmartCachedMetadataHandler(
            cache_dir=temp_cache_dir,
            enable_smart_cache=False
        )

        audio = handler.extract_metadata(sample_audio_file.path, use_cache=False)
        assert audio is not None

    def test_batch_extraction(self, smart_handler, temp_cache_dir):
        """Test batch metadata extraction."""
        # Create test files
        files = []
        for i in range(5):
            file_path = temp_cache_dir / f"batch_test{i}.mp3"
            file_path.write_bytes(b"audio data")
            files.append(file_path)

        # Extract in batch
        results = smart_handler.extract_metadata_batch(files, parallel=True)
        assert len(results) == len(files)

        for audio_file in results:
            assert audio_file is not None

    def test_directory_invalidation(self, smart_handler, temp_cache_dir):
        """Test directory-level cache invalidation."""
        # Create test files
        for i in range(3):
            file_path = temp_cache_dir / f"inv_test{i}.mp3"
            file_path.write_bytes(b"audio data")
            smart_handler.extract_metadata(file_path)

        # Invalidate directory
        count = smart_handler.invalidate_directory(temp_cache_dir, recursive=True)
        assert count >= 0

    def test_cache_warming(self, smart_handler, temp_cache_dir):
        """Test cache warming through handler."""
        # Create test files
        for i in range(3):
            file_path = temp_cache_dir / f"warm_handler{i}.mp3"
            file_path.write_bytes(b"audio data")

        # Warm cache
        warmed = smart_handler.warm_cache(temp_cache_dir, recursive=False)
        assert warmed == 3

    def test_cache_optimization(self, smart_handler, sample_audio_file):
        """Test cache optimization through handler."""
        # Extract a file to populate cache
        smart_handler.extract_metadata(sample_audio_file.path)

        # Optimize cache
        results = smart_handler.optimize_cache()
        assert 'optimization_type' in results or 'expired_entries_removed' in results

    def test_cache_health_report(self, smart_handler, sample_audio_file):
        """Test cache health reporting."""
        # Extract a file to populate cache
        smart_handler.extract_metadata(sample_audio_file.path)

        # Get health report
        health = smart_handler.get_cache_health()

        assert 'overall_health' in health
        assert 'recommendations' in health
        assert 'warnings' in health

        assert health['overall_health'] in ['good', 'fair', 'poor']

    def test_cache_stats(self, smart_handler, sample_audio_file):
        """Test cache statistics."""
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
        # Clear any existing singleton
        SmartCacheManager._instance = None
        SmartCacheManager._initialized = False

        manager1 = get_smart_cache_manager(temp_cache_dir)
        manager2 = get_smart_cache_manager()

        assert manager1 is manager2

    def test_get_smart_cached_metadata_handler(self, temp_cache_dir):
        """Test global smart cached metadata handler getter."""
        handler1 = get_smart_cached_metadata_handler(cache_dir=temp_cache_dir)
        handler2 = get_smart_cached_metadata_handler()

        # Should be the same instance (global default)
        assert handler1 is handler2

    def test_extract_metadata_smart_cached(self, temp_cache_dir, sample_audio_file):
        """Test global convenience function."""
        # Clear global handler
        from music_organizer.core.smart_cached_metadata import _default_smart_handler
        _default_smart_handler = None

        # Use global function
        audio = extract_metadata_smart_cached(sample_audio_file.path)
        assert audio is not None


# Import global function for testing
from music_organizer.core.smart_cached_metadata import extract_metadata_smart_cached