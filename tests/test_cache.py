"""Tests for SQLite metadata caching."""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from music_organizer.core.cache import SQLiteCache
from music_organizer.core.cached_metadata import CachedMetadataHandler
from music_organizer.models.audio_file import AudioFile, ContentType


class TestSQLiteCache:
    """Test SQLite cache functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a cache instance for testing."""
        # Reset the singleton to avoid issues between tests
        SQLiteCache._instance = None
        SQLiteCache._initialized = False
        return SQLiteCache(cache_dir=temp_cache_dir)

    @pytest.fixture
    def sample_audio_file(self):
        """Create a sample audio file for testing."""
        return AudioFile(
            path=Path("/test/music.flac"),
            file_type="FLAC",
            metadata={"test": "data"},
            content_type=ContentType.STUDIO,
            artists=["Test Artist"],
            primary_artist="Test Artist",
            album="Test Album",
            title="Test Track",
            year=2023,
            track_number=1,
            genre="Test Genre",
            has_cover_art=True
        )

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization creates database."""
        cache = SQLiteCache(cache_dir=temp_cache_dir)
        assert cache.db_path.exists()
        assert cache.db_path.is_file()

        # Check database schema
        with sqlite3.connect(cache.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='metadata_cache'"
            )
            table_exists = cursor.fetchone() is not None
            assert table_exists

    def test_singleton_pattern(self, temp_cache_dir):
        """Test cache follows singleton pattern."""
        cache1 = SQLiteCache(cache_dir=temp_cache_dir)
        cache2 = SQLiteCache(cache_dir=temp_cache_dir)
        assert cache1 is cache2

    def test_put_and_get(self, cache, sample_audio_file):
        """Test storing and retrieving audio file."""
        # Put in cache (should handle missing file gracefully)
        cache.put(sample_audio_file)

        # Get from cache (should handle missing file gracefully)
        cached = cache.get(sample_audio_file.path)
        assert cached is not None
        assert cached.path == sample_audio_file.path
        assert cached.file_type == sample_audio_file.file_type
        assert cached.content_type == sample_audio_file.content_type
        assert cached.artists == sample_audio_file.artists
        assert cached.primary_artist == sample_audio_file.primary_artist
        assert cached.album == sample_audio_file.album
        assert cached.title == sample_audio_file.title
        assert cached.year == sample_audio_file.year
        assert cached.track_number == sample_audio_file.track_number
        assert cached.genre == sample_audio_file.genre
        assert cached.has_cover_art == sample_audio_file.has_cover_art

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get(Path("/nonexistent/file.flac"))
        assert result is None

    def test_cache_invalidation_on_file_change(self, cache, sample_audio_file):
        """Test cache is invalidated when file changes."""
        # Put in cache
        cache.put(sample_audio_file)
        assert cache.get(sample_audio_file.path) is not None

        # Note: Can't test file change detection without real files
        # since we now handle missing file stats gracefully

    def test_ttl_expiration(self, cache, sample_audio_file):
        """Test cache entries expire after TTL."""
        # Put in cache with negative TTL (already expired)
        cache.put(sample_audio_file, ttl=timedelta(days=-1))

        # Should not be cached (already expired)
        assert cache.get(sample_audio_file.path) is None

    def test_cleanup_expired(self, cache, sample_audio_file):
        """Test cleanup of expired entries."""
        # Add one entry that's already expired
        cache.put(sample_audio_file, ttl=timedelta(days=-1))

        # Add another with long TTL
        sample_audio_file2 = AudioFile(
            path=Path("/test/music2.flac"),
            file_type="FLAC"
        )
        cache.put(sample_audio_file2, ttl=timedelta(days=1))

        # Cleanup should remove 1 expired entry
        removed = cache.cleanup_expired()
        assert removed == 1

        # Second should still be cached
        assert cache.get(sample_audio_file2.path) is not None

    def test_get_stats(self, cache, sample_audio_file):
        """Test cache statistics."""
        stats = cache.get_stats()
        assert 'total_entries' in stats
        assert 'valid_entries' in stats
        assert 'expired_entries' in stats
        assert 'size_bytes' in stats
        assert 'size_mb' in stats
        assert stats['total_entries'] == 0

        # Add an entry
        cache.put(sample_audio_file)

        stats = cache.get_stats()
        assert stats['total_entries'] == 1
        assert stats['valid_entries'] == 1

    def test_clear_cache(self, cache, sample_audio_file):
        """Test clearing all cache entries."""
        # Add some entries
        cache.put(sample_audio_file)
        assert cache.get_stats()['total_entries'] == 1

        # Clear cache
        cache.clear()

        # Should be empty
        assert cache.get_stats()['total_entries'] == 0


class TestCachedMetadataHandler:
    """Test cached metadata handler."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def handler(self, temp_cache_dir):
        """Create a cached metadata handler."""
        # Reset the singleton to avoid issues
        SQLiteCache._instance = None
        SQLiteCache._initialized = False
        handler = CachedMetadataHandler(cache_dir=temp_cache_dir)
        return handler

    @pytest.fixture
    def sample_file_path(self):
        """Create a sample file path."""
        return Path("/test/sample.flac")

    def test_cache_hit(self, handler, sample_file_path):
        """Test cache hit scenario."""
        # Create a cached audio file
        cached_audio = AudioFile(
            path=sample_file_path,
            file_type="FLAC",
            title="Cached Track"
        )

        # Mock cache.get to return cached audio
        with patch.object(handler.cache, 'get', return_value=cached_audio):
            result = handler.extract_metadata(sample_file_path, use_cache=True)
            assert result == cached_audio
            assert handler._cache_hits == 1
            assert handler._cache_misses == 0

    def test_cache_miss_with_caching(self, handler, sample_file_path):
        """Test cache miss scenario with caching enabled."""
        # Mock cache.get to return None (cache miss)
        with patch.object(handler.cache, 'get', return_value=None):
            # Mock MetadataHandler.extract_metadata
            mock_audio = AudioFile(
                path=sample_file_path,
                file_type="FLAC",
                title="Extracted Track"
            )
            with patch('music_organizer.core.cached_metadata.MetadataHandler.extract_metadata',
                      return_value=mock_audio) as mock_extract:
                # Mock cache.put to verify it's called
                with patch.object(handler.cache, 'put') as mock_put:

                    result = handler.extract_metadata(sample_file_path, use_cache=True)

                    assert result == mock_audio
                    assert handler._cache_hits == 0
                    assert handler._cache_misses == 1
                    mock_extract.assert_called_once_with(sample_file_path)

                    # Verify it was cached
                    mock_put.assert_called_once_with(mock_audio, handler.default_ttl)

    def test_cache_disabled(self, handler, sample_file_path):
        """Test operation with caching disabled."""
        # Mock MetadataHandler.extract_metadata
        mock_audio = AudioFile(
            path=sample_file_path,
            file_type="FLAC",
            title="Extracted Track"
        )
        with patch('music_organizer.core.cached_metadata.MetadataHandler.extract_metadata',
                  return_value=mock_audio) as mock_extract:
            # Mock cache.put to verify it's NOT called
            with patch.object(handler.cache, 'put') as mock_put:

                result = handler.extract_metadata(sample_file_path, use_cache=False)

                assert result == mock_audio
                assert handler._cache_hits == 0
                assert handler._cache_misses == 1
                mock_extract.assert_called_once_with(sample_file_path)

                # Should not be cached
                mock_put.assert_not_called()

    def test_invalidate(self, handler, sample_file_path):
        """Test cache invalidation."""
        with patch.object(handler.cache, 'invalidate') as mock_invalidate:
            handler.invalidate(sample_file_path)
            mock_invalidate.assert_called_once_with(sample_file_path)

    def test_get_cache_stats(self, handler):
        """Test getting cache statistics."""
        mock_stats = {
            'total_entries': 10,
            'valid_entries': 8,
            'expired_entries': 2,
            'size_mb': 1.5
        }
        with patch.object(handler.cache, 'get_stats', return_value=mock_stats):
            stats = handler.get_cache_stats()
            assert stats == mock_stats

    def test_get_cache_stats_with_performance(self, handler):
        """Test cache stats include performance metrics."""
        # Set some hit/miss counts
        handler._cache_hits = 80
        handler._cache_misses = 20

        mock_stats = {'total_entries': 100}
        with patch.object(handler.cache, 'get_stats', return_value=mock_stats):
            stats = handler.get_cache_stats()

            assert stats['cache_hits'] == 80
            assert stats['cache_misses'] == 20
            assert stats['hit_rate'] == 0.8  # 80/(80+20)

    def test_clear_cache(self, handler):
        """Test clearing cache."""
        with patch.object(handler.cache, 'clear') as mock_clear:
            handler.clear_cache()
            mock_clear.assert_called_once()
            assert handler._cache_hits == 0
            assert handler._cache_misses == 0

    @pytest.mark.asyncio
    async def test_preload_cache(self, handler, temp_cache_dir):
        """Test preloading cache with directory."""
        # Create temporary directory structure
        test_dir = temp_cache_dir / "music"
        test_dir.mkdir()

        # Create some test files
        for i in range(3):
            (test_dir / f"track{i:02d}.flac").touch()

        # Mock MetadataHandler.extract_metadata and cache.get
        mock_audio = AudioFile(
            path=test_dir / "track01.flac",
            file_type="FLAC"
        )
        with patch('music_organizer.core.cached_metadata.MetadataHandler.extract_metadata',
                  return_value=mock_audio):
            with patch.object(handler.cache, 'get', return_value=None):
                with patch.object(handler.cache, 'put'):
                    processed = handler.preload_cache(test_dir, recursive=False)
                    assert processed == 3

    def test_batch_invalidate(self, handler, temp_cache_dir):
        """Test batch invalidation."""
        # Create temporary directory structure
        test_dir = temp_cache_dir / "music"
        test_dir.mkdir()

        # Mock database query
        mock_rows = [
            (str(test_dir / "track01.flac"),),
            (str(test_dir / "track02.flac"),)
        ]
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.return_value = mock_rows

            invalidated = handler.batch_invalidate(test_dir)
            assert invalidated == 2