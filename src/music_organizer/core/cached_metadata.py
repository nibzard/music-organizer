"""Cached metadata handler that wraps MetadataHandler with SQLite caching.

Provides zero-copy metadata caching to avoid re-reading metadata from
unchanged files, significantly improving performance for large libraries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import timedelta
import logging
import sqlite3

from .cache import SQLiteCache
from .metadata import MetadataHandler
from ..models.audio_file import AudioFile


logger = logging.getLogger(__name__)


class CachedMetadataHandler:
    """Metadata handler with SQLite caching support."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl: Optional[timedelta] = None) -> None:
        """Initialize the cached metadata handler.

        Args:
            cache_dir: Directory for cache database (defaults to ~/.cache/music-organizer)
            ttl: Default time-to-live for cache entries (defaults to 30 days)
        """
        self.cache = SQLiteCache(cache_dir)
        self.default_ttl = ttl or timedelta(days=30)
        self._cache_hits = 0
        self._cache_misses = 0

    def extract_metadata(self, file_path: Path, use_cache: bool = True) -> AudioFile:
        """Extract metadata with caching support.

        Args:
            file_path: Path to the audio file
            use_cache: Whether to use cache (default: True)

        Returns:
            AudioFile with metadata
        """
        if use_cache:
            # Try to get from cache first
            cached_audio = self.cache.get(file_path)
            if cached_audio is not None:
                self._cache_hits += 1
                logger.debug(f"Cache hit for {file_path}")
                return cached_audio

        # Cache miss or disabled, extract from file
        self._cache_misses += 1
        logger.debug(f"Cache miss for {file_path}, extracting from file")

        # Use existing MetadataHandler to extract
        audio_file = MetadataHandler.extract_metadata(file_path)

        # Cache the result
        if use_cache:
            self.cache.put(audio_file, self.default_ttl)
            logger.debug(f"Cached metadata for {file_path}")

        return audio_file

    def invalidate(self, file_path: Path) -> None:
        """Invalidate cache for a specific file.

        Args:
            file_path: Path to invalidate
        """
        self.cache.invalidate(file_path)
        logger.debug(f"Invalidated cache for {file_path}")

    def cleanup_expired(self) -> int:
        """Clean up expired cache entries.

        Returns:
            Number of entries removed
        """
        count = self.cache.cleanup_expired()
        if count > 0:
            logger.info(f"Cleaned up {count} expired cache entries")
        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache and performance statistics.

        Returns:
            Dictionary with cache stats
        """
        stats = self.cache.get_stats()
        stats.update({
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0
            )
        })
        return stats

    def clear_cache(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Cleared all cache entries")
        self._cache_hits = 0
        self._cache_misses = 0

    def preload_cache(self, directory: Path, recursive: bool = True) -> int:
        """Preload cache with metadata from directory.

        Args:
            directory: Directory to scan
            recursive: Whether to scan recursively

        Returns:
            Number of files processed
        """
        from concurrent.futures import ThreadPoolExecutor
        import os

        logger.info(f"Preloading cache from {directory}")

        # Find all audio files
        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.mp4', '.aac', '.ogg', '.opus', '.wma'}
        files_to_process = []

        if recursive:
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    if Path(filename).suffix.lower() in audio_extensions:
                        files_to_process.append(Path(root) / filename)
        else:
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    files_to_process.append(file_path)

        logger.info(f"Found {len(files_to_process)} audio files to cache")

        # Process in parallel for speed
        processed = 0
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for file_path in files_to_process:
                # Skip if already cached
                if self.cache.get(file_path) is None:
                    futures.append(executor.submit(self.extract_metadata, file_path))

            for future in futures:
                try:
                    future.result()
                    processed += 1
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed}/{len(files_to_process)} files")
                except Exception as e:
                    logger.warning(f"Failed to cache file: {e}")

        logger.info(f"Preloaded cache with {processed} files")
        return processed

    def batch_invalidate(self, directory: Path, recursive: bool = True) -> int:
        """Invalidate all cache entries for files in a directory.

        Args:
            directory: Directory to invalidate
            recursive: Whether to process recursively

        Returns:
            Number of entries invalidated
        """
        import os

        logger.info(f"Invalidating cache for {directory}")

        # Find all files in cache from this directory
        with sqlite3.connect(self.cache.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_path FROM metadata_cache WHERE file_path LIKE ?",
                (f"{str(directory)}%",)
            )
            rows = cursor.fetchall()

        invalidated = 0
        for row in rows:
            file_path = Path(row[0])
            self.invalidate(file_path)
            invalidated += 1

        logger.info(f"Invalidated {invalidated} cache entries")
        return invalidated

    def find_cover_art(self, directory: Path) -> List[Path]:
        """Find cover art files in a directory.

        Delegates to the underlying MetadataHandler since cover art
        doesn't need caching.

        Args:
            directory: Directory to search for cover art

        Returns:
            List of paths to cover art files
        """
        return MetadataHandler.find_cover_art(directory)

    def close(self) -> None:
        """Close the cache handler."""
        self.cache.close()


# Global instance for easy access
_default_handler: Optional[CachedMetadataHandler] = None


def get_cached_metadata_handler(
    cache_dir: Optional[Path] = None,
    ttl: Optional[timedelta] = None
) -> CachedMetadataHandler:
    """Get the default cached metadata handler.

    Args:
        cache_dir: Directory for cache database
        ttl: Default time-to-live for cache entries

    Returns:
        CachedMetadataHandler instance
    """
    global _default_handler
    if _default_handler is None:
        _default_handler = CachedMetadataHandler(cache_dir, ttl)
    return _default_handler


def extract_metadata_cached(file_path: Path, use_cache: bool = True) -> AudioFile:
    """Convenient function to extract metadata with caching.

    Args:
        file_path: Path to the audio file
        use_cache: Whether to use cache

    Returns:
        AudioFile with metadata
    """
    handler = get_cached_metadata_handler()
    return handler.extract_metadata(file_path, use_cache)