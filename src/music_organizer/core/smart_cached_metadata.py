"""Smart cached metadata handler with intelligent caching strategies.

This module provides a drop-in replacement for CachedMetadataHandler that uses
the SmartCacheManager for optimal performance with adaptive TTL and
directory-level change detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import timedelta
import logging

from .smart_cache import SmartCacheManager, get_smart_cache_manager
from .metadata import MetadataHandler
from ..models.audio_file import AudioFile

logger = logging.getLogger(__name__)


class SmartCachedMetadataHandler:
    """Metadata handler with smart caching support."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        enable_smart_cache: bool = True,
        cache_warming_enabled: bool = True,
        auto_optimize: bool = True
    ) -> None:
        """Initialize the smart cached metadata handler.

        Args:
            cache_dir: Directory for cache database (defaults to ~/.cache/music-organizer)
            enable_smart_cache: Whether to use smart caching (default: True)
            cache_warming_enabled: Whether to enable automatic cache warming
            auto_optimize: Whether to automatically optimize cache performance
        """
        self.enable_smart_cache = enable_smart_cache

        if enable_smart_cache:
            self.smart_cache = get_smart_cache_manager(cache_dir)
            logger.info("Smart caching enabled")
        else:
            # Fallback to basic caching
            from .cached_metadata import get_cached_metadata_handler
            self.basic_cache = get_cached_metadata_handler(cache_dir)
            logger.info("Using basic caching")

        self.cache_warming_enabled = cache_warming_enabled
        self.auto_optimize = auto_optimize
        self._operations_since_optimize = 0
        self._optimize_threshold = 1000  # Optimize every 1000 operations

    def extract_metadata(self, file_path: Path, use_cache: bool = True) -> AudioFile:
        """Extract metadata with intelligent caching.

        Args:
            file_path: Path to the audio file
            use_cache: Whether to use cache (default: True)

        Returns:
            AudioFile with metadata
        """
        if not use_cache:
            # Skip cache entirely
            return MetadataHandler.extract_metadata(file_path)

        if self.enable_smart_cache:
            # Try smart cache first
            cached_audio = self.smart_cache.get(file_path)
            if cached_audio is not None:
                logger.debug(f"Smart cache hit for {file_path}")
                self._check_optimize()
                return cached_audio

            # Cache miss, extract from file
            logger.debug(f"Smart cache miss for {file_path}, extracting from file")
            audio_file = MetadataHandler.extract_metadata(file_path)

            # Cache with smart TTL calculation
            self.smart_cache.put(audio_file)
            logger.debug(f"Cached metadata with smart TTL for {file_path}")

        else:
            # Use basic cache
            cached_audio = self.basic_cache.get(file_path)
            if cached_audio is not None:
                logger.debug(f"Basic cache hit for {file_path}")
                self._check_optimize()
                return cached_audio

            logger.debug(f"Basic cache miss for {file_path}, extracting from file")
            audio_file = MetadataHandler.extract_metadata(file_path)
            self.basic_cache.put(audio_file)

        self._operations_since_optimize += 1
        return audio_file

    def extract_metadata_batch(
        self,
        file_paths: List[Path],
        use_cache: bool = True,
        parallel: bool = True
    ) -> List[AudioFile]:
        """Extract metadata for multiple files with optimized caching.

        Args:
            file_paths: List of file paths to process
            use_cache: Whether to use cache
            parallel: Whether to process in parallel

        Returns:
            List of AudioFile objects in the same order as input
        """
        if parallel and len(file_paths) > 10:
            from concurrent.futures import ThreadPoolExecutor
            import os

            # Group by directory for optimization
            directory_groups = {}
            for file_path in file_paths:
                directory = str(file_path.parent)
                if directory not in directory_groups:
                    directory_groups[directory] = []
                directory_groups[directory].append(file_path)

            # Process directories in parallel
            results = [None] * len(file_paths)
            path_to_index = {path: i for i, path in enumerate(file_paths)}

            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(directory_groups))) as executor:
                futures = []
                for directory, paths in directory_groups.items():
                    future = executor.submit(self._process_directory_batch, paths, use_cache)
                    futures.append((future, paths))

                for future, paths in futures:
                    try:
                        batch_results = future.result()
                        for audio_file in batch_results:
                            idx = path_to_index[audio_file.path]
                            results[idx] = audio_file
                    except Exception as e:
                        logger.error(f"Failed to process directory batch: {e}")
                        # Fall back to individual processing
                        for path in paths:
                            try:
                                audio_file = self.extract_metadata(path, use_cache)
                                idx = path_to_index[path]
                                results[idx] = audio_file
                            except Exception as e2:
                                logger.error(f"Failed to extract metadata for {path}: {e2}")

            return results

        else:
            # Sequential processing
            return [self.extract_metadata(path, use_cache) for path in file_paths]

    def _process_directory_batch(self, file_paths: List[Path], use_cache: bool) -> List[AudioFile]:
        """Process a batch of files from the same directory."""
        if self.enable_smart_cache:
            # Check if directory is stable
            if file_paths:
                directory = file_paths[0].parent
                directory_mtime = directory.stat().st_mtime

                # If directory unchanged, can use fast path
                if not self.smart_cache._should_check_directory(str(directory), directory_mtime):
                    results = []
                    for file_path in file_paths:
                        cached = self.smart_cache._get_from_cache_fast(
                            file_path,
                            file_path.stat().st_mtime,
                            file_path.stat().st_size
                        )
                        if cached:
                            results.append(cached)
                        else:
                            # Fall back to standard extraction
                            results.append(self.extract_metadata(file_path, use_cache))
                    return results

        # Standard processing
        return [self.extract_metadata(path, use_cache) for path in file_paths]

    def invalidate(self, file_path: Path) -> None:
        """Invalidate cache for a specific file.

        Args:
            file_path: Path to invalidate
        """
        if self.enable_smart_cache:
            self.smart_cache.invalidate(file_path)
        else:
            self.basic_cache.invalidate(file_path)
        logger.debug(f"Invalidated cache for {file_path}")

    def invalidate_directory(self, directory: Path, recursive: bool = True) -> int:
        """Invalidate all cache entries for files in a directory.

        Args:
            directory: Directory to invalidate
            recursive: Whether to process recursively

        Returns:
            Number of entries invalidated
        """
        if not recursive:
            # Just invalidate the directory itself for smart cache
            if self.enable_smart_cache:
                # Remove from directory cache
                dir_path = str(directory)
                if dir_path in self.smart_cache._directory_cache:
                    del self.smart_cache._directory_cache[dir_path]
                return 0
            return 0

        # For recursive invalidation, count affected files
        import os

        invalidated = 0
        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.mp4', '.aac', '.ogg', '.opus', '.wma'}

        if recursive:
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    if Path(filename).suffix.lower() in audio_extensions:
                        file_path = Path(root) / filename
                        self.invalidate(file_path)
                        invalidated += 1
        else:
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    self.invalidate(file_path)
                    invalidated += 1

        logger.info(f"Invalidated {invalidated} cache entries for {directory}")
        return invalidated

    def cleanup_expired(self) -> int:
        """Clean up expired cache entries.

        Returns:
            Number of entries removed
        """
        if self.enable_smart_cache:
            count = self.smart_cache.cleanup_expired()
            if count > 0:
                logger.info(f"Cleaned up {count} expired smart cache entries")
        else:
            count = self.basic_cache.cleanup_expired()
            if count > 0:
                logger.info(f"Cleaned up {count} expired basic cache entries")

        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache and performance statistics.

        Returns:
            Dictionary with cache stats
        """
        if self.enable_smart_cache:
            stats = self.smart_cache.get_smart_stats()
            stats.update({
                'cache_type': 'smart',
                'cache_warming_enabled': self.cache_warming_enabled,
                'auto_optimize': self.auto_optimize,
                'operations_since_optimize': self._operations_since_optimize
            })
        else:
            stats = self.basic_cache.get_cache_stats()
            stats.update({
                'cache_type': 'basic',
                'cache_warming_enabled': False,
                'auto_optimize': False
            })

        return stats

    def clear_cache(self) -> None:
        """Clear all cache entries."""
        if self.enable_smart_cache:
            self.smart_cache.clear()
            logger.info("Cleared all smart cache entries")
        else:
            self.basic_cache.clear_cache()
            logger.info("Cleared all basic cache entries")

        self._operations_since_optimize = 0

    def warm_cache(
        self,
        directory: Path,
        recursive: bool = True,
        max_files: int = 1000,
        force: bool = False
    ) -> int:
        """Warm cache with likely-to-be-accessed files.

        Args:
            directory: Directory to warm
            recursive: Whether to scan recursively
            max_files: Maximum number of files to warm
            force: Whether to force warming even if cache is warm

        Returns:
            Number of files warmed
        """
        if not self.cache_warming_enabled and not force:
            logger.info("Cache warming is disabled")
            return 0

        if self.enable_smart_cache:
            warmed = self.smart_cache.warm_cache(directory, recursive, max_files)
            logger.info(f"Warmed smart cache with {warmed} files")
            return warmed
        else:
            # Basic cache warming
            warmed = self.basic_cache.preload_cache(directory, recursive)
            if warmed > max_files:
                # Limit warmed files
                logger.info(f"Limited cache warming to {max_files} files")
                warmed = max_files
            return warmed

    def optimize_cache(self, force: bool = False) -> Dict[str, Any]:
        """Optimize cache performance.

        Args:
            force: Whether to force optimization regardless of threshold

        Returns:
            Optimization results
        """
        if not force and self._operations_since_optimize < self._optimize_threshold:
            logger.debug(f"Skipping optimization (threshold: {self._operations_since_optimize}/{self._optimize_threshold})")
            return {'skipped': True, 'reason': 'threshold_not_met'}

        if self.enable_smart_cache:
            results = self.smart_cache.optimize_cache()
            logger.info(f"Cache optimization completed: {results}")
        else:
            # Basic optimization - just cleanup
            removed = self.cleanup_expired()
            results = {
                'expired_entries_removed': removed,
                'optimization_type': 'basic'
            }

        self._operations_since_optimize = 0
        return results

    def _check_optimize(self) -> None:
        """Check if optimization should be triggered."""
        if self.auto_optimize and self._operations_since_optimize >= self._optimize_threshold:
            logger.debug("Triggering automatic cache optimization")
            self.optimize_cache()

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
        # Final optimization if needed
        if self.auto_optimize and self._operations_since_optimize > 0:
            self.optimize_cache(force=True)

        if not self.enable_smart_cache:
            self.basic_cache.close()

    def get_cache_health(self) -> Dict[str, Any]:
        """Get cache health metrics and recommendations.

        Returns:
            Cache health report with recommendations
        """
        stats = self.get_cache_stats()
        health = {
            'overall_health': 'good',
            'recommendations': [],
            'warnings': []
        }

        if self.enable_smart_cache:
            # Check stability
            if stats['avg_stability_score'] < 0.5:
                health['warnings'].append("Low average stability score - files changing frequently")
                health['recommendations'].append("Consider reducing cache TTL for volatile directories")

            # Check access frequency
            if stats['avg_access_frequency'] < 0.01:
                health['recommendations'].append("Consider cache warming for frequently accessed directories")

            # Check size
            if stats['size_mb'] > 1000:
                health['warnings'].append(f"Large cache size: {stats['size_mb']:.1f}MB")
                health['recommendations'].append("Consider running optimize_cache() to reclaim space")

            # Check hit rate (can be estimated)
            if stats['frequently_accessed_files'] < stats['total_entries'] * 0.1:
                health['recommendations'].append("Many files rarely accessed - consider selective caching")

            # Determine overall health
            if len(health['warnings']) > 2:
                health['overall_health'] = 'poor'
            elif len(health['warnings']) > 0:
                health['overall_health'] = 'fair'

        return health


# Global instance for easy access
_default_smart_handler: Optional[SmartCachedMetadataHandler] = None


def get_smart_cached_metadata_handler(
    cache_dir: Optional[Path] = None,
    enable_smart_cache: bool = True,
    cache_warming_enabled: bool = True,
    auto_optimize: bool = True
) -> SmartCachedMetadataHandler:
    """Get the default smart cached metadata handler.

    Args:
        cache_dir: Directory for cache database
        enable_smart_cache: Whether to use smart caching
        cache_warming_enabled: Whether to enable cache warming
        auto_optimize: Whether to enable automatic optimization

    Returns:
        SmartCachedMetadataHandler instance
    """
    global _default_smart_handler
    if _default_smart_handler is None:
        _default_smart_handler = SmartCachedMetadataHandler(
            cache_dir, enable_smart_cache, cache_warming_enabled, auto_optimize
        )
    return _default_smart_handler


def extract_metadata_smart_cached(file_path: Path, use_cache: bool = True) -> AudioFile:
    """Convenient function to extract metadata with smart caching.

    Args:
        file_path: Path to the audio file
        use_cache: Whether to use cache

    Returns:
        AudioFile with metadata
    """
    handler = get_smart_cached_metadata_handler()
    return handler.extract_metadata(file_path, use_cache)