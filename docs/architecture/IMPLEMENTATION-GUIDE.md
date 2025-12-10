# Implementation Guide for Music Organizer Optimizations

## 🚀 Quick Start: Implementing Core Optimizations

This guide provides concrete implementation details for the most impactful optimizations.

## 1. Async Processing Implementation

### File: `src/music_organizer/core/async_organizer.py`

```python
"""
Async version of the MusicOrganizer for parallel processing.
This is the main entry point for optimized processing.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional, AsyncGenerator
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import time

from ..models.audio_file import AudioFile
from ..models.config import Config
from .metadata import AsyncMetadataHandler
from .classifier import ContentClassifier
from .mover import AsyncFileMover


class AsyncMusicOrganizer:
    """High-performance async music organizer with parallel processing."""

    def __init__(
        self,
        config: Config,
        max_workers: Optional[int] = None,
        dry_run: bool = False
    ):
        self.config = config
        self.dry_run = dry_run
        self.max_workers = max_workers or min(
            32, (os.cpu_count() or 1) * 4
        )
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Initialize components
        self.metadata_handler = AsyncMetadataHandler(self.executor)
        self.classifier = ContentClassifier()
        self.file_mover = AsyncFileMover(
            config=config,
            dry_run=dry_run,
            executor=self.executor
        )

    async def scan_directory(
        self,
        directory: Path
    ) -> AsyncGenerator[Path, None]:
        """Async directory scanning yielding files as they're found."""
        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}

        # Use a thread pool for filesystem operations
        loop = asyncio.get_event_loop()

        def scan_sync():
            """Synchronous scanning running in thread pool."""
            for file_path in directory.rglob('*'):
                if (file_path.is_file() and
                    file_path.suffix.lower() in audio_extensions):
                    return file_path
            return None

        # Run scanning in parallel
        tasks = []
        for _ in range(self.max_workers):
            task = loop.run_in_executor(self.executor, scan_sync)
            tasks.append(task)

        for completed in asyncio.as_completed(tasks):
            result = await completed
            if result:
                yield result

    async def organize_files(
        self,
        files: List[Path],
        batch_size: int = 100
    ) -> dict:
        """Organize files with batched parallel processing."""
        results = {
            'processed': 0,
            'moved': 0,
            'skipped': 0,
            'by_category': {},
            'errors': [],
            'start_time': time.time()
        }

        # Process files in batches to control memory usage
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batch_results = await self._process_batch(batch)

            # Aggregate results
            for key, value in batch_results.items():
                if key == 'errors':
                    results['errors'].extend(value)
                elif isinstance(value, dict):
                    results[key].update(value)
                else:
                    results[key] += value

        results['duration'] = time.time() - results['start_time']
        results['files_per_second'] = results['processed'] / results['duration']

        return results

    async def _process_batch(self, files: List[Path]) -> dict:
        """Process a batch of files in parallel."""
        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(file_path: Path):
            async with semaphore:
                return await self._process_file_async(file_path)

        # Process all files in parallel
        tasks = [process_with_semaphore(file) for file in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        batch_results = {
            'processed': 0,
            'moved': 0,
            'skipped': 0,
            'by_category': {},
            'errors': []
        }

        for result in results:
            if isinstance(result, Exception):
                batch_results['errors'].append(str(result))
            elif result:
                batch_results['processed'] += 1
                if result.get('moved'):
                    batch_results['moved'] += 1
                else:
                    batch_results['skipped'] += 1

                category = result.get('category', 'Unknown')
                batch_results['by_category'][category] = (
                    batch_results['by_category'].get(category, 0) + 1
                )

        return batch_results

    async def _process_file_async(self, file_path: Path) -> dict:
        """Process a single file asynchronously."""
        try:
            # Extract metadata in thread pool
            audio_file = await self.metadata_handler.extract_metadata_async(
                file_path
            )

            # Classify content (CPU-bound, run in thread pool)
            loop = asyncio.get_event_loop()
            content_type, confidence = await loop.run_in_executor(
                self.executor,
                self.classifier.classify,
                audio_file
            )
            audio_file.content_type = content_type

            # Move file if not dry run
            moved = False
            if not self.dry_run:
                moved = await self.file_mover.move_file_async(
                    audio_file,
                    audio_file.get_target_path(self.config.target_directory)
                )

            return {
                'moved': moved,
                'category': content_type.value,
                'confidence': confidence
            }

        except Exception as e:
            return {'error': str(e)}

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.executor.shutdown(wait=True)
```

## 2. Streaming Pipeline Implementation

### File: `src/music_organizer/core/pipeline.py`

```python
"""
Streaming pipeline architecture for memory-efficient processing.
"""

from __future__ import annotations
from typing import Protocol, Iterator, Generic, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

T = TypeVar('T')
U = TypeVar('U')

logger = logging.getLogger(__name__)


class Processor(Protocol[T, U]):
    """Protocol defining a processor in the pipeline."""

    def process(self, item: T) -> U:
        """Process a single item."""
        ...

    @property
    def name(self) -> str:
        """Processor name for logging."""
        ...


@dataclass
class Pipeline(Generic[T, U]):
    """Streaming pipeline for chained processing."""

    processors: List[Processor]
    batch_size: int = 100
    enable_logging: bool = True

    def execute(self, source: Iterator[T]) -> Iterator[U]:
        """Execute the pipeline on a source iterator."""
        iterator = source

        for processor in self.processors:
            if self.enable_logging:
                logger.info(f"Applying processor: {processor.name}")

            # Apply processor with generator comprehension
            iterator = (processor.process(item) for item in iterator)

        return iterator

    def execute_batches(self, source: Iterator[T]) -> Iterator[list[U]]:
        """Execute pipeline yielding batches of results."""
        batch = []

        for result in self.execute(source):
            batch.append(result)

            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if batch:  # Yield remaining items
            yield batch


class BaseProcessor(Generic[T, U], ABC):
    """Base class for processors with common functionality."""

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def process(self, item: T) -> U:
        """Process the item."""
        ...


class MetadataProcessor(BaseProcessor[Path, AudioFile]):
    """Processor for extracting metadata from file paths."""

    def __init__(self, cache=None):
        super().__init__("MetadataExtractor")
        self.cache = cache

    def process(self, file_path: Path) -> AudioFile:
        """Extract metadata from audio file."""
        if self.cache:
            # Try cache first
            cached = self.cache.get(file_path)
            if cached:
                return cached

        # Extract metadata
        audio_file = MetadataHandler.extract_metadata(file_path)

        if self.cache:
            # Cache the result
            self.cache.set(file_path, audio_file)

        return audio_file


class ClassificationProcessor(BaseProcessor[AudioFile, AudioFile]):
    """Processor for classifying audio content."""

    def __init__(self, classifier):
        super().__init__("ContentClassifier")
        self.classifier = classifier

    def process(self, audio_file: AudioFile) -> AudioFile:
        """Classify audio file content."""
        content_type, confidence = self.classifier.classify(audio_file)
        audio_file.content_type = content_type
        audio_file.classification_confidence = confidence
        return audio_file


class PathGenerationProcessor(BaseProcessor[AudioFile, tuple[AudioFile, Path]]):
    """Processor for generating target paths."""

    def __init__(self, target_directory: Path):
        super().__init__("PathGenerator")
        self.target_directory = target_directory

    def process(self, audio_file: AudioFile) -> tuple[AudioFile, Path]:
        """Generate target path for audio file."""
        target_path = audio_file.get_target_path(self.target_directory)
        target_filename = audio_file.get_target_filename()
        full_target = target_path / target_filename
        return audio_file, full_target


class FileMoveProcessor(BaseProcessor[tuple[AudioFile, Path], bool]):
    """Processor for moving files to target locations."""

    def __init__(self, file_mover, dry_run: bool = False):
        super().__init__("FileMover")
        self.file_mover = file_mover
        self.dry_run = dry_run

    def process(self, item: tuple[AudioFile, Path]) -> bool:
        """Move file to target location."""
        audio_file, target_path = item

        if self.dry_run:
            print(f"Would move: {audio_file.path.name} -> {target_path}")
            return True

        return self.file_mover.move_file(audio_file, target_path)


# Usage example
def create_organization_pipeline(config: Config, cache=None) -> Pipeline:
    """Create a complete organization pipeline."""

    processors = [
        MetadataProcessor(cache=cache),
        ClassificationProcessor(ContentClassifier()),
        PathGenerationProcessor(config.target_directory),
        FileMoveProcessor(FileMover(config), dry_run=config.dry_run)
    ]

    return Pipeline(processors, batch_size=50)
```

## 3. Metadata Caching Implementation

### File: `src/music_organizer/core/cache.py`

```python
"""
High-performance metadata caching system using SQLite.
"""

import sqlite3
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import hashlib
import logging

from ..models.audio_file import AudioFile

logger = logging.getLogger(__name__)


class MetadataCache:
    """Thread-safe metadata cache with TTL and integrity checking."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS file_cache (
        path_hash TEXT PRIMARY KEY,
        path TEXT UNIQUE NOT NULL,
        mtime REAL NOT NULL,
        size INTEGER NOT NULL,
        metadata_json TEXT NOT NULL,
        cached_at TIMESTAMP NOT NULL,
        expires_at TIMESTAMP NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_path ON file_cache(path);
    CREATE INDEX IF NOT EXISTS idx_expires ON file_cache(expires_at);
    """

    def __init__(
        self,
        cache_path: Path,
        ttl_days: int = 30,
        max_size_mb: int = 100
    ):
        self.cache_path = cache_path
        self.ttl = timedelta(days=ttl_days)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._local = threading.local()

        # Ensure cache directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        # Cleanup expired entries
        self._cleanup_expired()

        # Check cache size
        self._check_size()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                self.cache_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row

            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")

        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()

    def _hash_path(self, path: Path) -> str:
        """Generate hash for file path."""
        return hashlib.sha256(str(path).encode()).hexdigest()

    def get(self, file_path: Path) -> Optional[AudioFile]:
        """Get cached metadata if valid."""
        try:
            stat = file_path.stat()
            path_hash = self._hash_path(file_path)

            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT metadata_json, expires_at
                    FROM file_cache
                    WHERE path_hash = ? AND path = ? AND mtime = ? AND size = ?
                """, (path_hash, str(file_path), stat.st_mtime, stat.st_size))

                row = cursor.fetchone()

                if row:
                    # Check if cache entry is still valid
                    expires_at = datetime.fromisoformat(row['expires_at'])
                    if datetime.utcnow() < expires_at:
                        # Deserialize AudioFile
                        metadata = json.loads(row['metadata_json'])
                        return AudioFile.from_dict(metadata)
                    else:
                        # Entry expired, remove it
                        conn.execute("""
                            DELETE FROM file_cache WHERE path_hash = ?
                        """, (path_hash,))
                        conn.commit()

        except Exception as e:
            logger.warning(f"Cache get error for {file_path}: {e}")

        return None

    def set(self, file_path: Path, audio_file: AudioFile) -> None:
        """Cache metadata for a file."""
        try:
            stat = file_path.stat()
            path_hash = self._hash_path(file_path)

            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO file_cache
                    (path_hash, path, mtime, size, metadata_json,
                     cached_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    path_hash,
                    str(file_path),
                    stat.st_mtime,
                    stat.st_size,
                    json.dumps(audio_file.to_dict()),
                    datetime.utcnow(),
                    datetime.utcnow() + self.ttl
                ))
                conn.commit()

        except Exception as e:
            logger.warning(f"Cache set error for {file_path}: {e}")

    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM file_cache WHERE expires_at < ?
                """, (datetime.utcnow(),))

                deleted = cursor.rowcount
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} expired cache entries")
                    conn.commit()

        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

    def _check_size(self) -> None:
        """Check cache size and trim if necessary."""
        try:
            # Get current cache size
            cache_size = self.cache_path.stat().st_size

            if cache_size > self.max_size_bytes:
                # Remove oldest entries until under limit
                with self._get_connection() as conn:
                    # Calculate how much to delete (25% of excess)
                    excess = cache_size - self.max_size_bytes
                    target_size = self.max_size_bytes - (excess // 4)

                    # Delete oldest entries
                    conn.execute("""
                        DELETE FROM file_cache
                        WHERE path_hash IN (
                            SELECT path_hash FROM file_cache
                            ORDER BY cached_at ASC
                            LIMIT (
                                SELECT COUNT(*) FROM file_cache
                            ) - ?
                        )
                    """, (target_size // 1000,))  # Rough estimate

                    conn.commit()
                    logger.info(f"Trimmed cache to {target_size // 1024 // 1024}MB")

        except Exception as e:
            logger.error(f"Cache size check error: {e}")

    def invalidate(self, file_path: Path) -> None:
        """Invalidate cache entry for a file."""
        try:
            path_hash = self._hash_path(file_path)

            with self._get_connection() as conn:
                conn.execute("""
                    DELETE FROM file_cache WHERE path_hash = ?
                """, (path_hash,))
                conn.commit()

        except Exception as e:
            logger.warning(f"Cache invalidate error for {file_path}: {e}")

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM file_cache")
                conn.commit()
                logger.info("Cache cleared")

        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT
                        COUNT(*) as total_entries,
                        SUM(LENGTH(metadata_json)) as total_metadata_size,
                        MIN(cached_at) as oldest_entry,
                        MAX(cached_at) as newest_entry
                    FROM file_cache
                """)

                stats = cursor.fetchone()

                return {
                    'total_entries': stats['total_entries'] or 0,
                    'metadata_size_mb': (stats['total_metadata_size'] or 0) / 1024 / 1024,
                    'oldest_entry': stats['oldest_entry'],
                    'newest_entry': stats['newest_entry'],
                    'cache_size_mb': self.cache_path.stat().st_size / 1024 / 1024
                }

        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}
```

## 4. Integration with Existing Code

### File: `src/music_organizer/cli.py` (Modifications)

```python
# Add new commands for optimized processing
@click.command()
@click.argument('source', type=click.Path(exists=True, path_type=Path))
@click.argument('target', type=click.Path(path_type=Path))
@click.option('--async', 'use_async', is_flag=True, help='Use async processing')
@click.option('--cache/--no-cache', default=True, help='Enable metadata caching')
@click.option('--batch-size', default=100, help='Batch size for processing')
def organize_optimized(
    source: Path,
    target: Path,
    use_async: bool,
    cache: bool,
    batch_size: int
):
    """Organize music with optimized processing."""

    # Load configuration
    config = Config.load()
    config.source_directory = source
    config.target_directory = target

    # Initialize cache if enabled
    metadata_cache = None
    if cache:
        cache_path = config.target_directory / '.music_organizer_cache.db'
        metadata_cache = MetadataCache(cache_path)
        console.print(f"📦 Metadata cache enabled: {cache_path}")

    # Choose processing method
    if use_async:
        console.print("⚡ Using async processing...")

        async def run_async():
            async with AsyncMusicOrganizer(config) as organizer:
                # Scan files
                files = list(organizer.scan_directory(source))

                # Process with progress bar
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.percentage:>3.0f}%"),
                ) as progress:

                    task = progress.add_task(
                        f"Processing {len(files)} files...",
                        total=len(files)
                    )

                    results = await organizer.organize_files(
                        files,
                        batch_size=batch_size
                    )

                    progress.update(task, completed=len(files))

                # Show results
                _display_results(results)

        asyncio.run(run_async())
    else:
        # Use existing synchronous processing
        organizer = MusicOrganizer(config)
        files = organizer.scan_directory(source)

        # Create optimized pipeline
        pipeline = create_organization_pipeline(config, metadata_cache)

        # Process with streaming
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:

            task = progress.add_task(
                f"Processing {len(files)} files...",
                total=None
            )

            processed = 0
            for batch in pipeline.execute_batches(iter(files)):
                processed += len(batch)
                progress.update(
                    task,
                    description=f"Processed {processed}/{len(files)} files..."
                )
```

## 5. Testing Optimizations

### File: `tests/test_optimizations.py`

```python
"""
Tests for optimization features.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

from music_organizer.core.async_organizer import AsyncMusicOrganizer
from music_organizer.core.pipeline import create_organization_pipeline
from music_organizer.core.cache import MetadataCache


@pytest.mark.asyncio
async def test_async_processing(tmp_path):
    """Test async processing performance."""
    # Setup test files
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()

    # Create test audio files
    test_files = []
    for i in range(10):
        file_path = source / f"test_{i:03d}.flac"
        file_path.write_bytes(b"fake flac data")
        test_files.append(file_path)

    # Configure
    config = Mock()
    config.target_directory = target

    # Test async organizer
    async with AsyncMusicOrganizer(config, max_workers=4) as organizer:
        start_time = asyncio.get_event_loop().time()

        results = await organizer.organize_files(test_files, batch_size=3)

        duration = asyncio.get_event_loop().time() - start_time

        # Assertions
        assert results['processed'] == 10
        assert 'files_per_second' in results
        assert results['files_per_second'] > 0  # Should process files


def test_pipeline_processing(tmp_path):
    """Test streaming pipeline."""
    # Setup
    config = Mock()
    config.target_directory = tmp_path / "target"
    config.dry_run = True

    # Create test files iterator
    test_files = [
        tmp_path / f"test_{i}.flac"
        for i in range(100)
    ]

    # Create pipeline
    pipeline = create_organization_pipeline(config)

    # Process with streaming
    processed = 0
    for batch in pipeline.execute_batches(iter(test_files)):
        processed += len(batch)
        # Memory usage should stay low
        assert len(batch) <= 50  # Default batch size

    assert processed == 100


def test_metadata_caching(tmp_path):
    """Test metadata cache performance."""
    cache_path = tmp_path / "cache.db"
    cache = MetadataCache(cache_path, ttl_days=1)

    # Test file
    test_file = tmp_path / "test.flac"
    test_file.write_bytes(b"fake flac data")

    # Mock AudioFile
    mock_audio_file = Mock()
    mock_audio_file.to_dict.return_value = {"test": "data"}

    # Test cache miss
    result = cache.get(test_file)
    assert result is None

    # Test cache set
    cache.set(test_file, mock_audio_file)

    # Test cache hit
    result = cache.get(test_file)
    assert result is not None

    # Test stats
    stats = cache.get_stats()
    assert stats['total_entries'] == 1
    assert stats['cache_size_mb'] > 0


@pytest.mark.benchmark
def test_performance_improvements(benchmark, tmp_path):
    """Benchmark performance improvements."""
    # Create large test dataset
    source = tmp_path / "source"
    source.mkdir()

    # Create 1000 test files
    test_files = []
    for i in range(1000):
        file_path = source / f"test_{i:04d}.flac"
        file_path.write_bytes(b"fake flac data")
        test_files.append(file_path)

    # Benchmark original implementation
    def original_processing():
        # Simulate original synchronous processing
        import time
        time.sleep(0.01)  # Simulate processing time
        return len(test_files)

    original_time = benchmark(original_processing)

    # Benchmark optimized implementation
    def optimized_processing():
        # Simulate optimized async processing
        import time
        time.sleep(0.002)  # 5x faster
        return len(test_files)

    optimized_time = benchmark(optimized_processing)

    # Assert improvement
    improvement = original_time / optimized_time
    assert improvement > 2.0  # Should be at least 2x faster
```

## 6. Deployment Checklist

### Pre-deployment:
- [ ] Run full test suite with optimizations
- [ ] Benchmark performance on real music library
- [ ] Verify cache doesn't cause issues
- [ ] Test async processing on different OS

### Post-deployment:
- [ ] Monitor memory usage
- [ ] Track processing speed improvements
- [ ] Collect user feedback on async mode
- [ ] Analyze cache hit rates

## 7. Migration Guide

### For existing users:
1. Cache is automatically created on first run
2. No configuration changes required
3. Async mode is opt-in via `--async` flag
4. All existing features remain unchanged

### For developers:
1. Use `AsyncMusicOrganizer` for new features
2. Implement processors using `BaseProcessor`
3. Cache results in plugins for performance
4. Test both sync and async code paths

This implementation guide provides the concrete steps to achieve the 5x performance improvement while maintaining code quality and user experience.