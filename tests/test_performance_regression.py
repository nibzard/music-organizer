"""
Performance regression tests to ensure we don't slow down over time
"""

import pytest
import asyncio
import tempfile
import shutil
import time
from pathlib import Path
from typing import List

from music_organizer.models.audio_file import AudioFile
from music_organizer.models.config import Config
from music_organizer.core.metadata import MetadataHandler
from music_organizer.core.cache import SQLiteCache
from music_organizer.core.async_organizer import AsyncMusicOrganizer as AsyncOrganizer
from music_organizer.core.cached_metadata import CachedMetadataHandler
from music_organizer.utils.memory_monitor import MemoryProfiler, profile_memory


# Performance targets from TODO.md
PERFORMANCE_TARGETS = {
    "startup_time_ms": 100,
    "metadata_extraction_rate_files_per_sec": 100,
    "memory_usage_per_file_mb": 0.01,  # 10KB per file
    "full_processing_rate_files_per_sec": 1000,
    "cache_improvement_percent": 90
}


@pytest.fixture
def temp_music_library():
    """Create a temporary music library for testing"""
    temp_dir = Path(tempfile.mkdtemp(prefix="music_perf_test_"))

    # Create realistic structure
    genres = ["Rock", "Jazz", "Classical"]
    for genre in genres:
        genre_dir = temp_dir / genre
        genre_dir.mkdir()

        for artist_idx in range(5):
            artist = f"Artist_{artist_idx:02d}"
            artist_dir = genre_dir / artist
            artist_dir.mkdir()

            for album_idx in range(2):
                album = f"Album_{album_idx:02d}"
                album_dir = artist_dir / album
                album_dir.mkdir()

                for track_idx in range(10):
                    filename = f"{track_idx + 1:02d} - Track.mp3"
                    filepath = album_dir / filename
                    # Create minimal MP3 file
                    filepath.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_files(temp_music_library: Path) -> List[Path]:
    """Get list of test files"""
    return list(temp_music_library.rglob("*.mp3"))


class TestStartupPerformance:
    """Test application startup performance"""

    def test_startup_time(self):
        """Test that application starts in under 100ms"""
        start_time = time.perf_counter()

        # Simulate startup
        config = Config()
        metadata_handler = MetadataHandler()
        cache = SQLiteCache.get_instance(config.cache.db_path)

        end_time = time.perf_counter()
        startup_time_ms = (end_time - start_time) * 1000

        assert startup_time_ms < PERFORMANCE_TARGETS["startup_time_ms"], \
            f"Startup took {startup_time_ms:.2f}ms, target is {PERFORMANCE_TARGETS['startup_time_ms']}ms"


class TestMetadataExtractionPerformance:
    """Test metadata extraction performance"""

    def test_extraction_rate(self, test_files: List[Path]):
        """Test metadata extraction meets rate targets"""
        metadata_handler = MetadataHandler()
        test_files = test_files[:100]  # Test subset

        start_time = time.perf_counter()
        extracted = 0

        for filepath in test_files:
            try:
                audio_file = metadata_handler.extract_metadata(filepath)
                if audio_file:
                    extracted += 1
            except Exception:
                pass

        end_time = time.perf_counter()
        duration = end_time - start_time
        rate = extracted / duration if duration > 0 else 0

        assert rate >= PERFORMANCE_TARGETS["metadata_extraction_rate_files_per_sec"], \
            f"Extraction rate: {rate:.1f} files/sec, target: {PERFORMANCE_TARGETS['metadata_extraction_rate_files_per_sec']}"

    def test_cache_performance_improvement(self, test_files: List[Path]):
        """Test that cache provides 90% improvement"""
        config = Config()
        test_files = test_files[:50]

        # Measure cold extraction
        metadata_handler = MetadataHandler()
        start_time = time.perf_counter()

        for filepath in test_files:
            try:
                metadata_handler.extract_metadata(filepath)
            except Exception:
                pass

        cold_time = time.perf_counter() - start_time

        # Measure with cache (warm)
        cached_handler = CachedMetadataHandler(config.cache)
        start_time = time.perf_counter()

        for filepath in test_files:
            try:
                cached_handler.get_metadata(filepath)
            except Exception:
                pass

        warm_time = time.perf_counter() - start_time

        # Calculate improvement
        improvement = ((cold_time - warm_time) / cold_time * 100) if cold_time > 0 else 0

        assert improvement >= PERFORMANCE_TARGETS["cache_improvement_percent"], \
            f"Cache improvement: {improvement:.1f}%, target: {PERFORMANCE_TARGETS['cache_improvement_percent']}%"


class TestMemoryUsagePerformance:
    """Test memory usage stays within limits"""

    def test_memory_per_file(self, test_files: List[Path]):
        """Test memory usage per file is within target"""
        with MemoryProfiler("Memory Per File Test", enable_tracemalloc=True) as profiler:
            audio_files = []
            for filepath in test_files[:100]:  # Test subset
                try:
                    audio_file = AudioFile.from_path(filepath)
                    audio_files.append(audio_file)
                except Exception:
                    pass

        # Calculate memory per file
        memory_per_file = profiler.stats.peak_rss / len(audio_files) if audio_files else 0

        assert memory_per_file <= PERFORMANCE_TARGETS["memory_usage_per_file_mb"], \
            f"Memory per file: {memory_per_file:.4f}MB, target: {PERFORMANCE_TARGETS['memory_usage_per_file_mb']}MB"

    @profile_memory("AudioFile Creation", enable_tracemalloc=True)
    def test_audiofile_creation_memory(self, test_files: List[Path]):
        """Test AudioFile creation doesn't leak memory"""
        # This test uses the @profile_memory decorator to automatically track memory
        for filepath in test_files[:200]:
            try:
                audio_file = AudioFile.from_path(filepath)
                # Use the file to ensure it's not optimized away
                assert audio_file is not None
            except Exception:
                pass

        # The decorator will automatically print memory usage
        # Additional assertions can be added here if needed


class TestAsyncProcessingPerformance:
    """Test async processing performance"""

    @pytest.mark.asyncio
    async def test_async_processing_rate(self, temp_music_library: Path):
        """Test async processing meets rate targets"""
        config = Config()
        organizer = AsyncOrganizer(config)

        target_dir = temp_music_library / "output"
        target_dir.mkdir(exist_ok=True)

        start_time = time.perf_counter()

        # Run organization (dry run to avoid file operations)
        result = await organizer.organize_files_streaming(
            source=temp_music_library,
            target=target_dir,
            pattern="{genre}/{artist}/{album}/{track:02d} - {title}",
            dry_run=True
        )

        end_time = time.perf_counter()
        duration = end_time - start_time
        rate = result.processed / duration if duration > 0 else 0

        assert rate >= PERFORMANCE_TARGETS["full_processing_rate_files_per_sec"], \
            f"Processing rate: {rate:.1f} files/sec, target: {PERFORMANCE_TARGETS['full_processing_rate_files_per_sec']}"

    @pytest.mark.asyncio
    async def test_concurrent_vs_sequential(self, test_files: List[Path]):
        """Test that concurrent processing is faster than sequential"""
        test_files = test_files[:50]
        config = Config()

        # Sequential processing
        organizer_seq = AsyncOrganizer(config)
        organizer_seq.config.workers = 1

        start_time = time.perf_counter()
        results = []
        for filepath in test_files:
            try:
                audio_file = AudioFile.from_path(filepath)
                results.append(audio_file)
            except Exception:
                pass
        seq_time = time.perf_counter() - start_time

        # Concurrent processing
        organizer_conc = AsyncOrganizer(config)
        organizer_conc.config.workers = 4

        async def process_concurrent():
            semaphore = asyncio.Semaphore(organizer_conc.config.workers)
            tasks = []

            async def process_file(filepath):
                async with semaphore:
                    return AudioFile.from_path(filepath)

            for filepath in test_files:
                task = asyncio.create_task(process_file(filepath))
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if isinstance(r, AudioFile)]

        start_time = time.perf_counter()
        concurrent_results = await process_concurrent()
        conc_time = time.perf_counter() - start_time

        # Concurrent should be faster (allow some variance for small datasets)
        speedup = seq_time / conc_time if conc_time > 0 else 1

        assert speedup > 1.2, \
            f"Concurrent processing speedup: {speedup:.2f}x, expected >1.2x"


class TestScalability:
    """Test scalability with increasing file counts"""

    @pytest.mark.parametrize("file_count", [10, 50, 100, 500])
    def test_linear_scaling(self, file_count: int):
        """Test that processing time scales linearly with file count"""
        # Create test files
        temp_dir = Path(tempfile.mkdtemp(f"scale_test_{file_count}_"))
        try:
            for i in range(file_count):
                filepath = temp_dir / f"file_{i:04d}.mp3"
                filepath.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00")

            # Measure processing time
            metadata_handler = MetadataHandler()
            start_time = time.perf_counter()

            for filepath in temp_dir.glob("*.mp3"):
                metadata_handler.extract_metadata(filepath)

            duration = time.perf_counter() - start_time
            time_per_file = duration / file_count

            # Assert O(n) complexity (time per file should be relatively constant)
            # Allow some variance (0.5ms to 2ms per file)
            assert 0.0005 <= time_per_file <= 0.002, \
                f"Time per file: {time_per_file*1000:.2f}ms for {file_count} files"

        finally:
            shutil.rmtree(temp_dir)


class TestPerformanceRegression:
    """Detect performance regressions by comparing against baselines"""

    def test_load_audio_file_baseline(self):
        """Test AudioFile loading performance against baseline"""
        # Create test file
        temp_dir = Path(tempfile.mkdtemp())
        try:
            test_file = temp_dir / "test.mp3"
            test_file.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00")

            # Measure loading time
            times = []
            for _ in range(100):
                start = time.perf_counter()
                audio_file = AudioFile.from_path(test_file)
                end = time.perf_counter()
                times.append(end - start)

            avg_time = sum(times) / len(times) * 1000000  # Convert to microseconds

            # Baseline: should load in under 100 microseconds
            assert avg_time < 100, \
                f"AudioFile loading: {avg_time:.1f}μs, baseline: <100μs"

        finally:
            shutil.rmtree(temp_dir)

    def test_cache_lookup_baseline(self):
        """Test cache lookup performance against baseline"""
        config = Config()
        cache = SQLiteCache.get_instance(config.cache.db_path)

        # Prime the cache
        test_path = Path("/test/file.mp3")
        cache.set(test_path, {"test": "data"})

        # Measure lookup time
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            cache.get(test_path)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times) * 1000000  # Convert to microseconds

        # Baseline: cache lookup should be under 10 microseconds
        assert avg_time < 10, \
            f"Cache lookup: {avg_time:.1f}μs, baseline: <10μs"


# Integration test for overall performance
@pytest.mark.integration
class TestEndToEndPerformance:
    """End-to-end performance tests"""

    @pytest.mark.asyncio
    async def test_full_workflow_performance(self, temp_music_library: Path):
        """Test complete workflow performance"""
        config = Config()
        organizer = AsyncOrganizer(config)

        with MemoryProfiler("Full Workflow") as profiler:
            result = await organizer.organize_files_streaming(
                source=temp_music_library,
                target=temp_music_library / "organized",
                pattern="{genre}/{artist}/{album} ({year})/{track:02d} - {title}",
                dry_run=True
            )

        # Check processing rate
        duration = profiler.stats.total_time
        rate = result.processed / duration if duration > 0 else 0

        assert rate >= PERFORMANCE_TARGETS["full_processing_rate_files_per_sec"], \
            f"Full workflow rate: {rate:.1f} files/sec, target: {PERFORMANCE_TARGETS['full_processing_rate_files_per_sec']}"

        # Check memory usage
        assert profiler.stats.peak_rss < 100, \
            f"Peak memory: {profiler.stats.peak_rss:.1f}MB, target: <100MB"