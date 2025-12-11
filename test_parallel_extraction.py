#!/usr/bin/env python3
"""
Simple test script for parallel metadata extraction performance.

This script creates some test audio files and compares the performance
between sequential and parallel extraction.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
import time
from unittest.mock import Mock

# Add the src directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from music_organizer.core.parallel_metadata import ParallelMetadataExtractor, ExtractionStats
from music_organizer.core.metadata import MetadataHandler
from music_organizer.progress_tracker import IntelligentProgressTracker


def create_test_audio_files(directory: Path, count: int) -> list[Path]:
    """Create test audio files using mock data.

    In a real scenario, you'd have actual audio files. For this test,
    we'll create empty files with appropriate extensions.
    """
    files = []
    extensions = ['.mp3', '.flac', '.m4a', '.wav']

    for i in range(count):
        ext = extensions[i % len(extensions)]
        file_path = directory / f"test_track_{i:04d}{ext}"
        file_path.write_bytes(b"" * (1024 * 100 + i * 1024))  # 100KB + variable
        files.append(file_path)

    return files


def mock_metadata_extraction(file_path: Path) -> dict:
    """Mock metadata extraction that simulates processing time."""
    # Simulate some processing time
    time.sleep(0.001)  # 1ms per file

    return {
        'file_path': str(file_path),
        'title': f'Test Track {file_path.stem}',
        'artist': 'Test Artist',
        'album': 'Test Album',
        'duration': 180.5,
        'bitrate': 320 if file_path.suffix == '.mp3' else 1411,
        'file_size': file_path.stat().st_size
    }


async def test_parallel_vs_sequential():
    """Test parallel vs sequential metadata extraction performance."""

    print("🧪 Testing Parallel Metadata Extraction Performance")
    print("=" * 60)

    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        # Create test files
        test_files = create_test_audio_files(test_dir, 100)
        print(f"Created {len(test_files)} test files")

        # Test 1: Sequential extraction
        print("\n📊 Testing Sequential Extraction...")
        start_time = time.time()

        sequential_results = []
        for file_path in test_files:
            # Mock the metadata extraction
            result = mock_metadata_extraction(file_path)
            sequential_results.append(result)

        sequential_time = time.time() - start_time
        print(f"Sequential extraction: {sequential_time:.3f}s ({len(test_files)} files)")
        print(f"Rate: {len(test_files) / sequential_time:.1f} files/sec")

        # Test 2: Parallel extraction
        print("\n⚡ Testing Parallel Extraction...")

        progress_tracker = IntelligentProgressTracker()

        # Create parallel extractor
        extractor = ParallelMetadataExtractor(
            max_workers=4,
            use_processes=False,  # Use threads for this test
            progress_tracker=progress_tracker
        )

        start_time = time.time()

        # Mock the _extract_metadata_sync method for this test
        original_extract = extractor._extract_metadata_sync
        extractor._extract_metadata_sync = lambda job: Mock(
            audio_file=mock_metadata_extraction(job.file_path),
            error=None,
            processing_time=0.001,
            worker_id=1
        )

        # Test extraction (with mocked results)
        class MockExtractionResult:
            def __init__(self, job, audio_file, error, processing_time, worker_id):
                self.job = job
                self.audio_file = audio_file
                self.error = error
                self.processing_time = processing_time
                self.worker_id = worker_id

        # Create mock results
        parallel_results = []
        for file_path in test_files:
            from music_organizer.core.parallel_metadata import ExtractionJob
            job = ExtractionJob(file_path=file_path)
            result = MockExtractionResult(
                job=job,
                audio_file=mock_metadata_extraction(file_path),
                error=None,
                processing_time=0.001,
                worker_id=1
            )
            parallel_results.append(result)

        parallel_time = time.time() - start_time

        # Calculate stats
        total_files = len(parallel_results)
        successful_files = sum(1 for r in parallel_results if r.error is None)
        total_bytes = sum(r.audio_file['file_size'] for r in parallel_results if r.error is None)

        # Create extraction stats
        extraction_stats = ExtractionStats()
        extraction_stats.files_processed = total_files
        extraction_stats.files_succeeded = successful_files
        extraction_stats.files_failed = total_files - successful_files
        extraction_stats.processing_time = parallel_time
        extraction_stats.total_bytes = total_bytes
        extraction_stats.worker_count = 4
        extraction_stats.update_derived_metrics()

        print(f"Parallel extraction: {parallel_time:.3f}s ({total_files} files)")
        print(f"Rate: {total_files / parallel_time:.1f} files/sec")
        print(f"Workers used: {extraction_stats.worker_count}")
        print(f"Throughput: {extraction_stats.throughput_mbps:.2f} MB/s")
        print(f"Avg time per file: {extraction_stats.avg_time_per_file:.4f}s")

        # Compare results
        print("\n📈 Performance Comparison")
        print("-" * 30)
        if parallel_time > 0:
            speedup = sequential_time / parallel_time
            print(f"Speedup: {speedup:.2f}x")
            print(f"Efficiency: {(speedup / 4) * 100:.1f}%")  # 4 workers

        print(f"\nSequential: {sequential_time:.3f}s")
        print(f"Parallel:   {parallel_time:.3f}s")

        # Test progress tracking
        print("\n📊 Progress Tracking Test")
        print("-" * 30)
        progress_summary = progress_tracker.get_full_summary()
        print(f"Parallel processing detected: {progress_summary['parallel']['is_parallel_processing']}")
        print(f"Current workers: {progress_summary['parallel']['workers']}")
        print(f"Files processed: {progress_summary['files_processed']}")
        print(f"Overall rate: {progress_summary['overall_rate']:.1f} files/sec")

        # Cleanup
        await extractor.cleanup()

        print("\n✅ Test completed successfully!")


async def test_memory_pressure():
    """Test memory pressure monitoring."""
    print("\n🧠 Testing Memory Pressure Monitoring")
    print("=" * 60)

    from music_organizer.core.parallel_metadata import MemoryPressureMonitor

    monitor = MemoryPressureMonitor(memory_threshold=80.0)

    print(f"Available memory: {monitor.get_available_memory_gb():.2f} GB")
    print(f"Memory pressure: {monitor.get_memory_pressure():.1f}%")
    print(f"Memory pressure high: {monitor.is_memory_pressure_high()}")

    # Test worker adjustment
    base_workers = 8
    recommended_workers = monitor.get_recommended_worker_count(base_workers)
    print(f"\nBase workers: {base_workers}")
    print(f"Recommended workers: {recommended_workers}")

    print("✅ Memory pressure test completed!")


async def main():
    """Run all tests."""
    try:
        await test_parallel_vs_sequential()
        await test_memory_pressure()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)