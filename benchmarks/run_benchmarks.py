#!/usr/bin/env python3
"""
Performance benchmarks for Music Organizer

Tests the performance targets defined in TODO.md:
- Process 10,000 files in < 10 seconds (1,000+ files/sec)
- Memory usage < 100MB for large libraries
- 90% speed improvement on cached runs
- Startup time < 100ms
"""

import asyncio
import os
import time
import tracemalloc
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import json
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from music_organizer.models.audio_file import AudioFile
from music_organizer.models.config import Config
from music_organizer.core.metadata import MetadataHandler
from music_organizer.core.cache import SQLiteCache
from music_organizer.core.async_organizer import AsyncMusicOrganizer
from music_organizer.core.cached_metadata import CachedMetadataHandler


@dataclass
class BenchmarkResult:
    """Stores the result of a benchmark run"""
    name: str
    value: float
    unit: str
    target: float
    passed: bool
    details: Dict[str, Any] = None


class BenchmarkRunner:
    """Runs performance benchmarks for the Music Organizer"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.temp_dir = None

    def setup_test_environment(self):
        """Set up temporary test environment with sample audio files"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="music_bench_"))

        # Create a realistic directory structure
        genres = ["Rock", "Jazz", "Classical", "Electronic", "Hip-Hop"]
        artists_per_genre = 20
        albums_per_artist = 3
        tracks_per_album = 10

        total_files = 0
        for genre in genres:
            genre_dir = self.temp_dir / genre
            genre_dir.mkdir()

            for artist_idx in range(artists_per_genre):
                artist = f"Artist_{artist_idx:02d}"
                artist_dir = genre_dir / artist
                artist_dir.mkdir()

                for album_idx in range(albums_per_artist):
                    year = 2000 + artist_idx + album_idx
                    album = f"Album_{album_idx:02d} ({year})"
                    album_dir = artist_dir / album
                    album_dir.mkdir()

                    for track_idx in range(tracks_per_album):
                        track_num = track_idx + 1
                        title = f"Track_{track_idx:02d}"
                        filename = f"{track_num:02d} - {title}.mp3"
                        filepath = album_dir / filename

                        # Create a dummy file (minimal MP3 header)
                        filepath.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00")
                        total_files += 1

        print(f"Created test environment with {total_files} files in {self.temp_dir}")
        return total_files

    def cleanup_test_environment(self):
        """Clean up temporary test environment"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up test environment: {self.temp_dir}")

    def benchmark_startup_time(self):
        """Benchmark application startup time (target: < 100ms)"""
        print("\n🚀 Benchmarking startup time...")

        # Measure import and initialization time
        start_time = time.perf_counter()

        # Simulate startup - just measure imports and basic init
        metadata_handler = MetadataHandler()

        # Config needs temp dirs
        source = self.temp_dir / "source"
        target = self.temp_dir / "target"
        cache_dir = self.temp_dir / "cache"
        source.mkdir(exist_ok=True)
        target.mkdir(exist_ok=True)
        cache_dir.mkdir(exist_ok=True)
        config = Config(source_directory=source, target_directory=target)

        cache = SQLiteCache(cache_dir)

        end_time = time.perf_counter()
        startup_time = (end_time - start_time) * 1000  # Convert to ms

        result = BenchmarkResult(
            name="Startup Time",
            value=startup_time,
            unit="ms",
            target=100,
            passed=startup_time < 100,
            details={"imports": ["Config", "MetadataHandler", "SQLiteCache"]}
        )

        self.results.append(result)
        print(f"   Startup time: {startup_time:.2f}ms (target: <100ms) {'✅' if result.passed else '❌'}")

    def benchmark_metadata_extraction(self, num_files: int = 1000):
        """Benchmark metadata extraction performance"""
        print(f"\n📊 Benchmarking metadata extraction ({num_files} files)...")

        source = self.temp_dir / "source"
        target = self.temp_dir / "target"
        source.mkdir(exist_ok=True)
        target.mkdir(exist_ok=True)
        config = Config(source_directory=source, target_directory=target)
        metadata_handler = MetadataHandler()

        # Get test files
        test_files = list(self.temp_dir.rglob("*.mp3"))[:num_files]

        # Measure cold extraction (no cache)
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
        cold_time = end_time - start_time
        cold_rate = extracted / cold_time if cold_time > 0 else 0

        # Measure with cache (warm)
        cache_dir = self.temp_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        cached_handler = CachedMetadataHandler(cache_dir)
        start_time = time.perf_counter()

        for filepath in test_files:
            try:
                audio_file = cached_handler.get_metadata(filepath)
                if audio_file:
                    extracted += 1
            except Exception:
                pass

        end_time = time.perf_counter()
        warm_time = end_time - start_time
        warm_rate = extracted / warm_time if warm_time > 0 else 0

        # Calculate cache improvement
        improvement = ((cold_time - warm_time) / cold_time * 100) if cold_time > 0 else 0

        result = BenchmarkResult(
            name="Metadata Extraction Rate",
            value=max(cold_rate, warm_rate),
            unit="files/sec",
            target=100,  # Target based on 10k files in 100s for extraction only
            passed=max(cold_rate, warm_rate) >= 100,
            details={
                "cold_rate": cold_rate,
                "warm_rate": warm_rate,
                "cache_improvement": improvement,
                "files_tested": len(test_files)
            }
        )

        self.results.append(result)
        print(f"   Cold extraction: {cold_rate:.1f} files/sec")
        print(f"   Warm extraction: {warm_rate:.1f} files/sec")
        print(f"   Cache improvement: {improvement:.1f}% (target: >90%) {'✅' if improvement >= 90 else '❌'}")

    def benchmark_memory_usage(self):
        """Benchmark memory usage during processing"""
        print("\n💾 Benchmarking memory usage...")

        tracemalloc.start()

        # Process a subset of files and measure memory
        test_files = list(self.temp_dir.rglob("*.mp3"))[:500]

        processed = 0
        for filepath in test_files:
            try:
                # Simulate processing
                audio_file = AudioFile.from_path(filepath)
                processed += 1

                if processed % 100 == 0:
                    current, peak = tracemalloc.get_traced_memory()
                    peak_mb = peak / 1024 / 1024
                    if peak_mb > 100:  # Early stop if we exceed target
                        break
            except Exception:
                pass

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        mb_per_file = peak_mb / processed if processed > 0 else 0

        # Extrapolate for 10k files
        extrapolated_mb = mb_per_file * 10000

        result = BenchmarkResult(
            name="Memory Usage",
            value=extrapolated_mb,
            unit="MB",
            target=100,
            passed=extrapolated_mb < 100,
            details={
                "peak_mb": peak_mb,
                "processed_files": processed,
                "mb_per_file": mb_per_file,
                "extrapolated_for_10k": extrapolated_mb
            }
        )

        self.results.append(result)
        print(f"   Peak memory: {peak_mb:.2f}MB for {processed} files")
        print(f"   Extrapolated for 10k files: {extrapolated_mb:.1f}MB (target: <100MB) {'✅' if result.passed else '❌'}")

    async def benchmark_full_processing(self, num_files: int = 1000):
        """Benchmark end-to-end processing performance"""
        print(f"\n⚡ Benchmarking full processing pipeline ({num_files} files)...")

        source = self.temp_dir / "source"
        target = self.temp_dir / "target"
        source.mkdir(exist_ok=True)
        target.mkdir(exist_ok=True)
        config = Config(source_directory=source, target_directory=target)
        organizer = AsyncMusicOrganizer(config, dry_run=True)

        # Prepare source and target
        source_dir = self.temp_dir
        target_dir = self.temp_dir / "output"
        target_dir.mkdir(exist_ok=True)

        # Get test files
        all_files = list(source_dir.rglob("*.mp3"))
        test_files = all_files[:num_files]

        # Measure processing time
        start_time = time.perf_counter()

        result = await organizer.organize_files(test_files)

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Calculate rate
        rate = result.get('processed', 0) / processing_time if processing_time > 0 else 0

        # Extrapolate to 10k files
        extrapolated_time = 10000 / rate if rate > 0 else float('inf')

        benchmark_result = BenchmarkResult(
            name="Full Processing Rate",
            value=rate,
            unit="files/sec",
            target=1000,  # 10k files in 10 seconds
            passed=extrapolated_time < 10,
            details={
                "processed": result.get('processed', 0),
                "errors": result.get('errors', []),
                "time_seconds": processing_time,
                "extrapolated_10k_seconds": extrapolated_time
            }
        )

        self.results.append(benchmark_result)
        print(f"   Processed: {result.get('processed', 0)} files in {processing_time:.2f}s")
        print(f"   Rate: {rate:.1f} files/sec")
        print(f"   Extrapolated for 10k files: {extrapolated_time:.1f}s (target: <10s) {'✅' if benchmark_result.passed else '❌'}")

    def generate_report(self):
        """Generate a benchmark report"""
        print("\n" + "="*60)
        print("📊 BENCHMARK REPORT")
        print("="*60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{status} {result.name}")
            print(f"   Value: {result.value:.2f} {result.unit}")
            print(f"   Target: {result.target} {result.unit}")

            if result.details:
                for key, value in result.details.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.2f}")
                    else:
                        print(f"   {key}: {value}")

        print(f"\n📈 SUMMARY: {passed}/{total} benchmarks passed")

        if passed == total:
            print("🎉 All performance targets achieved!")
        else:
            print("⚠️  Some performance targets not met. Review failed benchmarks.")

        # Save results to JSON
        report_path = Path("benchmark_results.json")
        with open(report_path, "w") as f:
            json.dump([
                {
                    "name": r.name,
                    "value": r.value,
                    "unit": r.unit,
                    "target": r.target,
                    "passed": r.passed,
                    "details": r.details
                }
                for r in self.results
            ], f, indent=2)

        print(f"\n💾 Results saved to {report_path}")

        return passed == total

    async def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("🏃 Running Music Organizer Performance Benchmarks")
        print("="*60)

        # Setup
        total_files = self.setup_test_environment()

        try:
            # Run individual benchmarks
            self.benchmark_startup_time()
            self.benchmark_metadata_extraction(min(1000, total_files // 10))
            self.benchmark_memory_usage()
            await self.benchmark_full_processing(min(1000, total_files // 5))

            # Generate report
            all_passed = self.generate_report()

            return all_passed

        finally:
            self.cleanup_test_environment()


async def main():
    """Main benchmark entry point"""
    runner = BenchmarkRunner()

    try:
        success = await runner.run_all_benchmarks()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Benchmarks interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Benchmark error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())