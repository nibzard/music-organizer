# Performance Tuning Guide

Guide to optimizing music-organizer performance for different library sizes and use cases.

## Table of Contents

1. [Overview](#overview)
2. [Performance Features](#performance-features)
3. [Worker Configuration](#worker-configuration)
4. [Caching Strategies](#caching-strategies)
5. [Memory Tuning](#memory-tuning)
6. [Bulk Operations](#bulk-operations)
7. [Library Size Guidelines](#library-size-guidelines)
8. [Performance Monitoring](#performance-monitoring)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The music-organizer is designed for high-performance processing of music libraries. Key performance targets:

- **Throughput**: 1,000+ files/second processing rate
- **Startup**: < 100ms initialization time
- **Memory**: < 100MB for large libraries (10k+ files)
- **Cache speed**: 90%+ improvement on cached runs

---

## Performance Features

### Built-in Optimizations

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Async I/O** | Non-blocking file operations | Better concurrency |
| **ThreadPoolExecutor** | Parallel worker pools | Multi-core utilization |
| **SQLite Caching** | Metadata persistence with TTL | Avoid re-reading unchanged files |
| **Streaming Pipeline** | Process files as they're scanned | Lower memory footprint |
| **Parallel Extraction** | Concurrent metadata reading | Faster processing |
| **Incremental Scanning** | Skip unchanged files | Faster subsequent runs |
| **Bulk Operations** | Batch directory creation & moves | Reduced filesystem overhead |

### Architecture Diagram

```
Source Files -> Scanner -> Metadata Extractor -> Classifier -> File Mover -> Target
                          | (cached)              | (parallel)   | (bulk ops)
                          v                       v             v
                     SQLite Cache         Worker Pool    Batch Groups
```

---

## Worker Configuration

### max_workers Parameter

Controls parallelism for file operations and metadata extraction.

**Location**: `EnhancedAsyncMusicOrganizer(max_workers=N)`

### Recommended Settings

| System | Workers | Use Case |
|--------|---------|----------|
| 2 cores | 2-4 | Small libraries (<1k files) |
| 4 cores | 4-8 | Medium libraries (1k-10k files) |
| 8+ cores | 8-16 | Large libraries (10k+ files) |
| Network storage | 2-4 | High latency storage |

**Formula**: `min(cpu_count, 8)` for most systems

### Memory Pressure

The system automatically reduces workers under memory pressure:

```python
# From parallel_metadata.py
def get_recommended_worker_count(base_workers: int) -> int:
    if memory_pressure > 80%:
        return max(1, base_workers // 2)
```

### Configuration Examples

```python
# Small library, fast storage
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    max_workers=2,
    enable_parallel_extraction=True
)

# Large library, local SSD
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    max_workers=8,
    enable_parallel_extraction=True,
    use_processes=False  # Threads are usually faster for I/O-bound work
)

# Network storage
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    max_workers=4,  # Lower for network latency
    enable_parallel_extraction=True
)
```

### CLI Usage

```bash
# Set worker count via CLI
music-organize organize source target --workers 8
```

---

## Caching Strategies

### Cache Types

| Cache Type | TTL | Use Case | Performance |
|------------|-----|----------|-------------|
| **None** | N/A | One-time processing | Baseline |
| **SQLite** | 30 days | Standard caching | 10-50x faster |
| **Smart Cache** | Adaptive (1-365 days) | Dynamic libraries | 50-200x faster |

### SQLite Cache

Basic caching with file modification time validation.

```python
from music_organizer.core.cached_metadata import CachedMetadataHandler

handler = CachedMetadataHandler(
    ttl=timedelta(days=30)  # Default
)
```

**Storage**: `~/.cache/music-organizer/metadata.db`

### Smart Cache

Intelligent caching with:
- Access frequency tracking
- Stability scoring
- Adaptive TTL (1-365 days)
- Directory-level change detection

```python
from music_organizer.core.smart_cached_metadata import SmartCachedMetadataHandler

handler = SmartCachedMetadataHandler(
    ttl=timedelta(days=30),
    enable_warming=True,      # Pre-load frequent files
    enable_optimization=True  # Auto-optimize cache
)
```

**Storage**: `~/.cache/music-organizer/smart_cache.db`

### Cache Configuration

```python
# Basic cache
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    use_cache=True,
    cache_ttl=30  # days
)

# Smart cache
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    use_cache=True,
    use_smart_cache=True,  # Enables smart caching
    cache_ttl=30
)
```

### Cache Warming

Pre-populate cache for frequently accessed files:

```python
from music_organizer.core.smart_cache import SmartCacheManager

cache = SmartCacheManager()
cache.warm_cache(
    directory=Path("~/Music"),
    recursive=True,
    max_files=1000  # Prioritize by frequency
)
```

### Cache Statistics

```python
# Get cache stats
stats = cache.get_smart_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Hit rate: {stats['valid_entries'] / stats['total_entries']:.1%}")
print(f"Size: {stats['size_mb']:.1f} MB")
```

### Cache Maintenance

```python
# Clean expired entries
expired_count = cache.cleanup_expired()

# Optimize database (reclaim space)
result = cache.optimize_cache()
print(f"Space saved: {result['space_saved_mb']:.1f} MB")
```

### Cache TTL Guidelines

| Scenario | Recommended TTL |
|----------|-----------------|
| Static library (rarely modified) | 90-365 days |
| Active library (frequent additions) | 7-30 days |
| Testing/development | 1-7 days |
| Network storage (metadata may change) | 1-7 days |

---

## Memory Tuning

### Memory Monitoring

Built-in memory pressure monitoring in `ParallelMetadataExtractor`:

```python
from music_organizer.core.parallel_metadata import ParallelMetadataExtractor

extractor = ParallelMetadataExtractor(
    max_workers=8,
    memory_threshold=80.0,  # Reduce workers at 80% RAM
    enable_memory_monitoring=True,
    batch_size=50  # Process in batches
)
```

### Batch Size Configuration

Controls how many files are processed per batch. Smaller = less memory.

| Library Size | Batch Size | Memory Impact |
|--------------|------------|---------------|
| < 1k files | 100-200 | Minimal |
| 1k-10k files | 50-100 | Low |
| 10k-100k files | 25-50 | Moderate |
| 100k+ files | 10-25 | Higher control |

### Configuration

```python
# Large library, memory-constrained
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    max_workers=4,
    enable_parallel_extraction=True
)

# Configure parallel extractor
extractor = ParallelMetadataExtractor(
    max_workers=4,
    batch_size=25,
    memory_threshold=70.0  # Be conservative
)
```

### Memory Usage by Component

| Component | Memory per 1k files | Notes |
|-----------|--------------------|-------|
| AudioFile objects | ~5 MB | slots=True reduces overhead |
| SQLite cache | ~10 MB | Indexed queries |
| Worker overhead | ~2 MB per worker | ThreadPoolExecutor |
| Progress tracking | ~1 MB | IntelligentProgressTracker |

**Total for 10k files**: ~50-70 MB baseline

---

## Bulk Operations

### When to Use Bulk Operations

| Scenario | Use Bulk? | Reason |
|----------|-----------|--------|
| 100+ files | Yes | Reduced syscall overhead |
| 1000+ files | Yes | Significant speedup |
| Network storage | Yes | Minimize round-trips |
| Local SSD | Optional | Less benefit on fast storage |

### Bulk Operation Config

```python
from music_organizer.core.bulk_operations import BulkOperationConfig, ConflictStrategy

config = BulkOperationConfig(
    max_workers=8,
    chunk_size=100,           # Files per batch
    conflict_strategy=ConflictStrategy.RENAME,
    verify_copies=False,      # Faster for moves
    create_dirs_batch=True,   # Batch directory creation
    skip_identical=True       # Skip unchanged files
)
```

### Using Bulk Operations

```python
# Via organizer
result = await organizer.organize_files_bulk(
    source_dir=Path("~/Music/unorganized"),
    target_dir=Path("~/Music/organized"),
    bulk_config=config
)
```

### Conflict Strategies

| Strategy | Speed | Use Case |
|----------|-------|----------|
| `SKIP` | Fastest | Keep existing files |
| `REPLACE` | Fast | Overwrite existing |
| `RENAME` | Medium | Keep both (add number) |
| `KEEP_BOTH` | Medium | Same as RENAME |

### Performance Tips

1. **Use `skip_identical=True`**: Avoids copying unchanged files
2. **Batch directory creation**: Reduces filesystem overhead
3. **Disable verification**: For moves, verification is redundant
4. **Group by directory**: Better filesystem locality

---

## Library Size Guidelines

### Small Libraries (< 1,000 files)

**Configuration**:
```python
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    max_workers=2,
    use_cache=True,      # Still beneficial
    enable_parallel_extraction=True
)
```

**Performance**: ~1-2 seconds total

**Tips**:
- Default settings work well
- Cache not critical but still helps

### Medium Libraries (1,000 - 10,000 files)

**Configuration**:
```python
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    max_workers=4,
    use_cache=True,
    use_smart_cache=True,  # Recommended
    enable_parallel_extraction=True
)
```

**Performance**: ~10-30 seconds (cold), ~1-5 seconds (warm)

**Tips**:
- Enable smart cache
- Consider cache warming
- Monitor memory usage

### Large Libraries (10,000 - 100,000 files)

**Configuration**:
```python
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    max_workers=8,
    use_cache=True,
    use_smart_cache=True,
    enable_parallel_extraction=True,
    use_processes=False  # Threads better for I/O-bound
)
```

**Performance**: ~1-5 minutes (cold), ~10-30 seconds (warm)

**Tips**:
- Use incremental scanning
- Batch size 25-50
- Enable memory monitoring
- Consider bulk operations
- Warm cache for frequent files

### Very Large Libraries (100,000+ files)

**Configuration**:
```python
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    max_workers=8,
    use_cache=True,
    use_smart_cache=True,
    enable_parallel_extraction=True
)

# Tuned parallel extractor
extractor = ParallelMetadataExtractor(
    max_workers=8,
    batch_size=25,
    memory_threshold=70.0,
    enable_memory_monitoring=True
)
```

**Performance**: ~5-20 minutes (cold), ~1-5 minutes (warm)

**Tips**:
- Process in batches by directory
- Use incremental scanning
- Aggressive cache TTL (90+ days)
- Monitor system resources
- Consider overnight runs for full scans

---

## Performance Monitoring

### Progress Tracking

Built-in progress tracker provides real-time metrics:

```python
from music_organizer.progress_tracker import IntelligentProgressTracker

tracker = organizer.progress_tracker

# Get current progress
stats = tracker.get_stats()
print(f"Progress: {stats['progress']:.1f}%")
print(f"Files/sec: {stats['files_per_second']:.1f}")
print(f"ETA: {stats['eta']:.0f} seconds")
```

### Parallel Metrics

When using parallel extraction:

```python
stats = extractor.get_stats()
print(f"Files processed: {stats.files_processed}")
print(f"Throughput: {stats.throughput_mbps:.2f} MB/s")
print(f"Avg time per file: {stats.avg_time_per_file*1000:.1f}ms")
print(f"Peak memory: {stats.memory_peak_mb:.1f} MB")
```

### Cache Statistics

```python
# SQLite cache
stats = cache.get_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Valid entries: {stats['valid_entries']}")
print(f"Hit rate: {stats['valid_entries'] / stats['total_entries']:.1%}")

# Smart cache
stats = smart_cache.get_smart_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Frequently accessed: {stats['frequently_accessed_files']}")
print(f"Stable files: {stats['stable_files']}")
```

### Benchmarking

Run the built-in benchmarks:

```bash
cd /path/to/music-organizer
python benchmarks/run_benchmarks.py
```

Expected output:
```
Startup Time: 45.23ms (target: <100ms) ✅
Cold extraction: 245.3 files/sec
Warm extraction: 421.8 files/sec
Cache improvement: 71.5% (target: >90%) ❌
Peak memory: 23.45MB for 500 files
Extrapolated for 10k files: 469.0MB (target: <100MB) ❌
```

---

## Troubleshooting

### Slow Processing

**Symptoms**: Low files/sec rate

**Solutions**:
1. Increase `max_workers` (up to CPU count)
2. Enable parallel extraction
3. Check disk I/O with `iostat` or `iotop`
4. Disable antivirus scanning of music directories
5. Use local storage instead of network

**Diagnostic**:
```python
stats = extractor.get_stats()
print(f"Avg time per file: {stats.avg_time_per_file*1000:.1f}ms")
print(f"Workers: {stats.worker_count}")
print(f"Throughput: {stats.throughput_mbps:.2f} MB/s")
```

### High Memory Usage

**Symptoms**: System swapping, OOM errors

**Solutions**:
1. Reduce `max_workers`
2. Reduce `batch_size`
3. Lower `memory_threshold`
4. Process in smaller batches

**Configuration**:
```python
extractor = ParallelMetadataExtractor(
    max_workers=4,        # Reduce from 8
    batch_size=25,        # Reduce from 50
    memory_threshold=70.0 # Lower from 80
)
```

### Cache Not Working

**Symptoms**: No speed improvement on subsequent runs

**Solutions**:
1. Check cache directory exists: `~/.cache/music-organizer/`
2. Verify cache not being cleared
3. Check file modification times
4. Increase cache TTL

**Diagnostic**:
```python
stats = cache.get_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Valid entries: {stats['valid_entries']}")
```

### Network Storage Performance

**Symptoms**: Slow processing on NAS/cloud storage

**Solutions**:
1. Reduce `max_workers` to 2-4
2. Increase batch size to reduce round-trips
3. Disable file verification
4. Consider local caching

**Configuration**:
```python
config = BulkOperationConfig(
    max_workers=4,
    chunk_size=200,        # Larger batches
    verify_copies=False   # Skip verification
)
```

### Metadata Extraction Failures

**Symptoms**: High failure rate in extraction

**Solutions**:
1. Check file permissions
2. Verify file formats are supported
3. Check for corrupted files
4. Enable error logging

**Diagnostic**:
```python
stats = extractor.get_stats()
print(f"Success rate: {stats.files_succeeded / stats.files_processed:.1%}")
print(f"Failed: {stats.files_failed}")
```

---

## Advanced Configuration

### Custom Executor

```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="music-org"
)

organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    max_workers=8
)
organizer.executor = executor
```

### Streaming Processing

For very large libraries, process as stream:

```python
async def process_stream(source_dir):
    async for file_path in scanner.scan_stream(source_dir):
        result = await organizer.process_single(file_path)
        yield result
```

### Incremental Scanning

Skip unchanged files:

```python
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    enable_incremental=True  # Use IncrementalScanner
)
```

---

## Quick Reference

### CLI Performance Options

```bash
# Worker count
music-organize organize source target --workers 8

# Incremental (skip unchanged)
music-organize organize source target --incremental

# Bulk operations (implied for large libraries)
music-organize organize source target --bulk

# Verbose progress
music-organize organize source target --verbose
```

### Python API Quick Config

```python
# Fastest for large libraries
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    max_workers=8,
    use_smart_cache=True,
    enable_parallel_extraction=True,
    cache_ttl=90
)

# Memory-constrained
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    max_workers=4,
    use_cache=True,
    enable_parallel_extraction=True
)

# Network storage
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    max_workers=4,
    use_cache=True,
    cache_ttl=30
)
```

---

## Further Reading

- [API Reference](api-reference.md) - Complete API documentation
- [Plugin Development Guide](plugin-development.md) - Extending functionality
- [CQRS Implementation](cqrs-implementation.md) - Architecture details
