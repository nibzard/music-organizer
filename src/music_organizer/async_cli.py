"""Async command line interface for music organizer."""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

from .core.async_organizer import AsyncMusicOrganizer, organize_files_async
from .core.metadata import MetadataHandler
from .core.classifier import ContentClassifier
from .core.bulk_operations import BulkOperationConfig, ConflictStrategy
from .core.bulk_organizer import BulkAsyncOrganizer
from .core.bulk_progress_tracker import BulkProgressTracker
from .core.organization_preview import OrganizationPreview, InteractivePreview
from .models.config import Config, load_config
from .exceptions import MusicOrganizerError
from .progress_tracker import IntelligentProgressTracker, ProgressStage
from .async_progress_renderer import AsyncProgressRenderer


class SimpleProgress:
    """Simple progress bar implementation without external dependencies."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()

    def update(self, advance: int = 1):
        """Update progress bar."""
        self.current += advance
        self._draw()

    def _draw(self):
        """Draw the progress bar."""
        if self.total == 0:
            return

        percent = min(100, (self.current * 100) // self.total)
        bar_length = 50
        filled = (percent * bar_length) // 100
        bar = '█' * filled + '░' * (bar_length - filled)

        elapsed = datetime.now() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f"{int(eta.total_seconds())}s"
        else:
            eta_str = "?:??"

        print(f"\r{self.description}: [{bar}] {percent}% ({self.current}/{self.total}) ETA: {eta_str}", end='')

        if self.current >= self.total:
            print()  # New line when complete


class SimpleConsole:
    """Simple console output without rich dependency."""

    @staticmethod
    def print(text: str, style: Optional[str] = None):
        """Print text with optional style."""
        if style == 'bold':
            print(f"\033[1m{text}\033[0m")
        elif style == 'green':
            print(f"\033[92m{text}\033[0m")
        elif style == 'red':
            print(f"\033[91m{text}\033[0m")
        elif style == 'yellow':
            print(f"\033[93m{text}\033[0m")
        elif style == 'cyan':
            print(f"\033[96m{text}\033[0m")
        else:
            print(text)

    @staticmethod
    def rule(title: str):
        """Print a horizontal rule with title."""
        width = 80
        padding = (width - len(title) - 2) // 2
        print(f"{'-' * padding} {title} {'-' * padding}")

    @staticmethod
    def table(rows: List[List[str]], headers: List[str]):
        """Print a simple table."""
        if not rows:
            return

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))

        # Print rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            print(row_line)


class AsyncMusicCLI:
    """Async CLI implementation for music organizer."""

    def __init__(self):
        self.console = SimpleConsole()

    async def organize(self,
                      source: Path,
                      target: Path,
                      config_path: Optional[Path] = None,
                      dry_run: bool = False,
                      interactive: bool = False,
                      backup: bool = True,
                      max_workers: int = 4,
                      use_processes: bool = False,
                      enable_parallel_extraction: bool = True,
                      memory_threshold: float = 80.0,
                      use_cache: bool = True,
                      cache_ttl: Optional[int] = None,
                      incremental: bool = False,
                      force_full_scan: bool = False,
                      bulk_mode: bool = False,
                      chunk_size: int = 200,
                      conflict_strategy: str = 'rename',
                      verify_copies: bool = False,
                      batch_dirs: bool = True,
                      preview_bulk: bool = False,
                      bulk_memory_threshold: int = 512,
                      smart_cache: Optional[bool] = None,
                      cache_warming: Optional[bool] = None,
                      cache_optimize: Optional[bool] = None,
                      warm_cache_dir: Optional[Path] = None,
                      cache_health: bool = False,
                      magic_mode: bool = False,
                      magic_analyze: bool = False,
                      magic_auto: bool = False,
                      magic_sample: Optional[int] = None,
                      magic_preview: bool = False,
                      magic_save_config: Optional[Path] = None,
                      magic_threshold: float = 0.6,
                      preview: bool = False,
                      preview_detailed: bool = False,
                      preview_interactive: bool = False,
                      export_preview: Optional[Path] = None) -> int:
        """Organize music files asynchronously."""
        try:
            # Load configuration
            if config_path and config_path.exists():
                config = load_config(config_path)
            else:
                config = Config(
                    source_directory=source,
                    target_directory=target,
                    file_operations=type('FileOps', (), {'backup': backup})()
                )

            # Validate directories
            if not source.exists():
                self.console.print(f"Error: Source directory does not exist: {source}", 'red')
                return 1

            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)

            # Handle Magic Mode
            if magic_mode or magic_analyze:
                return await self._handle_magic_mode(
                    source, target, config, dry_run, magic_analyze, magic_auto,
                    magic_sample, magic_preview, magic_save_config, magic_threshold,
                    backup, max_workers, use_processes, enable_parallel_extraction,
                    memory_threshold, use_cache, cache_ttl, incremental,
                    force_full_scan, bulk_mode, chunk_size, conflict_strategy,
                    verify_copies, batch_dirs, preview_bulk, bulk_memory_threshold,
                    smart_cache, cache_warming, cache_optimize, warm_cache_dir,
                    cache_health
                )

            # Handle Organization Preview
            if preview or preview_detailed or preview_interactive:
                return await self._handle_organization_preview(
                    source, target, config, preview_detailed, preview_interactive,
                    export_preview, max_workers, use_processes,
                    enable_parallel_extraction, memory_threshold, use_cache,
                    cache_ttl, incremental, force_full_scan, bulk_mode,
                    chunk_size, conflict_strategy, smart_cache, cache_warming,
                    cache_optimize, warm_cache_dir
                )

            # Initialize organizer
            async with AsyncMusicOrganizer(
                config,
                dry_run=dry_run,
                interactive=interactive,
                max_workers=max_workers,
                enable_parallel_extraction=enable_parallel_extraction,
                use_processes=use_processes,
                use_cache=use_cache,
                cache_ttl=cache_ttl
            ) as organizer:

                self.console.print(f"\n🎵 Music Organizer (Async Mode)")
                self.console.print(f"Source: {source}")
                self.console.print(f"Target: {target}")
                self.console.print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
                self.console.print(f"Workers: {max_workers}")
                if enable_parallel_extraction:
                    pool_type = "Process Pool" if use_processes else "Thread Pool"
                    self.console.print(f"Parallel: ENABLED ({pool_type})")
                    self.console.print(f"Memory Threshold: {memory_threshold}%")
                else:
                    self.console.print("Parallel: DISABLED")
                self.console.print(f"Cache: {'ENABLED' if use_cache else 'DISABLED'}")
                if use_cache:
                    ttl_str = f"{cache_ttl} days" if cache_ttl else "30 days (default)"
                    self.console.print(f"Cache TTL: {ttl_str}")

                # Show scan mode
                if incremental:
                    if force_full_scan:
                        self.console.print("Scan mode: Full (forced)", 'yellow')
                    else:
                        scan_info = organizer.get_scan_info(source)
                        if scan_info and scan_info['has_history']:
                            self.console.print(f"Scan mode: Incremental (last scan: {scan_info['last_scan'][:19]})", 'green')
                        else:
                            self.console.print("Scan mode: Full (no previous scan history)", 'yellow')
                else:
                    self.console.print("Scan mode: Full (standard scan)", 'yellow')

                # Count total files first
                self.console.print("\n🔍 Scanning files...")
                file_count = 0

                if incremental and not force_full_scan:
                    # Use incremental scanning
                    scan_info = organizer.get_scan_info(source)
                    async for _ in organizer.scan_directory_incremental(
                        source, force_full=False, filter_modified=True
                    ):
                        file_count += 1
                else:
                    # Use full scan
                    async for _ in organizer.scan_directory(source):
                        file_count += 1

                if file_count == 0:
                    if incremental and not force_full_scan:
                        self.console.print("No new or modified files found since last scan!", 'green')
                    else:
                        self.console.print("No audio files found!", 'yellow')
                    return 0

                scan_type = "new/modified" if incremental and not force_full_scan else "audio"
                self.console.print(f"Found {file_count} {scan_type} files")

                # Smart cache operations
                if use_cache:
                    # Determine smart cache settings
                    enable_smart_cache = smart_cache if smart_cache is not None else (not getattr(args, 'no_smart_cache', False))
                    enable_cache_warming = cache_warming if cache_warming is not None else (not getattr(args, 'no_cache_warming', False))
                    enable_cache_optimize = cache_optimize if cache_optimize is not None else (not getattr(args, 'no_cache_optimize', False))

                    if enable_smart_cache:
                        from .core.smart_cached_metadata import get_smart_cached_metadata_handler
                        smart_handler = get_smart_cached_metadata_handler(
                            enable_smart_cache=True,
                            cache_warming_enabled=enable_cache_warming,
                            auto_optimize=enable_cache_optimize
                        )
                        self.console.print(f"Smart Cache: ENABLED", 'green')

                        # Cache warming
                        if warm_cache_dir and warm_cache_dir.exists():
                            self.console.print(f"\n🔥 Warming cache for {warm_cache_dir}...")
                            warmed = smart_handler.warm_cache(warm_cache_dir, recursive=True)
                            self.console.print(f"Warmed cache with {warmed} files", 'green')
                        elif enable_cache_warming and not incremental:
                            # Auto-warm source directory if not incremental
                            self.console.print(f"\n🔥 Auto-warming cache for {source}...")
                            warmed = smart_handler.warm_cache(source, recursive=True, max_files=500)
                            self.console.print(f"Warmed cache with {warmed} files", 'green')

                        # Cache optimization
                        if enable_cache_optimize:
                            self.console.print("\n⚡ Optimizing cache...")
                            opt_results = smart_handler.optimize_cache(force=False)
                            if 'skipped' not in opt_results:
                                self.console.print(f"Optimization complete: {opt_results}", 'green')
                else:
                    self.console.print("Smart Cache: DISABLED", 'yellow')

                # Show bulk mode information if enabled
                if bulk_mode:
                    self.console.print(f"\n⚡ Bulk Operations: ENABLED")
                    self.console.print(f"  Chunk size: {chunk_size}")
                    self.console.print(f"  Conflict strategy: {conflict_strategy}")
                    self.console.print(f"  Verify copies: {verify_copies}")
                    self.console.print(f"  Batch directories: {batch_dirs}")
                    self.console.print(f"  Memory threshold: {bulk_memory_threshold}MB")

                # Preview mode for bulk operations
                if bulk_mode and preview_bulk:
                    self.console.print("\n🔮 Preview Mode - Analyzing bulk operation...")
                    preview = await organizer.get_bulk_operation_preview(source)

                    self.console.print("\n📋 Bulk Operation Preview:")
                    self.console.print(f"  Source files: {preview.get('source_files', 0)}")
                    self.console.print(f"  Estimated size: {preview.get('estimated_size_mb', 0):.1f} MB")
                    self.console.print(f"  Estimated duration: {preview.get('estimated_duration', 0):.1f} seconds")

                    content_dist = preview.get('content_distribution', {})
                    if content_dist:
                        self.console.print("\n📂 Content distribution:")
                        for content_type, count in content_dist.items():
                            self.console.print(f"  {content_type}: {count}")

                    if not dry_run:
                        confirm = input("\nContinue with bulk operation? (y/N): ")
                        if confirm.lower() != 'y':
                            self.console.print("Operation cancelled by user.", 'yellow')
                            return 0

                # Initialize intelligent progress tracker
                progress_tracker = IntelligentProgressTracker()
                progress_renderer = AsyncProgressRenderer()
                progress_tracker.add_render_callback(progress_renderer.render)
                progress_tracker.set_total_files(file_count)

                # Start scanning stage
                progress_tracker.start_stage(ProgressStage.SCANNING)

                # Process files using bulk or standard mode
                results = {
                    'processed': 0,
                    'moved': 0,
                    'skipped': 0,
                    'by_category': {
                        'Albums': 0,
                        'Live': 0,
                        'Collaborations': 0,
                        'Compilations': 0,
                        'Rarities': 0,
                        'Unknown': 0
                    },
                    'errors': []
                }

                # Switch to processing stage
                progress_tracker.finish_stage(ProgressStage.SCANNING)

                # Use bulk operations if enabled
                if bulk_mode:
                    # Create bulk operation configuration
                    from .core.bulk_progress_tracker import BulkProgressTracker
                    bulk_tracker = BulkProgressTracker(
                        update_interval=0.5,
                        batch_size=chunk_size
                    )

                    # Create bulk config
                    conflict_enum = ConflictStrategy(conflict_strategy)
                    bulk_config = BulkOperationConfig(
                        max_workers=max_workers,
                        chunk_size=chunk_size,
                        conflict_strategy=conflict_enum,
                        verify_copies=verify_copies,
                        create_dirs_batch=batch_dirs,
                        preserve_timestamps=True,
                        skip_identical=True,
                        memory_threshold_mb=bulk_memory_threshold
                    )

                    # Start bulk operation progress tracking
                    bulk_tracker.start_bulk_operation(file_count, estimated_batches=(file_count + chunk_size - 1) // chunk_size)

                    # Progress callback for bulk operations
                    async def bulk_progress(current, total, stage="Processing"):
                        bulk_tracker.update_batch_progress(current, total)
                        progress_tracker.set_completed(current)

                    # Execute bulk organization
                    self.console.print("\n⚡ Executing bulk organization...")
                    progress_tracker.start_stage(ProgressStage.MOVING, total=file_count)

                    results = await organizer.organize_files_bulk(
                        directory=source,
                        bulk_config=bulk_config,
                        incremental=incremental and not force_full_scan,
                        progress_callback=bulk_progress
                    )

                    progress_tracker.finish_stage(ProgressStage.MOVING)

                    # Get bulk performance report
                    bulk_summary = bulk_tracker.get_bulk_summary()
                    performance_report = bulk_tracker.get_performance_report()

                    # Show bulk statistics
                    if 'bulk_stats' in results:
                        stats = results['bulk_stats']
                        self.console.print("\n⚡ Bulk Operation Statistics:")
                        self.console.print(f"  Success rate: {stats.get('success_rate', 0):.1f}%")
                        self.console.print(f"  Throughput: {stats.get('throughput_mb_per_sec', 0):.1f} MB/s")
                        self.console.print(f"  Total size: {stats.get('total_size_mb', 0):.1f} MB")
                        self.console.print(f"  Conflicts resolved: {stats.get('conflicts_resolved', 0)}")

                    # Update results for display
                    results['processed'] = results.get('processed', file_count)
                    results['moved'] = results.get('moved', 0)
                    results['skipped'] = results.get('skipped', 0)
                    results['errors'] = results.get('errors', [])

                elif enable_parallel_extraction and incremental and not force_full_scan:
                    # Use the new parallel incremental organization
                    progress_tracker.start_stage(ProgressStage.PARALLEL_EXTRACTION, total=file_count)

                    results = await organizer.organize_incremental_parallel(
                        source,
                        force_full=False,
                        progress_callback=lambda progress, completed: progress_tracker.set_completed(
                            completed,
                            error=False
                        )
                    )

                    progress_tracker.finish_stage(ProgressStage.PARALLEL_EXTRACTION)
                else:
                    # Use the original streaming approach
                    progress_tracker.start_stage(ProgressStage.METADATA_EXTRACTION, total=file_count)

                    # Choose the appropriate scanner
                    if incremental and not force_full_scan:
                        file_scanner = organizer.scan_directory_incremental(
                            source, force_full=False, filter_modified=True
                        )
                    else:
                        file_scanner = organizer.scan_directory(source)

                    async for file_path, success, error in organizer.organize_files_streaming(
                        file_scanner,
                        batch_size=50
                    ):
                        results['processed'] += 1

                        # Update progress with file size
                        try:
                            file_size = file_path.stat().st_size
                        except:
                            file_size = 0

                        progress_tracker.set_completed(
                            results['processed'],
                            bytes_processed=file_size,
                            error=error is not None
                        )

                        if success:
                            results['moved'] += 1
                        else:
                            results['skipped'] += 1
                            if error:
                                results['errors'].append(f"{file_path.name}: {error}")

                    # Finish progress tracking
                    progress_tracker.finish_stage(ProgressStage.METADATA_EXTRACTION)

                # Clear progress renderer
                progress_renderer.clear()

                # Show results
                self.console.rule("\n📊 Results")

                # Summary table
                summary_data = [
                    ["Processed", str(results['processed'])],
                    ["Moved", str(results['moved'])],
                    ["Skipped", str(results['skipped'])],
                    ["Errors", str(len(results['errors']))]
                ]
                self.console.table(summary_data, ["Metric", "Count"])

                # Category breakdown
                if results['moved'] > 0:
                    self.console.print("\n📂 Files by category:")
                    for category, count in results['by_category'].items():
                        if count > 0:
                            self.console.print(f"  {category}: {count}", 'green')

                # Show errors if any
                if results['errors']:
                    self.console.print("\n❌ Errors encountered:", 'red')
                    for error in results['errors'][:10]:  # Show first 10 errors
                        self.console.print(f"  • {error}", 'red')
                    if len(results['errors']) > 10:
                        self.console.print(f"  ... and {len(results['errors']) - 10} more errors", 'red')

                # Get operation summary
                summary = await organizer.get_operation_summary()
                if not dry_run:
                    self.console.print(f"\n💾 Directories created: {summary.get('directories_created', 0)}")

                # Show cache statistics
                if 'cache' in summary:
                    cache_stats = summary['cache']
                    self.console.print("\n📋 Cache Statistics:")
                    self.console.print(f"  Cache hits: {cache_stats.get('cache_hits', 0)}")
                    self.console.print(f"  Cache misses: {cache_stats.get('cache_misses', 0)}")
                    hit_rate = cache_stats.get('hit_rate', 0) * 100
                    self.console.print(f"  Hit rate: {hit_rate:.1f}%")
                    self.console.print(f"  Valid entries: {cache_stats.get('valid_entries', 0)}")
                    self.console.print(f"  Expired entries: {cache_stats.get('expired_entries', 0)}")
                    self.console.print(f"  Cache size: {cache_stats.get('size_mb', 0):.2f} MB")

                # Show parallel extraction statistics
                if 'extraction_stats' in results:
                    extraction_stats = results['extraction_stats']
                    self.console.print("\n⚡ Parallel Extraction Statistics:")
                    self.console.print(f"  Workers used: {extraction_stats.get('worker_count', 0)}")
                    self.console.print(f"  Files processed: {extraction_stats.get('files_processed', 0)}")
                    self.console.print(f"  Files succeeded: {extraction_stats.get('files_succeeded', 0)}")
                    self.console.print(f"  Files failed: {extraction_stats.get('files_failed', 0)}")
                    self.console.print(f"  Processing time: {extraction_stats.get('processing_time', 0):.2f}s")
                    self.console.print(f"  Throughput: {extraction_stats.get('throughput_mbps', 0):.2f} MB/s")
                    self.console.print(f"  Avg time per file: {extraction_stats.get('avg_time_per_file', 0):.3f}s")
                    self.console.print(f"  Peak memory: {extraction_stats.get('memory_peak_mb', 0):.1f} MB")
                elif 'extraction' in summary:
                    # Show stats from operation summary
                    extraction_stats = summary['extraction']
                    self.console.print("\n⚡ Parallel Extraction Statistics:")
                    self.console.print(f"  Workers: {extraction_stats.get('workers', 0)}")
                    self.console.print(f"  Memory peak: {extraction_stats.get('memory_peak_mb', 0):.1f} MB")
                    self.console.print(f"  Worker efficiency: {extraction_stats.get('worker_efficiency', 0):.2f}")

                # Show cache health report if requested
                if use_cache and cache_health and enable_smart_cache:
                    from .core.smart_cached_metadata import get_smart_cached_metadata_handler
                    smart_handler = get_smart_cached_metadata_handler()
                    health = smart_handler.get_cache_health()
                    stats = smart_handler.get_cache_stats()

                    self.console.print("\n🏥 Cache Health Report:")
                    self.console.print(f"  Overall health: {health['overall_health'].upper()}",
                                     'green' if health['overall_health'] == 'good' else
                                     'yellow' if health['overall_health'] == 'fair' else 'red')

                    self.console.print(f"  Cache entries: {stats['total_entries']} total, {stats['valid_entries']} valid")
                    self.console.print(f"  Cache size: {stats['size_mb']:.1f} MB")
                    self.console.print(f"  Average stability: {stats['avg_stability_score']:.2f}")
                    self.console.print(f"  Frequently accessed: {stats['frequently_accessed_files']} files")

                    if health['recommendations']:
                        self.console.print("\n💡 Recommendations:")
                        for rec in health['recommendations']:
                            self.console.print(f"  • {rec}", 'cyan')

                    if health['warnings']:
                        self.console.print("\n⚠️  Warnings:")
                        for warn in health['warnings']:
                            self.console.print(f"  • {warn}", 'yellow')

                return 0

        except MusicOrganizerError as e:
            self.console.print(f"Error: {e}", 'red')
            return 1
        except KeyboardInterrupt:
            self.console.print("\nOperation cancelled by user", 'yellow')
            return 130
        except Exception as e:
            self.console.print(f"Unexpected error: {e}", 'red')
            if '--debug' in sys.argv:
                import traceback
                traceback.print_exc()
            return 1

    async def scan(self,
                  directory: Path,
                  config_path: Optional[Path] = None,
                  detailed: bool = False) -> int:
        """Scan and analyze music library asynchronously."""
        try:
            if not directory.exists():
                self.console.print(f"Error: Directory does not exist: {directory}", 'red')
                return 1

            self.console.print(f"🔍 Scanning: {directory}")

            # Initialize components
            metadata_handler = MetadataHandler()
            classifier = ContentClassifier()

            # Scan files
            file_count = 0
            total_size = 0
            formats = {}
            categories = {
                'Albums': 0,
                'Live': 0,
                'Collaborations': 0,
                'Compilations': 0,
                'Rarities': 0,
                'Unknown': 0
            }
            errors = []

            progress = SimpleProgress(0, "Scanning files")

            async with AsyncMusicOrganizer(Config(
                source_directory=directory,
                target_directory=directory.parent / "organized"
            )) as organizer:
                async for file_path in organizer.scan_directory(directory):
                    try:
                        # Get file size
                        stat = file_path.stat()
                        file_size = stat.st_size
                        total_size += file_size
                        file_count += 1

                        # Track format
                        ext = file_path.suffix.lower()
                        formats[ext] = formats.get(ext, 0) + 1

                        if detailed:
                            # Extract metadata and classify
                            audio_file = await asyncio.get_event_loop().run_in_executor(
                                None, metadata_handler.extract_metadata, file_path
                            )

                            content_type, _ = await asyncio.get_event_loop().run_in_executor(
                                None, classifier.classify, audio_file
                            )

                            # Update category count
                            category_name = organizer._get_category_name(content_type)
                            categories[category_name] += 1

                        progress.total = file_count + 1
                        progress._draw()

                    except Exception as e:
                        errors.append(f"{file_path.name}: {e}")

                progress.current = progress.total
                progress._draw()
                print()

            # Display results
            self.console.rule("📊 Scan Results")

            # Basic stats
            stats_data = [
                ["Total Files", f"{file_count:,}"],
                ["Total Size", self._format_size(total_size)],
                ["Formats", str(len(formats))],
                ["Errors", str(len(errors))]
            ]
            self.console.table(stats_data, ["Metric", "Value"])

            # File formats
            if formats:
                self.console.print("\n📄 File Formats:")
                for ext, count in sorted(formats.items(), key=lambda x: x[1], reverse=True):
                    percent = (count * 100) // file_count
                    self.console.print(f"  {ext}: {count} files ({percent}%)")

            # Categories (if detailed)
            if detailed and categories:
                self.console.print("\n📂 Content Categories:")
                for category, count in categories.items():
                    if count > 0:
                        percent = (count * 100) // sum(categories.values())
                        self.console.print(f"  {category}: {count} files ({percent}%)")

            # Errors
            if errors:
                self.console.print(f"\n❌ {len(errors)} errors encountered:", 'red')
                for error in errors[:5]:
                    self.console.print(f"  • {error}", 'red')
                if len(errors) > 5:
                    self.console.print(f"  ... and {len(errors) - 5} more", 'red')

            return 0

        except Exception as e:
            self.console.print(f"Error: {e}", 'red')
            return 1

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    async def handle_cache_command(self, command: str, confirm: bool = False) -> int:
        """Handle cache management commands."""
        from .core.cached_metadata import get_cached_metadata_handler

        cache_handler = get_cached_metadata_handler()

        if command == 'stats':
            stats = cache_handler.get_cache_stats()
            self.console.rule("📋 Cache Statistics")

            stats_data = [
                ["Total entries", str(stats.get('total_entries', 0))],
                ["Valid entries", str(stats.get('valid_entries', 0))],
                ["Expired entries", str(stats.get('expired_entries', 0))],
                ["Cache size", f"{stats.get('size_mb', 0):.2f} MB"],
                ["Cache hits", str(stats.get('cache_hits', 0))],
                ["Cache misses", str(stats.get('cache_misses', 0))],
            ]

            hit_rate = stats.get('hit_rate', 0) * 100
            stats_data.append(["Hit rate", f"{hit_rate:.1f}%"])

            self.console.table(stats_data, ["Metric", "Value"])

        elif command == 'cleanup':
            self.console.print("🧹 Cleaning up expired cache entries...")
            removed = cache_handler.cleanup_expired()
            if removed > 0:
                self.console.print(f"✅ Removed {removed} expired entries", 'green')
            else:
                self.console.print("✅ No expired entries found", 'green')

        elif command == 'clear':
            if not confirm:
                self.console.print("⚠️  This will clear ALL cache entries", 'yellow')
                self.console.print("Use --confirm to proceed", 'yellow')
                return 1

            self.console.print("🗑️  Clearing all cache entries...")
            cache_handler.clear_cache()
            self.console.print("✅ Cache cleared successfully", 'green')

        else:
            self.console.print(f"Unknown cache command: {command}", 'red')
            return 1

        return 0

    async def _handle_magic_mode(
        self,
        source: Path,
        target: Path,
        config: Config,
        dry_run: bool,
        magic_analyze: bool,
        magic_auto: bool,
        magic_sample: Optional[int],
        magic_preview: bool,
        magic_save_config: Optional[Path],
        magic_threshold: float,
        backup: bool,
        max_workers: int,
        use_processes: bool,
        enable_parallel_extraction: bool,
        memory_threshold: float,
        use_cache: bool,
        cache_ttl: Optional[int],
        incremental: bool,
        force_full_scan: bool,
        bulk_mode: bool,
        chunk_size: int,
        conflict_strategy: str,
        verify_copies: bool,
        batch_dirs: bool,
        preview_bulk: bool,
        bulk_memory_threshold: int,
        smart_cache: Optional[bool],
        cache_warming: Optional[bool],
        cache_optimize: Optional[bool],
        warm_cache_dir: Optional[Path],
        cache_health: bool
    ) -> int:
        """Handle Magic Mode organization."""
        try:
            from .core.magic_organizer import MagicMusicOrganizer

            # Initialize Magic Music Organizer
            magic_organizer = MagicMusicOrganizer(
                config=config,
                enable_smart_cache=smart_cache if smart_cache is not None else True,
                enable_bulk_operations=bulk_mode,
                magic_mode_confidence_threshold=magic_threshold
            )
            await magic_organizer.initialize()

            self.console.print("\n🪄 Magic Mode Activated")
            self.console.print(f"Source: {source}")
            self.console.print(f"Target: {target}")
            self.console.print(f"Mode: {'ANALYZE ONLY' if magic_analyze else 'DRY RUN' if dry_run else 'LIVE'}")

            # Analyze library
            self.console.print("\n🔍 Analyzing music library for Magic Mode...")
            suggestion = await magic_organizer.analyze_library_for_magic_mode(
                source,
                sample_size=magic_sample,
                force_analyze=True
            )

            # Show Magic Mode analysis
            await magic_organizer._show_magic_analysis(suggestion)

            # If only analyzing, exit here
            if magic_analyze:
                if magic_save_config:
                    config_path = magic_save_config.with_suffix('.json')
                    await magic_organizer.save_magic_config(config_path)
                    self.console.print(f"\n💾 Magic Mode configuration saved to: {config_path}", 'green')
                return 0

            # Preview mode
            if magic_preview:
                self.console.print("\n🔮 Generating Magic Mode preview...")
                preview = await magic_organizer.get_magic_mode_preview(
                    source,
                    target,
                    sample_size=20
                )

                self.console.print(f"\n📋 Preview - Sample Operations:")
                for i, op in enumerate(preview["sample_operations"][:10], 1):
                    self.console.print(f"  {i:2d}. {op['source']} -> {op['target']}")
                    self.console.print(f"      Size: {op['size_mb']:.1f} MB")

                self.console.print(f"\nTotal estimated operations: {preview['total_estimated_operations']:,}")

                if not dry_run:
                    confirm = input("\nContinue with Magic Mode organization? (y/N): ")
                    if confirm.lower() != 'y':
                        self.console.print("Operation cancelled by user.", 'yellow')
                        return 0

            # Execute Magic Mode organization
            if not dry_run or magic_preview:
                self.console.print("\n🚀 Starting Magic Mode organization...")

                result = await magic_organizer.organize_with_magic_mode(
                    source_dir=source,
                    target_dir=target,
                    dry_run=dry_run,
                    auto_accept=magic_auto,
                    sample_size=magic_sample,
                    chunk_size=chunk_size,
                    conflict_strategy=conflict_strategy,
                    verify_copies=verify_copies,
                    batch_directories=batch_dirs,
                    bulk_memory_threshold=bulk_memory_threshold
                )

                # Show results
                self._show_magic_results(result)

                # Save configuration if requested
                if magic_save_config:
                    config_path = magic_save_config.with_suffix('.json')
                    await magic_organizer.save_magic_config(config_path)
                    self.console.print(f"\n💾 Magic Mode configuration saved to: {config_path}", 'green')

                # Determine success
                stats = result.get("stats", {})
                success_rate = stats.get("success_rate", 0)
                if success_rate >= 0.9:  # 90% success rate
                    self.console.print("\n✅ Magic Mode organization completed successfully!", 'green')
                    return 0
                else:
                    self.console.print(f"\n⚠️ Magic Mode organization completed with {success_rate:.1%} success rate", 'yellow')
                    return 1 if success_rate < 0.5 else 0

            return 0

        except Exception as e:
            self.console.print(f"Error in Magic Mode: {e}", 'red')
            if magic_threshold < 0.8:  # Show error details for lower thresholds
                import traceback
                self.console.print(traceback.format_exc(), 'red')
            return 1

    def _show_magic_results(self, result: Dict[str, Any]):
        """Display Magic Mode organization results."""
        self.console.print("\n" + "="*60)
        self.console.print("🪄 MAGIC MODE RESULTS")
        self.console.print("="*60)

        if "stats" in result:
            stats = result["stats"]
            self.console.print(f"\n📊 ORGANIZATION STATISTICS:")
            self.console.print(f"  Total files processed: {stats.get('total_files', 0):,}")
            self.console.print(f"  Successfully organized: {stats.get('processed', 0):,}")
            self.console.print(f"  Errors: {stats.get('errors', 0):,}")
            self.console.print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
            self.console.print(f"  Strategy used: {stats.get('strategy_used', 'Unknown')}")
            self.console.print(f"  Confidence: {stats.get('confidence', 0):.1%}")

        if "organized_files" in result and len(result["organized_files"]) > 0:
            self.console.print(f"\n📁 SAMPLE ORGANIZED FILES:")
            for i, file_info in enumerate(result["organized_files"][:5], 1):
                source = Path(file_info["source"]).name
                target = Path(file_info["target"]).relative_to(Path(file_info["target"]).anchor)
                self.console.print(f"  {i}. {source} -> {target}")

            if len(result["organized_files"]) > 5:
                self.console.print(f"  ... and {len(result['organized_files']) - 5} more files")

        if "errors" in result and len(result["errors"]) > 0:
            self.console.print(f"\n❌ ERRORS ENCOUNTERED:")
            for i, error in enumerate(result["errors"][:3], 1):
                filename = Path(error["file"]).name
                self.console.print(f"  {i}. {filename}: {error['error']}")

            if len(result["errors"]) > 3:
                self.console.print(f"  ... and {len(result['errors']) - 3} more errors")

        self.console.print("="*60)

    async def _handle_organization_preview(self,
                                           source: Path,
                                           target: Path,
                                           config: Config,
                                           preview_detailed: bool,
                                           preview_interactive: bool,
                                           export_preview: Optional[Path],
                                           max_workers: int,
                                           use_processes: bool,
                                           enable_parallel_extraction: bool,
                                           memory_threshold: float,
                                           use_cache: bool,
                                           cache_ttl: Optional[int],
                                           incremental: bool,
                                           force_full_scan: bool,
                                           bulk_mode: bool,
                                           chunk_size: int,
                                           conflict_strategy: ConflictStrategy,
                                           smart_cache: Optional[bool],
                                           cache_warming: Optional[bool],
                                           cache_optimize: Optional[bool],
                                           warm_cache_dir: Optional[Path]) -> int:
        """Handle organization preview functionality."""
        try:
            self.console.print("📋 ORGANIZATION PREVIEW MODE", style="bold")
            self.console.print(f"Source: {source}")
            self.console.print(f"Target: {target}")
            self.console.print("-" * 40)

            # Initialize metadata handler and classifier
            metadata_handler = MetadataHandler()
            classifier = ContentClassifier()

            # Scan source directory
            self.console.print("\n🔍 Scanning source directory...")
            audio_files = []

            # Use simple directory scan since we just need to collect files
            import os
            from pathlib import Path

            for root, dirs, files in os.walk(source):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        # Skip if not an audio file
                        if file_path.suffix.lower() not in ['.mp3', '.flac', '.wav', '.m4a', '.aac', '.ogg']:
                            continue

                        # Extract metadata
                        audio_file = await asyncio.get_event_loop().run_in_executor(
                            None, metadata_handler.extract_metadata, file_path
                        )

                        # Classify content type
                        content_type, _ = await asyncio.get_event_loop().run_in_executor(
                            None, classifier.classify, audio_file
                        )

                        audio_files.append(audio_file)

                    except Exception as e:
                        self.console.print(f"Warning: Could not process {file_path}: {e}", 'yellow')

                if not audio_files:
                    self.console.print("No audio files found in source directory.")
                    return 0

                self.console.print(f"Found {len(audio_files)} audio files")

                # Generate target paths
                self.console.print("\n🎯 Generating organization plan...")
                target_mapping = {}
                conflicts = {}

                def generate_target_path(audio_file, target_dir: Path) -> Path:
                    """Simple target path generator for preview."""
                    # Get artist and album info
                    artist = audio_file.primary_artist or "Unknown Artist"
                    if not artist and audio_file.artists:
                        artist = audio_file.artists[0]

                    album = audio_file.album or "Unknown Album"
                    year = audio_file.year or ""

                    # Get content type directory
                    content_dir = "Albums"
                    if audio_file.content_type:
                        content_dir = {
                            "live": "Live",
                            "compilation": "Compilations",
                            "collaboration": "Collaborations"
                        }.get(audio_file.content_type.value, "Albums")

                    # Format filename
                    track_num = f"{audio_file.track_number:02d} " if audio_file.track_number else ""
                    title = audio_file.title or audio_file.path.stem
                    filename = f"{track_num}{title}{audio_file.path.suffix}"

                    # Build path
                    if year:
                        album_dir = f"{album} ({year})"
                    else:
                        album_dir = album

                    return target_dir / content_dir / artist / album_dir / filename

                for audio_file in audio_files:
                    try:
                        # Generate target path
                        target_path = generate_target_path(audio_file, target)
                        target_mapping[audio_file.path] = target_path
                    except Exception as e:
                        self.console.print(f"Warning: Could not generate target for {audio_file.path}: {e}", 'yellow')

                # Create organization preview
                preview = OrganizationPreview(config)
                await preview.collect_operations(audio_files, target_mapping, conflicts)

                # Display preview
                if preview_interactive:
                    self.console.print("\n🎮 Starting interactive preview...")
                    interactive_preview = InteractivePreview(preview)
                    proceed = await interactive_preview.run_interactive_preview()

                    if proceed:
                        self.console.print("\n✅ Proceeding with organization...")
                        # Here you could call the actual organization logic
                        # For now, just show what would happen
                        preview.display_preview(detailed=True)
                    else:
                        self.console.print("\n❌ Organization cancelled.")
                        return 0
                else:
                    preview.display_preview(detailed=preview_detailed)

                # Export preview if requested
                if export_preview:
                    preview.export_preview(export_preview)

                return 0

        except Exception as e:
            self.console.print(f"Error during preview: {e}", 'red')
            if isinstance(e, MusicOrganizerError):
                self.console.print(f"Details: {e}", 'red')
            return 1


def create_async_cli():
    """Create the async CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize your music library with smart metadata-based categorization (Async Mode)",
        epilog="Example: music-organize-async /path/to/music /path/to/organized --dry-run"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Organize command
    org_parser = subparsers.add_parser('organize', help='Organize music files')
    org_parser.add_argument('source', type=Path, help='Source directory containing music files')
    org_parser.add_argument('target', type=Path, help='Target directory for organized files')
    org_parser.add_argument('--config', type=Path, help='Configuration file path')
    org_parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    org_parser.add_argument('--interactive', action='store_true', help='Prompt for ambiguous categorizations')
    org_parser.add_argument('--no-backup', action='store_true', help='Disable backup creation')
    org_parser.add_argument('--workers', type=int, default=4, help='Number of worker threads (default: 4)')
    org_parser.add_argument('--processes', action='store_true', help='Use process pool instead of thread pool for parallel processing')
    org_parser.add_argument('--no-parallel', action='store_true', help='Disable parallel metadata extraction')
    org_parser.add_argument('--memory-threshold', type=float, default=80.0, help='Memory usage threshold percentage for dynamic worker adjustment (default: 80)')
    org_parser.add_argument('--cache', action='store_true', default=True, help='Enable metadata caching (default: enabled)')
    org_parser.add_argument('--no-cache', action='store_true', help='Disable metadata caching')
    org_parser.add_argument('--cache-ttl', type=int, help='Cache TTL in days (default: 30)')

    # Smart caching options
    smart_cache_group = org_parser.add_argument_group('Smart Caching', 'Intelligent caching with adaptive TTL and optimization')
    smart_cache_group.add_argument('--smart-cache', action='store_true', help='Enable smart caching with adaptive TTL (default: enabled when possible)')
    smart_cache_group.add_argument('--no-smart-cache', action='store_true', help='Disable smart caching, use basic cache instead')
    smart_cache_group.add_argument('--cache-warming', action='store_true', help='Enable automatic cache warming')
    smart_cache_group.add_argument('--no-cache-warming', action='store_true', help='Disable cache warming')
    smart_cache_group.add_argument('--cache-optimize', action='store_true', help='Enable automatic cache optimization (default: enabled)')
    smart_cache_group.add_argument('--no-cache-optimize', action='store_true', help='Disable automatic cache optimization')
    smart_cache_group.add_argument('--warm-cache-dir', type=Path, help='Directory to warm cache before organization')
    smart_cache_group.add_argument('--cache-health', action='store_true', help='Show cache health report after organization')
    org_parser.add_argument('--incremental', action='store_true', help='Only process new or modified files (incremental scan)')
    org_parser.add_argument('--force-full-scan', action='store_true', help='Force full scan instead of incremental')

    # Bulk operation arguments
    bulk_group = org_parser.add_argument_group('Bulk Operations', 'Advanced bulk file operation settings')
    bulk_group.add_argument('--bulk', action='store_true', help='Use bulk operations for maximum performance')
    bulk_group.add_argument('--chunk-size', type=int, default=200, help='Batch size for bulk operations (default: 200)')
    bulk_group.add_argument('--conflict-strategy', choices=['skip', 'rename', 'replace', 'keep_both'],
                           default='rename', help='Strategy for handling file conflicts (default: rename)')
    bulk_group.add_argument('--verify-copies', action='store_true', help='Verify file copies for bulk operations (default: disabled for moves)')
    bulk_group.add_argument('--no-batch-dirs', action='store_true', help='Disable batch directory creation')
    bulk_group.add_argument('--preview-bulk', action='store_true', help='Preview bulk operation before execution')
    bulk_group.add_argument('--bulk-memory-threshold', type=int, default=512, help='Memory threshold for bulk operations in MB (default: 512)')

    # Magic Mode arguments
    magic_group = org_parser.add_argument_group('Magic Mode', 'Zero-configuration intelligent organization')
    magic_group.add_argument('--magic', action='store_true', help='Enable Magic Mode for zero-configuration organization')
    magic_group.add_argument('--magic-analyze', action='store_true', help='Analyze library and show Magic Mode suggestions')
    magic_group.add_argument('--magic-auto', action='store_true', help='Auto-accept Magic Mode suggestions without confirmation')
    magic_group.add_argument('--magic-sample', type=int, default=None, help='Sample size for Magic Mode analysis (default: all files)')
    magic_group.add_argument('--magic-preview', action='store_true', help='Preview Magic Mode organization before execution')
    magic_group.add_argument('--magic-save-config', type=Path, help='Save Magic Mode configuration to file')
    magic_group.add_argument('--magic-threshold', type=float, default=0.6, help='Confidence threshold for Magic Mode auto-accept (default: 0.6)')

    # Organization preview arguments
    preview_group = org_parser.add_argument_group('Organization Preview', 'Preview and visualize organization before execution')
    preview_group.add_argument('--preview', action='store_true', help='Show organization preview with summary')
    preview_group.add_argument('--preview-detailed', action='store_true', help='Show detailed preview with all operations')
    preview_group.add_argument('--preview-interactive', action='store_true', help='Interactive preview with filtering and review')
    preview_group.add_argument('--export-preview', type=Path, help='Export preview data to JSON file')

    org_parser.add_argument('--debug', action='store_true', help='Enable debug output')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Analyze music library')
    scan_parser.add_argument('directory', type=Path, help='Directory to scan')
    scan_parser.add_argument('--config', type=Path, help='Configuration file path')
    scan_parser.add_argument('--detailed', action='store_true', help='Include detailed analysis')
    scan_parser.add_argument('--debug', action='store_true', help='Enable debug output')

    # Cache command
    cache_parser = subparsers.add_parser('cache', help='Manage metadata cache')
    cache_subparsers = cache_parser.add_subparsers(dest='cache_command', help='Cache operations')

    # Cache stats
    cache_subparsers.add_parser('stats', help='Show cache statistics')

    # Cache cleanup
    cleanup_parser = cache_subparsers.add_parser('cleanup', help='Clean up expired cache entries')

    # Cache clear
    clear_parser = cache_subparsers.add_parser('clear', help='Clear all cache entries')
    clear_parser.add_argument('--confirm', action='store_true', help='Confirm clearing all cache')

    # Add version
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Create CLI instance
    cli = AsyncMusicCLI()

    # Run appropriate command
    if args.command == 'organize':
        return asyncio.run(cli.organize(
            source=args.source,
            target=args.target,
            config_path=args.config,
            dry_run=args.dry_run,
            interactive=args.interactive,
            backup=not args.no_backup,
            max_workers=args.workers,
            use_processes=args.processes,
            enable_parallel_extraction=not args.no_parallel,
            memory_threshold=args.memory_threshold,
            use_cache=not args.no_cache,
            cache_ttl=args.cache_ttl,
            incremental=args.incremental,
            force_full_scan=args.force_full_scan,
            bulk_mode=args.bulk,
            chunk_size=args.chunk_size,
            conflict_strategy=args.conflict_strategy,
            verify_copies=args.verify_copies,
            batch_dirs=not args.no_batch_dirs,
            preview_bulk=args.preview_bulk,
            bulk_memory_threshold=args.bulk_memory_threshold,
            smart_cache=getattr(args, 'smart_cache', None),
            cache_warming=getattr(args, 'cache_warming', None) if hasattr(args, 'cache_warming') else None,
            cache_optimize=getattr(args, 'cache_optimize', None) if hasattr(args, 'cache_optimize') else None,
            warm_cache_dir=getattr(args, 'warm_cache_dir', None) if hasattr(args, 'warm_cache_dir') else None,
            cache_health=getattr(args, 'cache_health', False) if hasattr(args, 'cache_health') else False,
            magic_mode=getattr(args, 'magic', False) if hasattr(args, 'magic') else False,
            magic_analyze=getattr(args, 'magic_analyze', False) if hasattr(args, 'magic_analyze') else False,
            magic_auto=getattr(args, 'magic_auto', False) if hasattr(args, 'magic_auto') else False,
            magic_sample=getattr(args, 'magic_sample', None) if hasattr(args, 'magic_sample') else None,
            magic_preview=getattr(args, 'magic_preview', False) if hasattr(args, 'magic_preview') else False,
            magic_save_config=getattr(args, 'magic_save_config', None) if hasattr(args, 'magic_save_config') else None,
            magic_threshold=getattr(args, 'magic_threshold', 0.6) if hasattr(args, 'magic_threshold') else 0.6,
            preview=getattr(args, 'preview', False) if hasattr(args, 'preview') else False,
            preview_detailed=getattr(args, 'preview_detailed', False) if hasattr(args, 'preview_detailed') else False,
            preview_interactive=getattr(args, 'preview_interactive', False) if hasattr(args, 'preview_interactive') else False,
            export_preview=getattr(args, 'export_preview', None) if hasattr(args, 'export_preview') else None
        ))
    elif args.command == 'scan':
        return asyncio.run(cli.scan(
            directory=args.directory,
            config_path=args.config,
            detailed=args.detailed
        ))
    elif args.command == 'cache':
        if hasattr(args, 'cache_command') and args.cache_command:
            return asyncio.run(cli.handle_cache_command(
                command=args.cache_command,
                confirm=getattr(args, 'confirm', False)
            ))
        else:
            parser.error("Cache command required")

    return 0


def main():
    """Entry point for async CLI."""
    import sys
    sys.exit(create_async_cli())