"""Parallel metadata extraction with worker pools.

Provides high-performance metadata extraction using concurrent processing
with configurable worker pools and dynamic resource management.
"""

from __future__ import annotations

import asyncio
import os
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime
import time

from ..models.audio_file import AudioFile
from ..exceptions import MetadataError
from .metadata import MetadataHandler
from ..progress_tracker import IntelligentProgressTracker, ProgressStage

logger = logging.getLogger(__name__)


@dataclass
class ExtractionStats:
    """Statistics for metadata extraction operations."""
    files_processed: int = 0
    files_succeeded: int = 0
    files_failed: int = 0
    total_bytes: int = 0
    processing_time: float = 0.0
    avg_time_per_file: float = 0.0
    throughput_mbps: float = 0.0
    worker_count: int = 0
    memory_peak_mb: float = 0.0

    def update_derived_metrics(self):
        """Update derived metrics like averages and throughput."""
        if self.files_processed > 0 and self.processing_time > 0:
            self.avg_time_per_file = self.processing_time / self.files_processed
            # Calculate throughput in MB/s
            if self.total_bytes > 0:
                self.throughput_mbps = (self.total_bytes / (1024 * 1024)) / self.processing_time


@dataclass
class ExtractionJob:
    """Represents a single metadata extraction job."""
    file_path: Path
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of a metadata extraction job."""
    job: ExtractionJob
    audio_file: Optional[AudioFile]
    error: Optional[Exception]
    processing_time: float
    worker_id: int


class MemoryPressureMonitor:
    """Monitors system memory pressure for dynamic worker adjustment."""

    def __init__(self, memory_threshold: float = 80.0):
        """Initialize memory pressure monitor.

        Args:
            memory_threshold: Memory usage percentage threshold (default: 80%)
        """
        self.memory_threshold = memory_threshold
        self.initial_memory = psutil.virtual_memory().available

    def get_memory_pressure(self) -> float:
        """Get current memory pressure as percentage of threshold.

        Returns:
            Memory pressure percentage (0-100+)
        """
        mem = psutil.virtual_memory()
        return (mem.percent / self.memory_threshold) * 100

    def is_memory_pressure_high(self) -> bool:
        """Check if memory pressure is above threshold.

        Returns:
            True if memory pressure is high
        """
        return psutil.virtual_memory().percent > self.memory_threshold

    def get_available_memory_gb(self) -> float:
        """Get available memory in GB.

        Returns:
            Available memory in GB
        """
        return psutil.virtual_memory().available / (1024**3)

    def get_recommended_worker_count(self, base_workers: int) -> int:
        """Get recommended worker count based on memory pressure.

        Args:
            base_workers: Base number of workers

        Returns:
            Adjusted worker count
        """
        if self.is_memory_pressure_high():
            # Reduce workers under memory pressure
            pressure = self.get_memory_pressure()
            if pressure > 150:
                return 1  # Minimal workers under extreme pressure
            elif pressure > 120:
                return max(1, base_workers // 4)
            elif pressure > 100:
                return max(1, base_workers // 2)
            else:
                return max(1, int(base_workers * 0.75))
        return base_workers


class ParallelMetadataExtractor:
    """High-performance parallel metadata extractor with configurable worker pools."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        memory_threshold: float = 80.0,
        batch_size: int = 100,
        enable_memory_monitoring: bool = True,
        chunk_size: int = 32 * 1024,  # 32KB chunks for file reading
        progress_tracker: Optional[IntelligentProgressTracker] = None,
    ):
        """Initialize parallel metadata extractor.

        Args:
            max_workers: Maximum number of worker threads/processes (default: CPU count)
            use_processes: Use process pool instead of thread pool (default: False)
            memory_threshold: Memory usage percentage threshold (default: 80%)
            batch_size: Number of files to process in each batch (default: 100)
            enable_memory_monitoring: Enable memory pressure monitoring (default: True)
            chunk_size: Chunk size for file I/O operations (default: 32KB)
            progress_tracker: Optional external progress tracker to use
        """
        # Determine optimal worker count
        cpu_count = os.cpu_count() or 4
        self.max_workers = max_workers or min(cpu_count, 8)  # Cap at 8 to avoid overwhelming
        self.use_processes = use_processes
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        # Memory monitoring
        self.enable_memory_monitoring = enable_memory_monitoring
        self.memory_monitor = MemoryPressureMonitor(memory_threshold) if enable_memory_monitoring else None

        # Progress tracking
        self.progress_tracker = progress_tracker or IntelligentProgressTracker()

        # Statistics tracking
        self.stats = ExtractionStats()
        self.active_workers = 0
        self.memory_peak = 0.0

        # Task queue management
        self.task_queue: asyncio.Queue[ExtractionJob] = asyncio.Queue()
        self.result_queue: asyncio.Queue[ExtractionResult] = asyncio.Queue()

        # Worker pool
        self.executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None

        logger.info(f"Initialized ParallelMetadataExtractor: {self.max_workers} workers, "
                   f"{'processes' if use_processes else 'threads'}")

    def _extract_metadata_sync(self, job: ExtractionJob) -> ExtractionResult:
        """Synchronous metadata extraction for worker threads/processes.

        Args:
            job: Extraction job to process

        Returns:
            Extraction result with metadata or error
        """
        worker_id = os.getpid() if self.use_processes else id(asyncio.current_task())
        start_time = time.time()

        try:
            # Check file size for memory management
            file_size = job.file_path.stat().st_size

            # Extract metadata using the existing handler
            audio_file = MetadataHandler.extract_metadata(job.file_path)

            processing_time = time.time() - start_time

            result = ExtractionResult(
                job=job,
                audio_file=audio_file,
                error=None,
                processing_time=processing_time,
                worker_id=worker_id
            )

            # Update statistics
            result.job.metadata['file_size'] = file_size
            result.job.metadata['worker_id'] = worker_id

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to extract metadata from {job.file_path}: {e}")

            return ExtractionResult(
                job=job,
                audio_file=None,
                error=e,
                processing_time=processing_time,
                worker_id=worker_id
            )

    async def extract_batch(
        self,
        file_paths: List[Path],
        progress_callback: Optional[callable] = None
    ) -> List[ExtractionResult]:
        """Extract metadata from a batch of files in parallel.

        Args:
            file_paths: List of file paths to process
            progress_callback: Optional callback for progress updates

        Returns:
            List of extraction results
        """
        if not file_paths:
            return []

        # Update worker count based on memory pressure
        current_workers = self.max_workers
        if self.memory_monitor:
            current_workers = self.memory_monitor.get_recommended_worker_count(self.max_workers)

        logger.info(f"Extracting metadata from {len(file_paths)} files using {current_workers} workers")

        # Start parallel extraction stage in progress tracker
        self.progress_tracker.start_stage(ProgressStage.PARALLEL_EXTRACTION, len(file_paths))
        self.progress_tracker.update_parallel_metrics(workers=current_workers)

        start_time = time.time()
        peak_memory = 0.0

        # Create extraction jobs
        jobs = [ExtractionJob(file_path=path) for path in file_paths]

        # Use appropriate executor type
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        # Process in batches to manage memory
        all_results = []
        batch_count = 0

        for i in range(0, len(jobs), self.batch_size):
            batch = jobs[i:i + self.batch_size]
            batch_count += 1

            logger.debug(f"Processing batch {batch_count} with {len(batch)} files")

            # Check memory pressure before processing batch
            if self.memory_monitor and self.memory_monitor.is_memory_pressure_high():
                logger.warning(f"High memory pressure detected: {self.memory_monitor.get_memory_pressure():.1f}%")
                # Force garbage collection
                import gc
                gc.collect()

            # Process batch
            batch_results = await self._process_batch(batch, executor_class, current_workers)
            all_results.extend(batch_results)

            # Update peak memory usage and progress tracker
            if self.memory_monitor:
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                peak_memory = max(peak_memory, current_memory)
                cpu_percent = psutil.cpu_percent()

                # Update progress tracker with parallel metrics
                self.progress_tracker.update_parallel_metrics(
                    workers=current_workers,
                    memory_mb=current_memory,
                    cpu_percent=cpu_percent,
                    memory_peak_mb=peak_memory
                )

            # Update progress tracker with file progress
            processed_files = len([r for r in all_results if r.error is None])
            bytes_processed = sum(
                r.job.metadata.get('file_size', 0) for r in all_results if r.error is None
            )
            self.progress_tracker.set_completed(
                completed=processed_files,
                bytes_processed=bytes_processed,
                error=any(r.error is not None for r in batch_results)
            )

            # Call progress callback if provided
            if progress_callback:
                progress = min(100.0, (i + len(batch)) / len(jobs) * 100)
                await progress_callback(progress, len(all_results))

        # Update statistics
        processing_time = time.time() - start_time
        self._update_stats(all_results, processing_time, peak_memory)

        # Finish parallel extraction stage in progress tracker
        self.progress_tracker.finish_stage(ProgressStage.PARALLEL_EXTRACTION)

        logger.info(f"Batch extraction completed: {self.stats.files_succeeded}/{self.stats.files_processed} files "
                   f"in {processing_time:.2f}s ({self.stats.throughput_mbps:.2f} MB/s)")

        return all_results

    async def _process_batch(
        self,
        jobs: List[ExtractionJob],
        executor_class: Union[type[ThreadPoolExecutor], type[ProcessPoolExecutor]],
        worker_count: int
    ) -> List[ExtractionResult]:
        """Process a batch of jobs using worker pool.

        Args:
            jobs: List of extraction jobs
            executor_class: Type of executor to use
            worker_count: Number of workers to use

        Returns:
            List of extraction results
        """
        results = []

        with executor_class(max_workers=worker_count) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._extract_metadata_sync, job): job
                for job in jobs
            }

            # Process completed jobs as they finish
            for future in as_completed(future_to_job):
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, future.result
                    )
                    results.append(result)
                except Exception as e:
                    job = future_to_job[future]
                    logger.error(f"Worker pool error for {job.file_path}: {e}")
                    results.append(ExtractionResult(
                        job=job,
                        audio_file=None,
                        error=e,
                        processing_time=0.0,
                        worker_id=-1
                    ))

        return results

    async def extract_stream(
        self,
        file_paths: AsyncGenerator[Path, None],
        progress_callback: Optional[callable] = None
    ) -> AsyncGenerator[ExtractionResult, None]:
        """Extract metadata from a stream of files.

        Args:
            file_paths: Async generator of file paths
            progress_callback: Optional callback for progress updates

        Yields:
            Extraction results as they complete
        """
        batch = []
        processed_count = 0

        async for file_path in file_paths:
            batch.append(file_path)

            if len(batch) >= self.batch_size:
                # Process current batch
                batch_results = await self.extract_batch(batch, progress_callback)
                for result in batch_results:
                    processed_count += 1
                    yield result

                batch = []  # Clear batch

        # Process remaining files in last batch
        if batch:
            batch_results = await self.extract_batch(batch, progress_callback)
            for result in batch_results:
                processed_count += 1
                yield result

    def _update_stats(
        self,
        results: List[ExtractionResult],
        processing_time: float,
        peak_memory_mb: float
    ):
        """Update extraction statistics.

        Args:
            results: List of extraction results
            processing_time: Total processing time in seconds
            peak_memory_mb: Peak memory usage in MB
        """
        self.stats.files_processed = len(results)
        self.stats.files_succeeded = sum(1 for r in results if r.error is None)
        self.stats.files_failed = self.stats.files_processed - self.stats.files_succeeded
        self.stats.processing_time = processing_time
        self.stats.memory_peak_mb = peak_memory_mb
        self.stats.worker_count = self.max_workers

        # Calculate total bytes processed
        total_bytes = 0
        for result in results:
            if result.error is None and result.audio_file:
                file_size = result.job.metadata.get('file_size', 0)
                total_bytes += file_size

        self.stats.total_bytes = total_bytes
        self.stats.update_derived_metrics()

    def get_stats(self) -> ExtractionStats:
        """Get current extraction statistics.

        Returns:
            Copy of current statistics
        """
        return ExtractionStats(
            files_processed=self.stats.files_processed,
            files_succeeded=self.stats.files_succeeded,
            files_failed=self.stats.files_failed,
            total_bytes=self.stats.total_bytes,
            processing_time=self.stats.processing_time,
            avg_time_per_file=self.stats.avg_time_per_file,
            throughput_mbps=self.stats.throughput_mbps,
            worker_count=self.stats.worker_count,
            memory_peak_mb=self.stats.memory_peak_mb
        )

    def reset_stats(self):
        """Reset extraction statistics."""
        self.stats = ExtractionStats()
        self.memory_peak = 0.0

    async def cleanup(self):
        """Clean up resources and shutdown worker pools."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        # Clear queues
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("ParallelMetadataExtractor cleanup completed")