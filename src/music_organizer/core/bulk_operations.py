"""Enhanced bulk file operations with optimized parallel processing.

This module provides high-performance bulk file operations including:
- Batch directory creation to minimize filesystem overhead
- Intelligent conflict resolution with strategies (skip, rename, replace)
- Parallel file operations with configurable worker pools
- Progress tracking with detailed metrics
- Verification and rollback capabilities
"""

import os
import shutil
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, AsyncGenerator, Set
from datetime import datetime

from ..exceptions import FileOperationError
from ..models.audio_file import AudioFile, CoverArt


class ConflictStrategy(Enum):
    """Strategy for handling file conflicts during bulk operations."""
    SKIP = "skip"
    RENAME = "rename"
    REPLACE = "replace"
    KEEP_BOTH = "keep_both"


class OperationType(Enum):
    """Type of bulk operation."""
    MOVE = "move"
    COPY = "copy"
    MOVE_COPY = "move_copy"  # Move with verification copy


@dataclass
class BulkOperationConfig:
    """Configuration for bulk operations."""
    max_workers: int = 4
    chunk_size: int = 100  # Files per batch
    conflict_strategy: ConflictStrategy = ConflictStrategy.RENAME
    verify_copies: bool = True
    create_dirs_batch: bool = True
    preserve_timestamps: bool = True
    skip_identical: bool = True
    memory_threshold_mb: int = 512
    progress_report_interval: float = 1.0  # seconds


@dataclass
class FileOperation:
    """Represents a single file operation in a bulk batch."""
    source: Path
    target: Path
    operation_type: OperationType
    size: int = field(default=0, init=False)
    checksum: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        if self.source.exists():
            self.size = self.source.stat().st_size
            self.checksum = self._calculate_checksum() if self.size < 100 * 1024 * 1024 else None  # Only for <100MB files

    def _calculate_checksum(self) -> str:
        """Calculate MD5 checksum for verification."""
        hash_md5 = hashlib.md5()
        try:
            with open(self.source, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except (OSError, IOError):
            return None


@dataclass
class BulkOperationResult:
    """Result of a bulk file operation."""
    total_files: int
    successful: int
    skipped: int
    failed: int
    total_size_mb: float
    duration_seconds: float
    operations: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful / self.total_files) * 100

    @property
    def throughput_mb_per_sec(self) -> float:
        """Calculate throughput in MB/s."""
        if self.duration_seconds == 0:
            return 0.0
        return self.total_size_mb / self.duration_seconds


class BulkFileOperator:
    """High-performance bulk file operations manager."""

    def __init__(self, config: BulkOperationConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._operations: List[FileOperation] = []
        self._completed_operations: List[Dict] = []
        self._created_directories: Set[Path] = set()
        self._start_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def add_operation(self, source: Path, target: Path,
                          operation_type: OperationType = OperationType.MOVE) -> None:
        """Add a file operation to the batch."""
        async with self._lock:
            operation = FileOperation(source=source, target=target, operation_type=operation_type)
            self._operations.append(operation)

    async def add_operations_batch(self, operations: List[Tuple[Path, Path, OperationType]]) -> None:
        """Add multiple operations to the batch."""
        tasks = []
        for source, target, op_type in operations:
            task = self.add_operation(source, target, op_type)
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def execute_bulk_operation(self, progress_callback=None) -> BulkOperationResult:
        """Execute all queued operations with optimized batching."""
        if not self._operations:
            raise FileOperationError("No operations to execute")

        self._start_time = datetime.now()
        result = BulkOperationResult(
            total_files=len(self._operations),
            successful=0,
            skipped=0,
            failed=0,
            total_size_mb=0.0,
            duration_seconds=0.0
        )

        try:
            # Pre-operations: batch directory creation
            if self.config.create_dirs_batch:
                await self._create_directories_batch()

            # Group operations by target directory for better locality
            operations_by_dir = self._group_operations_by_directory()

            # Process each directory group
            for directory, operations in operations_by_dir.items():
                dir_result = await self._process_directory_batch(operations, progress_callback)
                result.successful += dir_result.successful
                result.skipped += dir_result.skipped
                result.failed += dir_result.failed
                result.total_size_mb += dir_result.total_size_mb
                result.operations.extend(dir_result.operations)
                result.errors.extend(dir_result.errors)

            # Calculate total duration
            result.duration_seconds = (datetime.now() - self._start_time).total_seconds()

            return result

        finally:
            # Cleanup
            self._operations.clear()
            self._created_directories.clear()

    async def _create_directories_batch(self) -> None:
        """Create all necessary directories in parallel."""
        directories = set()

        for op in self._operations:
            target_dir = op.target.parent
            if target_dir not in directories and not target_dir.exists():
                directories.add(target_dir)

        # Create directories in parallel
        async def create_dir(dir_path):
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self._created_directories.add(dir_path)
            except Exception as e:
                raise FileOperationError(f"Failed to create directory {dir_path}: {e}")

        tasks = [create_dir(dir_path) for dir_path in directories]
        await asyncio.gather(*tasks)

    def _group_operations_by_directory(self) -> Dict[Path, List[FileOperation]]:
        """Group operations by target directory for better filesystem locality."""
        groups = {}

        for op in self._operations:
            target_dir = op.target.parent
            if target_dir not in groups:
                groups[target_dir] = []
            groups[target_dir].append(op)

        return groups

    async def _process_directory_batch(self, operations: List[FileOperation],
                                     progress_callback=None) -> BulkOperationResult:
        """Process all operations in a single directory."""
        result = BulkOperationResult(
            total_files=len(operations),
            successful=0,
            skipped=0,
            failed=0,
            total_size_mb=0.0,
            duration_seconds=0.0
        )

        # Process in chunks to manage memory usage
        chunk_size = self.config.chunk_size
        for i in range(0, len(operations), chunk_size):
            chunk = operations[i:i + chunk_size]
            chunk_result = await self._process_operations_chunk(chunk, progress_callback)

            # Merge results
            result.successful += chunk_result.successful
            result.skipped += chunk_result.skipped
            result.failed += chunk_result.failed
            result.total_size_mb += chunk_result.total_size_mb
            result.operations.extend(chunk_result.operations)
            result.errors.extend(chunk_result.errors)

        return result

    async def _process_operations_chunk(self, chunk: List[FileOperation],
                                      progress_callback=None) -> BulkOperationResult:
        """Process a chunk of operations in parallel."""
        result = BulkOperationResult(
            total_files=len(chunk),
            successful=0,
            skipped=0,
            failed=0,
            total_size_mb=0.0,
            duration_seconds=0.0
        )

        # Create parallel tasks
        tasks = []
        for op in chunk:
            task = asyncio.create_task(self._process_single_operation(op))
            tasks.append(task)

        # Wait for all operations to complete
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, chunk_result in enumerate(chunk_results):
            if isinstance(chunk_result, Exception):
                result.failed += 1
                result.errors.append(f"Failed to process {chunk[i].source}: {chunk_result}")
            else:
                if chunk_result['status'] == 'success':
                    result.successful += 1
                elif chunk_result['status'] == 'skipped':
                    result.skipped += 1
                else:
                    result.failed += 1

                result.total_size_mb += chunk[i].size / (1024 * 1024)
                result.operations.append({
                    'source': str(chunk[i].source),
                    'target': str(chunk[i].target),
                    'operation': chunk[i].operation_type.value,
                    'status': chunk_result['status'],
                    'message': chunk_result.get('message', ''),
                    'timestamp': datetime.now().isoformat()
                })

            # Update progress
            if progress_callback:
                await progress_callback(i + 1, len(chunk))

        return result

    async def _process_single_operation(self, operation: FileOperation) -> Dict:
        """Process a single file operation with conflict resolution."""
        try:
            # Check if source exists
            if not operation.source.exists():
                return {'status': 'failed', 'message': 'Source file does not exist'}

            # Handle conflicts
            target_path = await self._resolve_conflict(operation)

            if target_path is None:
                # Operation was skipped due to conflict strategy
                return {'status': 'skipped', 'message': 'Skipped due to conflict'}

            # Perform the operation
            if operation.operation_type == OperationType.MOVE:
                await self._move_file(operation.source, target_path)
            elif operation.operation_type == OperationType.COPY:
                await self._copy_file(operation.source, target_path)
            elif operation.operation_type == OperationType.MOVE_COPY:
                await self._move_file_with_verification(operation.source, target_path)

            return {'status': 'success', 'message': f'Operation completed to {target_path}'}

        except Exception as e:
            return {'status': 'failed', 'message': str(e)}

    async def _resolve_conflict(self, operation: FileOperation) -> Optional[Path]:
        """Resolve file conflicts based on the configured strategy."""
        target_path = operation.target

        # If target doesn't exist, no conflict
        if not target_path.exists():
            return target_path

        # Skip identical files if configured
        if self.config.skip_identical:
            if await self._files_identical(operation.source, target_path):
                return None  # Skip operation

        # Apply conflict strategy
        if self.config.conflict_strategy == ConflictStrategy.SKIP:
            return None
        elif self.config.conflict_strategy == ConflictStrategy.REPLACE:
            return target_path
        elif self.config.conflict_strategy == ConflictStrategy.RENAME:
            return await self._find_unique_name(target_path)
        elif self.config.conflict_strategy == ConflictStrategy.KEEP_BOTH:
            return await self._find_unique_name(target_path)

        return target_path

    async def _files_identical(self, file1: Path, file2: Path) -> bool:
        """Check if two files are identical."""
        # Quick size check
        stat1 = file1.stat()
        stat2 = file2.stat()

        if stat1.st_size != stat2.st_size:
            return False

        # If we have checksums, use them
        if hasattr(self, '_checksum_cache'):
            checksum1 = self._checksum_cache.get(file1)
            checksum2 = self._checksum_cache.get(file2)
            if checksum1 and checksum2:
                return checksum1 == checksum2

        # Fall back to size + mtime check
        return abs(stat1.st_mtime - stat2.st_mtime) < 1.0

    async def _find_unique_name(self, path: Path) -> Path:
        """Find a unique filename by adding a number."""
        if not path.exists():
            return path

        base = path.stem
        ext = path.suffix
        parent = path.parent
        counter = 1

        while True:
            new_name = f"{base} ({counter}){ext}"
            new_path = parent / new_name

            def check_exists():
                return not new_path.exists()

            if await asyncio.get_event_loop().run_in_executor(
                self.executor, check_exists
            ):
                return new_path
            counter += 1

    async def _move_file(self, source: Path, target: Path) -> None:
        """Move a file with optional timestamp preservation."""
        def _do_move():
            shutil.move(str(source), str(target))
            if self.config.preserve_timestamps and source.exists():
                # Note: shutil.move preserves timestamps on Unix-like systems
                pass

        await asyncio.get_event_loop().run_in_executor(
            self.executor, _do_move
        )

    async def _copy_file(self, source: Path, target: Path) -> None:
        """Copy a file with verification."""
        def _do_copy():
            shutil.copy2(str(source), str(target))  # copy2 preserves metadata

        await asyncio.get_event_loop().run_in_executor(
            self.executor, _do_copy
        )

        # Verify copy if enabled
        if self.config.verify_copies:
            if not await self._verify_copy(source, target):
                raise FileOperationError(f"Copy verification failed for {source} -> {target}")

    async def _move_file_with_verification(self, source: Path, target: Path) -> None:
        """Move file with verification copy."""
        # First copy with verification
        await self._copy_file(source, target)

        # Then remove source
        def _do_remove():
            source.unlink()

        await asyncio.get_event_loop().run_in_executor(
            self.executor, _do_remove
        )

    async def _verify_copy(self, source: Path, target: Path) -> bool:
        """Verify that a copy operation was successful."""
        def _do_verify():
            if not target.exists():
                return False

            # Check file sizes
            source_size = source.stat().st_size
            target_size = target.stat().st_size

            if source_size != target_size:
                return False

            # For small files, verify checksum
            if source_size < 10 * 1024 * 1024:  # 10MB
                source_hash = self._calculate_file_hash(source)
                target_hash = self._calculate_file_hash(target)
                return source_hash == target_hash

            return True

        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _do_verify
        )

    def _calculate_file_hash(self, path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except (OSError, IOError):
            return ""

    async def get_estimated_duration(self, files: List[Path]) -> float:
        """Estimate duration for bulk operations based on file sizes."""
        total_size = 0
        count = 0

        for file_path in files:
            if file_path.exists():
                total_size += file_path.stat().st_size
                count += 1

        # Rough estimation: 50 MB/s per worker on average
        throughput_per_worker = 50 * 1024 * 1024  # bytes/sec
        total_throughput = throughput_per_worker * self.config.max_workers

        # Add overhead for directory creation and conflict resolution
        overhead_seconds = count * 0.1  # 100ms per file overhead

        if total_throughput > 0:
            return (total_size / total_throughput) + overhead_seconds
        return overhead_seconds

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.executor.shutdown(wait=True)


class BulkMoveOperator(BulkFileOperator):
    """Specialized bulk file mover with music-specific optimizations."""

    def __init__(self, config: Optional[BulkOperationConfig] = None):
        if config is None:
            config = BulkOperationConfig(
                conflict_strategy=ConflictStrategy.RENAME,
                verify_copies=False,  # Faster for moves
                preserve_timestamps=True
            )
        super().__init__(config)

    async def add_audio_move(self, audio_file: AudioFile, target_path: Path) -> None:
        """Add an audio file move operation."""
        await self.add_operation(audio_file.path, target_path, OperationType.MOVE)

    async def add_cover_art_move(self, cover_art: CoverArt, target_dir: Path) -> None:
        """Add a cover art move operation."""
        if cover_art and cover_art.path.exists():
            target_filename = self._get_cover_art_filename(cover_art)
            target_path = target_dir / target_filename
            await self.add_operation(cover_art.path, target_path, OperationType.MOVE)

    def _get_cover_art_filename(self, cover_art: CoverArt) -> str:
        """Get standardized filename for cover art."""
        if cover_art.type == 'front':
            return f'folder.{cover_art.format}'
        elif cover_art.type == 'back':
            return f'back.{cover_art.format}'
        elif cover_art.type == 'disc':
            return f'disc.{cover_art.format}'
        else:
            return f'cover.{cover_art.format}'


class BulkCopyOperator(BulkFileOperator):
    """Specialized bulk file copier with verification."""

    def __init__(self, config: Optional[BulkOperationConfig] = None):
        if config is None:
            config = BulkOperationConfig(
                conflict_strategy=ConflictStrategy.RENAME,
                verify_copies=True,  # Always verify copies
                preserve_timestamps=True
            )
        super().__init__(config)