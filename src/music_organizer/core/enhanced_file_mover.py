"""Enhanced async file operations with comprehensive operation history tracking."""

import os
import shutil
import json
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, AsyncGenerator, Union
from datetime import datetime

from .operation_history import (
    OperationHistoryTracker,
    OperationType,
    OperationStatus,
    OperationRecord,
    OperationSession,
    OperationRollbackService,
    operation_session,
    create_operation_record
)
from ..exceptions import FileOperationError
from ..models.audio_file import AudioFile, CoverArt
from ..domain.result import Result, Success, Failure


class EnhancedAsyncFileMover:
    """Enhanced file mover with comprehensive operation history tracking."""

    def __init__(self,
                 backup_enabled: bool = True,
                 backup_dir: Optional[Path] = None,
                 max_workers: int = 4,
                 history_tracker: Optional[OperationHistoryTracker] = None,
                 session_id: Optional[str] = None):
        """Initialize the enhanced file mover."""
        self.backup_enabled = backup_enabled
        self.backup_dir = backup_dir
        self.max_workers = max_workers
        self.history_tracker = history_tracker or OperationHistoryTracker()
        self.session_id = session_id
        self.session: Optional[OperationSession] = None
        self.rollback_service = OperationRollbackService(self.history_tracker)

        # Legacy support
        self.operations: List[Dict] = []
        self.started = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()

    async def start_operation(self, source_root: Path, target_root: Path,
                            metadata: Optional[Dict] = None) -> Result[OperationSession, Exception]:
        """Start a new operation session with operation history tracking."""
        async with self._lock:
            if self.started:
                return Failure("Operation already in progress")

            # Generate session ID if not provided
            if not self.session_id:
                self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # Start session in history tracker
            session_result = await self.history_tracker.start_session(
                self.session_id, source_root, target_root, metadata
            )

            if session_result.is_failure():
                return Failure(f"Failed to start operation session: {session_result.error()}")

            self.session = session_result.value()
            self.started = True
            self.operations = []

            # Handle backup directory
            if self.backup_enabled:
                if not self.backup_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.backup_dir = source_root.parent / f"backup_{timestamp}"

                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.backup_dir.mkdir(parents=True, exist_ok=True)
                )

                # Create backup manifest
                await self._create_backup_manifest(source_root)

            return Success(self.session)

    async def finish_operation(self, status: str = "completed") -> Result[OperationSession, Exception]:
        """Finish current operation session and record in history."""
        async with self._lock:
            if not self.started:
                return Failure("No operation in progress")

            # End session in history tracker
            if self.session:
                session_result = await self.history_tracker.end_session(
                    self.session.session_id, status
                )
                if session_result.is_failure():
                    return Failure(f"Failed to end session: {session_result.error()}")
                self.session = session_result.value()

            # Save legacy operation log
            if self.backup_enabled and self.backup_dir:
                await self._save_operation_log()

            self.started = False
            return Success(self.session)

    async def move_file(self, audio_file: AudioFile, target_path: Path,
                       verify_checksum: bool = False) -> Result[Path, Exception]:
        """Move an audio file with comprehensive tracking."""
        async with self._lock:
            if not self.started:
                return Failure("Must start operation before moving files")

        # Create operation record
        source_path = audio_file.path
        checksum_before = None

        if verify_checksum:
            checksum_before = await self._calculate_checksum_async(source_path)

        operation = create_operation_record(
            session_id=self.session_id,
            operation_type=OperationType.MOVE,
            source_path=source_path,
            target_path=target_path,
            metadata={
                "file_type": "audio",
                "file_format": audio_file.file_type,
                "verify_checksum": verify_checksum
            }
        )

        try:
            # Ensure target directory exists
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: target_path.parent.mkdir(parents=True, exist_ok=True)
            )

            # Resolve duplicates
            final_target = await self._resolve_duplicate(target_path)

            # Update operation with actual target
            operation = OperationRecord(
                id=operation.id,
                session_id=operation.session_id,
                timestamp=operation.timestamp,
                operation_type=operation.operation_type,
                source_path=operation.source_path,
                target_path=final_target,
                backup_path=operation.backup_path,
                status=OperationStatus.IN_PROGRESS,
                checksum_before=checksum_before,
                metadata=operation.metadata
            )

            # Record operation start
            await self.history_tracker.record_operation(operation)

            # Create backup if enabled
            if self.backup_enabled and self.backup_dir:
                backup_result = await self._backup_file_with_record(source_path, operation.id)
                if backup_result.is_success():
                    operation = OperationRecord(
                        id=operation.id,
                        session_id=operation.session_id,
                        timestamp=operation.timestamp,
                        operation_type=operation.operation_type,
                        source_path=operation.source_path,
                        target_path=operation.target_path,
                        backup_path=backup_result.value(),
                        status=OperationStatus.IN_PROGRESS,
                        checksum_before=checksum_before,
                        metadata=operation.metadata
                    )

            # Perform the move
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: shutil.move(str(source_path), str(final_target))
            )
            audio_file.path = final_target

            # Calculate checksum after move if verification enabled
            checksum_after = None
            if verify_checksum:
                checksum_after = await self._calculate_checksum_async(final_target)
                if checksum_before != checksum_after:
                    # Attempt rollback if checksum mismatch
                    await self._rollback_single_operation(operation)
                    return Failure(f"Checksum mismatch for {audio_file.path}")

            # Update file size
            file_size = final_target.stat().st_size if final_target.exists() else None

            # Update operation as completed
            completed_operation = OperationRecord(
                id=operation.id,
                session_id=operation.session_id,
                timestamp=operation.timestamp,
                operation_type=operation.operation_type,
                source_path=operation.source_path,
                target_path=operation.target_path,
                backup_path=operation.backup_path,
                status=OperationStatus.COMPLETED,
                checksum_before=checksum_before,
                checksum_after=checksum_after,
                file_size=file_size,
                metadata=operation.metadata
            )

            await self.history_tracker.record_operation(completed_operation)

            # Add to legacy operations list
            self.operations.append({
                'type': 'move',
                'original': str(source_path),
                'target': str(final_target),
                'timestamp': datetime.now().isoformat(),
                'operation_id': operation.id
            })

            return Success(final_target)

        except Exception as e:
            # Record operation as failed
            failed_operation = OperationRecord(
                id=operation.id,
                session_id=operation.session_id,
                timestamp=operation.timestamp,
                operation_type=operation.operation_type,
                source_path=operation.source_path,
                target_path=operation.target_path,
                backup_path=operation.backup_path,
                status=OperationStatus.FAILED,
                error_message=str(e),
                checksum_before=checksum_before,
                metadata=operation.metadata
            )

            await self.history_tracker.record_operation(failed_operation)
            return Failure(f"Failed to move {audio_file.path}: {str(e)}")

    async def move_cover_art(self, cover_art: CoverArt, target_dir: Path,
                           verify_checksum: bool = False) -> Result[Optional[Path], Exception]:
        """Move cover art with comprehensive tracking."""
        if not cover_art or not cover_art.path.exists():
            return Success(None)

        # Determine target filename
        target_filename = self._get_cover_art_filename(cover_art)
        target_path = target_dir / target_filename

        # Create operation record
        source_path = cover_art.path
        checksum_before = None

        if verify_checksum:
            checksum_before = await self._calculate_checksum_async(source_path)

        operation = create_operation_record(
            session_id=self.session_id,
            operation_type=OperationType.MOVE_COVER,
            source_path=source_path,
            target_path=target_path,
            metadata={
                "file_type": "cover",
                "cover_type": cover_art.type,
                "format": cover_art.format,
                "verify_checksum": verify_checksum
            }
        )

        try:
            # Handle duplicates
            final_target = await self._resolve_duplicate(target_path)

            # Update operation with actual target
            operation = OperationRecord(
                id=operation.id,
                session_id=operation.session_id,
                timestamp=operation.timestamp,
                operation_type=operation.operation_type,
                source_path=operation.source_path,
                target_path=final_target,
                backup_path=operation.backup_path,
                status=OperationStatus.IN_PROGRESS,
                checksum_before=checksum_before,
                metadata=operation.metadata
            )

            # Record operation start
            await self.history_tracker.record_operation(operation)

            # Create backup if enabled
            if self.backup_enabled and self.backup_dir:
                backup_result = await self._backup_file_with_record(source_path, operation.id)
                if backup_result.is_success():
                    operation = OperationRecord(
                        id=operation.id,
                        session_id=operation.session_id,
                        timestamp=operation.timestamp,
                        operation_type=operation.operation_type,
                        source_path=operation.source_path,
                        target_path=operation.target_path,
                        backup_path=backup_result.value(),
                        status=OperationStatus.IN_PROGRESS,
                        checksum_before=checksum_before,
                        metadata=operation.metadata
                    )

            # Perform the move
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: shutil.move(str(source_path), str(final_target))
            )
            cover_art.path = final_target

            # Calculate checksum after move if verification enabled
            checksum_after = None
            if verify_checksum:
                checksum_after = await self._calculate_checksum_async(final_target)
                if checksum_before != checksum_after:
                    await self._rollback_single_operation(operation)
                    return Failure(f"Checksum mismatch for cover art {cover_art.path}")

            # Update file size
            file_size = final_target.stat().st_size if final_target.exists() else None

            # Update operation as completed
            completed_operation = OperationRecord(
                id=operation.id,
                session_id=operation.session_id,
                timestamp=operation.timestamp,
                operation_type=operation.operation_type,
                source_path=operation.source_path,
                target_path=operation.target_path,
                backup_path=operation.backup_path,
                status=OperationStatus.COMPLETED,
                checksum_before=checksum_before,
                checksum_after=checksum_after,
                file_size=file_size,
                metadata=operation.metadata
            )

            await self.history_tracker.record_operation(completed_operation)

            # Add to legacy operations list
            self.operations.append({
                'type': 'move_cover',
                'original': str(source_path),
                'target': str(final_target),
                'timestamp': datetime.now().isoformat(),
                'operation_id': operation.id
            })

            return Success(final_target)

        except Exception as e:
            # Record operation as failed
            failed_operation = OperationRecord(
                id=operation.id,
                session_id=operation.session_id,
                timestamp=operation.timestamp,
                operation_type=operation.operation_type,
                source_path=operation.source_path,
                target_path=operation.target_path,
                backup_path=operation.backup_path,
                status=OperationStatus.FAILED,
                error_message=str(e),
                checksum_before=checksum_before,
                metadata=operation.metadata
            )

            await self.history_tracker.record_operation(failed_operation)
            return Failure(f"Failed to move cover art {cover_art.path}: {str(e)}")

    async def move_files_batch(self, moves: List[Tuple[AudioFile, Path]],
                             verify_checksum: bool = False) -> List[Result[Path, Exception]]:
        """Move multiple files in parallel with tracking."""
        tasks = []
        for audio_file, target_path in moves:
            task = asyncio.create_task(self.move_file(audio_file, target_path, verify_checksum))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    async def rollback_session(self, dry_run: bool = False) -> Result[Dict, Exception]:
        """Rollback the entire session using the rollback service."""
        if not self.session_id:
            return Failure("No session to rollback")

        return await self.rollback_service.rollback_session(self.session_id, dry_run)

    async def rollback_partial(self, operation_ids: List[str],
                             dry_run: bool = False) -> Result[Dict, Exception]:
        """Rollback specific operations."""
        if not self.session_id:
            return Failure("No session to rollback")

        return await self.rollback_service.rollback_partial(
            self.session_id, operation_ids, dry_run
        )

    async def get_operation_history(self) -> Result[List[OperationRecord], Exception]:
        """Get all operations for the current session."""
        if not self.session_id:
            return Failure("No session active")

        try:
            operations = await self.history_tracker.get_session_operations(self.session_id)
            return Success(operations)
        except Exception as e:
            return Failure(f"Failed to get operation history: {str(e)}")

    async def get_session_summary(self) -> Result[Dict, Exception]:
        """Get a summary of the current session."""
        if not self.session_id:
            return Failure("No session active")

        try:
            session = await self.history_tracker.get_session(self.session_id)
            if not session:
                return Failure("Session not found")

            operations = await self.history_tracker.get_session_operations(self.session_id)

            # Calculate summary statistics
            summary = {
                "session_id": session.session_id,
                "status": session.status,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "source_root": str(session.source_root),
                "target_root": str(session.target_root),
                "total_operations": len(operations),
                "completed_operations": sum(1 for op in operations if op.status == OperationStatus.COMPLETED),
                "failed_operations": sum(1 for op in operations if op.status == OperationStatus.FAILED),
                "pending_operations": sum(1 for op in operations if op.status == OperationStatus.PENDING),
                "in_progress_operations": sum(1 for op in operations if op.status == OperationStatus.IN_PROGRESS),
                "rolled_back_operations": sum(1 for op in operations if op.status == OperationStatus.ROLLED_BACK),
                "operation_types": {
                    op_type.value: sum(1 for op in operations if op.operation_type == op_type)
                    for op_type in OperationType
                },
                "total_size_bytes": sum(op.file_size or 0 for op in operations),
                "backup_enabled": self.backup_enabled,
                "backup_dir": str(self.backup_dir) if self.backup_dir else None
            }

            return Success(summary)

        except Exception as e:
            return Failure(f"Failed to get session summary: {str(e)}")

    # Private helper methods

    async def _calculate_checksum_async(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum asynchronously."""
        if not file_path.exists():
            return ""

        def _calculate():
            h = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    h.update(chunk)
            return h.hexdigest()

        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _calculate
        )

    async def _resolve_duplicate(self, target_path: Path) -> Path:
        """Resolve duplicate filenames by adding a number."""
        def _check_exists(path):
            return path.exists()

        if not await asyncio.get_event_loop().run_in_executor(
            self.executor, _check_exists, target_path
        ):
            return target_path

        base = target_path.stem
        ext = target_path.suffix
        parent = target_path.parent
        counter = 1

        while True:
            new_name = f"{base} ({counter}){ext}"
            new_path = parent / new_name

            if not await asyncio.get_event_loop().run_in_executor(
                self.executor, _check_exists, new_path
            ):
                return new_path
            counter += 1

    async def _backup_file_with_record(self, file_path: Path, operation_id: str) -> Result[Path, Exception]:
        """Create a backup and return the backup path."""
        if not self.backup_dir:
            return Failure("Backup directory not configured")

        def _do_backup():
            try:
                # Create relative path in backup
                relative_path = file_path.relative_to(file_path.anchor)
                backup_path = self.backup_dir / f"{operation_id}_{relative_path.name}"
                backup_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(str(file_path), str(backup_path))
                return backup_path
            except Exception as e:
                raise RuntimeError(f"Failed to backup {file_path}: {e}")

        try:
            backup_path = await asyncio.get_event_loop().run_in_executor(
                self.executor, _do_backup
            )
            return Success(backup_path)
        except Exception as e:
            return Failure(str(e))

    async def _rollback_single_operation(self, operation: OperationRecord) -> None:
        """Rollback a single failed operation."""
        if operation.operation_type == OperationType.MOVE:
            if operation.target_path and operation.target_path.exists():
                if operation.source_path and not operation.source_path.exists():
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: operation.source_path.parent.mkdir(parents=True, exist_ok=True)
                    )
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: operation.target_path.replace(operation.source_path)
                    )

    async def _create_backup_manifest(self, source_root: Path) -> None:
        """Create a manifest of all files before starting operations."""
        if not self.backup_dir:
            return

        def _scan_files():
            manifest = {
                'source_root': str(source_root),
                'timestamp': datetime.now().isoformat(),
                'files': []
            }

            # Find all audio files
            audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}
            cover_extensions = {'.jpg', '.jpeg', '.png', '.gif'}

            for file_path in source_root.rglob('*'):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in audio_extensions or ext in cover_extensions:
                        try:
                            stat = file_path.stat()
                            manifest['files'].append({
                                'path': str(file_path.relative_to(source_root)),
                                'size': stat.st_size,
                                'mtime': stat.st_mtime,
                                'type': 'audio' if ext in audio_extensions else 'cover'
                            })
                        except (OSError, ValueError):
                            pass

            return manifest

        manifest = await asyncio.get_event_loop().run_in_executor(
            self.executor, _scan_files
        )

        # Save manifest
        manifest_path = self.backup_dir / 'manifest.json'
        await self._save_json(manifest_path, manifest)

    async def _save_operation_log(self) -> None:
        """Save operation log to backup directory."""
        if not self.backup_dir or not self.operations:
            return

        # Get session summary
        summary_result = await self.get_session_summary()
        summary = summary_result.value() if summary_result.is_success() else {}

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'operations': self.operations,
            'summary': summary
        }

        log_path = self.backup_dir / 'operations.json'
        await self._save_json(log_path, log_data)

    async def _save_json(self, path: Path, data: Dict) -> None:
        """Save JSON data to file asynchronously."""
        def _write_json():
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

        await asyncio.get_event_loop().run_in_executor(
            self.executor, _write_json
        )

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

    # Legacy compatibility methods

    async def rollback(self) -> None:
        """Legacy rollback method for backward compatibility."""
        if not self.operations:
            return

        # Use new rollback system
        result = await self.rollback_session()
        if result.is_failure():
            print(f"Warning: Rollback failed: {result.error()}")

    async def get_operation_summary(self) -> Dict:
        """Legacy operation summary for backward compatibility."""
        summary_result = await self.get_session_summary()
        if summary_result.is_success():
            return summary_result.value()
        return {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Finish operation with error status if exception occurred
        if self.started and exc_type:
            await self.finish_operation("failed")
        elif self.started:
            await self.finish_operation("completed")

        self.executor.shutdown(wait=True)


# Context manager for easy session management
async def file_operation_session(source_root: Path, target_root: Path,
                                backup_enabled: bool = True,
                                session_id: Optional[str] = None,
                                metadata: Optional[Dict] = None):
    """Context manager for file operation sessions."""
    mover = EnhancedAsyncFileMover(
        backup_enabled=backup_enabled,
        session_id=session_id
    )

    session_result = await mover.start_operation(source_root, target_root, metadata)
    if session_result.is_failure():
        raise RuntimeError(f"Failed to start operation session: {session_result.error()}")

    try:
        yield mover
        await mover.finish_operation("completed")
    except Exception as e:
        await mover.finish_operation("failed")
        raise