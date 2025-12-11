"""Operation history tracking and management for the music organizer.

This module provides comprehensive operation tracking, history management,
and rollback capabilities for file operations in the music organizer.
"""

import asyncio
import json
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import hashlib
import os

from music_organizer.domain.result import Result, Success, Failure
from music_organizer.core.async_mover import AsyncFileMover


class OperationType(Enum):
    """Type of file operation."""
    MOVE = "move"
    COPY = "copy"
    DELETE = "delete"
    CREATE_DIR = "create_dir"
    MOVE_COVER = "move_cover"


class OperationStatus(Enum):
    """Status of an operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass(slots=True, frozen=True)
class OperationRecord:
    """Represents a single file operation record."""
    id: str
    session_id: str
    timestamp: datetime
    operation_type: OperationType
    source_path: Optional[Path]
    target_path: Optional[Path]
    backup_path: Optional[Path]
    status: OperationStatus
    error_message: Optional[str] = None
    checksum_before: Optional[str] = None
    checksum_after: Optional[str] = None
    file_size: Optional[int] = None
    metadata: Dict[str, Union[str, int, bool]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "operation_type": self.operation_type.value,
            "source_path": str(self.source_path) if self.source_path else None,
            "target_path": str(self.target_path) if self.target_path else None,
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "status": self.status.value,
            "error_message": self.error_message,
            "checksum_before": self.checksum_before,
            "checksum_after": self.checksum_after,
            "file_size": self.file_size,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "OperationRecord":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            operation_type=OperationType(data["operation_type"]),
            source_path=Path(data["source_path"]) if data["source_path"] else None,
            target_path=Path(data["target_path"]) if data["target_path"] else None,
            backup_path=Path(data["backup_path"]) if data["backup_path"] else None,
            status=OperationStatus(data["status"]),
            error_message=data.get("error_message"),
            checksum_before=data.get("checksum_before"),
            checksum_after=data.get("checksum_after"),
            file_size=data.get("file_size"),
            metadata=data.get("metadata", {})
        )


@dataclass(slots=True, frozen=True)
class OperationSession:
    """Represents a session of file operations."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    source_root: Path
    target_root: Path
    total_operations: int
    completed_operations: int
    failed_operations: int
    status: str  # "running", "completed", "failed", "rolled_back"
    metadata: Dict[str, Union[str, int, bool]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "source_root": str(self.source_root),
            "target_root": str(self.target_root),
            "total_operations": self.total_operations,
            "completed_operations": self.completed_operations,
            "failed_operations": self.failed_operations,
            "status": self.status,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "OperationSession":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
            source_root=Path(data["source_root"]),
            target_root=Path(data["target_root"]),
            total_operations=data["total_operations"],
            completed_operations=data["completed_operations"],
            failed_operations=data["failed_operations"],
            status=data["status"],
            metadata=data.get("metadata", {})
        )


class OperationHistoryTracker:
    """Tracks operation history with persistent storage."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the tracker with optional custom database path."""
        if db_path is None:
            cache_dir = Path.home() / ".cache" / "music-organizer"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / "operation_history.db"

        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS operation_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    source_root TEXT NOT NULL,
                    target_root TEXT NOT NULL,
                    total_operations INTEGER DEFAULT 0,
                    completed_operations INTEGER DEFAULT 0,
                    failed_operations INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS operation_records (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    source_path TEXT,
                    target_path TEXT,
                    backup_path TEXT,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    checksum_before TEXT,
                    checksum_after TEXT,
                    file_size INTEGER,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES operation_sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_session_id ON operation_records(session_id);
                CREATE INDEX IF NOT EXISTS idx_timestamp ON operation_records(timestamp);
                CREATE INDEX IF NOT EXISTS idx_status ON operation_records(status);
            """)

    async def start_session(self, session_id: str, source_root: Path, target_root: Path,
                           metadata: Optional[Dict] = None) -> Result[OperationSession, Exception]:
        """Start a new operation session."""
        try:
            session = OperationSession(
                session_id=session_id,
                start_time=datetime.now(),
                end_time=None,
                source_root=source_root,
                target_root=target_root,
                total_operations=0,
                completed_operations=0,
                failed_operations=0,
                status="running",
                metadata=metadata or {}
            )

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO operation_sessions
                       (session_id, start_time, source_root, target_root,
                        total_operations, completed_operations, failed_operations, status, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session.session_id,
                        session.start_time.isoformat(),
                        str(session.source_root),
                        str(session.target_root),
                        session.total_operations,
                        session.completed_operations,
                        session.failed_operations,
                        session.status,
                        json.dumps(session.metadata)
                    )
                )

            return Success(session)

        except Exception as e:
            return Failure(f"Failed to start session: {str(e)}")

    async def record_operation(self, operation: OperationRecord) -> Result[None, Exception]:
        """Record a single operation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Insert operation record
                conn.execute(
                    """INSERT INTO operation_records
                       (id, session_id, timestamp, operation_type, source_path, target_path,
                        backup_path, status, error_message, checksum_before, checksum_after,
                        file_size, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        operation.id,
                        operation.session_id,
                        operation.timestamp.isoformat(),
                        operation.operation_type.value,
                        str(operation.source_path) if operation.source_path else None,
                        str(operation.target_path) if operation.target_path else None,
                        str(operation.backup_path) if operation.backup_path else None,
                        operation.status.value,
                        operation.error_message,
                        operation.checksum_before,
                        operation.checksum_after,
                        operation.file_size,
                        json.dumps(operation.metadata)
                    )
                )

                # Update session counters
                if operation.status == OperationStatus.COMPLETED:
                    conn.execute(
                        """UPDATE operation_sessions
                           SET completed_operations = completed_operations + 1
                           WHERE session_id = ?""",
                        (operation.session_id,)
                    )
                elif operation.status == OperationStatus.FAILED:
                    conn.execute(
                        """UPDATE operation_sessions
                           SET failed_operations = failed_operations + 1
                           WHERE session_id = ?""",
                        (operation.session_id,)
                    )

            return Success(None)

        except Exception as e:
            return Failure(f"Failed to record operation: {str(e)}")

    async def end_session(self, session_id: str, status: str = "completed") -> Result[OperationSession, Exception]:
        """End an operation session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update session
                conn.execute(
                    """UPDATE operation_sessions
                       SET end_time = ?, status = ?
                       WHERE session_id = ?""",
                    (datetime.now().isoformat(), status, session_id)
                )

                # Get updated session
                cursor = conn.execute(
                    """SELECT * FROM operation_sessions WHERE session_id = ?""",
                    (session_id,)
                )
                row = cursor.fetchone()

                if row:
                    session = OperationSession(
                        session_id=row[0],
                        start_time=datetime.fromisoformat(row[1]),
                        end_time=datetime.fromisoformat(row[2]) if row[2] else None,
                        source_root=Path(row[3]),
                        target_root=Path(row[4]),
                        total_operations=row[5],
                        completed_operations=row[6],
                        failed_operations=row[7],
                        status=row[8],
                        metadata=json.loads(row[9]) if row[9] else {}
                    )
                    return Success(session)
                else:
                    return Failure(f"Session {session_id} not found")

        except Exception as e:
            return Failure(f"Failed to end session: {str(e)}")

    async def get_session(self, session_id: str) -> Optional[OperationSession]:
        """Get a specific session by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """SELECT * FROM operation_sessions WHERE session_id = ?""",
                    (session_id,)
                )
                row = cursor.fetchone()

                if row:
                    return OperationSession(
                        session_id=row[0],
                        start_time=datetime.fromisoformat(row[1]),
                        end_time=datetime.fromisoformat(row[2]) if row[2] else None,
                        source_root=Path(row[3]),
                        target_root=Path(row[4]),
                        total_operations=row[5],
                        completed_operations=row[6],
                        failed_operations=row[7],
                        status=row[8],
                        metadata=json.loads(row[9]) if row[9] else {}
                    )
                return None
        except Exception:
            return None

    async def get_session_operations(self, session_id: str,
                                   status_filter: Optional[OperationStatus] = None) -> List[OperationRecord]:
        """Get all operations for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if status_filter:
                    cursor = conn.execute(
                        """SELECT * FROM operation_records
                           WHERE session_id = ? AND status = ?
                           ORDER BY timestamp""",
                        (session_id, status_filter.value)
                    )
                else:
                    cursor = conn.execute(
                        """SELECT * FROM operation_records
                           WHERE session_id = ?
                           ORDER BY timestamp""",
                        (session_id,)
                    )

                operations = []
                for row in cursor.fetchall():
                    operations.append(OperationRecord(
                        id=row[0],
                        session_id=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        operation_type=OperationType(row[3]),
                        source_path=Path(row[4]) if row[4] else None,
                        target_path=Path(row[5]) if row[5] else None,
                        backup_path=Path(row[6]) if row[6] else None,
                        status=OperationStatus(row[7]),
                        error_message=row[8],
                        checksum_before=row[9],
                        checksum_after=row[10],
                        file_size=row[11],
                        metadata=json.loads(row[12]) if row[12] else {}
                    ))

                return operations
        except Exception:
            return []

    async def list_sessions(self, limit: int = 50) -> List[OperationSession]:
        """List recent sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """SELECT * FROM operation_sessions
                       ORDER BY start_time DESC
                       LIMIT ?""",
                    (limit,)
                )

                sessions = []
                for row in cursor.fetchall():
                    sessions.append(OperationSession(
                        session_id=row[0],
                        start_time=datetime.fromisoformat(row[1]),
                        end_time=datetime.fromisoformat(row[2]) if row[2] else None,
                        source_root=Path(row[3]),
                        target_root=Path(row[4]),
                        total_operations=row[5],
                        completed_operations=row[6],
                        failed_operations=row[7],
                        status=row[8],
                        metadata=json.loads(row[9]) if row[9] else {}
                    ))

                return sessions
        except Exception:
            return []

    async def delete_session(self, session_id: str) -> Result[None, Exception]:
        """Delete a session and all its operations."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete operations first (foreign key constraint)
                conn.execute(
                    """DELETE FROM operation_records WHERE session_id = ?""",
                    (session_id,)
                )
                # Delete session
                conn.execute(
                    """DELETE FROM operation_sessions WHERE session_id = ?""",
                    (session_id,)
                )

            return Success(None)

        except Exception as e:
            return Failure(f"Failed to delete session: {str(e)}")


class OperationRollbackService:
    """Service for rolling back file operations."""

    def __init__(self, history_tracker: OperationHistoryTracker):
        """Initialize with history tracker."""
        self.history_tracker = history_tracker

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        if not file_path.exists():
            return ""

        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()

    async def rollback_session(self, session_id: str,
                             dry_run: bool = False) -> Result[Dict[str, Union[int, List[str]]], Exception]:
        """Rollback all operations in a session."""
        try:
            # Get session
            session = await self.history_tracker.get_session(session_id)
            if not session:
                return Failure(f"Session {session_id} not found")

            # Get all completed operations (reverse order for correct rollback)
            operations = await self.history_tracker.get_session_operations(
                session_id,
                OperationStatus.COMPLETED
            )
            operations.reverse()

            if dry_run:
                return Success({
                    "session_id": session_id,
                    "total_operations": len(operations),
                    "operations_to_rollback": [op.id for op in operations],
                    "dry_run": True
                })

            # Track rollback results
            successful_rollbacks = []
            failed_rollbacks = []
            skipped_rollbacks = []

            for operation in operations:
                try:
                    if operation.operation_type == OperationType.MOVE:
                        # Move file back from target to source
                        if operation.target_path and operation.target_path.exists():
                            if operation.source_path and not operation.source_path.exists():
                                operation.source_path.parent.mkdir(parents=True, exist_ok=True)
                                operation.target_path.replace(operation.source_path)
                                successful_rollbacks.append(operation.id)
                            else:
                                skipped_rollbacks.append({
                                    "operation_id": operation.id,
                                    "reason": "Source file already exists"
                                })
                        else:
                            skipped_rollbacks.append({
                                "operation_id": operation.id,
                                "reason": "Target file no longer exists"
                            })

                    elif operation.operation_type == OperationType.COPY:
                        # Delete the copied file
                        if operation.target_path and operation.target_path.exists():
                            operation.target_path.unlink()
                            successful_rollbacks.append(operation.id)
                        else:
                            skipped_rollbacks.append({
                                "operation_id": operation.id,
                                "reason": "Target file no longer exists"
                            })

                    elif operation.operation_type == OperationType.MOVE_COVER:
                        # Move cover art back
                        if operation.target_path and operation.target_path.exists():
                            if operation.source_path and not operation.source_path.exists():
                                operation.source_path.parent.mkdir(parents=True, exist_ok=True)
                                operation.target_path.replace(operation.source_path)
                                successful_rollbacks.append(operation.id)
                            else:
                                skipped_rollbacks.append({
                                    "operation_id": operation.id,
                                    "reason": "Source cover already exists"
                                })
                        else:
                            skipped_rollbacks.append({
                                "operation_id": operation.id,
                                "reason": "Target cover no longer exists"
                            })

                    # Update operation status to rolled back
                    updated_record = OperationRecord(
                        id=operation.id,
                        session_id=operation.session_id,
                        timestamp=operation.timestamp,
                        operation_type=operation.operation_type,
                        source_path=operation.source_path,
                        target_path=operation.target_path,
                        backup_path=operation.backup_path,
                        status=OperationStatus.ROLLED_BACK,
                        error_message=None,
                        checksum_before=operation.checksum_before,
                        checksum_after=operation.checksum_after,
                        file_size=operation.file_size,
                        metadata={**operation.metadata, "rollback_timestamp": datetime.now().isoformat()}
                    )

                    await self.history_tracker.record_operation(updated_record)

                except Exception as e:
                    failed_rollbacks.append({
                        "operation_id": operation.id,
                        "error": str(e)
                    })

            # Update session status
            await self.history_tracker.end_session(session_id, "rolled_back")

            return Success({
                "session_id": session_id,
                "total_operations": len(operations),
                "successful_rollbacks": len(successful_rollbacks),
                "failed_rollbacks": len(failed_rollbacks),
                "skipped_rollbacks": len(skipped_rollbacks),
                "successful_operations": successful_rollbacks,
                "failed_operations": failed_rollbacks,
                "skipped_operations": skipped_rollbacks,
                "dry_run": False
            })

        except Exception as e:
            return Failure(f"Rollback failed: {str(e)}")

    async def rollback_partial(self, session_id: str,
                             operation_ids: List[str],
                             dry_run: bool = False) -> Result[Dict[str, Union[int, List[str]]], Exception]:
        """Rollback specific operations from a session."""
        try:
            # Get session
            session = await self.history_tracker.get_session(session_id)
            if not session:
                return Failure(f"Session {session_id} not found")

            # Get specific operations
            all_operations = await self.history_tracker.get_session_operations(session_id)
            operations = [op for op in all_operations if op.id in operation_ids]

            if not operations:
                return Failure("No matching operations found")

            if dry_run:
                return Success({
                    "session_id": session_id,
                    "total_operations": len(operations),
                    "operations_to_rollback": [op.id for op in operations],
                    "dry_run": True
                })

            # Similar rollback logic as full session rollback
            # ... (implementation would be similar to rollback_session)

            return Success({
                "session_id": session_id,
                "total_operations": len(operations),
                "message": "Partial rollback not yet implemented"
            })

        except Exception as e:
            return Failure(f"Partial rollback failed: {str(e)}")


@asynccontextmanager
async def operation_session(history_tracker: OperationHistoryTracker,
                          session_id: Optional[str] = None,
                          source_root: Optional[Path] = None,
                          target_root: Optional[Path] = None,
                          metadata: Optional[Dict] = None):
    """Context manager for tracking operation sessions."""
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Start session
    session_result = await history_tracker.start_session(
        session_id, source_root or Path(""), target_root or Path(""), metadata
    )

    if session_result.is_failure():
        raise RuntimeError(f"Failed to start session: {session_result.error()}")

    session = session_result.value()

    try:
        yield session
        # End session successfully
        await history_tracker.end_session(session_id, "completed")
    except Exception as e:
        # End session with error
        await history_tracker.end_session(session_id, "failed")
        raise


# Utility function to create operation records
def create_operation_record(session_id: str, operation_type: OperationType,
                          source_path: Optional[Path] = None,
                          target_path: Optional[Path] = None,
                          backup_path: Optional[Path] = None,
                          metadata: Optional[Dict] = None) -> OperationRecord:
    """Create an operation record with generated ID."""
    import uuid
    return OperationRecord(
        id=str(uuid.uuid4()),
        session_id=session_id,
        timestamp=datetime.now(),
        operation_type=operation_type,
        source_path=source_path,
        target_path=target_path,
        backup_path=backup_path,
        status=OperationStatus.PENDING,
        metadata=metadata or {}
    )