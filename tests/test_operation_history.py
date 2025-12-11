"""Tests for the operation history tracking system."""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import json

from music_organizer.core.operation_history import (
    OperationHistoryTracker,
    OperationRollbackService,
    OperationType,
    OperationStatus,
    OperationRecord,
    OperationSession,
    create_operation_record,
    operation_session
)
from music_organizer.core.enhanced_file_mover import EnhancedAsyncFileMover
from music_organizer.models.audio_file import AudioFile
from music_organizer.domain.result import Result


@pytest.fixture
async def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
async def test_db_path(temp_dir):
    """Create a temporary database path."""
    return temp_dir / "test_history.db"


@pytest.fixture
async def history_tracker(test_db_path):
    """Create an operation history tracker with test database."""
    return OperationHistoryTracker(test_db_path)


@pytest.fixture
async def rollback_service(history_tracker):
    """Create a rollback service."""
    return OperationRollbackService(history_tracker)


class TestOperationHistoryTracker:
    """Test cases for OperationHistoryTracker."""

    @pytest.mark.asyncio
    async def test_init_database(self, test_db_path):
        """Test database initialization."""
        tracker = OperationHistoryTracker(test_db_path)

        # Check if database file exists
        assert test_db_path.exists()

        # Check if tables exist by querying them
        import sqlite3
        with sqlite3.connect(test_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            assert "operation_sessions" in tables
            assert "operation_records" in tables

    @pytest.mark.asyncio
    async def test_start_session(self, history_tracker, temp_dir):
        """Test starting a new operation session."""
        source_root = temp_dir / "source"
        target_root = temp_dir / "target"

        result = await history_tracker.start_session(
            "test_session_1",
            source_root,
            target_root,
            {"test": True}
        )

        assert result.is_success()
        session = result.value()
        assert session.session_id == "test_session_1"
        assert session.source_root == source_root
        assert session.target_root == target_root
        assert session.status == "running"
        assert session.start_time is not None
        assert session.end_time is None
        assert session.metadata["test"] is True

    @pytest.mark.asyncio
    async def test_record_operation(self, history_tracker):
        """Test recording an operation."""
        # Start a session first
        await history_tracker.start_session(
            "test_session_2",
            Path("/source"),
            Path("/target")
        )

        # Create an operation record
        operation = create_operation_record(
            session_id="test_session_2",
            operation_type=OperationType.MOVE,
            source_path=Path("/source/file.mp3"),
            target_path=Path("/target/file.mp3")
        )

        # Record the operation
        result = await history_tracker.record_operation(operation)
        assert result.is_success()

        # Retrieve and verify
        operations = await history_tracker.get_session_operations("test_session_2")
        assert len(operations) == 1
        assert operations[0].session_id == "test_session_2"
        assert operations[0].operation_type == OperationType.MOVE

    @pytest.mark.asyncio
    async def test_end_session(self, history_tracker):
        """Test ending a session."""
        # Start a session
        await history_tracker.start_session(
            "test_session_3",
            Path("/source"),
            Path("/target")
        )

        # End the session
        result = await history_tracker.end_session("test_session_3", "completed")
        assert result.is_success()

        session = result.value()
        assert session.status == "completed"
        assert session.end_time is not None

    @pytest.mark.asyncio
    async def test_get_session(self, history_tracker):
        """Test retrieving a session."""
        # Start a session
        await history_tracker.start_session(
            "test_session_4",
            Path("/source"),
            Path("/target")
        )

        # Retrieve the session
        session = await history_tracker.get_session("test_session_4")
        assert session is not None
        assert session.session_id == "test_session_4"

    @pytest.mark.asyncio
    async def test_get_session_operations(self, history_tracker):
        """Test retrieving operations for a session."""
        # Start a session
        await history_tracker.start_session(
            "test_session_5",
            Path("/source"),
            Path("/target")
        )

        # Create multiple operations
        operations = [
            create_operation_record(
                session_id="test_session_5",
                operation_type=OperationType.MOVE,
                source_path=Path(f"/source/file{i}.mp3"),
                target_path=Path(f"/target/file{i}.mp3")
            )
            for i in range(3)
        ]

        # Update status for some operations
        operations[0] = OperationRecord(
            **operations[0].__dict__,
            status=OperationStatus.COMPLETED
        )
        operations[1] = OperationRecord(
            **operations[1].__dict__,
            status=OperationStatus.FAILED,
            error_message="Test error"
        )

        # Record operations
        for op in operations:
            await history_tracker.record_operation(op)

        # Get all operations
        all_ops = await history_tracker.get_session_operations("test_session_5")
        assert len(all_ops) == 3

        # Get only completed operations
        completed_ops = await history_tracker.get_session_operations(
            "test_session_5",
            OperationStatus.COMPLETED
        )
        assert len(completed_ops) == 1
        assert completed_ops[0].status == OperationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_list_sessions(self, history_tracker):
        """Test listing recent sessions."""
        # Create multiple sessions
        for i in range(5):
            await history_tracker.start_session(
                f"session_{i}",
                Path(f"/source_{i}"),
                Path(f"/target_{i}")
            )
            await history_tracker.end_session(f"session_{i}")

        # List sessions
        sessions = await history_tracker.list_sessions(limit=3)
        assert len(sessions) == 3

        # Check ordering (should be most recent first)
        assert sessions[0].session_id == "session_4"
        assert sessions[1].session_id == "session_3"
        assert sessions[2].session_id == "session_2"

    @pytest.mark.asyncio
    async def test_delete_session(self, history_tracker):
        """Test deleting a session."""
        # Start a session with operations
        await history_tracker.start_session(
            "session_to_delete",
            Path("/source"),
            Path("/target")
        )

        operation = create_operation_record(
            session_id="session_to_delete",
            operation_type=OperationType.MOVE,
            source_path=Path("/source/file.mp3"),
            target_path=Path("/target/file.mp3")
        )
        await history_tracker.record_operation(operation)

        # Verify session exists
        session = await history_tracker.get_session("session_to_delete")
        assert session is not None

        # Delete session
        result = await history_tracker.delete_session("session_to_delete")
        assert result.is_success()

        # Verify deletion
        session = await history_tracker.get_session("session_to_delete")
        assert session is None

        operations = await history_tracker.get_session_operations("session_to_delete")
        assert len(operations) == 0


class TestOperationRecord:
    """Test cases for OperationRecord."""

    def test_to_dict(self):
        """Test converting operation record to dictionary."""
        timestamp = datetime.now()
        operation = OperationRecord(
            id="test_id",
            session_id="test_session",
            timestamp=timestamp,
            operation_type=OperationType.MOVE,
            source_path=Path("/source/file.mp3"),
            target_path=Path("/target/file.mp3"),
            backup_path=Path("/backup/file.mp3"),
            status=OperationStatus.COMPLETED,
            checksum_before="abc123",
            checksum_after="def456",
            file_size=1024,
            metadata={"test": True}
        )

        data = operation.to_dict()
        assert data["id"] == "test_id"
        assert data["session_id"] == "test_session"
        assert data["timestamp"] == timestamp.isoformat()
        assert data["operation_type"] == "move"
        assert data["source_path"] == "/source/file.mp3"
        assert data["target_path"] == "/target/file.mp3"
        assert data["backup_path"] == "/backup/file.mp3"
        assert data["status"] == "completed"
        assert data["checksum_before"] == "abc123"
        assert data["checksum_after"] == "def456"
        assert data["file_size"] == 1024
        assert data["metadata"]["test"] is True

    def test_from_dict(self):
        """Test creating operation record from dictionary."""
        data = {
            "id": "test_id",
            "session_id": "test_session",
            "timestamp": datetime.now().isoformat(),
            "operation_type": "move",
            "source_path": "/source/file.mp3",
            "target_path": "/target/file.mp3",
            "backup_path": "/backup/file.mp3",
            "status": "completed",
            "error_message": None,
            "checksum_before": "abc123",
            "checksum_after": "def456",
            "file_size": 1024,
            "metadata": {"test": True}
        }

        operation = OperationRecord.from_dict(data)
        assert operation.id == "test_id"
        assert operation.session_id == "test_session"
        assert operation.operation_type == OperationType.MOVE
        assert operation.source_path == Path("/source/file.mp3")
        assert operation.target_path == Path("/target/file.mp3")
        assert operation.backup_path == Path("/backup/file.mp3")
        assert operation.status == OperationStatus.COMPLETED
        assert operation.checksum_before == "abc123"
        assert operation.checksum_after == "def456"
        assert operation.file_size == 1024
        assert operation.metadata["test"] is True


class TestOperationRollbackService:
    """Test cases for OperationRollbackService."""

    @pytest.mark.asyncio
    async def test_rollback_session_dry_run(self, history_tracker, rollback_service, temp_dir):
        """Test rollback session in dry run mode."""
        # Create source and target directories
        source_dir = temp_dir / "source"
        target_dir = temp_dir / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        # Start a session
        await history_tracker.start_session(
            "rollback_test",
            source_dir,
            target_dir
        )

        # Create a test file
        test_file = source_dir / "test.mp3"
        test_file.write_bytes(b"test audio data")

        # Create operation record
        operation = create_operation_record(
            session_id="rollback_test",
            operation_type=OperationType.MOVE,
            source_path=test_file,
            target_path=target_dir / "test.mp3"
        )

        # Update to completed
        operation = OperationRecord(
            **operation.__dict__,
            status=OperationStatus.COMPLETED
        )

        await history_tracker.record_operation(operation)

        # Perform dry run rollback
        result = await rollback_service.rollback_session("rollback_test", dry_run=True)
        assert result.is_success()

        data = result.value()
        assert data["session_id"] == "rollback_test"
        assert data["total_operations"] == 1
        assert data["operations_to_rollback"] == [operation.id]
        assert data["dry_run"] is True

    @pytest.mark.asyncio
    async def test_rollback_session_actual(self, history_tracker, rollback_service, temp_dir):
        """Test actual rollback session."""
        # Create source and target directories
        source_dir = temp_dir / "source"
        target_dir = temp_dir / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        # Start a session
        await history_tracker.start_session(
            "rollback_test_2",
            source_dir,
            target_dir
        )

        # Create a test file
        test_file = source_dir / "test.mp3"
        test_file.write_bytes(b"test audio data")

        # Create operation record
        operation = create_operation_record(
            session_id="rollback_test_2",
            operation_type=OperationType.MOVE,
            source_path=test_file,
            target_path=target_dir / "test.mp3"
        )

        # Update to completed
        operation = OperationRecord(
            **operation.__dict__,
            status=OperationStatus.COMPLETED
        )

        await history_tracker.record_operation(operation)

        # Actually move the file (simulate the operation)
        target_file = target_dir / "test.mp3"
        shutil.move(str(test_file), str(target_file))
        assert target_file.exists()
        assert not test_file.exists()

        # Perform actual rollback
        result = await rollback_service.rollback_session("rollback_test_2", dry_run=False)
        assert result.is_success()

        data = result.value()
        assert data["session_id"] == "rollback_test_2"
        assert data["successful_rollbacks"] == 1
        assert data["failed_rollbacks"] == 0
        assert data["skipped_rollbacks"] == 0
        assert data["dry_run"] is False

        # Verify file was rolled back
        assert test_file.exists()
        assert not target_file.exists()

        # Verify operation status was updated
        operations = await history_tracker.get_session_operations("rollback_test_2")
        assert len(operations) == 1
        assert operations[0].status == OperationStatus.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_session(self, rollback_service):
        """Test rolling back a non-existent session."""
        result = await rollback_service.rollback_session("nonexistent")
        assert result.is_failure()
        assert "not found" in result.error().lower()


class TestEnhancedAsyncFileMover:
    """Test cases for EnhancedAsyncFileMover."""

    @pytest.mark.asyncio
    async def test_start_operation(self, history_tracker, temp_dir):
        """Test starting an operation with enhanced file mover."""
        mover = EnhancedAsyncFileMover(
            history_tracker=history_tracker,
            session_id="enhanced_test"
        )

        source_root = temp_dir / "source"
        target_root = temp_dir / "target"

        result = await mover.start_operation(source_root, target_root, {"test": True})
        assert result.is_success()

        session = result.value()
        assert session.session_id == "enhanced_test"
        assert session.status == "running"

    @pytest.mark.asyncio
    async def test_move_file_with_tracking(self, history_tracker, temp_dir):
        """Test moving a file with comprehensive tracking."""
        mover = EnhancedAsyncFileMover(
            history_tracker=history_tracker,
            session_id="move_test",
            backup_enabled=False  # Disable backup for simplicity
        )

        # Setup directories
        source_dir = temp_dir / "source"
        target_dir = temp_dir / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        await mover.start_operation(source_dir, target_dir)

        # Create test audio file
        source_file = source_dir / "test.mp3"
        source_file.write_bytes(b"fake mp3 data")

        # Create AudioFile object
        audio_file = AudioFile(
            path=source_file,
            title="Test Song",
            artist="Test Artist",
            album="Test Album",
            year=2023,
            genre="Test"
        )

        target_path = target_dir / "test.mp3"

        # Move file
        result = await mover.move_file(audio_file, target_path)
        assert result.is_success()

        # Verify file was moved
        assert not source_file.exists()
        assert target_path.exists()
        assert audio_file.path == target_path

        # Check operation history
        history_result = await mover.get_operation_history()
        assert history_result.is_success()

        operations = history_result.value()
        assert len(operations) == 1
        assert operations[0].operation_type == OperationType.MOVE
        assert operations[0].status == OperationStatus.COMPLETED

        await mover.finish_operation()

    @pytest.mark.asyncio
    async def test_move_file_with_checksum_verification(self, history_tracker, temp_dir):
        """Test moving a file with checksum verification."""
        mover = EnhancedAsyncFileMover(
            history_tracker=history_tracker,
            session_id="checksum_test",
            backup_enabled=False
        )

        # Setup directories
        source_dir = temp_dir / "source"
        target_dir = temp_dir / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        await mover.start_operation(source_dir, target_dir)

        # Create test audio file
        source_file = source_dir / "test.mp3"
        test_data = b"fake mp3 data for checksum test"
        source_file.write_bytes(test_data)

        # Create AudioFile object
        audio_file = AudioFile(
            path=source_file,
            title="Test Song",
            artist="Test Artist"
        )

        target_path = target_dir / "test.mp3"

        # Move file with checksum verification
        result = await mover.move_file(audio_file, target_path, verify_checksum=True)
        assert result.is_success()

        # Check operation history has checksums
        history_result = await mover.get_operation_history()
        operations = history_result.value()

        assert len(operations) == 1
        operation = operations[0]
        assert operation.checksum_before is not None
        assert operation.checksum_after is not None
        assert operation.checksum_before == operation.checksum_after

        await mover.finish_operation()

    @pytest.mark.asyncio
    async def test_get_session_summary(self, history_tracker, temp_dir):
        """Test getting session summary."""
        mover = EnhancedAsyncFileMover(
            history_tracker=history_tracker,
            session_id="summary_test",
            backup_enabled=False
        )

        # Setup directories
        source_dir = temp_dir / "source"
        target_dir = temp_dir / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        await mover.start_operation(source_dir, target_dir)

        # Create and move a test file
        source_file = source_dir / "test.mp3"
        source_file.write_bytes(b"test data")

        audio_file = AudioFile(
            path=source_file,
            title="Test",
            artist="Artist"
        )

        target_path = target_dir / "test.mp3"
        await mover.move_file(audio_file, target_path)

        # Get session summary
        result = await mover.get_session_summary()
        assert result.is_success()

        summary = result.value()
        assert summary["session_id"] == "summary_test"
        assert summary["total_operations"] == 1
        assert summary["completed_operations"] == 1
        assert summary["failed_operations"] == 0
        assert summary["operation_types"]["move"] == 1

        await mover.finish_operation()


class TestOperationSessionContextManager:
    """Test cases for operation session context manager."""

    @pytest.mark.asyncio
    async def test_operation_session_context(self, history_tracker, temp_dir):
        """Test using the operation session context manager."""
        source_root = temp_dir / "source"
        target_root = temp_dir / "target"

        async with operation_session(
            history_tracker,
            session_id="context_test",
            source_root=source_root,
            target_root=target_root,
            metadata={"context": True}
        ) as session:
            assert session.session_id == "context_test"
            assert session.metadata["context"] is True

            # Create an operation record
            operation = create_operation_record(
                session_id="context_test",
                operation_type=OperationType.MOVE,
                source_path=source_root / "file.mp3",
                target_path=target_root / "file.mp3"
            )

            operation = OperationRecord(
                **operation.__dict__,
                status=OperationStatus.COMPLETED
            )

            await history_tracker.record_operation(operation)

        # Verify session was ended
        session = await history_tracker.get_session("context_test")
        assert session is not None
        assert session.status == "completed"
        assert session.end_time is not None