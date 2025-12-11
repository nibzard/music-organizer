"""Scan tracking for incremental scanning functionality.

Tracks file modification times and scan history to enable incremental scans
that only process new or modified files.
"""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ScanRecord:
    """Represents a scan record for a file."""
    file_path: str
    file_mtime: float
    file_size: int
    last_scanned: datetime
    scan_id: str
    hash_checksum: Optional[str] = None  # For quick change detection


class ScanTracker:
    """Tracks file scans to enable incremental scanning."""

    _instance: Optional[ScanTracker] = None
    _lock = threading.Lock()

    def __new__(cls, cache_dir: Optional[Path] = None) -> ScanTracker:
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize the scan tracker."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Default cache directory
            if cache_dir is None:
                cache_dir = Path.home() / ".cache" / "music-organizer"

            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # SQLite database path
            self.db_path = self.cache_dir / "scan_history.db"

            # Initialize database
            self._init_db()

            self._initialized = True

    def _init_db(self) -> None:
        """Initialize SQLite database with schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Create scan history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_history (
                    file_path TEXT PRIMARY KEY,
                    file_mtime REAL NOT NULL,
                    file_size INTEGER NOT NULL,
                    last_scanned TIMESTAMP NOT NULL,
                    scan_id TEXT NOT NULL,
                    hash_checksum TEXT,
                    directory_path TEXT NOT NULL
                )
            """)

            # Create scan sessions table for tracking complete scans
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_sessions (
                    scan_id TEXT PRIMARY KEY,
                    directory_path TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    files_scanned INTEGER DEFAULT 0,
                    files_modified INTEGER DEFAULT 0,
                    files_added INTEGER DEFAULT 0,
                    files_removed INTEGER DEFAULT 0
                )
            """)

            # Create indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_history_file_path
                ON scan_history(file_path)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_history_directory
                ON scan_history(directory_path)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_history_last_scanned
                ON scan_history(last_scanned)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_sessions_directory
                ON scan_sessions(directory_path)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_sessions_started_at
                ON scan_sessions(started_at)
            """)

            conn.commit()

    def start_scan_session(self, directory_path: Path) -> str:
        """Start a new scan session.

        Args:
            directory_path: Path to the directory being scanned

        Returns:
            Scan session ID
        """
        scan_id = f"scan_{datetime.now().isoformat()}_{hash(str(directory_path))}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO scan_sessions
                (scan_id, directory_path, started_at)
                VALUES (?, ?, ?)
                """,
                (scan_id, str(directory_path), datetime.now())
            )
            conn.commit()

        logger.debug(f"Started scan session {scan_id} for {directory_path}")
        return scan_id

    def complete_scan_session(self, scan_id: str, stats: Dict[str, int]) -> None:
        """Complete a scan session with statistics.

        Args:
            scan_id: The scan session ID
            stats: Dictionary with scan statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE scan_sessions
                SET completed_at = ?,
                    files_scanned = ?,
                    files_modified = ?,
                    files_added = ?,
                    files_removed = ?
                WHERE scan_id = ?
                """,
                (
                    datetime.now(),
                    stats.get('scanned', 0),
                    stats.get('modified', 0),
                    stats.get('added', 0),
                    stats.get('removed', 0),
                    scan_id
                )
            )
            conn.commit()

        logger.debug(f"Completed scan session {scan_id}: {stats}")

    def update_file_record(self, file_path: Path, scan_id: str,
                          hash_checksum: Optional[str] = None) -> None:
        """Update or create a scan record for a file.

        Args:
            file_path: Path to the file
            scan_id: Current scan session ID
            hash_checksum: Optional hash for quick change detection
        """
        try:
            stat = file_path.stat()
            file_mtime = stat.st_mtime
            file_size = stat.st_size
        except (OSError, FileNotFoundError):
            logger.warning(f"Could not stat file: {file_path}")
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scan_history
                (file_path, file_mtime, file_size, last_scanned, scan_id,
                 hash_checksum, directory_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(file_path),
                    file_mtime,
                    file_size,
                    datetime.now(),
                    scan_id,
                    hash_checksum,
                    str(file_path.parent)
                )
            )
            conn.commit()

    def get_modified_files(self, directory_path: Path,
                          since: Optional[datetime] = None) -> List[Path]:
        """Get files that have been modified since last scan.

        Args:
            directory_path: Directory to check
            since: Only check modifications since this time (uses last scan if None)

        Returns:
            List of modified file paths
        """
        modified_files = []

        # Get all files in directory
        try:
            all_files = []
            for file_path in directory_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in {
                    '.flac', '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.opus', '.wma'
                }:
                    all_files.append(file_path)
        except (OSError, PermissionError) as e:
            logger.error(f"Error scanning directory {directory_path}: {e}")
            return []

        if not all_files:
            return []

        # Get last scan time for directory if not specified
        if since is None:
            since = self.get_last_scan_time(directory_path)

        # If no previous scan, all files are considered new
        if since is None:
            logger.info(f"No previous scan found for {directory_path}, all {len(all_files)} files are new")
            return all_files

        # Query database for files in this directory
        directory_str = str(directory_path)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT file_path, file_mtime, file_size
                FROM scan_history
                WHERE directory_path = ? AND last_scanned >= ?
                """,
                (directory_str, since)
            )
            stored_files = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

        # Check each file for modifications
        for file_path in all_files:
            file_path_str = str(file_path)

            try:
                stat = file_path.stat()
                current_mtime = stat.st_mtime
                current_size = stat.st_size

                # Check if file is new or modified
                if file_path_str not in stored_files:
                    # New file
                    modified_files.append(file_path)
                else:
                    stored_mtime, stored_size = stored_files[file_path_str]
                    # Check if file was modified
                    if current_mtime != stored_mtime or current_size != stored_size:
                        modified_files.append(file_path)

            except (OSError, FileNotFoundError):
                # File might have been deleted, skip it
                continue

        # Check for files that were removed (optional, for reporting)
        # This would require a separate query to find files in DB but not on disk

        logger.info(f"Found {len(modified_files)} modified/new files in {directory_path}")
        return modified_files

    def get_last_scan_time(self, directory_path: Path) -> Optional[datetime]:
        """Get the last scan time for a directory.

        Args:
            directory_path: Directory to check

        Returns:
            Last scan datetime or None if never scanned
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT MAX(started_at)
                FROM scan_sessions
                WHERE directory_path = ? AND completed_at IS NOT NULL
                """,
                (str(directory_path),)
            )
            result = cursor.fetchone()

            if result and result[0]:
                # Convert string back to datetime
                return datetime.fromisoformat(result[0])

            return None

    def get_scan_statistics(self, directory_path: Optional[Path] = None) -> Dict[str, Any]:
        """Get scan statistics.

        Args:
            directory_path: Optional directory to filter stats

        Returns:
            Dictionary with scan statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total files tracked
            if directory_path:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM scan_history WHERE directory_path = ?",
                    (str(directory_path),)
                )
                total_files = cursor.fetchone()[0]

                # Last scan time
                cursor = conn.execute(
                    """
                    SELECT MAX(started_at)
                    FROM scan_sessions
                    WHERE directory_path = ? AND completed_at IS NOT NULL
                    """,
                    (str(directory_path),)
                )
                last_scan = cursor.fetchone()[0]
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM scan_history")
                total_files = cursor.fetchone()[0]

                cursor = conn.execute(
                    "SELECT MAX(started_at) FROM scan_sessions WHERE completed_at IS NOT NULL"
                )
                last_scan = cursor.fetchone()[0]

            # Recent scan sessions
            cursor = conn.execute(
                """
                SELECT directory_path, started_at, files_scanned, files_modified, files_added
                FROM scan_sessions
                WHERE completed_at IS NOT NULL
                ORDER BY started_at DESC
                LIMIT 10
                """
            )
            recent_scans = [
                {
                    'directory': row[0],
                    'started_at': row[1],
                    'files_scanned': row[2],
                    'files_modified': row[3],
                    'files_added': row[4]
                }
                for row in cursor.fetchall()
            ]

            return {
                'total_files_tracked': total_files,
                'last_scan': last_scan,
                'recent_scans': recent_scans
            }

    def cleanup_old_records(self, days: int = 90) -> int:
        """Clean up old scan records.

        Args:
            days: Remove records older than this many days

        Returns:
            Number of records removed
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            # Remove old scan history
            cursor = conn.execute(
                "DELETE FROM scan_history WHERE last_scanned < ?",
                (cutoff_date,)
            )
            history_removed = cursor.rowcount

            # Remove old scan sessions
            cursor = conn.execute(
                "DELETE FROM scan_sessions WHERE started_at < ?",
                (cutoff_date,)
            )
            sessions_removed = cursor.rowcount

            conn.commit()

            total_removed = history_removed + sessions_removed
            logger.info(f"Cleaned up {total_removed} old scan records")

            return total_removed

    def clear_directory(self, directory_path: Path) -> None:
        """Clear all scan records for a directory.

        Args:
            directory_path: Directory to clear records for
        """
        with sqlite3.connect(self.db_path) as conn:
            # Remove scan history
            conn.execute(
                "DELETE FROM scan_history WHERE directory_path = ?",
                (str(directory_path),)
            )

            # Remove scan sessions
            conn.execute(
                "DELETE FROM scan_sessions WHERE directory_path = ?",
                (str(directory_path),)
            )

            conn.commit()

        logger.info(f"Cleared scan records for {directory_path}")

    def close(self) -> None:
        """Close database connections."""
        # SQLite connections are closed automatically with context managers
        pass

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()