#!/usr/bin/env python3
"""Simple test for incremental scanning core functionality."""

import sys
import tempfile
import sqlite3
from pathlib import Path
import time
from datetime import datetime, timedelta

# Direct imports without going through the full package
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import directly
from music_organizer.core.scan_tracker import ScanTracker, ScanRecord
from music_organizer.core.incremental_scanner import IncrementalScanner


def test_sqlite_schema():
    """Test that the SQLite schema is created correctly."""
    print("\n=== Testing SQLite Schema ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Initialize tracker
        tracker = ScanTracker(cache_dir=temp_path)

        # Verify database exists and has correct tables
        with sqlite3.connect(tracker.db_path) as conn:
            # Check tables exist
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            print(f"Tables created: {tables}")
            assert "scan_history" in tables, "scan_history table should exist"
            assert "scan_sessions" in tables, "scan_sessions table should exist"

            # Check scan_history table schema
            cursor = conn.execute("PRAGMA table_info(scan_history)")
            columns = [row[1] for row in cursor.fetchall()]
            print(f"scan_history columns: {columns}")

            required_columns = [
                "file_path", "file_mtime", "file_size", "last_scanned",
                "scan_id", "directory_path"
            ]
            for col in required_columns:
                assert col in columns, f"Column {col} should exist in scan_history"

            # Check scan_sessions table schema
            cursor = conn.execute("PRAGMA table_info(scan_sessions)")
            columns = [row[1] for row in cursor.fetchall()]
            print(f"scan_sessions columns: {columns}")

            required_columns = [
                "scan_id", "directory_path", "started_at", "completed_at"
            ]
            for col in required_columns:
                assert col in columns, f"Column {col} should exist in scan_sessions"

        print("✅ SQLite schema test passed!")


def test_scan_tracker_basic():
    """Test basic scan tracker functionality without audio files."""
    print("\n=== Testing ScanTracker Basic Operations ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Initialize tracker
        tracker = ScanTracker(cache_dir=temp_path)

        # Test singleton pattern
        tracker2 = ScanTracker(cache_dir=temp_path)
        assert tracker is tracker2, "ScanTracker should be singleton"

        # Create a test directory and files
        test_dir = temp_path / "test"
        test_dir.mkdir()

        # Create some test files (not audio files, just for testing)
        file1 = test_dir / "file1.txt"
        file2 = test_dir / "file2.txt"
        file1.write_text("test1")
        file2.write_text("test2")

        print(f"Created test files: {file1}, {file2}")

        # Test scan session
        scan_id = tracker.start_scan_session(test_dir)
        print(f"Started scan session: {scan_id}")
        assert scan_id.startswith("scan_"), "Scan ID should start with 'scan_'"

        # Test file record tracking
        tracker.update_file_record(file1, scan_id)
        tracker.update_file_record(file2, scan_id)

        # Complete session
        stats = {'scanned': 2, 'modified': 2, 'added': 2, 'removed': 0}
        tracker.complete_scan_session(scan_id, stats)

        # Test statistics
        stats = tracker.get_scan_statistics(test_dir)
        print(f"Scan statistics: {stats}")
        assert stats['total_files_tracked'] == 2, "Should track 2 files"
        assert stats['last_scan'] is not None, "Should have last scan time"

        # Test last scan time
        last_scan = tracker.get_last_scan_time(test_dir)
        print(f"Last scan time: {last_scan}")
        assert last_scan is not None, "Should have last scan time"

        # Test getting modified files (no previous scan data for these files)
        modified = tracker.get_modified_files(test_dir)
        print(f"Modified files (no filter): {len(modified)}")
        # Should return all files since they're not audio files (we're not filtering here)

        # Clear directory
        tracker.clear_directory(test_dir)
        stats = tracker.get_scan_statistics(test_dir)
        assert stats['total_files_tracked'] == 0, "Should have 0 files after clear"

        print("✅ ScanTracker basic operations test passed!")


def test_incremental_scanner_direct():
    """Test incremental scanner directly without audio files."""
    print("\n=== Testing IncrementalScanner Direct ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        test_dir = temp_path / "music"
        test_dir.mkdir()

        # Create files with audio extensions
        file1 = test_dir / "song1.flac"
        file2 = test_dir / "song2.mp3"
        file1.write_bytes(b"fake flac data 1")
        file2.write_bytes(b"fake mp3 data 2")

        print(f"Created audio files: {file1}, {file2}")

        # Initialize scanner
        scanner = IncrementalScanner(cache_dir=temp_path)

        # Test get_scan_info before any scan
        info = scanner.get_scan_info(test_dir)
        print(f"Initial scan info: {info}")
        assert info['has_history'] is False, "Should have no history initially"

        # Test file hash calculation
        hash1 = scanner._calculate_file_hash(file1, quick=True)
        hash2 = scanner._calculate_file_hash(file1, quick=False)
        print(f"Quick hash: {hash1}")
        print(f"Full hash: {hash2}")
        assert len(hash1) == 32, "Quick hash should be 32 chars (MD5)"
        assert len(hash2) == 32, "Full hash should be 32 chars (MD5)"

        # Test force full scan
        scanner.force_full_scan_next(test_dir)
        info = scanner.get_scan_info(test_dir)
        print(f"After force full scan: {info}")

        print("✅ IncrementalScanner direct test passed!")


def main():
    """Run all tests."""
    print("Starting simple incremental scanning tests...")

    try:
        test_sqlite_schema()
        test_scan_tracker_basic()
        test_incremental_scanner_direct()
        print("\n🎉 All tests passed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())