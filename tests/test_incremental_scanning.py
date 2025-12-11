"""Tests for incremental scanning functionality."""

import unittest
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import json

from music_organizer.core.scan_tracker import ScanTracker, ScanRecord
from music_organizer.core.incremental_scanner import IncrementalScanner
from music_organizer.exceptions import MusicOrganizerError


class TestScanTracker(unittest.TestCase):
    """Test cases for ScanTracker."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tracker = ScanTracker(cache_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        # Clean up database
        if hasattr(self.tracker, 'db_path') and self.tracker.db_path.exists():
            self.tracker.db_path.unlink()

    def test_singleton_pattern(self):
        """Test that ScanTracker follows singleton pattern."""
        tracker2 = ScanTracker(cache_dir=self.temp_dir)
        self.assertIs(self.tracker, tracker2)

    def test_scan_session_management(self):
        """Test scan session creation and completion."""
        directory = Path("/test/directory")

        # Start session
        scan_id = self.tracker.start_scan_session(directory)
        self.assertIsNotNone(scan_id)
        self.assertTrue(scan_id.startswith("scan_"))

        # Complete session
        stats = {'scanned': 10, 'modified': 3, 'added': 2, 'removed': 0}
        self.tracker.complete_scan_session(scan_id, stats)

        # Verify session was recorded
        last_scan = self.tracker.get_last_scan_time(directory)
        self.assertIsNotNone(last_scan)

    def test_file_record_tracking(self):
        """Test file record tracking."""
        # Create a temporary file
        test_file = self.temp_dir / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        scan_id = self.tracker.start_scan_session(self.temp_dir)

        # Update file record
        self.tracker.update_file_record(test_file, scan_id)

        # Get scan statistics
        stats = self.tracker.get_scan_statistics(self.temp_dir)
        self.assertEqual(stats['total_files_tracked'], 1)

    def test_get_modified_files_new_directory(self):
        """Test getting modified files for new directory."""
        # Create some test files
        for i in range(3):
            test_file = self.temp_dir / f"test{i}.mp3"
            test_file.write_bytes(b"fake audio data")

        # Get modified files (no previous scan)
        modified = self.tracker.get_modified_files(self.temp_dir)
        self.assertEqual(len(modified), 3)

    def test_get_modified_files_incremental(self):
        """Test incremental file modification detection."""
        # Create test files
        test_file = self.temp_dir / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        # First scan
        scan_id = self.tracker.start_scan_session(self.temp_dir)
        self.tracker.update_file_record(test_file, scan_id)
        self.tracker.complete_scan_session(scan_id, {})

        # Get modified files (should be empty)
        modified = self.tracker.get_modified_files(self.temp_dir)
        self.assertEqual(len(modified), 0)

        # Modify the file
        import time
        time.sleep(0.1)  # Ensure different timestamp
        test_file.write_bytes(b"modified fake audio data")

        # Get modified files (should include the modified file)
        modified = self.tracker.get_modified_files(self.temp_dir)
        self.assertEqual(len(modified), 1)
        self.assertEqual(modified[0], test_file)

    def test_cleanup_old_records(self):
        """Test cleanup of old scan records."""
        # Create a file and record it
        test_file = self.temp_dir / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        scan_id = self.tracker.start_scan_session(self.temp_dir)
        self.tracker.update_file_record(test_file, scan_id)
        self.tracker.complete_scan_session(scan_id, {})

        # Manually insert old record
        with sqlite3.connect(self.tracker.db_path) as conn:
            old_date = datetime.now() - timedelta(days=100)
            conn.execute(
                "INSERT INTO scan_history VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("/old/file.mp3", 0, 0, old_date, "old_scan", None, "/old")
            )

        # Cleanup records older than 90 days
        removed = self.tracker.cleanup_old_records(days=90)
        self.assertGreater(removed, 0)

    def test_clear_directory(self):
        """Test clearing scan records for a directory."""
        # Create test file and record it
        test_file = self.temp_dir / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        scan_id = self.tracker.start_scan_session(self.temp_dir)
        self.tracker.update_file_record(test_file, scan_id)
        self.tracker.complete_scan_session(scan_id, {})

        # Verify we have records
        stats = self.tracker.get_scan_statistics(self.temp_dir)
        self.assertGreater(stats['total_files_tracked'], 0)

        # Clear directory
        self.tracker.clear_directory(self.temp_dir)

        # Verify records are gone
        stats = self.tracker.get_scan_statistics(self.temp_dir)
        self.assertEqual(stats['total_files_tracked'], 0)


class TestIncrementalScanner(unittest.TestCase):
    """Test cases for IncrementalScanner."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.scanner = IncrementalScanner(cache_dir=self.temp_dir)

        # Create some test audio files
        self.audio_files = []
        for i in range(3):
            file_path = self.temp_dir / f"test{i}.flac"
            file_path.write_bytes(b"fake flac data")
            self.audio_files.append(file_path)

    def tearDown(self):
        """Clean up test environment."""
        # Clean up test files
        for file_path in self.audio_files:
            if file_path.exists():
                file_path.unlink()

        # Clean up database
        if hasattr(self.scanner, 'scan_tracker'):
            if hasattr(self.scanner.scan_tracker, 'db_path'):
                if self.scanner.scan_tracker.db_path.exists():
                    self.scanner.scan_tracker.db_path.unlink()

    def test_full_scan(self):
        """Test full directory scanning."""
        async def test_scan():
            files = []
            async for file_path, is_modified in self.scanner.scan_directory_incremental(
                self.temp_dir, force_full=True
            ):
                files.append(file_path)

            self.assertEqual(len(files), 3)
            for file_path in files:
                self.assertTrue(file_path.suffix.lower() in {'.flac', '.mp3', '.wav'})

        asyncio.run(test_scan())

    def test_incremental_scan_first_time(self):
        """Test incremental scan on first run (should scan all files)."""
        async def test_scan():
            files = []
            async for file_path, is_modified in self.scanner.scan_directory_incremental(
                self.temp_dir, force_full=False
            ):
                files.append(file_path)
                # All files should be marked as modified on first scan
                self.assertTrue(is_modified)

            self.assertEqual(len(files), 3)

        asyncio.run(test_scan())

    def test_incremental_scan_no_changes(self):
        """Test incremental scan with no file changes."""
        async def test_scan():
            # First scan
            files1 = []
            async for file_path, is_modified in self.scanner.scan_directory_incremental(
                self.temp_dir, force_full=False
            ):
                files1.append(file_path)

            # Second scan (no changes)
            files2 = []
            modified_count = 0
            async for file_path, is_modified in self.scanner.scan_directory_incremental(
                self.temp_dir, force_full=False
            ):
                files2.append(file_path)
                if is_modified:
                    modified_count += 1

            # Should still find all files but none marked as modified
            self.assertEqual(len(files2), 3)
            self.assertEqual(modified_count, 0)

        asyncio.run(test_scan())

    def test_incremental_scan_with_modifications(self):
        """Test incremental scan with file modifications."""
        async def test_scan():
            # First scan
            files1 = []
            async for file_path, is_modified in self.scanner.scan_directory_incremental(
                self.temp_dir, force_full=False
            ):
                files1.append(file_path)

            # Modify one file
            import time
            time.sleep(0.1)  # Ensure different timestamp
            self.audio_files[0].write_bytes(b"modified fake flac data")

            # Second scan
            modified_files = []
            async for file_path, is_modified in self.scanner.scan_directory_incremental(
                self.temp_dir, force_full=False
            ):
                if is_modified:
                    modified_files.append(file_path)

            # Should find only the modified file
            self.assertEqual(len(modified_files), 1)
            self.assertEqual(modified_files[0], self.audio_files[0])

        asyncio.run(test_scan())

    def test_batch_scanning(self):
        """Test batch scanning functionality."""
        async def test_scan():
            batches = []
            async for batch in self.scanner.scan_directory_batch_incremental(
                self.temp_dir, batch_size=2, filter_modified=True
            ):
                batches.append(batch)

            # Should have 2 batches: 2 files + 1 file
            self.assertEqual(len(batches), 2)
            self.assertEqual(len(batches[0]), 2)
            self.assertEqual(len(batches[1]), 1)

        asyncio.run(test_scan())

    def test_get_scan_info(self):
        """Test getting scan information."""
        # No scan yet
        info = self.scanner.get_scan_info(self.temp_dir)
        self.assertIsNone(info['last_scan'])
        self.assertFalse(info['has_history'])

        # Perform a scan
        async def test_scan():
            files = []
            async for file_path, is_modified in self.scanner.scan_directory_incremental(
                self.temp_dir, force_full=True
            ):
                files.append(file_path)

        asyncio.run(test_scan())

        # Should have scan info now
        info = self.scanner.get_scan_info(self.temp_dir)
        self.assertIsNotNone(info['last_scan'])
        self.assertTrue(info['has_history'])
        self.assertEqual(info['files_tracked'], 3)

    def test_force_full_scan(self):
        """Test forcing a full scan."""
        async def test_scan():
            # First scan
            files1 = []
            async for file_path, is_modified in self.scanner.scan_directory_incremental(
                self.temp_dir, force_full=False
            ):
                files1.append(file_path)

            # Clear scan history
            self.scanner.force_full_scan_next(self.temp_dir)

            # Second scan should be full
            modified_files = []
            async for file_path, is_modified in self.scanner.scan_directory_incremental(
                self.temp_dir, force_full=False
            ):
                modified_files.append(file_path)
                # All files should be marked as modified
                self.assertTrue(is_modified)

            self.assertEqual(len(modified_files), 3)

        asyncio.run(test_scan())

    def test_invalid_directory(self):
        """Test handling of invalid directory."""
        async def test_scan():
            with self.assertRaises(MusicOrganizerError):
                async for _ in self.scanner.scan_directory_incremental(
                    Path("/nonexistent/directory"), force_full=True
                ):
                    pass

        asyncio.run(test_scan())

    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        # Test with existing file
        hash1 = self.scanner._calculate_file_hash(self.audio_files[0])
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 32)  # MD5 hash length

        # Test with modified file
        import time
        time.sleep(0.1)
        self.audio_files[0].write_bytes(b"modified data")
        hash2 = self.scanner._calculate_file_hash(self.audio_files[0])
        self.assertNotEqual(hash1, hash2)

        # Test quick vs full hash
        quick_hash = self.scanner._calculate_file_hash(self.audio_files[0], quick=True)
        full_hash = self.scanner._calculate_file_hash(self.audio_files[0], quick=False)
        self.assertIsInstance(quick_hash, str)
        self.assertIsInstance(full_hash, str)


if __name__ == '__main__':
    unittest.main()