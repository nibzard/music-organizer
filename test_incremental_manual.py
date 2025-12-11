#!/usr/bin/env python3
"""Manual test script for incremental scanning functionality."""

import sys
import tempfile
from pathlib import Path
import shutil
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from music_organizer.core.scan_tracker import ScanTracker
from music_organizer.core.incremental_scanner import IncrementalScanner


def test_scan_tracker():
    """Test basic scan tracker functionality."""
    print("\n=== Testing ScanTracker ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Initialize tracker
        tracker = ScanTracker(cache_dir=temp_path)

        # Create test files
        test_dir = temp_path / "music"
        test_dir.mkdir()

        file1 = test_dir / "song1.flac"
        file2 = test_dir / "song2.mp3"
        file1.write_bytes(b"fake flac data 1")
        file2.write_bytes(b"fake mp3 data 2")

        print(f"Created test files: {file1}, {file2}")

        # Test first scan (should find all files)
        print("\n--- First scan ---")
        modified = tracker.get_modified_files(test_dir)
        print(f"Modified files found: {len(modified)}")
        assert len(modified) == 2, "Should find 2 files on first scan"

        # Start a scan session and record files
        scan_id = tracker.start_scan_session(test_dir)
        print(f"Started scan session: {scan_id}")

        tracker.update_file_record(file1, scan_id)
        tracker.update_file_record(file2, scan_id)

        stats = {'scanned': 2, 'modified': 2, 'added': 2, 'removed': 0}
        tracker.complete_scan_session(scan_id, stats)

        # Test second scan (should find no modified files)
        print("\n--- Second scan (no changes) ---")
        modified = tracker.get_modified_files(test_dir)
        print(f"Modified files found: {len(modified)}")
        assert len(modified) == 0, "Should find 0 files on second scan"

        # Modify one file
        time.sleep(0.1)  # Ensure different timestamp
        file1.write_bytes(b"modified flac data")
        print(f"\nModified: {file1}")

        # Test third scan (should find modified file)
        print("\n--- Third scan (one file modified) ---")
        modified = tracker.get_modified_files(test_dir)
        print(f"Modified files found: {len(modified)}")
        assert len(modified) == 1, "Should find 1 modified file"
        assert modified[0] == file1, "Should be the modified file"

        # Test scan info
        info = tracker.get_scan_statistics(test_dir)
        print(f"\nScan statistics:")
        print(f"  Total files tracked: {info['total_files_tracked']}")
        print(f"  Last scan: {info['last_scan']}")

        print("✅ ScanTracker tests passed!")


def test_incremental_scanner():
    """Test incremental scanner functionality."""
    print("\n=== Testing IncrementalScanner ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Initialize scanner
        scanner = IncrementalScanner(cache_dir=temp_path)

        # Create test files
        test_dir = temp_path / "music"
        test_dir.mkdir()

        file1 = test_dir / "song1.flac"
        file2 = test_dir / "song2.mp3"
        file3 = test_dir / "song3.wav"
        file1.write_bytes(b"fake flac data 1")
        file2.write_bytes(b"fake mp3 data 2")
        file3.write_bytes(b"fake wav data 3")

        print(f"Created test files: {file1}, {file2}, {file3}")

        # Test first scan (full scan)
        import asyncio

        async def test_scans():
            print("\n--- First scan (incremental mode) ---")
            files = []
            modified_count = 0
            async for file_path, is_modified in scanner.scan_directory_incremental(
                test_dir, force_full=False
            ):
                files.append(file_path)
                if is_modified:
                    modified_count += 1
                print(f"  {file_path.name} (modified: {is_modified})")

            print(f"Total files found: {len(files)}")
            print(f"Files marked as modified: {modified_count}")
            assert len(files) == 3, "Should find 3 files"
            assert modified_count == 3, "All files should be marked as modified on first scan"

            # Test second scan (incremental - no changes)
            print("\n--- Second scan (incremental, no changes) ---")
            files = []
            modified_count = 0
            async for file_path, is_modified in scanner.scan_directory_incremental(
                test_dir, force_full=False
            ):
                files.append(file_path)
                if is_modified:
                    modified_count += 1
                print(f"  {file_path.name} (modified: {is_modified})")

            print(f"Total files found: {len(files)}")
            print(f"Files marked as modified: {modified_count}")
            assert len(files) == 3, "Should still find 3 files"
            assert modified_count == 0, "No files should be marked as modified"

            # Modify a file
            time.sleep(0.1)  # Ensure different timestamp
            file2.write_bytes(b"modified mp3 data")
            print(f"\nModified: {file2}")

            # Test third scan (incremental - with changes)
            print("\n--- Third scan (incremental, one file changed) ---")
            files = []
            modified_files = []
            async for file_path, is_modified in scanner.scan_directory_incremental(
                test_dir, force_full=False
            ):
                files.append(file_path)
                if is_modified:
                    modified_files.append(file_path)
                print(f"  {file_path.name} (modified: {is_modified})")

            print(f"Total files found: {len(files)}")
            print(f"Files marked as modified: {len(modified_files)}")
            assert len(files) == 3, "Should still find 3 files"
            assert len(modified_files) == 1, "Should find 1 modified file"
            assert modified_files[0] == file2, "Should be the modified file"

            # Test batch scanning
            print("\n--- Batch scanning (modified files only) ---")
            batch_count = 0
            total_files = 0
            async for batch in scanner.scan_directory_batch_incremental(
                test_dir, batch_size=2, filter_modified=True
            ):
                batch_count += 1
                total_files += len(batch)
                print(f"  Batch {batch_count}: {len(batch)} files")

            print(f"Total batches: {batch_count}")
            print(f"Total files in batches: {total_files}")

            # Test scan info
            info = scanner.get_scan_info(test_dir)
            print(f"\nScan info:")
            print(f"  Directory: {info['directory']}")
            print(f"  Last scan: {info['last_scan']}")
            print(f"  Files tracked: {info['files_tracked']}")
            print(f"  Has history: {info['has_history']}")

            # Test force full scan
            print("\n--- Force full scan ---")
            scanner.force_full_scan_next(test_dir)

            files = []
            modified_count = 0
            async for file_path, is_modified in scanner.scan_directory_incremental(
                test_dir, force_full=False
            ):
                files.append(file_path)
                if is_modified:
                    modified_count += 1

            print(f"Files after force full: {len(files)}")
            print(f"Modified after force full: {modified_count}")
            assert modified_count == 3, "All files should be marked as modified after force full"

            print("✅ IncrementalScanner tests passed!")

        asyncio.run(test_scans())


def main():
    """Run all tests."""
    print("Starting incremental scanning manual tests...")

    try:
        test_scan_tracker()
        test_incremental_scanner()
        print("\n🎉 All tests passed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())