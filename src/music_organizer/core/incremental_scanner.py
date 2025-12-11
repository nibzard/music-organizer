"""Incremental scanner for processing only new or modified files.

Integrates with the scan tracker to provide efficient incremental scanning
that avoids reprocessing unchanged files.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Set, Tuple
import logging
from datetime import datetime

from .scan_tracker import ScanTracker
from ..exceptions import MusicOrganizerError

logger = logging.getLogger(__name__)


class IncrementalScanner:
    """Scanner that supports incremental scanning functionality."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the incremental scanner.

        Args:
            cache_dir: Directory for scan tracking cache
        """
        self.scan_tracker = ScanTracker(cache_dir)
        self.audio_extensions = {
            '.flac', '.mp3', '.wav', '.m4a', '.aac',
            '.ogg', '.opus', '.wma', '.m4p', '.aiff'
        }

    def _calculate_file_hash(self, file_path: Path, quick: bool = True) -> str:
        """Calculate hash for quick change detection.

        Args:
            file_path: File to hash
            quick: If True, use metadata-based quick hash (faster)

        Returns:
            Hex string of the hash
        """
        if quick:
            # Quick hash using file stats (size, mtime) and partial content
            try:
                stat = file_path.stat()
                hash_input = f"{file_path}_{stat.st_size}_{stat.st_mtime}"

                # Add first 1KB of content for better detection
                try:
                    with open(file_path, 'rb') as f:
                        hash_input += f.read(1024).hex()
                except (OSError, PermissionError):
                    pass

                return hashlib.md5(hash_input.encode()).hexdigest()
            except (OSError, FileNotFoundError):
                return hashlib.md5(str(file_path).encode()).hexdigest()
        else:
            # Full file hash (slower but more accurate)
            hash_md5 = hashlib.md5()
            try:
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                return hash_md5.hexdigest()
            except (OSError, FileNotFoundError, PermissionError):
                return hashlib.md5(str(file_path).encode()).hexdigest()

    async def scan_directory_incremental(
        self,
        directory: Path,
        force_full: bool = False,
        quick_hash: bool = True
    ) -> AsyncGenerator[Tuple[Path, bool], None]:
        """Scan directory incrementally, yielding (file_path, is_modified) tuples.

        Args:
            directory: Directory to scan
            force_full: Force full scan instead of incremental
            quick_hash: Use quick hash for change detection

        Yields:
            Tuple of (file_path, is_modified) where is_modified indicates
            if the file is new or has been modified since last scan
        """
        if not directory.exists():
            raise MusicOrganizerError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise MusicOrganizerError(f"Path is not a directory: {directory}")

        # Start scan session
        scan_id = self.scan_tracker.start_scan_session(directory)

        try:
            if force_full:
                logger.info(f"Performing full scan of {directory}")
                async for file_path in self._full_scan(directory):
                    # Update scan record
                    file_hash = self._calculate_file_hash(file_path, quick_hash)
                    self.scan_tracker.update_file_record(file_path, scan_id, file_hash)
                    yield file_path, True  # All files are considered "modified" in full scan
            else:
                logger.info(f"Performing incremental scan of {directory}")
                async for file_path, is_modified in self._incremental_scan(
                    directory, scan_id, quick_hash
                ):
                    yield file_path, is_modified

            # Complete scan session
            self.scan_tracker.complete_scan_session(scan_id, {
                'scanned': 0,  # Would need to track during iteration
                'modified': 0,  # Would need to track during iteration
                'added': 0,    # Would need to track during iteration
                'removed': 0   # Hard to track without full disk scan
            })

        except Exception as e:
            logger.error(f"Error during scan session {scan_id}: {e}")
            # Don't complete session on error
            raise

    async def _full_scan(self, directory: Path) -> AsyncGenerator[Path, None]:
        """Perform a full scan of all audio files in directory.

        Args:
            directory: Directory to scan

        Yields:
            Paths to all audio files
        """
        def _scan_files():
            files = []
            try:
                for file_path in directory.rglob('*'):
                    if (file_path.is_file() and
                        file_path.suffix.lower() in self.audio_extensions):
                        files.append(file_path)
            except (OSError, PermissionError) as e:
                logger.error(f"Error scanning directory {directory}: {e}")
                raise MusicOrganizerError(f"Error scanning directory: {e}")
            return files

        # Run the file system scan in a thread pool
        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(None, _scan_files)

        logger.info(f"Full scan found {len(files)} audio files in {directory}")

        # Yield files
        for file_path in files:
            yield file_path

    async def _incremental_scan(
        self,
        directory: Path,
        scan_id: str,
        quick_hash: bool
    ) -> AsyncGenerator[Tuple[Path, bool], None]:
        """Perform incremental scan, checking for modified files.

        Args:
            directory: Directory to scan
            scan_id: Current scan session ID
            quick_hash: Use quick hash for change detection

        Yields:
            Tuple of (file_path, is_modified)
        """
        # Get modified files from tracker
        modified_files = self.scan_tracker.get_modified_files(directory)
        modified_set = set(modified_files)

        def _scan_all_files():
            all_files = []
            try:
                for file_path in directory.rglob('*'):
                    if (file_path.is_file() and
                        file_path.suffix.lower() in self.audio_extensions):
                        all_files.append(file_path)
            except (OSError, PermissionError) as e:
                logger.error(f"Error scanning directory {directory}: {e}")
                raise MusicOrganizerError(f"Error scanning directory: {e}")
            return all_files

        # Get all files to detect removed files (for reporting)
        loop = asyncio.get_event_loop()
        all_files = await loop.run_in_executor(None, _scan_all_files)

        logger.info(
            f"Incremental scan: {len(modified_files)} modified, "
            f"{len(all_files) - len(modified_files)} unchanged"
        )

        # Process all files, marking which ones are modified
        for file_path in all_files:
            is_modified = file_path in modified_set

            if is_modified:
                # Update scan record for modified files
                file_hash = self._calculate_file_hash(file_path, quick_hash)
                self.scan_tracker.update_file_record(file_path, scan_id, file_hash)

            yield file_path, is_modified

    async def scan_directory_batch_incremental(
        self,
        directory: Path,
        batch_size: int = 100,
        force_full: bool = False,
        filter_modified: bool = True,
        quick_hash: bool = True
    ) -> AsyncGenerator[List[Tuple[Path, bool]], None]:
        """Scan directory incrementally, yielding batches of (file_path, is_modified).

        Args:
            directory: Directory to scan
            batch_size: Number of files per batch
            force_full: Force full scan instead of incremental
            filter_modified: If True, only yield modified files
            quick_hash: Use quick hash for change detection

        Yields:
            Lists of (file_path, is_modified) tuples
        """
        batch = []
        modified_only_batch = []

        async for file_path, is_modified in self.scan_directory_incremental(
            directory, force_full, quick_hash
        ):
            if filter_modified:
                # Only include modified files
                if is_modified:
                    modified_only_batch.append((file_path, is_modified))
                    if len(modified_only_batch) >= batch_size:
                        yield modified_only_batch
                        modified_only_batch = []
            else:
                # Include all files with modification status
                batch.append((file_path, is_modified))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

        # Yield the last batch
        if filter_modified and modified_only_batch:
            yield modified_only_batch
        elif not filter_modified and batch:
            yield batch

    def get_scan_info(self, directory: Path) -> dict:
        """Get information about the last scan of a directory.

        Args:
            directory: Directory to check

        Returns:
            Dictionary with scan information
        """
        last_scan = self.scan_tracker.get_last_scan_time(directory)
        stats = self.scan_tracker.get_scan_statistics(directory)

        return {
            'directory': str(directory),
            'last_scan': last_scan.isoformat() if last_scan else None,
            'files_tracked': stats['total_files_tracked'],
            'has_history': last_scan is not None
        }

    def force_full_scan_next(self, directory: Path) -> None:
        """Clear scan history to force full scan next time.

        Args:
            directory: Directory to clear history for
        """
        logger.info(f"Clearing scan history for {directory} - next scan will be full")
        self.scan_tracker.clear_directory(directory)

    def cleanup_old_records(self, days: int = 90) -> int:
        """Clean up old scan records.

        Args:
            days: Remove records older than this many days

        Returns:
            Number of records removed
        """
        return self.scan_tracker.cleanup_old_records(days)