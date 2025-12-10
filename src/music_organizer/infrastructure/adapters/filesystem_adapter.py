"""
Filesystem Adapter - Anti-Corruption Layer for filesystem operations.

This adapter isolates the domain from filesystem-specific concerns,
providing a clean interface for file operations.
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from concurrent.futures import ThreadPoolExecutor

from ..domain.catalog import AudioPath, FileFormat


class FilesystemAdapter:
    """
    Adapter for filesystem operations.

    This adapter provides a clean interface for filesystem operations
    while protecting the domain from OS-specific details.
    """

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
        file_filter: Optional[callable] = None
    ) -> List[Path]:
        """
        Scan a directory for audio files.

        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            file_filter: Optional filter function for files

        Returns:
            List of file paths
        """
        def _scan():
            files = []
            if recursive:
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        if file_filter is None or file_filter(file_path):
                            files.append(file_path)
            else:
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        if file_filter is None or file_filter(file_path):
                            files.append(file_path)
            return files

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _scan)

    async def copy_file(self, source: Path, destination: Path) -> bool:
        """
        Copy a file from source to destination.

        Creates parent directories if needed.
        """
        def _copy():
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                return True
            except Exception as e:
                print(f"Error copying {source} to {destination}: {e}")
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _copy)

    async def move_file(self, source: Path, destination: Path) -> bool:
        """
        Move a file from source to destination.

        Creates parent directories if needed.
        """
        def _move():
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                return True
            except Exception as e:
                print(f"Error moving {source} to {destination}: {e}")
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _move)

    async def delete_file(self, file_path: Path) -> bool:
        """
        Delete a file.
        """
        def _delete():
            try:
                file_path.unlink()
                return True
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _delete)

    async def create_directory(self, directory: Path) -> bool:
        """
        Create a directory and any necessary parents.
        """
        def _create():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                return True
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _create)

    async def get_file_info(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get information about a file.
        """
        def _get_info():
            try:
                stat = file_path.stat()
                return {
                    "size": stat.st_size,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified_time": stat.st_mtime,
                    "created_time": stat.st_ctime,
                    "is_readable": os.access(file_path, os.R_OK),
                    "is_writable": os.access(file_path, os.W_OK),
                }
            except Exception as e:
                print(f"Error getting info for {file_path}: {e}")
                return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _get_info)

    async def find_duplicates_by_size(self, files: List[Path]) -> Dict[int, List[Path]]:
        """
        Group files by size to find potential duplicates.
        """
        def _group_by_size():
            size_groups = {}
            for file_path in files:
                try:
                    size = file_path.stat().st_size
                    if size not in size_groups:
                        size_groups[size] = []
                    size_groups[size].append(file_path)
                except Exception:
                    continue
            return size_groups

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _group_by_size)

    async def ensure_unique_path(self, base_path: Path) -> Path:
        """
        Ensure a path is unique by adding a suffix if needed.
        """
        def _ensure_unique():
            if not base_path.exists():
                return base_path

            counter = 1
            while True:
                stem = base_path.stem
                suffix = base_path.suffix
                new_name = f"{stem}_{counter}{suffix}"
                new_path = base_path.parent / new_name

                if not new_path.exists():
                    return new_path

                counter += 1

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _ensure_unique)

    async def batch_copy(
        self,
        file_pairs: List[tuple[Path, Path]],
        preserve_structure: bool = False
    ) -> List[bool]:
        """
        Copy multiple files in parallel.
        """
        tasks = []
        for source, dest in file_pairs:
            if preserve_structure:
                # Ensure the relative path structure is preserved
                # This is a simplified implementation
                pass
            tasks.append(self.copy_file(source, dest))

        return await asyncio.gather(*tasks)

    async def batch_move(self, file_pairs: List[tuple[Path, Path]]) -> List[bool]:
        """
        Move multiple files in parallel.
        """
        tasks = [self.move_file(source, dest) for source, dest in file_pairs]
        return await asyncio.gather(*tasks)

    def create_audio_filter(self, supported_formats: Optional[List[FileFormat]] = None) -> callable:
        """
        Create a filter function for audio files.
        """
        if supported_formats is None:
            supported_formats = [
                FileFormat.FLAC, FileFormat.MP3, FileFormat.M4A,
                FileFormat.WAV, FileFormat.OGG, FileFormat.OPUS,
                FileFormat.WMA
            ]

        extensions = {f".{fmt.value}" for fmt in supported_formats}

        def _audio_filter(file_path: Path) -> bool:
            return file_path.suffix.lower() in extensions

        return _audio_filter

    async def validate_audio_file(self, file_path: Path) -> bool:
        """
        Validate that a file is a readable audio file.
        """
        def _validate():
            try:
                # Check file exists and is readable
                if not file_path.exists() or not file_path.is_file():
                    return False

                # Check file has audio extension
                audio_path = AudioPath(file_path)
                if not audio_path.is_known_format:
                    return False

                # Check file is not empty
                if file_path.stat().st_size == 0:
                    return False

                # Try to open with mutagen if available
                try:
                    from mutagen import File as MutagenFile
                    audio_file = MutagenFile(file_path)
                    return audio_file is not None
                except ImportError:
                    # Mutagen not available, trust the extension
                    return True

            except Exception:
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _validate)