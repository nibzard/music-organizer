"""Async file operations for moving and organizing music files."""

import os
import shutil
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, AsyncGenerator
from datetime import datetime

from ..exceptions import FileOperationError
from ..models.audio_file import AudioFile, CoverArt


class AsyncFileMover:
    """Handle async file operations with backup and rollback support."""

    def __init__(self,
                 backup_enabled: bool = True,
                 backup_dir: Optional[Path] = None,
                 max_workers: int = 4):
        self.backup_enabled = backup_enabled
        self.backup_dir = backup_dir
        self.operations: List[Dict] = []
        self.started = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()

    async def start_operation(self, source_root: Path) -> None:
        """Start a new operation session with optional backup."""
        async with self._lock:
            if self.started:
                raise FileOperationError("Operation already in progress")

            self.started = True
            self.operations = []

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

    async def finish_operation(self) -> None:
        """Finish current operation session."""
        async with self._lock:
            if not self.started:
                raise FileOperationError("No operation in progress")

            if self.backup_enabled and self.backup_dir:
                # Save operation log
                await self._save_operation_log()

            self.started = False

    async def move_file(self, audio_file: AudioFile, target_path: Path) -> Path:
        """Move an audio file to its target location asynchronously."""
        async with self._lock:
            if not self.started:
                raise FileOperationError("Must start operation before moving files")

        # Ensure target directory exists
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: target_path.parent.mkdir(parents=True, exist_ok=True)
        )

        # Check if target already exists
        final_target = await self._resolve_duplicate(target_path)

        # Perform the move
        try:
            if self.backup_enabled and self.backup_dir:
                # Create backup copy first
                await self._backup_file(audio_file.path)

            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: shutil.move(str(audio_file.path), str(final_target))
            )
            audio_file.path = final_target

            # Record operation
            async with self._lock:
                self.operations.append({
                    'type': 'move',
                    'original': str(audio_file.path),
                    'target': str(final_target),
                    'timestamp': datetime.now().isoformat()
                })

            return final_target

        except Exception as e:
            raise FileOperationError(f"Failed to move {audio_file.path}: {e}")

    async def move_files_batch(self,
                              moves: List[Tuple[AudioFile, Path]]) -> List[Tuple[bool, Optional[Path], Optional[str]]]:
        """Move multiple files in parallel."""
        tasks = []
        for audio_file, target_path in moves:
            task = asyncio.create_task(self.move_file(audio_file, target_path))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error messages
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append((False, None, str(result)))
            else:
                processed_results.append((True, result, None))

        return processed_results

    async def move_cover_art(self, cover_art: CoverArt, target_dir: Path) -> Optional[Path]:
        """Move cover art to target directory asynchronously."""
        if not cover_art or not cover_art.path.exists():
            return None

        # Determine target filename
        target_filename = self._get_cover_art_filename(cover_art)
        target_path = target_dir / target_filename

        # Handle duplicates
        target_path = await self._resolve_duplicate(target_path)

        try:
            if self.backup_enabled and self.backup_dir:
                await self._backup_file(cover_art.path)

            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: shutil.move(str(cover_art.path), str(target_path))
            )
            cover_art.path = target_path

            # Record operation
            async with self._lock:
                self.operations.append({
                    'type': 'move_cover',
                    'original': str(cover_art.path),
                    'target': str(target_path),
                    'timestamp': datetime.now().isoformat()
                })

            return target_path

        except Exception as e:
            raise FileOperationError(f"Failed to move cover art {cover_art.path}: {e}")

    async def rollback(self) -> None:
        """Rollback all performed operations asynchronously."""
        if not self.operations:
            return

        # Reverse operations list
        reversed_ops = list(reversed(self.operations))

        # Create rollback tasks
        async def rollback_operation(op):
            if op['type'] in ['move', 'move_cover']:
                original = Path(op['original'])
                target = Path(op['target'])

                if target.exists() and not original.exists():
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: target.parent.mkdir(parents=True, exist_ok=True)
                    )
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: shutil.move(str(target), str(original))
                    )

        # Execute rollbacks in parallel (limit concurrency to avoid filesystem issues)
        semaphore = asyncio.Semaphore(4)
        async def bounded_rollback(op):
            async with semaphore:
                try:
                    await rollback_operation(op)
                except Exception as e:
                    # Log error but continue rollback
                    print(f"Warning: Failed to rollback {op['target']}: {e}")

        await asyncio.gather(*[bounded_rollback(op) for op in reversed_ops])

        async with self._lock:
            self.operations = []

    async def get_operation_summary(self) -> Dict:
        """Get summary of performed operations."""
        async with self._lock:
            summary = {
                'total_files': 0,
                'total_cover_art': 0,
                'total_size_mb': 0,
                'directories_created': set(),
                'errors': []
            }

            for op in self.operations:
                if op['type'] == 'move':
                    summary['total_files'] += 1
                elif op['type'] == 'move_cover':
                    summary['total_cover_art'] += 1

                # Add parent directory to set
                target = Path(op['target'])
                summary['directories_created'].add(str(target.parent))

            summary['directories_created'] = len(summary['directories_created'])
            return summary

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

    async def _backup_file(self, file_path: Path) -> None:
        """Create a backup of the file asynchronously."""
        if not self.backup_dir:
            return

        def _do_backup():
            try:
                # Create relative path in backup
                relative_path = file_path.relative_to(file_path.anchor)
                backup_path = self.backup_dir / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file if backup doesn't exist
                if not backup_path.exists():
                    shutil.copy2(str(file_path), str(backup_path))
            except Exception as e:
                print(f"Warning: Failed to backup {file_path}: {e}")

        await asyncio.get_event_loop().run_in_executor(
            self.executor, _do_backup
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

        summary = await self.get_operation_summary()
        log_data = {
            'timestamp': datetime.now().isoformat(),
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

    async def validate_integrity(self, source_root: Path) -> Tuple[bool, List[str]]:
        """Validate that all files are intact after operations."""
        errors = []

        if self.backup_enabled and self.backup_dir:
            # Check against backup manifest
            manifest_path = self.backup_dir / 'manifest.json'

            def _load_manifest():
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        return json.load(f)
                return None

            manifest = await asyncio.get_event_loop().run_in_executor(
                self.executor, _load_manifest
            )

            if manifest:
                for file_info in manifest['files']:
                    original_path = source_root / file_info['path']

                    def _check_file():
                        if not original_path.exists():
                            return None
                        try:
                            stat = original_path.stat()
                            return stat.st_size == file_info['size']
                        except OSError:
                            return False

                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor, _check_file
                    )

                    if result is None:
                        errors.append(f"Missing file: {file_info['path']}")
                    elif result is False:
                        errors.append(f"Size mismatch: {file_info['path']}")
            else:
                errors.append("Cannot read backup manifest")

        return len(errors) == 0, errors

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)


class AsyncDirectoryOrganizer:
    """Helper class for creating and validating directory structures asynchronously."""

    @staticmethod
    async def create_directory_structure(base_path: Path) -> None:
        """Create the standard music directory structure asynchronously."""
        directories = [
            "Albums",
            "Live",
            "Collaborations",
            "Compilations",
            "Rarities"
        ]

        async with ThreadPoolExecutor() as executor:
            tasks = []
            for dir_name in directories:
                dir_path = base_path / dir_name
                task = asyncio.get_event_loop().run_in_executor(
                    executor,
                    dir_path.mkdir,
                    parents=True,
                    exist_ok=True
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

    @staticmethod
    async def validate_structure(base_path: Path) -> Dict[str, bool]:
        """Validate that required directories exist asynchronously."""
        required_dirs = [
            "Albums",
            "Live",
            "Collaborations",
            "Compilations",
            "Rarities"
        ]

        async def check_dir(dir_name):
            dir_path = base_path / dir_name
            return dir_name, dir_path.exists()

        tasks = [check_dir(dir_name) for dir_name in required_dirs]
        results = await asyncio.gather(*tasks)

        return dict(results)

    @staticmethod
    async def get_empty_directories(base_path: Path) -> List[Path]:
        """Find empty directories that can be cleaned up asynchronously."""
        def scan_directory():
            empty_dirs = []
            audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}
            cover_extensions = {'.jpg', '.jpeg', '.png', '.gif'}

            for root, dirs, files in os.walk(base_path):
                # Skip if it has audio files or cover art
                has_media = False
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in audio_extensions or ext in cover_extensions:
                        has_media = True
                        break

                if not has_media and not dirs:
                    empty_dirs.append(Path(root))

            return empty_dirs

        return await asyncio.get_event_loop().run_in_executor(
            None, scan_directory
        )