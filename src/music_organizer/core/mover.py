"""File operations for moving and organizing music files."""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..exceptions import FileOperationError
from ..models.audio_file import AudioFile, CoverArt


class FileMover:
    """Handle safe file operations with backup and rollback support."""

    def __init__(self, backup_enabled: bool = True, backup_dir: Optional[Path] = None):
        self.backup_enabled = backup_enabled
        self.backup_dir = backup_dir
        self.operations: List[Dict] = []
        self.started = False

    def start_operation(self, source_root: Path) -> None:
        """Start a new operation session with optional backup."""
        if self.started:
            raise FileOperationError("Operation already in progress")

        self.started = True
        self.operations = []

        if self.backup_enabled:
            if not self.backup_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.backup_dir = source_root.parent / f"backup_{timestamp}"

            self.backup_dir.mkdir(parents=True, exist_ok=True)

            # Create backup manifest
            self._create_backup_manifest(source_root)

    def finish_operation(self) -> None:
        """Finish current operation session."""
        if not self.started:
            raise FileOperationError("No operation in progress")

        if self.backup_enabled and self.backup_dir:
            # Save operation log
            self._save_operation_log()

        self.started = False

    def move_file(self, audio_file: AudioFile, target_path: Path) -> Path:
        """Move an audio file to its target location."""
        if not self.started:
            raise FileOperationError("Must start operation before moving files")

        # Ensure target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if target already exists
        final_target = self._resolve_duplicate(target_path)

        # Perform the move
        try:
            if self.backup_enabled and self.backup_dir:
                # Create backup copy first
                self._backup_file(audio_file.path)

            shutil.move(str(audio_file.path), str(final_target))
            audio_file.path = final_target

            # Record operation
            self.operations.append({
                'type': 'move',
                'original': str(audio_file.path),
                'target': str(final_target),
                'timestamp': datetime.now().isoformat()
            })

            return final_target

        except Exception as e:
            raise FileOperationError(f"Failed to move {audio_file.path}: {e}")

    def move_cover_art(self, cover_art: CoverArt, target_dir: Path) -> Optional[Path]:
        """Move cover art to target directory."""
        if not cover_art or not cover_art.path.exists():
            return None

        # Determine target filename
        target_filename = self._get_cover_art_filename(cover_art)
        target_path = target_dir / target_filename

        # Handle duplicates
        target_path = self._resolve_duplicate(target_path)

        try:
            if self.backup_enabled and self.backup_dir:
                self._backup_file(cover_art.path)

            shutil.move(str(cover_art.path), str(target_path))
            cover_art.path = target_path

            # Record operation
            self.operations.append({
                'type': 'move_cover',
                'original': str(cover_art.path),
                'target': str(target_path),
                'timestamp': datetime.now().isoformat()
            })

            return target_path

        except Exception as e:
            raise FileOperationError(f"Failed to move cover art {cover_art.path}: {e}")

    def rollback(self) -> None:
        """Rollback all performed operations."""
        if not self.operations:
            return

        # Reverse operations list
        reversed_ops = reversed(self.operations)

        for op in reversed_ops:
            try:
                if op['type'] in ['move', 'move_cover']:
                    original = Path(op['original'])
                    target = Path(op['target'])

                    if target.exists() and not original.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(target), str(original))

            except Exception as e:
                # Log error but continue rollback
                print(f"Warning: Failed to rollback {op['target']}: {e}")

        self.operations = []

    def get_operation_summary(self) -> Dict:
        """Get summary of performed operations."""
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

    def _resolve_duplicate(self, target_path: Path) -> Path:
        """Resolve duplicate filenames by adding a number."""
        if not target_path.exists():
            return target_path

        base = target_path.stem
        ext = target_path.suffix
        parent = target_path.parent
        counter = 1

        while True:
            new_name = f"{base} ({counter}){ext}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1

    def _backup_file(self, file_path: Path) -> None:
        """Create a backup of the file."""
        if not self.backup_dir:
            return

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

    def _create_backup_manifest(self, source_root: Path) -> None:
        """Create a manifest of all files before starting operations."""
        if not self.backup_dir:
            return

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

        # Save manifest
        manifest_path = self.backup_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _save_operation_log(self) -> None:
        """Save operation log to backup directory."""
        if not self.backup_dir or not self.operations:
            return

        log_path = self.backup_dir / 'operations.json'
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'operations': self.operations,
            'summary': self.get_operation_summary()
        }

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

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

    def validate_integrity(self, source_root: Path) -> Tuple[bool, List[str]]:
        """Validate that all files are intact after operations."""
        errors = []

        if self.backup_enabled and self.backup_dir:
            # Check against backup manifest
            manifest_path = self.backup_dir / 'manifest.json'
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)

                    for file_info in manifest['files']:
                        original_path = source_root / file_info['path']
                        if not original_path.exists():
                            errors.append(f"Missing file: {file_info['path']}")
                        else:
                            try:
                                stat = original_path.stat()
                                if stat.st_size != file_info['size']:
                                    errors.append(f"Size mismatch: {file_info['path']}")
                            except OSError:
                                errors.append(f"Cannot access: {file_info['path']}")

                except (json.JSONDecodeError, KeyError):
                    errors.append("Cannot read backup manifest")

        return len(errors) == 0, errors


class DirectoryOrganizer:
    """Helper class for creating and validating directory structures."""

    @staticmethod
    def create_directory_structure(base_path: Path) -> None:
        """Create the standard music directory structure."""
        directories = [
            "Albums",
            "Live",
            "Collaborations",
            "Compilations",
            "Rarities"
        ]

        for dir_name in directories:
            dir_path = base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_structure(base_path: Path) -> Dict[str, bool]:
        """Validate that required directories exist."""
        required_dirs = [
            "Albums",
            "Live",
            "Collaborations",
            "Compilations",
            "Rarities"
        ]

        validation = {}
        for dir_name in required_dirs:
            validation[dir_name] = (base_path / dir_name).exists()

        return validation

    @staticmethod
    def get_empty_directories(base_path: Path) -> List[Path]:
        """Find empty directories that can be cleaned up."""
        empty_dirs = []

        for root, dirs, files in os.walk(base_path):
            # Skip if it has audio files or cover art
            audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}
            cover_extensions = {'.jpg', '.jpeg', '.png', '.gif'}

            has_media = False
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in audio_extensions or ext in cover_extensions:
                    has_media = True
                    break

            if not has_media and not dirs:
                empty_dirs.append(Path(root))

        return empty_dirs