"""Organization Context Domain Services.

This module defines domain services for the Organization bounded context.
Domain services contain business logic that doesn't naturally fit in entities or value objects.
"""

import asyncio
import hashlib
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from ..result import Result, success, failure, OrganizationError
from .entities import OrganizationRule, FolderStructure, MovedFile, OrganizationSession, ConflictResolution, OperationStatus
from .value_objects import TargetPath, OrganizationPattern, ConflictStrategy


class RecordingLoaderService:
    """Service for loading Recording entities from audio files.

    This service bridges the catalog context (Recording entities) with the
    organization context by extracting metadata from audio files and
    converting them to domain Recording entities.
    """

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._adapter = None  # Lazy import to avoid circular dependencies

    def _get_adapter(self):
        """Lazy load the adapter to avoid circular imports."""
        if self._adapter is None:
            from ...infrastructure.adapters.audio_file_adapter import AudioFileToRecordingAdapter
            self._adapter = AudioFileToRecordingAdapter()
        return self._adapter

    async def load_from_path(self, file_path: Path) -> Optional[Any]:
        """Load a Recording entity from a single audio file path.

        Returns a catalog.Recording entity or None if the file is not supported.
        """
        def _load():
            try:
                from ...core.metadata import MetadataHandler
                audio_file = MetadataHandler.extract_metadata(file_path)
                return self._get_adapter().to_recording(audio_file)
            except Exception:
                # Skip unsupported files silently
                return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _load)

    async def load_from_paths(self, file_paths: List[Path]) -> List[Any]:
        """Load Recording entities from multiple audio file paths in parallel.

        Returns a list of catalog.Recording entities (unsupported files are filtered out).
        """
        tasks = [self.load_from_path(path) for path in file_paths]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    async def load_from_directory(
        self,
        directory: Path,
        recursive: bool = True,
        pattern: str = "*"
    ) -> List[Any]:
        """Load all Recording entities from audio files in a directory.

        Args:
            directory: The directory to scan
            recursive: Whether to scan subdirectories
            pattern: Glob pattern for matching files (default: all files)

        Returns:
            List of catalog.Recording entities
        """
        # Find all audio files
        if recursive:
            file_paths = list(directory.rglob(pattern))
        else:
            file_paths = list(directory.glob(pattern))

        # Filter to audio files only
        audio_extensions = {'.flac', '.mp3', '.m4a', '.mp4', '.ogg', '.opus',
                           '.wav', '.aiff', '.aif', '.wma'}
        audio_files = [p for p in file_paths if p.suffix.lower() in audio_extensions]

        return await self.load_from_paths(audio_files)


class PathGenerationService:
    """Service for generating organization paths."""

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def generate_target_path(
        self,
        source_path: Path,
        metadata: Any,  # catalog.Metadata
        rules: List[OrganizationRule],
        folder_structure: Optional[FolderStructure] = None,
        base_target_path: Optional[Path] = None
    ) -> Optional[TargetPath]:
        """Generate the target path for a file based on rules."""
        # Find the best matching rule
        matching_rule = None
        for rule in sorted(rules, key=lambda r: r.priority, reverse=True):
            if rule.matches(metadata, source_path):
                matching_rule = rule
                break

        if not matching_rule or not matching_rule.pattern:
            return None

        # Generate path using the rule's pattern
        pattern = matching_rule.pattern
        target_path = pattern.generate_path(metadata, source_path)

        # Apply folder structure if provided
        if folder_structure and base_target_path:
            folder_path = folder_structure.get_folder_path(metadata, base_target_path)
            target_path = folder_path / target_path.name

        # Create TargetPath object
        return TargetPath(
            path=target_path,
            pattern_used=pattern,
            original_path=source_path
        )

    async def batch_generate_paths(
        self,
        files: List[Tuple[Path, Any]],  # List of (source_path, metadata)
        rules: List[OrganizationRule],
        folder_structure: Optional[FolderStructure] = None,
        base_target_path: Optional[Path] = None
    ) -> List[Optional[TargetPath]]:
        """Generate target paths for multiple files in parallel."""
        tasks = [
            self.generate_target_path(
                source_path,
                metadata,
                rules,
                folder_structure,
                base_target_path
            )
            for source_path, metadata in files
        ]

        return await asyncio.gather(*tasks)

    def create_pattern_from_template(
        self,
        path_template: str,
        filename_template: str,
        level: Any  # OrganizationLevel
    ) -> OrganizationPattern:
        """Create an OrganizationPattern from template strings."""
        return OrganizationPattern(
            path_pattern=path_template,
            filename_pattern=filename_template,
            level=level
        )


class OrganizationService:
    """Service for executing file organization operations."""

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self.path_service = PathGenerationService(max_workers)
        self.recording_loader = RecordingLoaderService(max_workers)

    async def organize_file(
        self,
        source_path: Path,
        target_path: TargetPath,
        conflict_strategy: ConflictStrategy = ConflictStrategy.SKIP,
        dry_run: bool = False
    ) -> Result[MovedFile, OrganizationError]:
        """Organize a single file."""
        moved_file = MovedFile(
            source_path=source_path,
            target_path=target_path.path,
            status=OperationStatus.PENDING
        )

        try:
            # Check if source exists
            if not source_path.exists():
                moved_file.mark_failed(f"Source file does not exist: {source_path}")
                return success(moved_file)  # Return as success with error info in the object

            # Create target directory if needed
            if not dry_run:
                target_path.path.parent.mkdir(parents=True, exist_ok=True)

            # Handle conflicts
            if target_path.exists:
                resolution = await self._resolve_conflict(
                    source_path,
                    target_path,
                    conflict_strategy
                )

                if resolution.strategy == ConflictStrategy.SKIP:
                    moved_file.mark_skipped("File already exists at target")
                    moved_file.set_conflict_resolution(resolution)
                    return success(moved_file)
                else:
                    target_path = TargetPath(
                        path=resolution.final_path,
                        pattern_used=target_path.pattern_used,
                        conflict_strategy=resolution.strategy,
                        original_path=target_path.original_path
                    )
                    moved_file.conflict_resolution = resolution

            # Calculate checksum before move if not in dry run
            if not dry_run:
                moved_file.checksum_before = await self._calculate_checksum(source_path)

            # Perform the move
            if not dry_run:
                shutil.move(str(source_path), str(target_path.path))

                # Verify the move
                if not target_path.path.exists():
                    moved_file.mark_failed("File move failed - target does not exist")
                    return success(moved_file)

                # Calculate checksum after move
                moved_file.checksum_after = await self._calculate_checksum(target_path.path)

                # Verify checksums match
                if moved_file.checksum_before != moved_file.checksum_after:
                    moved_file.mark_failed("File corruption detected during move")
                    return success(moved_file)

            moved_file.mark_completed()
            return success(moved_file)

        except Exception as e:
            moved_file.mark_failed(str(e))
            return failure(OrganizationError(f"Failed to organize file {source_path}: {e}"))

    async def organize_session(self, session: OrganizationSession, dry_run: bool = False) -> None:
        """Execute an entire organization session."""
        session.start()

        try:
            # Load all recordings from source directory using catalog context integration
            recordings = await self.recording_loader.load_from_directory(
                session.source_directory,
                recursive=True
            )

            # Prepare files for organization with their metadata
            files_to_organize = []
            for recording in recordings:
                source_path = recording.path.path
                # Use the recording's metadata for path generation
                files_to_organize.append((source_path, recording.metadata))

            # Generate target paths for all files
            target_paths = await self.path_service.batch_generate_paths(
                files_to_organize,
                session.rules,
                session.folder_structure,
                session.target_directory
            )

            # Organize files
            for (source_path, metadata), target_path in zip(files_to_organize, target_paths):
                if target_path:
                    moved_file = await self.organize_file(
                        source_path,
                        target_path,
                        session.default_conflict_strategy,
                        dry_run
                    )
                    session.add_moved_file(moved_file)

        finally:
            session.complete()

    async def _resolve_conflict(
        self,
        source_path: Path,
        target_path: TargetPath,
        strategy: ConflictStrategy
    ) -> ConflictResolution:
        """Resolve a file conflict."""
        if strategy == ConflictStrategy.SKIP:
            return ConflictResolution(
                source_path=source_path,
                target_path=target_path.path,
                conflicting_path=target_path.path,
                strategy=strategy,
                final_path=target_path.path
            )

        elif strategy == ConflictStrategy.REPLACE:
            # Delete existing file and use original path
            if target_path.path.exists():
                target_path.path.unlink()
            return ConflictResolution(
                source_path=source_path,
                target_path=target_path.path,
                conflicting_path=target_path.path,
                strategy=strategy,
                final_path=target_path.path
            )

        elif strategy == ConflictStrategy.RENAME:
            # Generate a new name
            counter = 1
            while True:
                new_name = f"{target_path.stem}_{counter}{target_path.suffix}"
                new_path = target_path.parent / new_name
                if not new_path.exists():
                    break
                counter += 1

            return ConflictResolution(
                source_path=source_path,
                target_path=target_path.path,
                conflicting_path=target_path.path,
                strategy=strategy,
                final_path=new_path
            )

        elif strategy == ConflictStrategy.KEEP_BOTH:
            # Move to a subfolder
            dupe_folder = target_path.parent / "duplicates"
            new_path = dupe_folder / target_path.path.name

            return ConflictResolution(
                source_path=source_path,
                target_path=target_path.path,
                conflicting_path=target_path.path,
                strategy=strategy,
                final_path=new_path
            )

        else:  # APPEND_SUFFIX
            # Append a suffix to the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"{target_path.stem}_{timestamp}{target_path.suffix}"
            new_path = target_path.parent / new_name

            return ConflictResolution(
                source_path=source_path,
                target_path=target_path.path,
                conflicting_path=target_path.path,
                strategy=strategy,
                final_path=new_path
            )

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        def _calc_checksum():
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _calc_checksum)

    async def preview_organization(
        self,
        session: OrganizationSession
    ) -> List[Dict[str, Any]]:
        """Preview what would be organized without actually moving files."""
        preview_items = []

        # Load all recordings from source directory using catalog context integration
        recordings = await self.recording_loader.load_from_directory(
            session.source_directory,
            recursive=True
        )

        # Generate preview for each recording
        for recording in recordings:
            source_path = recording.path.path
            metadata = recording.metadata

            rule = session.find_rule_for_file(metadata, source_path)
            if rule:
                target_path = await self.path_service.generate_target_path(
                    source_path,
                    metadata,
                    [rule],
                    session.folder_structure,
                    session.target_directory
                )

                if target_path:
                    preview_items.append({
                        "source_path": str(source_path),
                        "target_path": str(target_path.path),
                        "rule": rule.name,
                        "conflict": target_path.exists,
                        "action": "Would move" if not target_path.exists else "Would resolve conflict"
                    })

        return preview_items