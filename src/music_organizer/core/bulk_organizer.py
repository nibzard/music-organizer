"""Enhanced async organizer with bulk file operations integration.

This module provides high-performance music organization by leveraging
bulk file operations for improved throughput and reduced filesystem overhead.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
from datetime import datetime

from ..models.audio_file import AudioFile, CoverArt
from ..models.config import Config
from ..exceptions import MusicOrganizerError
from ..domain.entities import Recording, Release, AudioLibrary
from ..domain.value_objects import AudioPath, Metadata
from .async_organizer import AsyncMusicOrganizer
from .bulk_operations import BulkMoveOperator, BulkOperationConfig, ConflictStrategy
from .parallel_metadata import ParallelMetadataExtractor
from .incremental_scanner import IncrementalScanner


class BulkAsyncOrganizer:
    """High-performance async music organizer with bulk operations."""

    def __init__(self,
                 config: Config,
                 bulk_config: Optional[BulkOperationConfig] = None,
                 dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.bulk_config = bulk_config or BulkOperationConfig()

        # Initialize components
        self.parallel_extractor = ParallelMetadataExtractor(
            max_workers=self.bulk_config.max_workers
        )
        self.incremental_scanner = IncrementalScanner()
        self.bulk_mover = BulkMoveOperator(self.bulk_config)

        # Domain entities for organization
        self.audio_library: Optional[AudioLibrary] = None
        self.releases: Dict[str, Release] = {}
        self.recordings: List[Recording] = []

    async def organize_bulk(self,
                           source_dir: Path,
                           target_dir: Path,
                           incremental: bool = False,
                           progress_callback=None) -> Dict[str, Any]:
        """Organize music library using bulk operations for maximum performance."""
        start_time = datetime.now()

        try:
            # Initialize audio library
            self.audio_library = AudioLibrary(
                name=str(source_dir.name),
                root_path=target_dir
            )

            # Scan for files (incremental if requested)
            if incremental:
                files = await self._scan_incremental(source_dir)
            else:
                files = await self._scan_full(source_dir)

            if not files:
                return {
                    'status': 'completed',
                    'total_files': 0,
                    'processed': 0,
                    'moved': 0,
                    'skipped': 0,
                    'errors': [],
                    'duration_seconds': 0,
                    'bulk_stats': {}
                }

            # Extract metadata in parallel
            recordings = await self._extract_metadata_parallel(files, progress_callback)

            # Group recordings by release for organization
            release_groups = self._group_by_release(recordings)

            # Prepare bulk move operations
            bulk_operations = await self._prepare_bulk_operations(release_groups, target_dir)

            # Execute bulk operations
            bulk_result = await self._execute_bulk_operations(
                bulk_operations, progress_callback
            )

            # Update domain entities
            await self._update_domain_entities(bulk_result)

            # Compile final results
            duration = (datetime.now() - start_time).total_seconds()
            results = {
                'status': 'completed',
                'total_files': len(files),
                'processed': bulk_result.successful + bulk_result.failed,
                'moved': bulk_result.successful,
                'skipped': bulk_result.skipped,
                'errors': bulk_result.errors,
                'duration_seconds': duration,
                'bulk_stats': {
                    'success_rate': bulk_result.success_rate,
                    'throughput_mb_per_sec': bulk_result.throughput_mb_per_sec,
                    'total_size_mb': bulk_result.total_size_mb,
                    'conflicts_resolved': len([op for op in bulk_result.operations
                                             if 'renamed' in op.get('message', '').lower()])
                },
                'library_stats': self.audio_library.get_statistics() if self.audio_library else {}
            }

            return results

        except Exception as e:
            raise MusicOrganizerError(f"Bulk organization failed: {e}")

        finally:
            # Cleanup resources
            await self.parallel_extractor.cleanup()
            await self.bulk_mover.cleanup()

    async def _scan_incremental(self, source_dir: Path) -> List[Path]:
        """Perform incremental scanning for new/modified files."""
        files = []

        async for file_path, is_modified in self.incremental_scanner.scan_directory_incremental(
            source_dir
        ):
            if is_modified:
                files.append(file_path)

        return files

    async def _scan_full(self, source_dir: Path) -> List[Path]:
        """Perform full directory scanning."""
        files = []

        # Use parallel scanning for large directories
        async for batch in self.incremental_scanner.scan_directory_batch_incremental(
            source_dir, batch_size=1000
        ):
            files.extend(batch)

        return files

    async def _extract_metadata_parallel(self,
                                        files: List[Path],
                                        progress_callback=None) -> List[Recording]:
        """Extract metadata from all files in parallel."""
        # Create AudioPath objects
        audio_paths = [AudioPath(str(f)) for f in files]

        # Extract metadata in parallel
        metadata_results = await self.parallel_extractor.extract_metadata_parallel(
            audio_paths, progress_callback
        )

        # Convert to Recording entities
        recordings = []
        for audio_path, metadata_result in zip(audio_paths, metadata_results):
            if metadata_result.is_success():
                metadata = metadata_result.value()
                recording = Recording(
                    path=audio_path,
                    metadata=metadata
                )
                recordings.append(recording)
                self.recordings.append(recording)
            else:
                # Log error but continue processing
                print(f"Failed to extract metadata from {audio_path}: {metadata_result.error()}")

        return recordings

    def _group_by_release(self, recordings: List[Recording]) -> Dict[str, List[Recording]]:
        """Group recordings by their release (album)."""
        groups = {}

        for recording in recordings:
            # Create release key from metadata
            album = recording.metadata.album or "Unknown Album"
            primary_artist = recording.metadata.artists[0] if recording.metadata.artists else "Unknown Artist"

            # Create unique release key
            release_key = f"{primary_artist} - {album}"

            if release_key not in groups:
                groups[release_key] = []

            groups[release_key].append(recording)

        return groups

    async def _prepare_bulk_operations(self,
                                     release_groups: Dict[str, List[Recording]],
                                     target_dir: Path) -> List[Tuple[Path, Path, str]]:
        """Prepare bulk move operations organized by directory structure."""
        operations = []

        for release_key, recordings in release_groups.items():
            # Determine target directory for this release
            if recordings:
                first_recording = recordings[0]
                content_type = first_recording.metadata.content_type.value
                target_subdir = self._get_content_type_directory(content_type)

                # Create release directory
                primary_artist = first_recording.metadata.artists[0] if first_recording.metadata.artists else "Unknown Artist"
                album = first_recording.metadata.album or "Unknown Album"
                year = first_recording.metadata.year or ""

                if year:
                    release_dir_name = f"{primary_artist} - {album} ({year})"
                else:
                    release_dir_name = f"{primary_artist} - {album}"

                release_dir = target_dir / target_subdir / release_dir_name

                # Add operations for each recording
                for recording in recordings:
                    target_filename = self._generate_target_filename(recording)
                    target_path = release_dir / target_filename

                    operations.append((recording.path.path, target_path, "move"))

                    # Add cover art operations if applicable
                    cover_art = self._find_cover_art(recording.path.path)
                    if cover_art:
                        operations.append((cover_art, release_dir, "move"))

        return operations

    async def _execute_bulk_operations(self,
                                     operations: List[Tuple[Path, Path, str]],
                                     progress_callback=None) -> Any:
        """Execute bulk file operations."""
        if self.dry_run:
            # Just show what would be done
            for source, target, op_type in operations:
                print(f"Would {op_type}: {source.name} -> {target}")

            # Return mock result
            from .bulk_operations import BulkOperationResult
            return BulkOperationResult(
                total_files=len(operations),
                successful=len(operations),
                skipped=0,
                failed=0,
                total_size_mb=0.0,
                duration_seconds=0.0
            )

        # Add operations to bulk mover
        await self.bulk_mover.add_operations_batch(operations)

        # Execute with progress tracking
        if progress_callback:
            async def bulk_progress(current, total):
                await progress_callback(current, total, "Moving files")

            return await self.bulk_mover.execute_bulk_operation(bulk_progress)
        else:
            return await self.bulk_mover.execute_bulk_operation()

    async def _update_domain_entities(self, bulk_result) -> None:
        """Update domain entities with operation results."""
        if not self.audio_library:
            return

        # Process successful operations
        for op in bulk_result.operations:
            if op['status'] == 'success':
                # Find corresponding recording
                source_path = Path(op['source'])
                target_path = Path(op['target'])

                for recording in self.recordings:
                    if recording.path.path == source_path:
                        # Update recording path
                        recording.path = AudioPath(str(target_path))

                        # Add to audio library
                        self.audio_library.add_recording(recording)
                        break

    def _get_content_type_directory(self, content_type: str) -> str:
        """Map content type to target directory."""
        type_map = {
            'studio': 'Albums',
            'live': 'Live',
            'collaboration': 'Collaborations',
            'compilation': 'Compilations',
            'rarity': 'Rarities'
        }
        return type_map.get(content_type, 'Albums')

    def _generate_target_filename(self, recording: Recording) -> str:
        """Generate target filename for a recording."""
        metadata = recording.metadata

        # Format: track_number title.extension
        track_str = ""
        if metadata.track_number:
            track_str = f"{metadata.track_number.formatted()} "

        title = metadata.title or "Unknown"

        # Get file extension
        ext = recording.path.format.value.lower()

        return f"{track_str}{title}.{ext}"

    def _find_cover_art(self, audio_file_path: Path) -> Optional[Path]:
        """Find cover art in the same directory as the audio file."""
        audio_dir = audio_file_path.parent
        cover_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
        cover_names = {'folder', 'cover', 'front', 'albumart'}

        for file_path in audio_dir.iterdir():
            if file_path.is_file():
                if file_path.suffix.lower() in cover_extensions:
                    if file_path.stem.lower() in cover_names:
                        return file_path

        return None

    async def get_organization_preview(self,
                                      source_dir: Path,
                                      target_dir: Path) -> Dict[str, Any]:
        """Get a preview of what the organization would look like."""
        preview = {
            'source_files': 0,
            'target_structure': {},
            'estimated_size_mb': 0,
            'estimated_duration': 0,
            'content_distribution': {}
        }

        # Scan files
        files = await self._scan_full(source_dir)
        preview['source_files'] = len(files)

        # Extract metadata for a sample to estimate
        sample_size = min(100, len(files))
        sample_files = files[:sample_size]

        if sample_files:
            recordings = await self._extract_metadata_parallel(sample_files)

            # Analyze distribution
            for recording in recordings:
                content_type = recording.metadata.content_type.value
                preview['content_distribution'][content_type] = \
                    preview['content_distribution'].get(content_type, 0) + 1

                # Estimate size
                if recording.path.size_mb:
                    preview['estimated_size_mb'] += recording.path.size_mb * (len(files) / sample_size)

            # Estimate duration
            preview['estimated_duration'] = await self.bulk_mover.get_estimated_duration(files)

        return preview

    async def rollback_last_operation(self) -> bool:
        """Rollback the last bulk operation."""
        try:
            await self.bulk_mover.rollback()
            return True
        except Exception as e:
            print(f"Rollback failed: {e}")
            return False

    async def get_library_statistics(self) -> Dict[str, Any]:
        """Get comprehensive library statistics."""
        if self.audio_library:
            return self.audio_library.get_statistics()
        return {}