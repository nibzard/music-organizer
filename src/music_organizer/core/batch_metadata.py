"""Batch metadata operations for bulk tagging and metadata updates.

This module provides high-performance batch operations for:
- Bulk metadata updates across multiple files
- Pattern-based metadata transformations
- Tag operations (add, remove, modify)
- Genre and artist standardization
- Batch validation and preview
- Undo/redo support for metadata changes
"""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Set
from datetime import datetime

from ..exceptions import MetadataError, FileOperationError
from ..models.audio_file import AudioFile
from ..infrastructure.adapters.mutagen_adapter import MutagenMetadataAdapter
from ..domain.value_objects import ArtistName, TrackNumber, Metadata
from .operation_history import OperationHistoryTracker


class OperationType(Enum):
    """Type of metadata operation."""
    SET = "set"          # Set a field value
    ADD = "add"          # Add to multi-value field
    REMOVE = "remove"    # Remove from multi-value field
    CLEAR = "clear"      # Clear field value
    TRANSFORM = "transform"  # Transform field value
    COPY = "copy"        # Copy value from one field to another
    RENAME = "rename"    # Rename artist/album/genre


class ConflictStrategy(Enum):
    """Strategy for handling metadata conflicts."""
    SKIP = "skip"        # Skip if field has value
    REPLACE = "replace"  # Replace existing value
    MERGE = "merge"      # Merge with existing value
    ASK = "ask"          # Ask user (for interactive mode)


@dataclass
class MetadataOperation:
    """Represents a single metadata operation."""
    field: str                   # Field name (e.g., 'genre', 'artist')
    operation: OperationType     # Type of operation
    value: Any = None           # Operation value (optional)
    condition: Dict[str, Any] = field(default_factory=dict)  # Condition to apply operation
    pattern: Optional[str] = None  # Pattern for transform operations
    conflict_strategy: ConflictStrategy = ConflictStrategy.REPLACE


@dataclass
class BatchMetadataConfig:
    """Configuration for batch metadata operations."""
    max_workers: int = 4
    batch_size: int = 100
    dry_run: bool = False
    backup_before_update: bool = True
    validate_before_update: bool = True
    continue_on_error: bool = True
    preserve_modified_time: bool = True
    operation_timeout: float = 30.0  # seconds per file
    create_undo_log: bool = True


@dataclass
class BatchResult:
    """Result of batch metadata operations."""
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    conflicts: int = 0
    duration_seconds: float = 0.0
    operations_performed: List[Dict] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful / self.total_files) * 100

    @property
    def throughput_files_per_sec(self) -> float:
        """Calculate throughput in files per second."""
        if self.duration_seconds == 0:
            return 0.0
        return self.total_files / self.duration_seconds


class BatchMetadataProcessor:
    """High-performance batch metadata processor."""

    def __init__(self, config: BatchMetadataConfig, adapter: Optional[MutagenMetadataAdapter] = None):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.adapter = adapter if adapter is not None else MutagenMetadataAdapter()
        self._operation_history: List[Dict] = []
        self._start_time: Optional[datetime] = None

    async def apply_operations(self,
                             files: List[Path],
                             operations: List[MetadataOperation],
                             progress_callback: Optional[Callable] = None) -> BatchResult:
        """Apply metadata operations to multiple files."""
        self._start_time = datetime.now()
        result = BatchResult(total_files=len(files))

        # Create undo log if enabled
        if self.config.create_undo_log and not self.config.dry_run:
            await self._create_backup_metadata(files)

        # Process files in batches
        for i in range(0, len(files), self.config.batch_size):
            batch = files[i:i + self.config.batch_size]
            batch_result = await self._process_batch(batch, operations)

            # Merge results
            result.successful += batch_result.successful
            result.failed += batch_result.failed
            result.skipped += batch_result.skipped
            result.conflicts += batch_result.conflicts
            result.operations_performed.extend(batch_result.operations_performed)
            result.errors.extend(batch_result.errors)
            result.warnings.extend(batch_result.warnings)

            # Update progress
            if progress_callback:
                await progress_callback(i + len(batch), len(files))

            # Stop on error if not configured to continue
            if batch_result.failed > 0 and not self.config.continue_on_error:
                result.warnings.append("Stopping batch due to errors and continue_on_error=False")
                break

        result.duration_seconds = (datetime.now() - self._start_time).total_seconds()
        return result

    async def _process_batch(self,
                           files: List[Path],
                           operations: List[MetadataOperation]) -> BatchResult:
        """Process a batch of files with metadata operations."""
        result = BatchResult(total_files=len(files))

        # Create parallel tasks
        tasks = []
        for file_path in files:
            task = asyncio.create_task(
                self._process_file(file_path, operations)
            )
            tasks.append(task)

        # Wait for all tasks with timeout
        try:
            batch_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.operation_timeout * len(files)
            )
        except asyncio.TimeoutError:
            result.failed = len(files)
            result.errors.append({
                "error": "Batch operation timed out",
                "files_count": len(files)
            })
            return result

        # Process results
        for i, file_result in enumerate(batch_results):
            if isinstance(file_result, Exception):
                result.failed += 1
                result.errors.append({
                    "file": str(files[i]),
                    "error": str(file_result)
                })
            else:
                if file_result['status'] == 'success':
                    result.successful += 1
                    result.operations_performed.append({
                        "file": str(files[i]),
                        "operations": file_result['operations'],
                        "timestamp": datetime.now().isoformat()
                    })
                elif file_result['status'] == 'skipped':
                    result.skipped += 1
                elif file_result['status'] == 'conflict':
                    result.conflicts += 1
                else:
                    result.failed += 1
                    result.errors.append({
                        "file": str(files[i]),
                        "error": file_result.get('error', 'Unknown error')
                    })

        return result

    async def _process_file(self,
                          file_path: Path,
                          operations: List[MetadataOperation]) -> Dict:
        """Process a single file with metadata operations."""
        try:
            # Check if file exists
            if not file_path.exists():
                return {'status': 'failed', 'error': 'File does not exist'}

            # Read current metadata
            current_metadata = await self.adapter.read_metadata(file_path)
            if current_metadata is None:
                return {'status': 'failed', 'error': 'Cannot read metadata'}

            # Apply operations
            updated_metadata, applied_ops = await self._apply_operations(
                current_metadata, operations
            )

            # Check if any changes were made
            if not applied_ops:
                return {'status': 'skipped', 'error': 'No changes needed'}

            # Validate metadata
            if self.config.validate_before_update:
                validation_errors = self._validate_metadata(updated_metadata)
                if validation_errors:
                    return {
                        'status': 'failed',
                        'error': f'Validation failed: {", ".join(validation_errors)}'
                    }

            # Write metadata (unless dry run)
            if not self.config.dry_run:
                success = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.adapter.write_metadata,
                    file_path,
                    updated_metadata
                )

                if not success:
                    return {'status': 'failed', 'error': 'Failed to write metadata'}

                # Preserve modification time if configured
                if self.config.preserve_modified_time:
                    import os
                    original_mtime = file_path.stat().st_mtime
                    original_atime = file_path.stat().st_atime
                    os.utime(file_path, (original_atime, original_mtime))

            return {
                'status': 'success',
                'operations': [op.field for op in applied_ops]
            }

        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    async def _apply_operations(self,
                              metadata: Metadata,
                              operations: List[MetadataOperation]) -> Tuple[Metadata, List[MetadataOperation]]:
        """Apply operations to metadata and return updated metadata and applied operations."""
        applied_operations = []
        metadata_dict = metadata.to_dict()

        for operation in operations:
            # Check condition
            if not self._check_condition(metadata_dict, operation.condition):
                continue

            # Apply operation based on type
            updated = False
            field_name = operation.field

            if operation.operation == OperationType.SET:
                updated = self._set_field(metadata_dict, field_name, operation.value, operation.conflict_strategy)
            elif operation.operation == OperationType.ADD:
                updated = self._add_to_field(metadata_dict, field_name, operation.value)
            elif operation.operation == OperationType.REMOVE:
                updated = self._remove_from_field(metadata_dict, field_name, operation.value)
            elif operation.operation == OperationType.CLEAR:
                if field_name in metadata_dict and metadata_dict[field_name]:
                    metadata_dict[field_name] = None
                    updated = True
            elif operation.operation == OperationType.TRANSFORM:
                updated = self._transform_field(metadata_dict, field_name, operation.pattern)
            elif operation.operation == OperationType.COPY:
                updated = self._copy_field(metadata_dict, field_name, operation.value)
            elif operation.operation == OperationType.RENAME:
                updated = self._rename_field_value(metadata_dict, field_name, operation.value, operation.pattern)

            if updated:
                applied_operations.append(operation)

        # Convert back to Metadata object
        updated_metadata = self._dict_to_metadata(metadata_dict)
        return updated_metadata, applied_operations

    def _check_condition(self, metadata_dict: Dict, condition: Dict) -> bool:
        """Check if metadata satisfies the condition."""
        if not condition:
            return True

        for field, expected in condition.items():
            actual = metadata_dict.get(field)

            if isinstance(expected, dict):
                # Support for operators like {'contains': 'rock'}
                if 'contains' in expected:
                    if not actual or expected['contains'] not in str(actual):
                        return False
                elif 'equals' in expected:
                    if actual != expected['equals']:
                        return False
                elif 'regex' in expected:
                    if not actual or not re.search(expected['regex'], str(actual)):
                        return False
            else:
                # Simple equality check
                if actual != expected:
                    return False

        return True

    def _set_field(self, metadata_dict: Dict, field: str, value: Any, strategy: ConflictStrategy) -> bool:
        """Set a field value with conflict resolution."""
        current = metadata_dict.get(field)

        if current is not None and current != value:
            if strategy == ConflictStrategy.SKIP:
                return False
            elif strategy == ConflictStrategy.MERGE:
                # For multi-value fields, merge
                if field in ['artists', 'genres']:
                    if isinstance(value, list):
                        if isinstance(current, list):
                            metadata_dict[field] = list(set(current + value))
                        else:
                            metadata_dict[field] = value
                    else:
                        if isinstance(current, list):
                            metadata_dict[field] = list(set(current + [value]))
                        else:
                            metadata_dict[field] = [current, value]
                    return True
                # For other fields, replace
                metadata_dict[field] = value
                return True
            elif strategy == ConflictStrategy.ASK:
                # In non-interactive mode, treat as skip
                return False

        metadata_dict[field] = value
        return True

    def _add_to_field(self, metadata_dict: Dict, field: str, value: Any) -> bool:
        """Add value to multi-value field."""
        current = metadata_dict.get(field, [])

        if field in ['artists', 'genres']:
            if not isinstance(current, list):
                current = [current] if current else []

            if value not in current:
                current.append(value)
                metadata_dict[field] = current
                return True
        else:
            # For single-value fields, convert to list
            if not isinstance(current, list):
                current = [current] if current else []

            if value not in current:
                current.append(value)
                metadata_dict[field] = current
                return True

        return False

    def _remove_from_field(self, metadata_dict: Dict, field: str, value: Any) -> bool:
        """Remove value from field."""
        current = metadata_dict.get(field)

        if not current:
            return False

        if isinstance(current, list):
            if value in current:
                current.remove(value)
                metadata_dict[field] = current if current else None
                return True
        elif current == value:
            metadata_dict[field] = None
            return True

        return False

    def _transform_field(self, metadata_dict: Dict, field: str, pattern: str) -> bool:
        """Transform field value using regex pattern."""
        if not pattern:
            return False

        current = metadata_dict.get(field)
        if not current:
            return False

        # Parse pattern: "s/find/replace/g" for substitution
        if pattern.startswith('s/') and pattern.count('/') >= 2:
            parts = pattern.split('/')
            if len(parts) >= 3:
                find = parts[1]
                replace = parts[2] if len(parts) < 4 else parts[2]
                flags = parts[3] if len(parts) > 3 else ''

                # Compile regex with flags
                regex_flags = 0
                if 'i' in flags:
                    regex_flags |= re.IGNORECASE
                if 'g' in flags:
                    # Global replace (default behavior)
                    pass

                # Apply transformation
                if isinstance(current, list):
                    new_values = []
                    for val in current:
                        new_val = re.sub(find, replace, str(val), flags=regex_flags)
                        new_values.append(new_val)
                    metadata_dict[field] = new_values
                else:
                    new_value = re.sub(find, replace, str(current), flags=regex_flags)
                    metadata_dict[field] = new_value

                return True

        return False

    def _copy_field(self, metadata_dict: Dict, target_field: str, source_field: str) -> bool:
        """Copy value from source field to target field."""
        source_value = metadata_dict.get(source_field)
        if source_value and source_field != target_field:
            metadata_dict[target_field] = source_value
            return True
        return False

    def _rename_field_value(self, metadata_dict: Dict, field: str, old_value: str, new_value: str) -> bool:
        """Rename a specific value in a field."""
        current = metadata_dict.get(field)
        if not current:
            return False

        if isinstance(current, list):
            if old_value in current:
                # Replace old value with new value
                current[current.index(old_value)] = new_value
                metadata_dict[field] = current
                return True
        elif current == old_value:
            metadata_dict[field] = new_value
            return True

        return False

    def _validate_metadata(self, metadata: Metadata) -> List[str]:
        """Validate metadata and return list of errors."""
        errors = []

        # Check required fields
        if not metadata.title:
            errors.append("Title is required")

        if not metadata.artists:
            errors.append("At least one artist is required")

        # Validate year
        if metadata.year and (metadata.year < 1900 or metadata.year > datetime.now().year + 1):
            errors.append(f"Invalid year: {metadata.year}")

        # Validate track number
        if metadata.track_number and metadata.track_number.number < 0:
            errors.append(f"Invalid track number: {metadata.track_number}")

        # Validate disc number
        if metadata.disc_number and metadata.disc_number < 0:
            errors.append(f"Invalid disc number: {metadata.disc_number}")

        return errors

    async def _create_backup_metadata(self, files: List[Path]) -> None:
        """Create backup of current metadata for undo support."""
        backup_data = {}

        for file_path in files:
            metadata = await self.adapter.read_metadata(file_path)
            if metadata:
                backup_data[str(file_path)] = metadata.to_dict()

        # Store backup in operation history
        if hasattr(self, '_operation_tracker'):
            await self._operation_tracker.store_backup(backup_data)

    def _dict_to_metadata(self, metadata_dict: Dict) -> Metadata:
        """Convert dictionary to Metadata object."""
        # Convert artists
        artists = frozenset()
        if 'artists' in metadata_dict and metadata_dict['artists']:
            if isinstance(metadata_dict['artists'], list):
                artists = frozenset(ArtistName(a) for a in metadata_dict['artists'])
            else:
                artists = frozenset({ArtistName(metadata_dict['artists'])})

        # Convert album artist
        albumartist = None
        if 'albumartist' in metadata_dict and metadata_dict['albumartist']:
            albumartist = ArtistName(metadata_dict['albumartist'])

        # Convert track number
        track_number = None
        if 'track_number' in metadata_dict and metadata_dict['track_number']:
            track_number = TrackNumber(str(metadata_dict['track_number']))

        return Metadata(
            title=metadata_dict.get('title'),
            artists=artists,
            album=metadata_dict.get('album'),
            year=metadata_dict.get('year'),
            genre=metadata_dict.get('genre'),
            track_number=track_number,
            disc_number=metadata_dict.get('disc_number'),
            total_discs=metadata_dict.get('total_discs'),
            albumartist=albumartist,
            composer=metadata_dict.get('composer'),
            duration_seconds=metadata_dict.get('duration_seconds'),
            bitrate=metadata_dict.get('bitrate'),
            sample_rate=metadata_dict.get('sample_rate'),
            channels=metadata_dict.get('channels'),
            date=metadata_dict.get('date'),
            location=metadata_dict.get('location'),
            file_hash=metadata_dict.get('file_hash'),
            acoustic_fingerprint=metadata_dict.get('acoustic_fingerprint')
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.executor.shutdown(wait=True)


class MetadataOperationBuilder:
    """Builder for creating metadata operations."""

    @staticmethod
    def set_genre(genre: str, conflict_strategy: ConflictStrategy = ConflictStrategy.REPLACE) -> MetadataOperation:
        """Create operation to set genre."""
        return MetadataOperation(
            field='genre',
            operation=OperationType.SET,
            value=genre,
            conflict_strategy=conflict_strategy
        )

    @staticmethod
    def set_year(year: int, conflict_strategy: ConflictStrategy = ConflictStrategy.REPLACE) -> MetadataOperation:
        """Create operation to set year."""
        return MetadataOperation(
            field='year',
            operation=OperationType.SET,
            value=year,
            conflict_strategy=conflict_strategy
        )

    @staticmethod
    def add_artist(artist: str) -> MetadataOperation:
        """Create operation to add artist."""
        return MetadataOperation(
            field='artists',
            operation=OperationType.ADD,
            value=artist
        )

    @staticmethod
    def remove_artist(artist: str) -> MetadataOperation:
        """Create operation to remove artist."""
        return MetadataOperation(
            field='artists',
            operation=OperationType.REMOVE,
            value=artist
        )

    @staticmethod
    def standardize_genre(mapping: Dict[str, str]) -> MetadataOperation:
        """Create operation to standardize genres."""
        return MetadataOperation(
            field='genre',
            operation=OperationType.TRANSFORM,
            pattern=MetadataOperationBuilder._create_genre_pattern(mapping)
        )

    @staticmethod
    def capitalize_fields(fields: List[str]) -> List[MetadataOperation]:
        """Create operations to capitalize text fields."""
        operations = []
        for field in fields:
            operations.append(MetadataOperation(
                field=field,
                operation=OperationType.TRANSFORM,
                pattern='s/\\b(\\w)/\\U\\1/g'  # Capitalize first letter of each word
            ))
        return operations

    @staticmethod
    def fix_track_numbers(prefix: str = "") -> MetadataOperation:
        """Create operation to fix track numbers."""
        pattern = f's/^({prefix})?(\\d+).*/{prefix}\\2/g'
        return MetadataOperation(
            field='track_number',
            operation=OperationType.TRANSFORM,
            pattern=pattern
        )

    @staticmethod
    def copy_album_to_artist_if_missing() -> MetadataOperation:
        """Create operation to copy album to artist if artist is missing."""
        return MetadataOperation(
            field='artists',
            operation=OperationType.COPY,
            value='album'
        )

    @staticmethod
    def rename_artist(old_name: str, new_name: str) -> MetadataOperation:
        """Create operation to rename artist."""
        return MetadataOperation(
            field='artists',
            operation=OperationType.RENAME,
            value=old_name,
            pattern=new_name
        )

    @staticmethod
    def _create_genre_pattern(mapping: Dict[str, str]) -> str:
        """Create regex pattern for genre mapping."""
        # This is simplified - in reality, you'd want to build a more complex pattern
        patterns = []
        for old_genre, new_genre in mapping.items():
            patterns.append(f's/^{re.escape(old_genre)}$/{re.escape(new_genre)}/g')
        return '|'.join(patterns)