"""Organization Context Entities.

This module defines the core entities for the Organization bounded context.
The Organization context is responsible for managing file organization rules and operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from .value_objects import TargetPath, OrganizationPattern, ConflictStrategy


class OperationStatus(Enum):
    """Status of organization operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class ConflictResolution:
    """Represents how a file conflict was resolved."""

    source_path: Path
    target_path: Path
    conflicting_path: Path
    strategy: ConflictStrategy
    final_path: Path
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def description(self) -> str:
        """Get a human-readable description of the resolution."""
        if self.strategy == ConflictStrategy.SKIP:
            return f"Skipped moving {self.source_path.name} (conflict with existing file)"
        elif self.strategy == ConflictStrategy.REPLACE:
            return f"Replaced existing {self.conflicting_path.name}"
        elif self.strategy == ConflictStrategy.RENAME:
            return f"Renamed to {self.final_path.name} to avoid conflict"
        elif self.strategy == ConflictStrategy.KEEP_BOTH:
            return f"Kept both files: {self.target_path.name} and {self.final_path.name}"
        else:  # APPEND_SUFFIX
            return f"Appended suffix: {self.final_path.name}"


@dataclass
class MovedFile:
    """Represents a file that has been moved as part of organization."""

    source_path: Path
    target_path: Path
    timestamp: datetime = field(default_factory=datetime.now)
    status: OperationStatus = OperationStatus.PENDING
    error_message: Optional[str] = None
    conflict_resolution: Optional[ConflictResolution] = None
    checksum_before: Optional[str] = None
    checksum_after: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        """Check if the move was successful."""
        return self.status == OperationStatus.COMPLETED

    @property
    def has_conflict(self) -> bool:
        """Check if there was a conflict during the move."""
        return self.conflict_resolution is not None

    def mark_completed(self) -> None:
        """Mark the move as completed."""
        self.status = OperationStatus.COMPLETED

    def mark_failed(self, error: str) -> None:
        """Mark the move as failed with an error message."""
        self.status = OperationStatus.FAILED
        self.error_message = error

    def mark_skipped(self, reason: str) -> None:
        """Mark the move as skipped."""
        self.status = OperationStatus.SKIPPED
        self.error_message = reason

    def set_conflict_resolution(self, resolution: ConflictResolution) -> None:
        """Set the conflict resolution for this move."""
        self.conflict_resolution = resolution


@dataclass
class OrganizationRule:
    """
    Represents a rule for organizing files.

    Rules define how files should be organized based on conditions
    and specify the patterns to apply.
    """

    # Rule identity
    name: str
    description: Optional[str] = None
    enabled: bool = True
    priority: int = 0  # Higher priority rules are evaluated first

    # Rule conditions
    genre_patterns: List[str] = field(default_factory=list)
    artist_patterns: List[str] = field(default_factory=list)
    year_range: Optional[tuple[int, int]] = None  # (start_year, end_year)
    file_types: List[str] = field(default_factory=list)  # File extensions

    # Organization pattern
    pattern: Optional[OrganizationPattern] = None

    # Rule metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: Optional[datetime] = None
    usage_count: int = 0

    def matches(
        self,
        metadata: Any,  # catalog.Metadata
        file_path: Optional[Path] = None
    ) -> bool:
        """Check if this rule matches the given file."""
        if not self.enabled:
            return False

        # Check genre patterns
        if self.genre_patterns and metadata.genre:
            if not any(pattern.lower() in metadata.genre.lower() for pattern in self.genre_patterns):
                return False

        # Check artist patterns
        if self.artist_patterns and metadata.artists:
            artists_str = " ".join(str(a) for a in metadata.artists).lower()
            if not any(pattern.lower() in artists_str for pattern in self.artist_patterns):
                return False

        # Check year range
        if self.year_range and metadata.year:
            start_year, end_year = self.year_range
            if not (start_year <= metadata.year <= end_year):
                return False

        # Check file types
        if self.file_types and file_path:
            if file_path.suffix.lower() not in self.file_types:
                return False

        return True

    def apply(self, metadata: Any, source_path: Path) -> Optional[TargetPath]:
        """Apply this rule to generate a target path."""
        if not self.pattern:
            return None

        if not self.matches(metadata, source_path):
            return None

        # Generate target path using the pattern
        target_path = self.pattern.generate_path(metadata, source_path)

        # Create TargetPath object
        return TargetPath(
            path=target_path,
            pattern_used=self.pattern,
            original_path=source_path
        )

    def increment_usage(self) -> None:
        """Increment the usage count of this rule."""
        self.usage_count += 1
        self.last_modified = datetime.now()

    def update_pattern(self, pattern: OrganizationPattern) -> None:
        """Update the organization pattern for this rule."""
        self.pattern = pattern
        self.last_modified = datetime.now()


@dataclass
class FolderStructure:
    """
    Represents a folder structure configuration.

    A folder structure defines the hierarchy and organization of directories
    in the target location.
    """

    # Structure identity
    name: str
    description: Optional[str] = None

    # Structure definition
    hierarchy: List[str]  # List of folder names or patterns
    root_path: Optional[Path] = None

    # Organization settings
    create_empty_folders: bool = False
    collapse_single_folders: bool = True  # If only one subfolder, flatten it
    max_depth: Optional[int] = None

    # Special folders
    folders_for_compilations: bool = True
    folders_for_soundtracks: bool = True
    folders_for_live: bool = True
    folders_for_dj_mixes: bool = True

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: Optional[datetime] = None

    def get_folder_path(
        self,
        metadata: Any,  # catalog.Metadata
        base_path: Path
    ) -> Path:
        """Get the full folder path for a recording."""
        folder_path = base_path

        # Determine special folder handling
        special_folder = self._get_special_folder(metadata)
        if special_folder:
            folder_path = folder_path / special_folder

        # Process hierarchy
        for folder in self.hierarchy:
            # Replace variables in folder names
            resolved_folder = self._resolve_folder_name(folder, metadata)

            if resolved_folder:
                folder_path = folder_path / resolved_folder

            # Check depth limit
            if self.max_depth and len(folder_path.relative_to(base_path).parts) >= self.max_depth:
                break

        # Apply folder collapsing if enabled
        if self.collapse_single_folders:
            folder_path = self._collapse_single_folders(folder_path, base_path)

        return folder_path

    def _get_special_folder(self, metadata: Any) -> Optional[str]:
        """Determine if a special folder should be used."""
        if self.folders_for_compilations and metadata.is_compilation:
            return "Compilations"
        elif self.folders_for_soundtracks and "soundtrack" in (metadata.genre or "").lower():
            return "Soundtracks"
        elif self.folders_for_live and metadata.is_live:
            return "Live Recordings"
        elif self.folders_for_dj_mixes and "dj mix" in (metadata.genre or "").lower():
            return "DJ Mixes"

        return None

    def _resolve_folder_name(self, folder: str, metadata: Any) -> str:
        """Resolve variables in a folder name."""
        # Simple variable replacement
        replacements = {
            "artist": str(metadata.artists[0]) if metadata.artists else "Unknown Artist",
            "albumartist": str(metadata.albumartist) if metadata.albumartist else "",
            "album": metadata.album or "",
            "year": str(metadata.year) if metadata.year else "",
            "decade": f"{(metadata.year // 10) * 10}s" if metadata.year else "",
            "genre": metadata.genre or "",
        }

        for key, value in replacements.items():
            folder = folder.replace(f"{{{key}}}", str(value))

        # Clean up the folder name
        invalid_chars = '<>:"|?*'
        for char in invalid_chars:
            folder = folder.replace(char, '')

        folder = folder.replace('/', '_')

        return folder

    def _collapse_single_folders(self, path: Path, base: Path) -> Path:
        """Collapse single intermediate folders."""
        # This is a simplified implementation
        # In practice, you'd want to check if folders contain only one subfolder
        # recursively collapse them
        return path

    def validate_structure(self) -> List[str]:
        """Validate the folder structure definition."""
        errors = []

        if not self.hierarchy:
            errors.append("Folder hierarchy cannot be empty")

        # Check for invalid patterns
        for folder in self.hierarchy:
            if not folder or folder.isspace():
                errors.append(f"Invalid folder name in hierarchy: '{folder}'")

        return errors

    def update_hierarchy(self, hierarchy: List[str]) -> None:
        """Update the folder hierarchy."""
        self.hierarchy = hierarchy.copy()
        self.last_modified = datetime.now()


@dataclass
class OrganizationSession:
    """
    Represents a session of organizing files.

    A session tracks all file moves and their status.
    """

    # Session identity
    id: str
    source_directory: Path
    target_directory: Path

    # Session configuration
    rules: List[OrganizationRule] = field(default_factory=list)
    folder_structure: Optional[FolderStructure] = None
    default_conflict_strategy: ConflictStrategy = ConflictStrategy.SKIP

    # Session state
    moved_files: List[MovedFile] = field(default_factory=list)
    status: OperationStatus = OperationStatus.PENDING

    # Session metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_count: int = 0
    success_count: int = 0
    skip_count: int = 0

    @property
    def is_active(self) -> bool:
        """Check if the session is currently active."""
        return self.status == OperationStatus.IN_PROGRESS

    @property
    def is_completed(self) -> bool:
        """Check if the session is completed."""
        return self.status == OperationStatus.COMPLETED

    @property
    def total_files(self) -> int:
        """Get the total number of files processed."""
        return len(self.moved_files)

    @property
    def progress_percentage(self) -> float:
        """Get the progress as a percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.success_count + self.error_count + self.skip_count) / self.total_files * 100

    def start(self) -> None:
        """Start the organization session."""
        self.status = OperationStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def complete(self) -> None:
        """Complete the organization session."""
        self.status = OperationStatus.COMPLETED
        self.completed_at = datetime.now()

    def add_moved_file(self, moved_file: MovedFile) -> None:
        """Add a moved file to the session."""
        self.moved_files.append(moved_file)

        # Update counters
        if moved_file.is_successful:
            self.success_count += 1
        elif moved_file.status == OperationStatus.FAILED:
            self.error_count += 1
        elif moved_file.status == OperationStatus.SKIPPED:
            self.skip_count += 1

    def find_rule_for_file(self, metadata: Any, file_path: Path) -> Optional[OrganizationRule]:
        """Find the best matching rule for a file."""
        # Sort rules by priority (highest first)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            if rule.matches(metadata, file_path):
                return rule

        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the organization session."""
        return {
            "session_id": self.id,
            "source_directory": str(self.source_directory),
            "target_directory": str(self.target_directory),
            "status": self.status.value,
            "total_files": self.total_files,
            "successful": self.success_count,
            "failed": self.error_count,
            "skipped": self.skip_count,
            "progress_percentage": self.progress_percentage,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_minutes": (
                (self.completed_at - self.started_at).total_seconds() / 60
                if self.started_at and self.completed_at
                else None
            ),
        }