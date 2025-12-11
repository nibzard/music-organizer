"""
Domain Events - Specific event implementations.

This module defines the specific domain events used throughout the music organizer.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

from .event_bus import DomainEvent


@dataclass(kw_only=True)
class RecordingAdded(DomainEvent):
    """Event fired when a recording is added to the catalog."""
    recording_id: str
    file_path: str
    title: str
    artist: str
    album: Optional[str] = None
    year: Optional[int] = None

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "file_path": self.file_path,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "year": self.year,
        }


@dataclass(kw_only=True)
class RecordingModified(DomainEvent):
    """Event fired when a recording's metadata is modified."""
    recording_id: str
    modified_fields: List[str]
    old_values: Dict[str, Any] = field(default_factory=dict)
    new_values: Dict[str, Any] = field(default_factory=dict)
    modification_source: str = "user"  # user, plugin, auto

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "modified_fields": self.modified_fields,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "modification_source": self.modification_source,
        }


@dataclass(kw_only=True)
class RecordingDeleted(DomainEvent):
    """Event fired when a recording is deleted from the catalog."""
    recording_id: str
    file_path: str
    title: str
    artist: str
    reason: str = "user_deleted"  # user_deleted, duplicate_removed, error

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "file_path": self.file_path,
            "title": self.title,
            "artist": self.artist,
            "reason": self.reason,
        }


@dataclass(kw_only=True)
class DuplicateDetected(DomainEvent):
    """Event fired when duplicate recordings are detected."""
    duplicate_group_id: str
    recording_ids: List[str]
    similarity_scores: Dict[str, float]
    similarity_threshold: float
    detection_method: str = "metadata"  # metadata, fingerprint, hybrid

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "duplicate_group_id": self.duplicate_group_id,
            "recording_ids": self.recording_ids,
            "similarity_scores": self.similarity_scores,
            "similarity_threshold": self.similarity_threshold,
            "detection_method": self.detection_method,
        }


@dataclass(kw_only=True)
class DuplicateResolved(DomainEvent):
    """Event fired when duplicate recordings are resolved."""
    duplicate_group_id: str
    kept_recording_id: Optional[str]  # None if all kept
    removed_recording_ids: List[str]
    resolution_strategy: str  # keep_best, keep_all, remove_duplicates

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "duplicate_group_id": self.duplicate_group_id,
            "kept_recording_id": self.kept_recording_id,
            "removed_recording_ids": self.removed_recording_ids,
            "resolution_strategy": self.resolution_strategy,
        }


@dataclass(kw_only=True)
class ClassificationCompleted(DomainEvent):
    """Event fired when a recording is classified."""
    recording_id: str
    content_type: str
    genres: List[str]
    confidence: float
    classification_source: str = "automatic"  # automatic, user, plugin

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "content_type": self.content_type,
            "genres": self.genres,
            "confidence": self.confidence,
            "classification_source": self.classification_source,
        }


@dataclass(kw_only=True)
class OrganizationCompleted(DomainEvent):
    """Event fired when file organization is completed."""
    source_directory: str
    target_directory: str
    organized_count: int
    moved_count: int
    skipped_count: int
    error_count: int
    conflicts_resolved: int

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "source_directory": self.source_directory,
            "target_directory": self.target_directory,
            "organized_count": self.organized_count,
            "moved_count": self.moved_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "conflicts_resolved": self.conflicts_resolved,
        }


@dataclass(kw_only=True)
class FileMoved(DomainEvent):
    """Event fired when a file is moved during organization."""
    recording_id: str
    source_path: str
    target_path: str
    conflict_resolved: bool = False
    conflict_strategy: Optional[str] = None

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "conflict_resolved": self.conflict_resolved,
            "conflict_strategy": self.conflict_strategy,
        }


@dataclass(kw_only=True)
class MetadataEnhanced(DomainEvent):
    """Event fired when metadata is enhanced from external sources."""
    recording_id: str
    enhanced_fields: List[str]
    source: str  # musicbrainz, acoustid, user_input
    enhancement_confidence: float
    old_values: Dict[str, Any] = field(default_factory=dict)
    new_values: Dict[str, Any] = field(default_factory=dict)

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "enhanced_fields": self.enhanced_fields,
            "source": self.source,
            "enhancement_confidence": self.enhancement_confidence,
            "old_values": self.old_values,
            "new_values": self.new_values,
        }


@dataclass(kw_only=True)
class LibraryScanned(DomainEvent):
    """Event fired when a library scan is completed."""
    source_directory: str
    total_files_found: int
    total_files_imported: int
    duplicates_found: int
    errors: List[str] = field(default_factory=list)
    scan_duration_seconds: Optional[float] = None

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "source_directory": self.source_directory,
            "total_files_found": self.total_files_found,
            "total_files_imported": self.total_files_imported,
            "duplicates_found": self.duplicates_found,
            "errors": self.errors,
            "scan_duration_seconds": self.scan_duration_seconds,
        }


@dataclass(kw_only=True)
class PluginExecuted(DomainEvent):
    """Event fired when a plugin is executed."""
    plugin_name: str
    plugin_type: str  # metadata, classification, organization, output
    recording_ids: List[str]
    execution_duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "plugin_type": self.plugin_type,
            "recording_ids": self.recording_ids,
            "execution_duration_ms": self.execution_duration_ms,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass(kw_only=True)
class UserCorrectionApplied(DomainEvent):
    """Event fired when a user corrects an automatic classification or organization."""
    recording_id: str
    correction_type: str  # genre, content_type, organization_path
    original_value: Any
    corrected_value: Any
    correction_source: str  # manual_ui, bulk_edit

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "correction_type": self.correction_type,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "correction_source": self.correction_source,
        }


@dataclass(kw_only=True)
class PerformanceWarning(DomainEvent):
    """Event fired when a performance issue is detected."""
    operation: str
    duration_ms: float
    threshold_ms: float
    details: Dict[str, Any] = field(default_factory=dict)

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "threshold_ms": self.threshold_ms,
            "details": self.details,
        }