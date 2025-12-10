"""
Integration Service - Bounded Context Integration.

This service provides integration points between bounded contexts,
ensuring proper communication and data flow while maintaining context boundaries.
"""

from typing import Any, Dict, List, Optional, Type, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..catalog import Recording as CatalogRecording, Metadata
from ..organization import TargetPath, OrganizationRule
from ..classification import ContentTypeEnum, GenreClassification


class ContextEvent:
    """Base class for events that cross context boundaries."""
    pass


@dataclass
class RecordingAddedEvent(ContextEvent):
    """Event fired when a recording is added to the catalog."""
    recording_id: str
    recording: CatalogRecording
    timestamp: datetime


@dataclass
class RecordingClassifiedEvent(ContextEvent):
    """Event fired when a recording is classified."""
    recording_id: str
    content_type: ContentTypeEnum
    genres: List[str]
    confidence: float
    timestamp: datetime


@dataclass
class RecordingMovedEvent(ContextEvent):
    """Event fired when a recording is moved by organization."""
    recording_id: str
    source_path: str
    target_path: str
    timestamp: datetime


@dataclass
class DuplicatesDetectedEvent(ContextEvent):
    """Event fired when duplicates are detected."""
    duplicate_group_id: str
    recording_ids: List[str]
    similarity_threshold: float
    timestamp: datetime


class ContextIntegrationService:
    """
    Manages integration between bounded contexts.

    This service provides a clean API for contexts to communicate
    without creating tight coupling between them.
    """

    def __init__(self):
        self._event_handlers: Dict[Type[ContextEvent], List[Callable]] = {}
        self._context_adapters: Dict[str, Dict[str, Callable]] = {
            "catalog": {},
            "organization": {},
            "classification": {}
        }
        self._shared_data: Dict[str, Any] = {}

    def register_event_handler(self, event_type: Type[ContextEvent], handler: Callable) -> None:
        """Register a handler for cross-context events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def publish_event(self, event: ContextEvent) -> None:
        """Publish an event to all registered handlers."""
        event_type = type(event)
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    # Log error but continue with other handlers
                    print(f"Error in event handler: {e}")

    def register_adapter(
        self,
        source_context: str,
        target_context: str,
        adapter: Callable
    ) -> None:
        """Register an adapter for converting data between contexts."""
        if source_context not in self._context_adapters:
            self._context_adapters[source_context] = {}
        self._context_adapters[source_context][target_context] = adapter

    def adapt_data(self, data: Any, source_context: str, target_context: str) -> Any:
        """Adapt data from one context to another."""
        adapter = self._context_adapters.get(source_context, {}).get(target_context)
        if adapter:
            return adapter(data)
        return data

    def set_shared_data(self, key: str, value: Any) -> None:
        """Store data that needs to be shared across contexts."""
        self._shared_data[key] = value

    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """Retrieve shared data."""
        return self._shared_data.get(key, default)


# Standard adapters for common conversions

def catalog_to_organization(recording: CatalogRecording) -> Dict[str, Any]:
    """
    Convert a Catalog Recording to Organization context data.
    """
    return {
        "source_path": str(recording.path.path),
        "metadata": {
            "title": recording.title,
            "artists": [str(a) for a in recording.artists],
            "album": recording.album,
            "year": recording.year,
            "genre": recording.metadata.genre,
            "track_number": recording.track_number.formatted() if recording.track_number else None,
            "content_type": recording.content_type,
        },
        "classification": {
            "genres": list(recording.genre_classifications),
            "energy_level": recording.energy_level,
        }
    }


def classification_to_catalog(classification_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Classification result to Catalog context updates.
    """
    updates = {}

    if classification_result.get("content_type") != ContentTypeEnum.UNKNOWN:
        updates["content_type"] = classification_result["content_type"].value

    if classification_result.get("genres"):
        updates["genre_classifications"] = classification_result["genres"]

    if classification_result.get("energy_level"):
        updates["energy_level"] = classification_result["energy_level"].value

    return updates


def organization_to_catalog(moved_file: Any) -> Dict[str, Any]:
    """
    Convert Organization moved file to Catalog context updates.
    """
    return {
        "new_path": str(moved_file.target_path),
        "move_history": {
            "from": str(moved_file.source_path),
            "to": str(moved_file.target_path),
            "timestamp": moved_file.timestamp.isoformat(),
        }
    }


def create_standard_integrations(integration_service: ContextIntegrationService) -> None:
    """
    Register standard integrations between contexts.
    """
    # Register standard adapters
    integration_service.register_adapter(
        "catalog",
        "organization",
        catalog_to_organization
    )

    integration_service.register_adapter(
        "classification",
        "catalog",
        classification_to_catalog
    )

    integration_service.register_adapter(
        "organization",
        "catalog",
        organization_to_catalog
    )

    # Register standard event handlers
    def handle_recording_added(event: RecordingAddedEvent) -> None:
        """Handle when a recording is added to the catalog."""
        # Could trigger classification, organization checks, etc.
        pass

    def handle_recording_classified(event: RecordingClassifiedEvent) -> None:
        """Handle when a recording is classified."""
        # Could trigger organization rule application, genre-based grouping, etc.
        pass

    def handle_recording_moved(event: RecordingMovedEvent) -> None:
        """Handle when a recording is moved."""
        # Could trigger catalog path updates, playlist refresh, etc.
        pass

    def handle_duplicates_detected(event: DuplicatesDetectedEvent) -> None:
        """Handle when duplicates are detected."""
        # Could trigger duplicate resolution workflow, user notifications, etc.
        pass

    integration_service.register_event_handler(RecordingAddedEvent, handle_recording_added)
    integration_service.register_event_handler(RecordingClassifiedEvent, handle_recording_classified)
    integration_service.register_event_handler(RecordingMovedEvent, handle_recording_moved)
    integration_service.register_event_handler(DuplicatesDetectedEvent, handle_duplicates_detected)