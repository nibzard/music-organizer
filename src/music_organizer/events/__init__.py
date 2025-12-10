"""
Event System - Domain Events Architecture

This package implements an event-driven architecture for loose coupling
between bounded contexts and components of the music organizer.
"""

from .event_bus import EventBus, DomainEvent, EventHandler
from .domain_events import (
    RecordingAdded,
    RecordingModified,
    RecordingDeleted,
    DuplicateDetected,
    ClassificationCompleted,
    OrganizationCompleted,
    MetadataEnhanced,
)

__all__ = [
    # Core event system
    "EventBus",
    "DomainEvent",
    "EventHandler",
    # Domain events
    "RecordingAdded",
    "RecordingModified",
    "RecordingDeleted",
    "DuplicateDetected",
    "ClassificationCompleted",
    "OrganizationCompleted",
    "MetadataEnhanced",
]