"""
Domain module for Music Organizer.

This package contains domain models, value objects, and domain services
following Domain-Driven Design principles.
"""

from .value_objects import (
    AudioPath,
    ArtistName,
    ContentPattern,
    FileFormat,
    Metadata,
    TrackNumber,
)
from .entities import (
    AudioLibrary,
    Collection,
    DuplicateResolutionMode,
    Recording,
    Release,
)

__all__ = [
    "AudioPath",
    "ArtistName",
    "ContentPattern",
    "FileFormat",
    "Metadata",
    "TrackNumber",
    "AudioLibrary",
    "Collection",
    "DuplicateResolutionMode",
    "Recording",
    "Release",
]