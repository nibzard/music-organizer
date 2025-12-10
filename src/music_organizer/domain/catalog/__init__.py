"""
Catalog Context - Managing the music catalog and metadata.

This bounded context is responsible for:
- Managing recordings, releases, and artists
- Handling metadata operations
- Providing catalog-specific services
- Maintaining the integrity of the music catalog
"""

from .entities import Recording, Release, Artist, Catalog
from .value_objects import AudioPath, ArtistName, TrackNumber, Metadata, FileFormat
from .repositories import RecordingRepository, ReleaseRepository, ArtistRepository
from .services import CatalogService, MetadataService

__all__ = [
    # Entities
    "Recording",
    "Release",
    "Artist",
    "Catalog",
    # Value Objects
    "AudioPath",
    "ArtistName",
    "TrackNumber",
    "Metadata",
    "FileFormat",
    # Repositories
    "RecordingRepository",
    "ReleaseRepository",
    "ArtistRepository",
    # Services
    "CatalogService",
    "MetadataService",
]