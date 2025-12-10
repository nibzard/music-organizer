"""
Repository Implementations - Infrastructure Layer

This package contains repository implementations for data access,
following the Repository pattern from Domain-Driven Design.
"""

from .catalog_repository import (
    InMemoryRecordingRepository,
    InMemoryReleaseRepository,
    InMemoryArtistRepository,
    InMemoryCatalogRepository,
)
from .file_based_repository import FileBasedRecordingRepository

__all__ = [
    # Catalog repositories
    "InMemoryRecordingRepository",
    "InMemoryReleaseRepository",
    "InMemoryArtistRepository",
    "InMemoryCatalogRepository",
    "FileBasedRecordingRepository",
]