"""Catalog Context Repository Interfaces.

This module defines repository interfaces for the Catalog bounded context.
Repositories provide abstraction over data storage and retrieval.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Iterator

from .entities import Recording, Release, Artist, Catalog
from .value_objects import ArtistName, AudioPath


class RecordingRepository(ABC):
    """Repository for Recording entities."""

    @abstractmethod
    async def save(self, recording: Recording) -> None:
        """Save a recording."""
        pass

    @abstractmethod
    async def find_by_id(self, recording_id: str) -> Optional[Recording]:
        """Find a recording by its ID."""
        pass

    @abstractmethod
    async def find_by_path(self, path: AudioPath) -> Optional[Recording]:
        """Find a recording by its file path."""
        pass

    @abstractmethod
    async def find_by_artist(self, artist: ArtistName) -> List[Recording]:
        """Find all recordings by an artist."""
        pass

    @abstractmethod
    async def find_by_title(self, title: str) -> List[Recording]:
        """Find recordings by title (partial match)."""
        pass

    @abstractmethod
    async def find_duplicates(self, similarity_threshold: float = 0.85) -> List[List[Recording]]:
        """Find groups of duplicate recordings."""
        pass

    @abstractmethod
    async def find_all(self, limit: Optional[int] = None, offset: int = 0) -> Iterator[Recording]:
        """Find all recordings with optional pagination."""
        pass

    @abstractmethod
    async def delete(self, recording: Recording) -> None:
        """Delete a recording."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total count of recordings."""
        pass


class ReleaseRepository(ABC):
    """Repository for Release entities."""

    @abstractmethod
    async def save(self, release: Release) -> None:
        """Save a release."""
        pass

    @abstractmethod
    async def find_by_id(self, release_id: str) -> Optional[Release]:
        """Find a release by its ID."""
        pass

    @abstractmethod
    async def find_by_title_and_artist(self, title: str, artist: ArtistName) -> Optional[Release]:
        """Find a release by title and artist."""
        pass

    @abstractmethod
    async def find_by_artist(self, artist: ArtistName) -> List[Release]:
        """Find all releases by an artist."""
        pass

    @abstractmethod
    async def find_by_year(self, year: int) -> List[Release]:
        """Find releases by year."""
        pass

    @abstractmethod
    async def find_by_genre(self, genre: str) -> List[Release]:
        """Find releases by genre."""
        pass

    @abstractmethod
    async def find_all(self, limit: Optional[int] = None, offset: int = 0) -> Iterator[Release]:
        """Find all releases with optional pagination."""
        pass

    @abstractmethod
    async def delete(self, release: Release) -> None:
        """Delete a release."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total count of releases."""
        pass


class ArtistRepository(ABC):
    """Repository for Artist entities."""

    @abstractmethod
    async def save(self, artist: Artist) -> None:
        """Save an artist."""
        pass

    @abstractmethod
    async def find_by_name(self, name: ArtistName) -> Optional[Artist]:
        """Find an artist by name."""
        pass

    @abstractmethod
    async def find_by_partial_name(self, partial_name: str) -> List[Artist]:
        """Find artists with names matching partial string."""
        pass

    @abstractmethod
    async def find_collaborators(self, artist: Artist) -> List[Artist]:
        """Find all artists who have collaborated with the given artist."""
        pass

    @abstractmethod
    async def find_all(self, limit: Optional[int] = None, offset: int = 0) -> Iterator[Artist]:
        """Find all artists with optional pagination."""
        pass

    @abstractmethod
    async def delete(self, artist: Artist) -> None:
        """Delete an artist."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total count of artists."""
        pass


class CatalogRepository(ABC):
    """Repository for Catalog entity (root aggregate)."""

    @abstractmethod
    async def save(self, catalog: Catalog) -> None:
        """Save the catalog."""
        pass

    @abstractmethod
    async def load(self, catalog_name: str) -> Optional[Catalog]:
        """Load a catalog by name."""
        pass

    @abstractmethod
    async def get_statistics(self, catalog_name: str) -> Dict[str, Any]:
        """Get catalog statistics."""
        pass