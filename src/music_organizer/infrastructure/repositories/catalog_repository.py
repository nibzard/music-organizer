"""
Catalog Repository Implementations.

This module provides in-memory and file-based repository implementations
for the Catalog bounded context entities.
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any
from datetime import datetime
import asyncio

from ...domain.catalog.entities import Recording, Release, Artist, Catalog
from ...domain.catalog.repositories import (
    RecordingRepository,
    ReleaseRepository,
    ArtistRepository,
    CatalogRepository,
)
from ...domain.catalog.value_objects import ArtistName


class InMemoryRecordingRepository(RecordingRepository):
    """In-memory implementation of RecordingRepository for testing and development."""

    def __init__(self):
        self._recordings: Dict[str, Recording] = {}
        self._path_index: Dict[str, str] = {}  # Path string -> Recording ID

    async def save(self, recording: Recording) -> None:
        """Save a recording."""
        recording_id = str(id(recording))
        self._recordings[recording_id] = recording
        self._path_index[str(recording.path.path)] = recording_id

    async def find_by_id(self, recording_id: str) -> Optional[Recording]:
        """Find a recording by its ID."""
        return self._recordings.get(recording_id)

    async def find_by_path(self, path: Any) -> Optional[Recording]:
        """Find a recording by its file path."""
        path_str = str(path.path) if hasattr(path, 'path') else str(path)
        recording_id = self._path_index.get(path_str)
        return self._recordings.get(recording_id) if recording_id else None

    async def find_by_artist(self, artist: ArtistName) -> List[Recording]:
        """Find all recordings by an artist."""
        artist_str = str(artist).lower()
        matching_recordings = []

        for recording in self._recordings.values():
            if any(str(a).lower() == artist_str for a in recording.artists):
                matching_recordings.append(recording)

        return matching_recordings

    async def find_by_title(self, title: str) -> List[Recording]:
        """Find recordings by title (partial match)."""
        title_lower = title.lower()
        matching_recordings = []

        for recording in self._recordings.values():
            if title_lower in recording.title.lower():
                matching_recordings.append(recording)

        return matching_recordings

    async def find_duplicates(self, similarity_threshold: float = 0.85) -> List[List[Recording]]:
        """Find groups of duplicate recordings."""
        duplicate_groups = []
        processed = set()

        recordings = list(self._recordings.values())

        for i, recording1 in enumerate(recordings):
            if recording1 in processed:
                continue

            group = [recording1]
            processed.add(recording1)

            for recording2 in recordings[i+1:]:
                if recording2 in processed:
                    continue

                similarity = recording1.calculate_similarity(recording2)
                if similarity >= similarity_threshold:
                    group.append(recording2)
                    processed.add(recording2)

            if len(group) > 1:
                duplicate_groups.append(group)

        return duplicate_groups

    async def find_all(self, limit: Optional[int] = None, offset: int = 0) -> Iterator[Recording]:
        """Find all recordings with optional pagination."""
        recordings = list(self._recordings.values())[offset:]
        if limit:
            recordings = recordings[:limit]

        for recording in recordings:
            yield recording

    async def delete(self, recording: Recording) -> None:
        """Delete a recording."""
        recording_id = str(id(recording))
        if recording_id in self._recordings:
            del self._recordings[recording_id]
            # Clean up path index
            path_str = str(recording.path.path)
            if path_str in self._path_index:
                del self._path_index[path_str]

    async def count(self) -> int:
        """Get total count of recordings."""
        return len(self._recordings)


class InMemoryReleaseRepository(ReleaseRepository):
    """In-memory implementation of ReleaseRepository."""

    def __init__(self):
        self._releases: Dict[str, Release] = {}
        self._title_artist_index: Dict[str, str] = {}  # "title-artist" -> Release ID

    async def save(self, release: Release) -> None:
        """Save a release."""
        release_id = str(id(release))
        self._releases[release_id] = release

        # Update title-artist index
        key = f"{release.title.lower()}-{str(release.primary_artist).lower()}"
        self._title_artist_index[key] = release_id

    async def find_by_id(self, release_id: str) -> Optional[Release]:
        """Find a release by its ID."""
        return self._releases.get(release_id)

    async def find_by_title_and_artist(self, title: str, artist: ArtistName) -> Optional[Release]:
        """Find a release by title and artist."""
        key = f"{title.lower()}-{str(artist).lower()}"
        release_id = self._title_artist_index.get(key)
        return self._releases.get(release_id) if release_id else None

    async def find_by_artist(self, artist: ArtistName) -> List[Release]:
        """Find all releases by an artist."""
        artist_str = str(artist).lower()
        matching_releases = []

        for release in self._releases.values():
            if str(release.primary_artist).lower() == artist_str:
                matching_releases.append(release)

        return matching_releases

    async def find_by_year(self, year: int) -> List[Release]:
        """Find releases by year."""
        return [r for r in self._releases.values() if r.year == year]

    async def find_by_genre(self, genre: str) -> List[Release]:
        """Find releases by genre."""
        genre_lower = genre.lower()
        matching_releases = []

        for release in self._releases.values():
            if release.genre and genre_lower in release.genre.lower():
                matching_releases.append(release)

        return matching_releases

    async def find_all(self, limit: Optional[int] = None, offset: int = 0) -> Iterator[Release]:
        """Find all releases with optional pagination."""
        releases = list(self._releases.values())[offset:]
        if limit:
            releases = releases[:limit]

        for release in releases:
            yield release

    async def delete(self, release: Release) -> None:
        """Delete a release."""
        release_id = str(id(release))
        if release_id in self._releases:
            del self._releases[release_id]

            # Clean up title-artist index
            key = f"{release.title.lower()}-{str(release.primary_artist).lower()}"
            if key in self._title_artist_index:
                del self._title_artist_index[key]

    async def count(self) -> int:
        """Get total count of releases."""
        return len(self._releases)


class InMemoryArtistRepository(ArtistRepository):
    """In-memory implementation of ArtistRepository."""

    def __init__(self):
        self._artists: Dict[str, Artist] = {}
        self._name_index: Dict[str, str] = {}  # Normalized name -> Artist ID

    async def save(self, artist: Artist) -> None:
        """Save an artist."""
        artist_id = str(id(artist))
        self._artists[artist_id] = artist

        # Update name index
        name_key = str(artist.name).lower()
        self._name_index[name_key] = artist_id

    async def find_by_name(self, name: ArtistName) -> Optional[Artist]:
        """Find an artist by name."""
        name_key = str(name).lower()
        artist_id = self._name_index.get(name_key)
        return self._artists.get(artist_id) if artist_id else None

    async def find_by_partial_name(self, partial_name: str) -> List[Artist]:
        """Find artists with names matching partial string."""
        partial_lower = partial_name.lower()
        matching_artists = []

        for artist in self._artists.values():
            if partial_lower in str(artist.name).lower():
                matching_artists.append(artist)

        return matching_artists

    async def find_collaborators(self, artist: Artist) -> List[Artist]:
        """Find all artists who have collaborated with the given artist."""
        collaborators = []
        artist_name = str(artist.name)

        for other_artist in self._artists.values():
            if other_artist != artist and artist_name in other_artist.collaborators:
                collaborators.append(other_artist)

        return collaborators

    async def find_all(self, limit: Optional[int] = None, offset: int = 0) -> Iterator[Artist]:
        """Find all artists with optional pagination."""
        artists = list(self._artists.values())[offset:]
        if limit:
            artists = artists[:limit]

        for artist in artists:
            yield artist

    async def delete(self, artist: Artist) -> None:
        """Delete an artist."""
        artist_id = str(id(artist))
        if artist_id in self._artists:
            del self._artists[artist_id]

            # Clean up name index
            name_key = str(artist.name).lower()
            if name_key in self._name_index:
                del self._name_index[name_key]

    async def count(self) -> int:
        """Get total count of artists."""
        return len(self._artists)


class InMemoryCatalogRepository(CatalogRepository):
    """In-memory implementation of CatalogRepository."""

    def __init__(self):
        self._catalogs: Dict[str, Catalog] = {}

    async def save(self, catalog: Catalog) -> None:
        """Save the catalog."""
        self._catalogs[catalog.name] = catalog

    async def load(self, catalog_name: str) -> Optional[Catalog]:
        """Load a catalog by name."""
        return self._catalogs.get(catalog_name)

    async def get_statistics(self, catalog_name: str) -> Dict[str, Any]:
        """Get catalog statistics."""
        catalog = await self.load(catalog_name)
        if catalog:
            return catalog.get_statistics()
        return {}