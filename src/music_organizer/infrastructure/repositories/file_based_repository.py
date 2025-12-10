"""
File-based Repository Implementations.

This module provides file-based repository implementations for persistence
using JSON files for storage.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any
from datetime import datetime
import aiofiles
from concurrent.futures import ThreadPoolExecutor

from ...domain.catalog.entities import Recording, Release, Artist, Catalog
from ...domain.catalog.repositories import (
    RecordingRepository,
    ReleaseRepository,
    ArtistRepository,
    CatalogRepository,
)
from ...domain.catalog.value_objects import ArtistName, AudioPath, TrackNumber, Metadata, FileFormat
from ...infrastructure.adapters.audio_file_adapter import AudioFileToRecordingAdapter


class FileBasedRecordingRepository(RecordingRepository):
    """File-based implementation of RecordingRepository using JSON storage."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._recordings_file = self.storage_dir / "recordings.json"
        self._index_file = self.storage_dir / "recordings_index.json"
        self._adapter = AudioFileToRecordingAdapter()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._cache: Optional[Dict[str, Any]] = None

    async def _load_data(self) -> Dict[str, Any]:
        """Load all recordings data from files."""
        if self._cache is not None:
            return self._cache

        def _load():
            # Load recordings
            if self._recordings_file.exists():
                with open(self._recordings_file, 'r') as f:
                    recordings = json.load(f)
            else:
                recordings = {}

            # Load index
            if self._index_file.exists():
                with open(self._index_file, 'r') as f:
                    index = json.load(f)
            else:
                index = {}

            return {
                "recordings": recordings,
                "index": index
            }

        loop = asyncio.get_event_loop()
        self._cache = await loop.run_in_executor(self._executor, _load)
        return self._cache

    async def _save_data(self, data: Dict[str, Any]) -> None:
        """Save recordings data to files."""
        def _save():
            # Save recordings
            with open(self._recordings_file, 'w') as f:
                json.dump(data["recordings"], f, indent=2)

            # Save index
            with open(self._index_file, 'w') as f:
                json.dump(data["index"], f, indent=2)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, _save)
        self._cache = data

    async def save(self, recording: Recording) -> None:
        """Save a recording."""
        data = await self._load_data()

        # Convert recording to dictionary
        recording_dict = self._adapter.to_audio_file(recording)
        recording_dict["id"] = str(id(recording))

        # Save recording
        data["recordings"][recording_dict["id"]] = recording_dict

        # Update index
        path_str = str(recording.path.path)
        data["index"][path_str] = recording_dict["id"]

        # Save to file
        await self._save_data(data)

    async def find_by_id(self, recording_id: str) -> Optional[Recording]:
        """Find a recording by its ID."""
        data = await self._load_data()
        recording_dict = data["recordings"].get(recording_id)

        if recording_dict:
            return self._dict_to_recording(recording_dict)

        return None

    async def find_by_path(self, path: Any) -> Optional[Recording]:
        """Find a recording by its file path."""
        data = await self._load_data()
        path_str = str(path.path) if hasattr(path, 'path') else str(path)
        recording_id = data["index"].get(path_str)

        if recording_id:
            return await self.find_by_id(recording_id)

        return None

    async def find_by_artist(self, artist: ArtistName) -> List[Recording]:
        """Find all recordings by an artist."""
        data = await self._load_data()
        artist_str = str(artist).lower()
        matching_recordings = []

        for recording_dict in data["recordings"].values():
            metadata = recording_dict.get("metadata", {})
            artists = metadata.get("artists", [])
            if any(artist_str in a.lower() for a in artists):
                recording = self._dict_to_recording(recording_dict)
                if recording:
                    matching_recordings.append(recording)

        return matching_recordings

    async def find_by_title(self, title: str) -> List[Recording]:
        """Find recordings by title (partial match)."""
        data = await self._load_data()
        title_lower = title.lower()
        matching_recordings = []

        for recording_dict in data["recordings"].values():
            metadata = recording_dict.get("metadata", {})
            recording_title = metadata.get("title", "")
            if title_lower in recording_title.lower():
                recording = self._dict_to_recording(recording_dict)
                if recording:
                    matching_recordings.append(recording)

        return matching_recordings

    async def find_duplicates(self, similarity_threshold: float = 0.85) -> List[List[Recording]]:
        """Find groups of duplicate recordings."""
        # Load all recordings
        all_recordings = []
        data = await self._load_data()

        for recording_dict in data["recordings"].values():
            recording = self._dict_to_recording(recording_dict)
            if recording:
                all_recordings.append(recording)

        # Find duplicates using the domain logic
        duplicate_groups = []
        processed = set()

        for i, recording1 in enumerate(all_recordings):
            if recording1 in processed:
                continue

            group = [recording1]
            processed.add(recording1)

            for recording2 in all_recordings[i+1:]:
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
        data = await self._load_data()
        recordings_list = list(data["recordings"].values())[offset:]

        if limit:
            recordings_list = recordings_list[:limit]

        for recording_dict in recordings_list:
            recording = self._dict_to_recording(recording_dict)
            if recording:
                yield recording

    async def delete(self, recording: Recording) -> None:
        """Delete a recording."""
        data = await self._load_data()
        recording_id = str(id(recording))

        if recording_id in data["recordings"]:
            del data["recordings"][recording_id]

            # Clean up index
            path_str = str(recording.path.path)
            if path_str in data["index"]:
                del data["index"][path_str]

            await self._save_data(data)

    async def count(self) -> int:
        """Get total count of recordings."""
        data = await self._load_data()
        return len(data["recordings"])

    def _dict_to_recording(self, recording_dict: Dict[str, Any]) -> Optional[Recording]:
        """Convert dictionary to Recording entity."""
        try:
            # This is a simplified conversion
            # In practice, you'd need to properly reconstruct all nested objects

            # Create AudioPath
            path = AudioPath(recording_dict["path"])

            # Create metadata (simplified)
            metadata_dict = recording_dict.get("metadata", {})
            metadata = Metadata(
                title=metadata_dict.get("title"),
                artists=[ArtistName(a) for a in metadata_dict.get("artists", [])],
                album=metadata_dict.get("album"),
                year=metadata_dict.get("year"),
                genre=metadata_dict.get("genre"),
                # ... other fields
            )

            # Create recording
            recording = Recording(path=path, metadata=metadata)

            # Set additional attributes
            if "is_processed" in recording_dict:
                recording.is_processed = recording_dict["is_processed"]
            if "target_path" in recording_dict:
                recording.target_path = Path(recording_dict["target_path"])

            return recording

        except Exception as e:
            print(f"Error converting recording dict to entity: {e}")
            return None


class FileBasedCatalogRepository(CatalogRepository):
    """File-based implementation of CatalogRepository."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._catalogs_file = self.storage_dir / "catalogs.json"
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def save(self, catalog: Catalog) -> None:
        """Save the catalog."""
        def _save():
            # Load existing catalogs
            catalogs = {}
            if self._catalogs_file.exists():
                with open(self._catalogs_file, 'r') as f:
                    catalogs = json.load(f)

            # Convert catalog to dictionary
            catalog_dict = {
                "name": catalog.name,
                "created_at": catalog.created_at.isoformat(),
                "last_updated": catalog.last_updated.isoformat() if catalog.last_updated else None,
                "artist_count": catalog.artist_count,
                "release_count": catalog.release_count,
                "recording_count": catalog.recording_count,
                # In practice, you'd also save the actual entities
                # This is a simplified version
            }

            catalogs[catalog.name] = catalog_dict

            # Save to file
            with open(self._catalogs_file, 'w') as f:
                json.dump(catalogs, f, indent=2)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, _save)

    async def load(self, catalog_name: str) -> Optional[Catalog]:
        """Load a catalog by name."""
        def _load():
            if not self._catalogs_file.exists():
                return None

            with open(self._catalogs_file, 'r') as f:
                catalogs = json.load(f)

            catalog_dict = catalogs.get(catalog_name)
            if not catalog_dict:
                return None

            # Create catalog (simplified)
            catalog = Catalog(name=catalog_dict["name"])

            # Set metadata
            if catalog_dict.get("created_at"):
                catalog.created_at = datetime.fromisoformat(catalog_dict["created_at"])
            if catalog_dict.get("last_updated"):
                catalog.last_updated = datetime.fromisoformat(catalog_dict["last_updated"])

            return catalog

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _load)

    async def get_statistics(self, catalog_name: str) -> Dict[str, Any]:
        """Get catalog statistics."""
        catalog = await self.load(catalog_name)
        if catalog:
            return catalog.get_statistics()
        return {}