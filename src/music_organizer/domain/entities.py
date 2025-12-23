"""Domain entities for Music Organizer.

This module defines the core domain entities following Domain-Driven Design
principles. Each entity represents a core business concept with identity
and lifecycle.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Iterator
from datetime import datetime
from enum import Enum

from .value_objects import (
    AudioPath,
    ArtistName,
    TrackNumber,
    Metadata,
    ContentPattern,
    FileFormat,
)


class DuplicateResolutionMode(Enum):
    """Strategies for handling duplicate files."""
    SKIP = "skip"
    RENAME = "rename"
    REPLACE = "replace"
    KEEP_BOTH = "keep_both"


@dataclass
class Recording:
    """
    Represents a single audio recording (a track).

    A Recording is the core entity representing a distinct audio recording
    with its own metadata. It's identified by its combination of
    acoustic fingerprint and metadata.
    """

    # Core identity
    path: AudioPath
    metadata: Metadata

    # Runtime state
    is_processed: bool = False
    is_duplicate: bool = False
    duplicate_of: Optional["Recording"] = None
    processing_errors: List[str] = field(default_factory=list)

    # Classification
    content_type: Optional[str] = None
    genre_classifications: Set[str] = field(default_factory=set)
    energy_level: Optional[str] = None  # "low", "medium", "high"

    # Organization
    target_path: Optional[Path] = None
    move_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def title(self) -> str:
        """Get the recording title."""
        return self.metadata.title or self.path.path.stem

    @property
    def artists(self) -> List[ArtistName]:
        """Get the recording artists."""
        if not self.metadata.artists:
            return [ArtistName("Unknown Artist")]
        return list(self.metadata.artists)

    @property
    def primary_artist(self) -> ArtistName:
        """Get the primary artist."""
        return self.artists[0]

    @property
    def album(self) -> Optional[str]:
        """Get the album name."""
        return self.metadata.album

    @property
    def year(self) -> Optional[int]:
        """Get the release year."""
        return self.metadata.year

    @property
    def track_number(self) -> Optional[TrackNumber]:
        """Get the track number."""
        return self.metadata.track_number

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return self.metadata.duration_seconds or 0.0

    @property
    def file_hash(self) -> str:
        """Get file hash for duplicate detection."""
        return self.metadata.file_hash or ""

    @property
    def acoustic_fingerprint(self) -> str:
        """Get acoustic fingerprint for matching."""
        return self.metadata.acoustic_fingerprint or ""

    def get_display_name(self) -> str:
        """Get human-readable display name."""
        # Check if we have real artists and a real title (not from filename)
        has_real_artist = (
            self.metadata.artists and
            len(self.metadata.artists) > 0
        )
        has_real_title = self.metadata.title is not None

        if has_real_artist and has_real_title:
            return f"{self.primary_artist} - {self.title}"
        if has_real_title:
            return self.title
        return self.path.path.name

    def add_genre_classification(self, genre: str) -> None:
        """Add a genre classification."""
        self.genre_classifications.add(genre.lower())

    def has_genre(self, genre: str) -> bool:
        """Check if recording has a specific genre."""
        return genre.lower() in self.genre_classifications

    def mark_as_processed(self) -> None:
        """Mark recording as processed."""
        self.is_processed = True

    def add_processing_error(self, error: str) -> None:
        """Add a processing error."""
        self.processing_errors.append(error)

    def set_duplicate(self, duplicate_of: "Recording") -> None:
        """Mark as duplicate of another recording."""
        self.is_duplicate = True
        self.duplicate_of = duplicate_of

    def record_move(self, from_path: Path, to_path: Path) -> None:
        """Record a file move operation."""
        self.move_history.append({
            "from": str(from_path),
            "to": str(to_path),
            "timestamp": datetime.now().isoformat()
        })

    def calculate_similarity(self, other: "Recording") -> float:
        """Calculate similarity with another recording."""
        if not isinstance(other, Recording):
            return 0.0

        # Exact match on hash
        if self.file_hash and self.file_hash == other.file_hash:
            return 1.0

        # Acoustic fingerprint match
        if (self.acoustic_fingerprint and other.acoustic_fingerprint
            and self.acoustic_fingerprint == other.acoustic_fingerprint):
            return 0.95

        # Metadata-based similarity
        similarity = 0.0

        # Title similarity (40% weight)
        if self.title and other.title:
            title_sim = self._calculate_string_similarity(self.title, other.title)
            similarity += title_sim * 0.4

        # Artist similarity (35% weight)
        if self.artists and other.artists:
            artist_sim = self._calculate_artist_similarity(self.artists, other.artists)
            similarity += artist_sim * 0.35

        # Album similarity (15% weight)
        if self.album and other.album:
            album_sim = self._calculate_string_similarity(self.album.lower(), other.album.lower())
            similarity += album_sim * 0.15

        # Duration similarity (10% weight)
        if self.duration_seconds and other.duration_seconds:
            duration_diff = abs(self.duration_seconds - other.duration_seconds)
            duration_sim = max(0, 1 - duration_diff / 30)  # 30 second tolerance
            similarity += duration_sim * 0.1

        return min(1.0, similarity)

    def _calculate_string_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings."""
        # Simple Levenshtein-like distance
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        # Normalize
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()

        # Check for exact match after normalization
        if s1 == s2:
            return 1.0

        # Check if one contains the other
        if s1 in s2 or s2 in s1:
            return 0.8

        # Simple word-based similarity
        words1 = set(s1.split())
        words2 = set(s2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _calculate_artist_similarity(self, artists1: List[ArtistName], artists2: List[ArtistName]) -> float:
        """Calculate similarity between artist lists."""
        if not artists1 or not artists2:
            return 0.0

        # Compare normalized names
        names1 = [str(a).lower().strip() for a in artists1]
        names2 = [str(a).lower().strip() for a in artists2]

        # Check for exact matches
        matches = sum(1 for n1 in names1 for n2 in names2 if n1 == n2)

        if matches > 0:
            return matches / max(len(names1), len(names2))

        # Check for partial matches (e.g., "The Beatles" vs "Beatles")
        for n1 in names1:
            for n2 in names2:
                if n1.replace("the ", "") == n2.replace("the ", ""):
                    return 0.8

        return 0.0


@dataclass
class Release:
    """
    Represents a music release (album, EP, single).

    A Release groups recordings that were released together
    as part of the same album or collection.
    """

    # Core identity
    title: str
    primary_artist: ArtistName
    year: Optional[int] = None

    # Additional metadata
    release_type: str = "album"  # "album", "ep", "single", "compilation", "live"
    genre: Optional[str] = None
    total_tracks: Optional[int] = None
    total_discs: Optional[int] = None
    disc_number: Optional[int] = None

    # Associated recordings
    recordings: List[Recording] = field(default_factory=list)

    # Filesystem locations
    source_paths: List[AudioPath] = field(default_factory=list)
    target_path: Optional[Path] = None

    @property
    def display_name(self) -> str:
        """Get display name with year."""
        year_str = f" ({self.year})" if self.year else ""
        return f"{self.title}{year_str}"

    @property
    def artist_display(self) -> str:
        """Get artist display name."""
        return str(self.primary_artist)

    @property
    def track_count(self) -> int:
        """Get number of recordings."""
        return len(self.recordings)

    @property
    def total_duration_seconds(self) -> float:
        """Get total duration of all recordings."""
        return sum(r.duration_seconds for r in self.recordings)

    @property
    def total_size_mb(self) -> float:
        """Get total file size."""
        return sum(r.path.size_mb for r in self.recordings)

    def add_recording(self, recording: Recording) -> None:
        """Add a recording to this release."""
        if recording not in self.recordings:
            self.recordings.append(recording)
            # Update total tracks if not set
            if self.total_tracks is None:
                self.total_tracks = len(self.recordings)

    def remove_recording(self, recording: Recording) -> None:
        """Remove a recording from this release."""
        if recording in self.recordings:
            self.recordings.remove(recording)

    def get_recording_by_track(self, track_number: TrackNumber) -> Optional[Recording]:
        """Get recording by track number."""
        for recording in self.recordings:
            if recording.track_number and recording.track_number == track_number:
                return recording
        return None

    def sort_recordings(self) -> None:
        """Sort recordings by track number, then by title."""
        def sort_key(recording: Recording) -> tuple:
            track_num = recording.track_number.number if recording.track_number else 999
            return (track_num, recording.title.lower())

        self.recordings.sort(key=sort_key)

    def get_duplicate_groups(self, similarity_threshold: float = 0.85) -> List[List[Recording]]:
        """Find groups of duplicate recordings."""
        groups = []
        processed = set()

        for i, recording1 in enumerate(self.recordings):
            if id(recording1) in processed:
                continue

            group = [recording1]
            processed.add(id(recording1))

            for recording2 in self.recordings[i+1:]:
                if id(recording2) in processed:
                    continue

                similarity = recording1.calculate_similarity(recording2)
                if similarity >= similarity_threshold:
                    group.append(recording2)
                    processed.add(id(recording2))

            if len(group) > 1:
                groups.append(group)

        return groups

    def merge_with(self, other: "Release") -> None:
        """Merge another release into this one."""
        # Keep the best metadata
        if not self.year and other.year:
            self.year = other.year
        if not self.genre and other.genre:
            self.genre = other.genre
        # Prefer the larger total_tracks (likely an explicitly set value)
        if other.total_tracks and (not self.total_tracks or other.total_tracks > self.total_tracks):
            self.total_tracks = other.total_tracks

        # Add unique recordings
        for recording in other.recordings:
            if recording not in self.recordings:
                self.add_recording(recording)

        # Merge source paths
        for path in other.source_paths:
            if path not in self.source_paths:
                self.source_paths.append(path)


@dataclass
class Collection:
    """
    Represents a music collection.

    A Collection is a curated grouping of releases, often representing
    a user's music library or a specific subset like "2020s Pop".
    """

    # Core identity
    name: str
    description: Optional[str] = None

    # Organization
    releases: List[Release] = field(default_factory=list)
    subcollections: List["Collection"] = field(default_factory=list)
    parent_collection: Optional["Collection"] = None

    # Classification
    genre_patterns: List[ContentPattern] = field(default_factory=list)
    year_range: Optional[tuple[int, int]] = None  # (start_year, end_year)

    # Paths
    source_directories: List[Path] = field(default_factory=list)
    target_directory: Optional[Path] = None

    @property
    def release_count(self) -> int:
        """Get number of releases."""
        return len(self.releases)

    @property
    def recording_count(self) -> int:
        """Get total number of recordings."""
        return sum(release.track_count for release in self.releases)

    @property
    def total_duration_seconds(self) -> float:
        """Get total duration of all recordings."""
        return sum(release.total_duration_seconds for release in self.releases)

    @property
    def total_size_gb(self) -> float:
        """Get total size in GB."""
        total_mb = sum(release.total_size_mb for release in self.releases)
        return total_mb / 1024

    def add_release(self, release: Release) -> None:
        """Add a release to this collection."""
        if release not in self.releases:
            self.releases.append(release)

    def remove_release(self, release: Release) -> None:
        """Remove a release from this collection."""
        if release in self.releases:
            self.releases.remove(release)

    def add_subcollection(self, subcollection: "Collection") -> None:
        """Add a subcollection."""
        if subcollection not in self.subcollections:
            self.subcollections.append(subcollection)
            subcollection.parent_collection = self

    def get_all_releases(self) -> Iterator[Release]:
        """Get all releases including from subcollections."""
        yield from self.releases
        for subcollection in self.subcollections:
            yield from subcollection.get_all_releases()

    def get_all_recordings(self) -> Iterator[Recording]:
        """Get all recordings including from subcollections."""
        for release in self.get_all_releases():
            yield from release.recordings

    def filter_by_genre(self, genre: str) -> List[Release]:
        """Filter releases by genre."""
        filtered = []
        genre_lower = genre.lower()

        for release in self.releases:
            if release.genre and genre_lower in release.genre.lower():
                filtered.append(release)

        return filtered

    def filter_by_year(self, year: int) -> List[Release]:
        """Filter releases by year."""
        return [r for r in self.releases if r.year == year]

    def filter_by_artist(self, artist: ArtistName) -> List[Release]:
        """Filter releases by artist."""
        artist_str = str(artist).lower()
        filtered = []

        for release in self.releases:
            if artist_str in str(release.primary_artist).lower():
                filtered.append(release)

        return filtered


@dataclass
class AudioLibrary:
    """
    Represents the entire audio library.

    An AudioLibrary is the root entity representing a user's complete
    music collection, with operations for organization, duplicate detection,
    and maintenance.
    """

    # Library identity
    name: str
    root_path: Path

    # Library contents
    collections: List[Collection] = field(default_factory=list)
    standalone_recordings: List[Recording] = field(default_factory=list)

    # Organization settings
    duplicate_resolution_mode: DuplicateResolutionMode = DuplicateResolutionMode.SKIP
    organized_directory: Optional[Path] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_scanned: Optional[datetime] = None
    scan_count: int = 0

    @property
    def total_recordings(self) -> int:
        """Get total number of recordings."""
        collection_recordings = sum(
            collection.recording_count
            for collection in self.collections
        )
        return collection_recordings + len(self.standalone_recordings)

    @property
    def total_releases(self) -> int:
        """Get total number of releases."""
        return sum(collection.release_count for collection in self.collections)

    @property
    def total_size_gb(self) -> float:
        """Get total library size in GB."""
        collection_size = sum(collection.total_size_gb for collection in self.collections)
        standalone_size = sum(r.path.size_mb for r in self.standalone_recordings) / 1024
        return collection_size + standalone_size

    def add_collection(self, collection: Collection) -> None:
        """Add a collection to the library."""
        if collection not in self.collections:
            self.collections.append(collection)

    def remove_collection(self, collection: Collection) -> None:
        """Remove a collection from the library."""
        if collection in self.collections:
            self.collections.remove(collection)

    def add_standalone_recording(self, recording: Recording) -> None:
        """Add a standalone recording (not part of any release)."""
        if recording not in self.standalone_recordings:
            self.standalone_recordings.append(recording)

    def get_all_recordings(self) -> Iterator[Recording]:
        """Get all recordings in the library."""
        # Recordings from collections
        for collection in self.collections:
            yield from collection.get_all_recordings()

        # Standalone recordings
        yield from self.standalone_recordings

    def find_duplicates(self, similarity_threshold: float = 0.85) -> Dict[str, List[Recording]]:
        """Find all duplicate recordings across the library."""
        duplicates = {}
        processed = set()

        for recording in self.get_all_recordings():
            if id(recording) in processed or recording.is_duplicate:
                continue

            duplicate_group = [recording]
            processed.add(id(recording))

            # Find all similar recordings
            for other in self.get_all_recordings():
                if id(other) in processed or other == recording:
                    continue

                similarity = recording.calculate_similarity(other)
                if similarity >= similarity_threshold:
                    duplicate_group.append(other)
                    processed.add(id(other))

            # Only add groups with actual duplicates
            if len(duplicate_group) > 1:
                key = f"{recording.primary_artist} - {recording.title}"
                duplicates[key] = duplicate_group

        return duplicates

    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        recordings = list(self.get_all_recordings())

        # Format distribution
        format_counts = {}
        for recording in recordings:
            fmt = recording.path.format.value
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

        # Year distribution
        year_counts = {}
        for recording in recordings:
            if recording.year:
                decade = (recording.year // 10) * 10
                decade_key = f"{decade}s"
                year_counts[decade_key] = year_counts.get(decade_key, 0) + 1

        # Genre distribution
        genre_counts = {}
        for recording in recordings:
            for genre in recording.genre_classifications:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        # Artist counts
        artist_counts = {}
        for recording in recordings:
            artist = str(recording.primary_artist)
            artist_counts[artist] = artist_counts.get(artist, 0) + 1

        # Top artists (by number of recordings)
        top_artists = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_recordings": len(recordings),
            "total_releases": self.total_releases,
            "total_collections": len(self.collections),
            "total_size_gb": round(self.total_size_gb, 2),
            "average_duration_seconds": sum(r.duration_seconds for r in recordings) / len(recordings) if recordings else 0,
            "format_distribution": format_counts,
            "decade_distribution": year_counts,
            "genre_distribution": genre_counts,
            "top_artists": top_artists,
            "duplicate_count": sum(1 for r in recordings if r.is_duplicate),
            "processed_count": sum(1 for r in recordings if r.is_processed),
            "error_count": sum(len(r.processing_errors) for r in recordings),
        }

    def mark_scan_completed(self) -> None:
        """Mark that a library scan was completed."""
        self.last_scanned = datetime.now()
        self.scan_count += 1

    def get_releases_by_artist(self, artist: ArtistName) -> List[Release]:
        """Get all releases by a specific artist."""
        artist_str = str(artist).lower()
        matching_releases = []

        for collection in self.collections:
            for release in collection.releases:
                if artist_str in str(release.primary_artist).lower():
                    matching_releases.append(release)

        return matching_releases

    def get_recently_added(self, days: int = 7) -> List[Recording]:
        """Get recordings added in the last N days."""
        cutoff = datetime.now().timestamp() - (days * 24 * 3600)
        recent = []

        for recording in self.get_all_recordings():
            try:
                file_mtime = recording.path.path.stat().st_mtime
                if file_mtime >= cutoff:
                    recent.append(recording)
            except OSError:
                continue

        # Sort by modification time (newest first)
        recent.sort(key=lambda r: r.path.path.stat().st_mtime, reverse=True)
        return recent