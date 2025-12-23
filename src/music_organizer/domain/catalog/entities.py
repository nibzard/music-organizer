"""Catalog Context Entities.

This module defines the core entities for the Catalog bounded context.
The Catalog context is responsible for managing recordings, releases, and artists.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Iterator
from datetime import datetime
from enum import Enum
from uuid import uuid4

from ..value_objects import (
    AudioPath,
    ArtistName,
    TrackNumber,
    Metadata,
    FileFormat,
)


class DuplicateResolutionMode(Enum):
    """Strategies for handling duplicate files."""
    SKIP = "skip"
    RENAME = "rename"
    REPLACE = "replace"
    KEEP_BOTH = "keep_both"


@dataclass(kw_only=True)
class Recording:
    """
    Represents a single audio recording (a track).

    A Recording is the core entity representing a distinct audio recording
    with its own metadata. It's identified by its combination of
    acoustic fingerprint and metadata.
    """

    # Entity ID
    id: str = field(default_factory=lambda: str(uuid4()))

    # Core identity
    path: AudioPath
    metadata: Metadata

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

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
        return self.metadata.artists or [ArtistName("Unknown Artist")]

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
        if self.artists and self.title:
            return f"{self.primary_artist} - {self.title}"
        return self.title or self.path.path.name

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

    def update_metadata(self, new_metadata: Metadata) -> None:
        """Update the recording's metadata."""
        self.metadata = new_metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert recording to dictionary."""
        return {
            "id": self.id,
            "path": str(self.path.path),
            "format": self.path.format.name,
            "size_mb": self.path.size_mb,
            "title": self.title,
            "artists": [str(a) for a in self.artists],
            "album": self.album,
            "year": self.year,
            "genre": self.metadata.genre,
            "track_number": self.track_number.number if self.track_number else None,
            "duration_seconds": self.metadata.duration_seconds,
            "bitrate": self.metadata.bitrate,
            "is_processed": self.is_processed,
            "is_duplicate": self.is_duplicate,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class Artist:
    """
    Represents a musical artist or group.

    An Artist aggregates all recordings and releases associated with them.
    """

    # Core identity
    name: ArtistName

    # Artist metadata
    biography: Optional[str] = None
    country: Optional[str] = None
    formed_year: Optional[int] = None
    disband_year: Optional[int] = None

    # Associated data
    recordings: List[Recording] = field(default_factory=list)
    releases: List["Release"] = field(default_factory=list)

    # Relationships
    collaborators: Set[ArtistName] = field(default_factory=set)

    @property
    def recording_count(self) -> int:
        """Get number of recordings."""
        return len(self.recordings)

    @property
    def release_count(self) -> int:
        """Get number of releases."""
        return len(self.releases)

    @property
    def total_duration_seconds(self) -> float:
        """Get total duration of all recordings."""
        return sum(r.duration_seconds for r in self.recordings)

    @property
    def genres(self) -> Set[str]:
        """Get all genres associated with this artist."""
        genres = set()
        for recording in self.recordings:
            genres.update(recording.genre_classifications)
        return genres

    def add_recording(self, recording: Recording) -> None:
        """Add a recording to this artist."""
        if recording not in self.recordings:
            self.recordings.append(recording)

            # Track collaborators
            for artist in recording.artists:
                if artist != self.name:
                    self.collaborators.add(artist)

    def add_release(self, release: "Release") -> None:
        """Add a release to this artist."""
        if release not in self.releases:
            self.releases.append(release)

    def is_collaborator(self, other: ArtistName) -> bool:
        """Check if another artist is a collaborator."""
        return other in self.collaborators

    def get_years_active(self) -> Optional[tuple[int, int]]:
        """Get years active as (start, end) tuple."""
        return (self.formed_year, self.disband_year) if self.formed_year else None


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
            if recording.track_number and recording.track_number.equals(track_number):
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
            if recording1 in processed:
                continue

            group = [recording1]
            processed.add(recording1)

            for recording2 in self.recordings[i+1:]:
                if recording2 in processed:
                    continue

                similarity = recording1.calculate_similarity(recording2)
                if similarity >= similarity_threshold:
                    group.append(recording2)
                    processed.add(recording2)

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

        # Add unique recordings
        for recording in other.recordings:
            if recording not in self.recordings:
                self.add_recording(recording)

        # Merge source paths
        for path in other.source_paths:
            if path not in self.source_paths:
                self.source_paths.append(path)


@dataclass
class Catalog:
    """
    Represents the music catalog.

    A Catalog is the root entity for the catalog context, containing
    all artists, releases, and recordings in the music library.
    """

    # Catalog identity
    name: str

    # Catalog contents
    artists: Dict[str, Artist] = field(default_factory=dict)
    releases: Dict[str, Release] = field(default_factory=dict)
    recordings: List[Recording] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: Optional[datetime] = None

    @property
    def artist_count(self) -> int:
        """Get number of unique artists."""
        return len(self.artists)

    @property
    def release_count(self) -> int:
        """Get number of releases."""
        return len(self.releases)

    @property
    def recording_count(self) -> int:
        """Get number of recordings."""
        return len(self.recordings)

    def add_recording(self, recording: Recording) -> None:
        """Add a recording to the catalog."""
        if recording not in self.recordings:
            self.recordings.append(recording)
            self.last_updated = datetime.now()

            # Update artist associations
            for artist_name in recording.artists:
                artist_key = str(artist_name).lower()
                if artist_key not in self.artists:
                    self.artists[artist_key] = Artist(name=artist_name)
                self.artists[artist_key].add_recording(recording)

            # Update release associations
            if recording.album:
                release_key = f"{recording.primary_artist} - {recording.album}"
                if release_key not in self.releases:
                    self.releases[release_key] = Release(
                        title=recording.album,
                        primary_artist=recording.primary_artist,
                        year=recording.year
                    )
                self.releases[release_key].add_recording(recording)
                self.artists[str(recording.primary_artist).lower()].add_release(
                    self.releases[release_key]
                )

    def get_artist(self, name: ArtistName) -> Optional[Artist]:
        """Get an artist by name."""
        return self.artists.get(str(name).lower())

    def get_release(self, title: str, artist: ArtistName) -> Optional[Release]:
        """Get a release by title and artist."""
        key = f"{artist} - {title}"
        return self.releases.get(key)

    def find_similar_recordings(self, recording: Recording, threshold: float = 0.85) -> List[Recording]:
        """Find recordings similar to the given one."""
        similar = []
        for other in self.recordings:
            if other != recording:
                similarity = recording.calculate_similarity(other)
                if similarity >= threshold:
                    similar.append(other)
        return similar

    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        # Format distribution
        format_counts = {}
        for recording in self.recordings:
            fmt = recording.path.format.value if recording.path.format else "unknown"
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

        # Year distribution
        year_counts = {}
        for recording in self.recordings:
            if recording.year:
                decade = (recording.year // 10) * 10
                decade_key = f"{decade}s"
                year_counts[decade_key] = year_counts.get(decade_key, 0) + 1

        # Genre distribution
        genre_counts = {}
        for recording in self.recordings:
            for genre in recording.genre_classifications:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        # Artist counts
        artist_counts = {}
        for recording in self.recordings:
            artist = str(recording.primary_artist)
            artist_counts[artist] = artist_counts.get(artist, 0) + 1

        # Top artists (by number of recordings)
        top_artists = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_recordings": len(self.recordings),
            "total_artists": len(self.artists),
            "total_releases": len(self.releases),
            "average_duration_seconds": sum(r.duration_seconds for r in self.recordings) / len(self.recordings) if self.recordings else 0,
            "format_distribution": format_counts,
            "decade_distribution": year_counts,
            "genre_distribution": genre_counts,
            "top_artists": top_artists,
            "duplicate_count": sum(1 for r in self.recordings if r.is_duplicate),
            "processed_count": sum(1 for r in self.recordings if r.is_processed),
            "error_count": sum(len(r.processing_errors) for r in self.recordings),
        }