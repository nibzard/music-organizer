"""
Domain value objects for Music Organizer.

This module implements value objects following Domain-Driven Design principles.
Value objects are immutable objects that are defined by their attributes rather than identity.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union


class FileFormat(Enum):
    """Supported audio file formats."""
    FLAC = "flac"
    MP3 = "mp3"
    MP4 = "mp4"
    M4A = "m4a"
    WAV = "wav"
    AIFF = "aiff"
    OGG = "ogg"
    OPUS = "opus"
    WMA = "wma"


@dataclass(frozen=True, slots=True)
class AudioPath:
    """
    Value object representing a file path for audio files.

    Provides domain-specific operations and validation for audio file paths.
    """

    _path: Path = field(repr=False)
    _format: Optional[FileFormat] = None
    _unknown_format: Optional[str] = field(default=None, repr=False)

    def __init__(self, path: Union[str, Path]) -> None:
        """Create an AudioPath with validation."""
        if isinstance(path, str):
            path = Path(path)

        if not isinstance(path, Path):
            raise TypeError(f"Path must be str or Path, got {type(path)}")

        if not path.suffix:
            raise ValueError("AudioPath must have a file extension")

        # Normalize the path
        path = path.resolve()

        # Determine format
        extension = path.suffix.lower().lstrip('.')
        try:
            format_enum = FileFormat(extension)
            object.__setattr__(self, '_format', format_enum)
        except ValueError:
            # Allow unknown formats but store as string
            object.__setattr__(self, '_format', None)
            object.__setattr__(self, '_unknown_format', extension)

        object.__setattr__(self, '_path', path)

    @property
    def path(self) -> Path:
        """Get the underlying Path object."""
        return self._path

    @property
    def format(self) -> Optional[FileFormat]:
        """Get the file format."""
        return self._format

    @property
    def is_known_format(self) -> bool:
        """Check if the format is a known supported format."""
        return self._format is not None

    @property
    def extension(self) -> str:
        """Get the file extension (including dot)."""
        return self._path.suffix

    @property
    def filename(self) -> str:
        """Get the filename without directory."""
        return self._path.name

    @property
    def stem(self) -> str:
        """Get the filename without extension."""
        return self._path.stem

    @property
    def parent(self) -> Path:
        """Get the parent directory."""
        return self._path.parent

    @property
    def size_bytes(self) -> int:
        """Get file size in bytes."""
        try:
            return self._path.stat().st_size
        except (OSError, FileNotFoundError):
            return 0

    @property
    def size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    def exists(self) -> bool:
        """Check if the file exists."""
        return self._path.exists()

    def is_absolute(self) -> bool:
        """Check if the path is absolute."""
        return self._path.is_absolute()

    def relative_to(self, other: Union[str, Path]) -> AudioPath:
        """Return a relative path to another path."""
        try:
            return AudioPath(self._path.relative_to(Path(other)))
        except ValueError as e:
            raise ValueError(f"Cannot make path {self._path} relative to {other}: {e}")

    def with_name(self, name: str) -> AudioPath:
        """Return a new AudioPath with a different name."""
        return AudioPath(self._path.with_name(name))

    def with_suffix(self, suffix: str) -> AudioPath:
        """Return a new AudioPath with a different suffix."""
        return AudioPath(self._path.with_suffix(suffix))

    def joinpath(self, *args: Union[str, Path]) -> AudioPath:
        """Join this path with additional parts."""
        return AudioPath(self._path.joinpath(*args))

    def __str__(self) -> str:
        """String representation."""
        return str(self._path)

    def __eq__(self, other: Any) -> bool:
        """Equality based on normalized path."""
        if not isinstance(other, AudioPath):
            return False
        return self._path == other._path

    def __hash__(self) -> int:
        """Hash based on normalized path."""
        return hash(self._path)


@dataclass(frozen=True, slots=True)
class TrackNumber:
    """
    Value object representing a track number.

    Handles various formats like "5", "5/12", "05", etc.
    """

    _number: int
    _total: Optional[int] = None

    def __init__(self, value: Union[str, int, Tuple[int, int]]) -> None:
        """Create a TrackNumber from various formats."""
        if isinstance(value, str):
            # Handle formats like "5", "5/12", "05", "5 of 12"
            if '/' in value:
                parts = value.split('/')
                try:
                    num = int(parts[0])
                    total = int(parts[1]) if len(parts) > 1 else None
                except ValueError:
                    # Try parsing with "of"
                    match = re.match(r'(\d+)\s*of\s*(\d+)', value, re.IGNORECASE)
                    if match:
                        num = int(match.group(1))
                        total = int(match.group(2))
                    else:
                        nums = re.findall(r'\d+', value)
                        if nums:
                            num = int(nums[0])
                            total = None
                        else:
                            raise ValueError(f"Cannot parse track number from: {value}")
            else:
                # Check for "of" pattern first
                match = re.match(r'(\d+)\s*of\s*(\d+)', value, re.IGNORECASE)
                if match:
                    num = int(match.group(1))
                    total = int(match.group(2))
                else:
                    # Extract first number found
                    nums = re.findall(r'\d+', value)
                    if nums:
                        num = int(nums[0])
                        total = None
                    else:
                        raise ValueError(f"Cannot parse track number from: {value}")
        elif isinstance(value, int):
            num = value
            total = None
        elif isinstance(value, tuple) and len(value) == 2:
            num, total = value
        else:
            raise TypeError(f"TrackNumber expects str, int, or tuple, got {type(value)}")

        object.__setattr__(self, '_number', max(0, num))
        object.__setattr__(self, '_total', total and max(0, total))

    @property
    def number(self) -> int:
        """Get the track number."""
        return self._number

    @property
    def total(self) -> Optional[int]:
        """Get the total tracks if known."""
        return self._total

    @property
    def has_total(self) -> bool:
        """Check if total tracks is known."""
        return self._total is not None

    def formatted(self, width: int = 2) -> str:
        """Get zero-padded representation."""
        if width <= 0:
            width = 2
        return f"{self._number:0{width}d}"

    def formatted_with_total(self, width: int = 2) -> str:
        """Get representation with total (e.g., '05/12')."""
        if width <= 0:
            width = 2
        if self._total:
            return f"{self._number:0{width}d}/{self._total:0{width}d}"
        return self.formatted(width)

    def __str__(self) -> str:
        """String representation."""
        if self._total:
            return f"{self._number}/{self._total}"
        return str(self._number)

    def __int__(self) -> int:
        """Integer conversion."""
        return self._number


@dataclass(frozen=True, slots=True)
class ArtistName:
    """
    Value object representing an artist name with normalization.
    """

    _name: str

    def __init__(self, name: str) -> None:
        """Create an ArtistName with normalization."""
        if not isinstance(name, str):
            raise TypeError(f"ArtistName must be str, got {type(name)}")

        # Normalize whitespace
        normalized = " ".join(name.strip().split())

        # Handle special cases
        if not normalized:
            raise ValueError("Artist name cannot be empty")

        object.__setattr__(self, '_name', normalized)

    @property
    def name(self) -> str:
        """Get the normalized name."""
        return self._name

    @property
    def sortable(self) -> str:
        """Get a version suitable for sorting (ignores articles)."""
        articles = {"the", "a", "an"}
        words = self._name.lower().split()

        # Don't treat "The" as an article if it's part of a name like "The The"
        if words and words[0] in articles and len(words) > 1 and self._name.lower() != f"{words[0]} {words[0]}":
            return f"{self._name[len(words[0]) + 1:]}, {words[0].title()}"

        return self._name

    @property
    def first_letter(self) -> str:
        """Get first letter for alphabetical grouping."""
        # Use first letter of sortable name
        sortable = self.sortable.lstrip()
        return sortable[0].upper() if sortable else "#"

    def starts_with(self, prefix: str) -> bool:
        """Check if name starts with prefix (case-insensitive)."""
        return self._name.lower().startswith(prefix.lower())

    def contains(self, substring: str) -> bool:
        """Check if name contains substring (case-insensitive)."""
        return substring.lower() in self._name.lower()

    def __str__(self) -> str:
        """String representation."""
        return self._name

    def __eq__(self, other: Any) -> bool:
        """Equality based on normalized name (case-insensitive)."""
        if not isinstance(other, ArtistName):
            return False
        return self._name.lower() == other._name.lower()

    def __hash__(self) -> int:
        """Hash based on normalized name."""
        return hash(self._name.lower())


@dataclass(frozen=True, slots=True)
class Metadata:
    """
    Value object representing audio metadata.

    Immutable collection of audio metadata with validation.
    """

    # Core metadata
    title: Optional[str] = None
    artists: FrozenSet[ArtistName] = frozenset()
    albumartist: Optional[ArtistName] = None
    album: Optional[str] = None
    year: Optional[int] = None
    track_number: Optional[TrackNumber] = None
    genre: Optional[str] = None

    # Additional metadata
    disc_number: Optional[int] = None
    total_discs: Optional[int] = None
    composer: Optional[str] = None
    comment: Optional[str] = None
    bpm: Optional[int] = None
    key: Optional[str] = None

    # Technical metadata
    duration_seconds: Optional[float] = None
    bitrate: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None

    # Live/recording metadata
    date: Optional[str] = None
    location: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        # Validate year
        if self.year is not None:
            if not (0 <= self.year <= 9999):
                raise ValueError(f"Invalid year: {self.year}")

        # Validate disc numbers
        if self.disc_number is not None and self.disc_number < 1:
            raise ValueError(f"Invalid disc number: {self.disc_number}")

        if self.total_discs is not None and self.total_discs < 1:
            raise ValueError(f"Invalid total discs: {self.total_discs}")

        # Validate technical metadata
        if self.bitrate is not None and self.bitrate < 0:
            raise ValueError(f"Invalid bitrate: {self.bitrate}")

        if self.sample_rate is not None and self.sample_rate < 0:
            raise ValueError(f"Invalid sample rate: {self.sample_rate}")

        if self.channels is not None and self.channels not in {1, 2, 4, 6, 8}:
            raise ValueError(f"Invalid channel count: {self.channels}")

    @property
    def primary_artist(self) -> Optional[ArtistName]:
        """Get the primary artist (first in list or albumartist)."""
        if self.albumartist:
            return self.albumartist
        if self.artists:
            return next(iter(self.artists))
        return None

    @property
    def is_live(self) -> bool:
        """Check if this is likely a live recording."""
        return (
            self.location is not None or
            self.date is not None or
            (self.title and any(x in self.title.lower() for x in ["live", "concert"]))
        )

    @property
    def is_compilation(self) -> bool:
        """Check if this is likely from a compilation."""
        return (
            "various" in (self.albumartist.name if self.albumartist else "").lower() or
            len(self.artists) > 1 or
            (self.album and any(x in self.album.lower() for x in ["compilation", "soundtrack", "ost"]))
        )

    @property
    def has_multiple_artists(self) -> bool:
        """Check if track has multiple artists."""
        return len(self.artists) > 1

    @property
    def artist_names(self) -> List[str]:
        """Get list of artist names as strings."""
        return [artist.name for artist in self.artists]

    def formatted_artists(self, separator: str = ", ") -> str:
        """Get formatted string of all artists."""
        return separator.join(self.artist_names)

    def formatted_title(self) -> str:
        """Get formatted title with optional suffixes."""
        if not self.title:
            return ""

        title = self.title

        # Add live info
        if self.is_live and "live" not in title.lower():
            parts = []
            if self.date:
                parts.append(self.date)
            if self.location:
                parts.append(self.location)
            if parts:
                title += f" (Live{' - ' if len(parts) > 1 else ''}{' - '.join(parts)})"

        # Add track number
        if self.track_number and not title.startswith(str(self.track_number.number)):
            title = f"{self.track_number.formatted()} {title}"

        return title

    def formatted_duration(self) -> str:
        """Get formatted duration (MM:SS)."""
        if not self.duration_seconds:
            return ""

        minutes = int(self.duration_seconds // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{minutes}:{seconds:02d}"

    def get_hash(self) -> str:
        """Get a hash of core metadata for duplicate detection."""
        core_data = (
            self.formatted_artists().lower(),
            (self.title or "").lower(),
            (self.album or "").lower(),
            self.track_number.number if self.track_number else None,
            self.year
        )
        return hashlib.sha256(str(core_data).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        from dataclasses import asdict
        result = asdict(self)
        # Convert FrozenSet and ArtistName objects to serializable formats
        result['artists'] = [str(a) for a in self.artists]
        if self.albumartist:
            result['albumartist'] = str(self.albumartist)
        if self.track_number:
            result['track_number'] = self.track_number.to_dict()
        return result

    def with_field(self, **kwargs) -> Metadata:
        """Create a new Metadata with updated fields."""
        # Get current field values
        current = {
            'title': self.title,
            'artists': self.artists,
            'albumartist': self.albumartist,
            'album': self.album,
            'year': self.year,
            'track_number': self.track_number,
            'genre': self.genre,
            'disc_number': self.disc_number,
            'total_discs': self.total_discs,
            'composer': self.composer,
            'comment': self.comment,
            'bpm': self.bpm,
            'key': self.key,
            'duration_seconds': self.duration_seconds,
            'bitrate': self.bitrate,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'date': self.date,
            'location': self.location
        }

        # Convert string artists to ArtistName objects
        if 'artists' in kwargs:
            artists = kwargs['artists']
            if isinstance(artists, str):
                kwargs['artists'] = frozenset({ArtistName(artists)})
            elif isinstance(artists, (list, set, frozenset)):
                kwargs['artists'] = frozenset(
                    ArtistName(a) if isinstance(a, str) else a
                    for a in artists
                )

        # Convert string albumartist to ArtistName
        if 'albumartist' in kwargs and kwargs['albumartist']:
            albumartist = kwargs['albumartist']
            if isinstance(albumartist, str):
                kwargs['albumartist'] = ArtistName(albumartist)

        # Merge and return new Metadata
        return Metadata(**{**current, **kwargs})

    def __str__(self) -> str:
        """String representation."""
        parts = []

        if self.title:
            parts.append(self.title)

        if self.artists:
            parts.append(f"by {self.formatted_artists()}")

        if self.album:
            parts.append(f"from '{self.album}'")

        if self.year:
            parts.append(f"({self.year})")

        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class ContentPattern:
    """
    Value object for content classification patterns.

    Defines patterns for identifying different types of audio content.
    """

    name: str
    patterns: FrozenSet[str] = frozenset()
    priority: int = 0

    def matches(self, text: str) -> bool:
        """Check if any pattern matches the text."""
        text_lower = text.lower()
        return any(pattern.lower() in text_lower for pattern in self.patterns)

    def __str__(self) -> str:
        """String representation."""
        return f"Pattern({self.name}, {len(self.patterns)} rules)"