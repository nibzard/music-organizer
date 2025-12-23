"""Audio file model representing music tracks with metadata."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from ..exceptions import MetadataError


class ContentType(Enum):
    """Types of music content."""
    STUDIO = "studio"
    LIVE = "live"
    COLLABORATION = "collaboration"
    COMPILATION = "compilation"
    RARITY = "rarity"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class AudioFile:
    """Representation of an audio file with its metadata."""

    path: Path
    file_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_type: ContentType = ContentType.UNKNOWN
    artists: List[str] = field(default_factory=list)
    primary_artist: Optional[str] = None
    album: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    date: Optional[str] = None  # For live recordings
    location: Optional[str] = None  # For live recordings
    track_number: Optional[int] = None
    genre: Optional[str] = None
    has_cover_art: bool = False

    @property
    def filename(self) -> str:
        """Get the filename without path."""
        return self.path.name

    @property
    def extension(self) -> str:
        """Get the file extension."""
        return self.path.suffix.lower()

    @property
    def size_mb(self) -> float:
        """Get file size in MB."""
        try:
            return self.path.stat().st_size / (1024 * 1024)
        except (OSError, FileNotFoundError):
            return 0.0

    def get_display_name(self) -> str:
        """Get a human-readable display name."""
        if self.title and self.artists:
            return f"{self.artists[0]} - {self.title}"
        elif self.title:
            return self.title
        else:
            return self.filename

    def get_target_path(self, base_dir: Path) -> Path:
        """Generate target path based on content type and metadata."""
        if self.content_type == ContentType.LIVE and self.date:
            artist = self.primary_artist or "Unknown Artist"
            location = self.location or "Unknown Location"
            return base_dir / "Live" / artist / f"{self.date} - {location}"

        elif self.content_type == ContentType.COLLABORATION:
            # For collaborations, include all artists (but limit to 3 for readability)
            if len(self.artists) <= 3:
                artists_str = ", ".join(self.artists)
            else:
                artists_str = ", ".join(self.artists[:3]) + " et al."
            album = self.album or "Unknown Album"
            year_str = f" ({self.year})" if self.year else ""
            return base_dir / "Collaborations" / f"{album}{year_str} - {artists_str}"

        elif self.content_type == ContentType.COMPILATION:
            artist = self.primary_artist or self.artists[0] if self.artists else "Various Artists"
            album = self.album or "Unknown Compilation"
            year_str = f" ({self.year})" if self.year else ""
            return base_dir / "Compilations" / artist / f"{album}{year_str}"

        elif self.content_type == ContentType.RARITY:
            artist = self.primary_artist or self.artists[0] if self.artists else "Unknown Artist"
            album = self.album or "Unknown Release"
            # Include edition info if available in metadata
            edition = self.metadata.get("edition", "")
            edition_str = f" ({edition})" if edition else ""
            return base_dir / "Rarities" / artist / f"{album}{edition_str}"

        else:  # STUDIO or UNKNOWN
            artist = self.primary_artist or self.artists[0] if self.artists else "Unknown Artist"
            album = self.album or "Unknown Album"
            year_str = f" ({self.year})" if self.year else ""
            return base_dir / "Albums" / artist / f"{album}{year_str}"

    def get_target_filename(self) -> str:
        """Generate target filename based on track metadata."""
        # Get track number with zero padding
        track_str = ""
        if self.track_number is not None:
            track_str = f"{self.track_number:02d}. "

        # Clean title for filename
        title = self.title or self.path.stem
        # Replace filesystem-incompatible characters
        safe_title = title.replace('/', '_').replace('\\', '_').replace(':', ' -')
        safe_title = safe_title.replace('?', '').replace('*', '').replace('|', '-')
        safe_title = safe_title.replace('<', '').replace('>', '').replace('"', "'")

        return f"{track_str}{safe_title}{self.extension}"

    @classmethod
    def from_path(cls, path: Path) -> "AudioFile":
        """Create AudioFile instance from file path."""
        if not path.exists():
            raise MetadataError(f"File does not exist: {path}")

        if not path.is_file():
            raise MetadataError(f"Path is not a file: {path}")

        # Determine file type from extension
        extension = path.suffix.lower()
        if extension in ['.flac']:
            file_type = 'FLAC'
        elif extension in ['.mp3']:
            file_type = 'MP3'
        elif extension in ['.wav']:
            file_type = 'WAV'
        elif extension in ['.m4a', '.mp4', '.aac']:
            file_type = 'MP4'
        elif extension in ['.ogg']:
            file_type = 'OGG'
        elif extension in ['.opus']:
            file_type = 'OPUS'
        elif extension in ['.wma']:
            file_type = 'WMA'
        elif extension in ['.ape']:
            file_type = 'APE'
        elif extension in ['.aiff', '.aif']:
            file_type = 'AIFF'
        else:
            file_type = 'UNKNOWN'

        # Create basic instance
        audio_file = cls(path=path, file_type=file_type)

        # Parse basic info from filename as fallback
        audio_file._parse_from_filename()

        return audio_file

    def _parse_from_filename(self) -> None:
        """Extract basic info from filename patterns."""
        # Pattern: "01. Artist - Title.flac"
        import re

        filename = self.path.stem

        # Try track number pattern
        track_match = re.match(r'^(\d+)\.\s*(.+)', filename)
        if track_match:
            self.track_number = int(track_match.group(1))
            rest = track_match.group(2)
        else:
            rest = filename

        # Try artist - title pattern
        if ' - ' in rest:
            parts = rest.split(' - ', 1)
            if len(parts) == 2 and not self.artists:
                self.artists = [parts[0].strip()]
                self.title = parts[1].strip()
        elif not self.title:
            self.title = rest


@dataclass(slots=True)
class CoverArt:
    """Cover art information for an album."""
    path: Path
    type: str  # 'front', 'back', 'disc', etc.
    format: str  # 'jpg', 'png', etc.
    size: int  # Size in bytes

    @classmethod
    def from_file(cls, path: Path) -> Optional["CoverArt"]:
        """Create CoverArt instance from image file."""
        if not path.exists():
            return None

        extension = path.suffix.lower()
        if extension not in ['.jpg', '.jpeg', '.png', '.gif']:
            return None

        # Determine type from filename
        filename_lower = path.name.lower()
        if 'front' in filename_lower or 'cover' in filename_lower or 'folder' in filename_lower:
            art_type = 'front'
        elif 'back' in filename_lower:
            art_type = 'back'
        elif 'disc' in filename_lower:
            art_type = 'disc'
        else:
            art_type = 'other'

        try:
            size = path.stat().st_size
            return cls(
                path=path,
                type=art_type,
                format=extension[1:],  # Remove dot
                size=size
            )
        except (OSError, FileNotFoundError):
            return None