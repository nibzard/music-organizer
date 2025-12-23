#!/usr/bin/env python3
"""Music Library Organizer - Single File Distribution

A standalone tool for organizing music libraries using metadata-aware categorization.

Dependencies:
    - mutagen>=1.47.0 (for metadata extraction)

Usage:
    python music_organize.py organize <source> <target> [options]
    python music_organize.py scan <directory>
    python music_organize.py inspect <file>
    python music_organize.py validate <directory>
"""

__version__ = "0.1.0"

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# EXCEPTIONS
# =============================================================================

class MusicOrganizerError(Exception):
    """Base exception for music organizer errors."""
    pass


class MetadataError(MusicOrganizerError):
    """Raised when there's an error extracting or processing metadata."""
    pass


class FileOperationError(MusicOrganizerError):
    """Raised when file operations fail."""
    pass


class ClassificationError(MusicOrganizerError):
    """Raised when content classification fails."""
    pass


# =============================================================================
# CONSOLE UTILITIES
# =============================================================================

class SimpleProgress:
    """Simple progress bar implementation without external dependencies."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        self.last_update = 0

    def update(self, advance: int = 1, force: bool = False):
        """Update progress bar."""
        self.current += advance
        if force or (self.current - self.last_update) >= max(1, self.total // 100):
            self._draw()
            self.last_update = self.current

    def set_description(self, description: str):
        """Update progress description."""
        self.description = description
        self._draw()

    def _draw(self):
        """Draw the progress bar."""
        if self.total == 0:
            return

        percent = min(100, (self.current * 100) // self.total)
        bar_length = 50
        filled = (percent * bar_length) // 100
        bar = '█' * filled + '░' * (bar_length - filled)

        elapsed = datetime.now() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f"{int(eta.total_seconds())}s"
        else:
            eta_str = "?:??"

        print(f"\r{self.description}: [{bar}] {percent}% ({self.current}/{self.total}) ETA: {eta_str}", end='', flush=True)

        if self.current >= self.total:
            print()

    def finish(self):
        """Mark progress as finished."""
        self.current = self.total
        self._draw()


class SimpleConsole:
    """Simple console output without rich dependency."""

    COLORS = {
        'bold': '\033[1m',
        'dim': '\033[2m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'reset': '\033[0m',
    }

    @staticmethod
    def print(text: str, style: Optional[str] = None):
        """Print text with optional style."""
        if style and style in SimpleConsole.COLORS:
            print(f"{SimpleConsole.COLORS[style]}{text}{SimpleConsole.COLORS['reset']}")
        else:
            print(text)

    @staticmethod
    def rule(title: str = "", character: str = "-"):
        """Print a horizontal rule with optional title."""
        width = 80
        if title:
            title = f" {title} "
            padding = (width - len(title)) // 2
            line = character * padding + title + character * padding
            if len(line) < width:
                line += character * (width - len(line))
        else:
            line = character * width
        print(line)

    @staticmethod
    def table(rows: List[List[str]], headers: List[str], title: Optional[str] = None):
        """Print a simple table."""
        if not rows:
            return

        if title:
            SimpleConsole.rule(title)

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))

        for row in rows:
            padded_row = row + [""] * (len(headers) - len(row))
            row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(padded_row))
            print(row_line)

        print()

    @staticmethod
    def prompt(message: str, default: Optional[Any] = None) -> str:
        """Prompt for user input."""
        if default:
            prompt_msg = f"{message} [{default}]: "
        else:
            prompt_msg = f"{message}: "

        result = input(prompt_msg)
        return result if result else str(default) if default else result

    @staticmethod
    def confirm(message: str, default: bool = False) -> bool:
        """Prompt for yes/no confirmation."""
        suffix = " [Y/n]" if default else " [y/N]"
        response = input(f"{message}{suffix}: ").strip().lower()

        if not response:
            return default

        return response.startswith('y')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DirectoryConfig:
    """Configuration for directory names."""
    albums: str = "Albums"
    live: str = "Live"
    collaborations: str = "Collaborations"
    compilations: str = "Compilations"
    rarities: str = "Rarities"


@dataclass
class FileOperationsConfig:
    """Configuration for file operations."""
    strategy: str = "move"
    backup: bool = True
    handle_duplicates: str = "number"


@dataclass
class Config:
    """Main configuration model."""
    source_directory: Path
    target_directory: Path
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    file_operations: FileOperationsConfig = field(default_factory=FileOperationsConfig)


# =============================================================================
# AUDIO FILE MODEL
# =============================================================================

class ContentType(Enum):
    """Types of music content."""
    STUDIO = "studio"
    LIVE = "live"
    COLLABORATION = "collaboration"
    COMPILATION = "compilation"
    RARITY = "rarity"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class CoverArt:
    """Cover art information for an album."""
    path: Path
    type: str
    format: str
    size: int

    @classmethod
    def from_file(cls, path: Path) -> Optional["CoverArt"]:
        """Create CoverArt instance from image file."""
        if not path.exists():
            return None

        extension = path.suffix.lower()
        if extension not in ['.jpg', '.jpeg', '.png', '.gif']:
            return None

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
                format=extension[1:],
                size=size
            )
        except (OSError, FileNotFoundError):
            return None


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
    date: Optional[str] = None
    location: Optional[str] = None
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
        track_str = ""
        if self.track_number is not None:
            track_str = f"{self.track_number:02d}. "

        title = self.title or self.path.stem
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

        extension = path.suffix.lower()
        type_map = {
            '.flac': 'FLAC', '.mp3': 'MP3', '.wav': 'WAV',
            '.m4a': 'MP4', '.mp4': 'MP4', '.aac': 'MP4',
            '.ogg': 'OGG', '.opus': 'OPUS', '.wma': 'WMA',
            '.aiff': 'AIFF', '.aif': 'AIFF'
        }
        file_type = type_map.get(extension, 'UNKNOWN')

        audio_file = cls(path=path, file_type=file_type)
        audio_file._parse_from_filename()
        return audio_file

    def _parse_from_filename(self) -> None:
        """Extract basic info from filename patterns."""
        filename = self.path.stem

        track_match = re.match(r'^(\d+)\.\s*(.+)', filename)
        if track_match:
            self.track_number = int(track_match.group(1))
            rest = track_match.group(2)
        else:
            rest = filename

        if ' - ' in rest:
            parts = rest.split(' - ', 1)
            if len(parts) == 2 and not self.artists:
                self.artists = [parts[0].strip()]
                self.title = parts[1].strip()
        elif not self.title:
            self.title = rest


# =============================================================================
# METADATA HANDLER
# =============================================================================

class MetadataHandler:
    """Handle all metadata operations using mutagen."""

    @staticmethod
    def extract_metadata(file_path: Path) -> AudioFile:
        """Extract metadata from audio file."""
        try:
            from mutagen import File as MutagenFile
            from mutagen.flac import FLAC
            from mutagen.id3 import ID3
            from mutagen.mp4 import MP4
            from mutagen.wave import WAVE

            audio_file = AudioFile.from_path(file_path)
            mutagen_file = MutagenFile(file_path)
            if mutagen_file is None:
                raise MetadataError(f"Unsupported file format: {file_path}")

            if isinstance(mutagen_file, FLAC):
                audio_file = MetadataHandler._extract_flac_metadata(audio_file, mutagen_file)
            elif isinstance(mutagen_file, ID3):
                audio_file = MetadataHandler._extract_id3_metadata(audio_file, mutagen_file)
            elif isinstance(mutagen_file, MP4):
                audio_file = MetadataHandler._extract_mp4_metadata(audio_file, mutagen_file)
            elif isinstance(mutagen_file, WAVE):
                audio_file = MetadataHandler._extract_wave_metadata(audio_file, mutagen_file)

            audio_file = MetadataHandler._post_process_metadata(audio_file)
            return audio_file

        except ImportError:
            raise MetadataError("mutagen library is required. Install with: pip install mutagen")
        except Exception as e:
            raise MetadataError(f"Failed to extract metadata from {file_path}: {e}")

    @staticmethod
    def _extract_flac_metadata(audio_file: AudioFile, flac_file) -> AudioFile:
        """Extract metadata from FLAC file (Vorbis comments)."""
        if not flac_file.tags:
            return audio_file

        tags = flac_file.tags

        raw_artists = MetadataHandler._get_list_field(tags, ['ARTIST'])
        audio_file.artists = []
        for artist in raw_artists:
            if ',' in artist:
                split_artists = [a.strip() for a in artist.split(',') if a.strip()]
                audio_file.artists.extend(split_artists)
            else:
                audio_file.artists.append(artist)

        seen = set()
        unique_artists = []
        for artist in audio_file.artists:
            if artist not in seen:
                seen.add(artist)
                unique_artists.append(artist)
        audio_file.artists = unique_artists

        albumartist = MetadataHandler._get_single_field(tags, ['ALBUMARTIST'])
        if albumartist:
            audio_file.primary_artist = albumartist
        else:
            if audio_file.artists:
                audio_file.primary_artist = audio_file.artists[0]

        audio_file.album = MetadataHandler._get_single_field(tags, ['ALBUM'])
        audio_file.title = MetadataHandler._get_single_field(tags, ['TITLE'])
        audio_file.genre = MetadataHandler._get_single_field(tags, ['GENRE'])

        date = MetadataHandler._get_single_field(tags, ['DATE'])
        if date:
            year_match = re.match(r'(\d{4})', date)
            if year_match:
                audio_file.year = int(year_match.group(1))
            audio_file.date = date

        tracknumber = MetadataHandler._get_single_field(tags, ['TRACKNUMBER'])
        if tracknumber:
            track_match = re.match(r'(\d+)', tracknumber)
            if track_match:
                audio_file.track_number = int(track_match.group(1))

        audio_file.location = MetadataHandler._get_single_field(tags, ['LOCATION', 'VENUE'])

        if hasattr(flac_file, 'pictures') and flac_file.pictures:
            audio_file.has_cover_art = True

        audio_file.metadata = dict(tags)
        return audio_file

    @staticmethod
    def _extract_id3_metadata(audio_file: AudioFile, id3_file) -> AudioFile:
        """Extract metadata from MP3 file (ID3 tags)."""
        if not id3_file.tags:
            return audio_file

        tags = id3_file.tags

        audio_file.artists = MetadataHandler._get_id3_text(tags, ['TPE1'])

        primary_artist_list = MetadataHandler._get_id3_text(tags, ['TPE2'])
        audio_file.primary_artist = primary_artist_list[0] if primary_artist_list else \
                                   (audio_file.artists[0] if audio_file.artists else None)

        album_list = MetadataHandler._get_id3_text(tags, ['TALB'])
        audio_file.album = album_list[0] if album_list else None

        title_list = MetadataHandler._get_id3_text(tags, ['TIT2'])
        audio_file.title = title_list[0] if title_list else None

        genre_list = MetadataHandler._get_id3_text(tags, ['TCON'])
        audio_file.genre = genre_list[0] if genre_list else None

        year = MetadataHandler._get_id3_text(tags, ['TDRC', 'TYER'])
        if year:
            year_match = re.match(r'(\d{4})', str(year[0]) if isinstance(year, list) else str(year))
            if year_match:
                audio_file.year = int(year_match.group(1))

        track = MetadataHandler._get_id3_text(tags, ['TRCK'])
        if track:
            track_match = re.match(r'(\d+)', str(track[0]) if isinstance(track, list) else str(track))
            if track_match:
                audio_file.track_number = int(track_match.group(1))

        if 'APIC:' in tags:
            audio_file.has_cover_art = True

        audio_file.metadata = {
            'artists': audio_file.artists,
            'album': audio_file.album,
            'title': audio_file.title,
            'year': audio_file.year,
            'track': audio_file.track_number,
            'genre': audio_file.genre,
        }

        return audio_file

    @staticmethod
    def _extract_mp4_metadata(audio_file: AudioFile, mp4_file) -> AudioFile:
        """Extract metadata from M4A/MP4 file."""
        if not mp4_file.tags:
            return audio_file

        tags = mp4_file.tags

        audio_file.artists = MetadataHandler._get_mp4_field(tags, ['\xa9ART'])
        audio_file.primary_artist = MetadataHandler._get_mp4_field(tags, ['aART']) or \
                                   (audio_file.artists[0] if audio_file.artists else None)
        audio_file.album = MetadataHandler._get_mp4_field(tags, ['\xa9alb'])
        audio_file.title = MetadataHandler._get_mp4_field(tags, ['\xa9nam'])
        audio_file.genre = MetadataHandler._get_mp4_field(tags, ['\xa9gen'])

        date = MetadataHandler._get_mp4_field(tags, ['\xa9day'])
        if date:
            year_match = re.match(r'(\d{4})', str(date))
            if year_match:
                audio_file.year = int(year_match.group(1))

        track = MetadataHandler._get_mp4_field(tags, ['trkn'])
        if track and isinstance(track, tuple) and len(track) >= 1:
            audio_file.track_number = track[0]

        if 'covr' in tags:
            audio_file.has_cover_art = True

        audio_file.metadata = dict(tags)
        return audio_file

    @staticmethod
    def _extract_wave_metadata(audio_file: AudioFile, wave_file) -> AudioFile:
        """Extract metadata from WAV/AIFF file."""
        if hasattr(wave_file, 'tags') and wave_file.tags:
            tags = wave_file.tags
            audio_file.title = tags.get('TIT2', [None])[0]
            audio_file.artists = tags.get('TPE1', [])
            audio_file.album = tags.get('TALB', [None])[0]

            if audio_file.artists:
                audio_file.primary_artist = audio_file.artists[0]

        return audio_file

    @staticmethod
    def _get_list_field(tags: Dict, keys: List[str]) -> List[str]:
        """Get a field that can have multiple values."""
        for key in keys:
            if key in tags:
                values = tags[key]
                if isinstance(values, list):
                    return [str(v) for v in values]
                return [str(values)]
        return []

    @staticmethod
    def _get_single_field(tags: Dict, keys: List[str]) -> Optional[str]:
        """Get a single-value field."""
        for key in keys:
            if key in tags:
                value = tags[key]
                if isinstance(value, list) and value:
                    return str(value[0])
                return str(value) if value else None
        return None

    @staticmethod
    def _get_id3_text(tags: Dict, frame_ids: List[str]) -> List[str]:
        """Get text from ID3 frame."""
        for frame_id in frame_ids:
            if frame_id in tags:
                frame = tags[frame_id]
                if hasattr(frame, 'text'):
                    return [str(t) for t in frame.text]
        return []

    @staticmethod
    def _get_mp4_field(tags: Dict, keys: List[str]) -> List[str]:
        """Get field from MP4 tags."""
        for key in keys:
            if key in tags:
                value = tags[key]
                if isinstance(value, list):
                    return [str(v) for v in value]
                return [str(value)]
        return []

    @staticmethod
    def _post_process_metadata(audio_file: AudioFile) -> AudioFile:
        """Clean up and standardize extracted metadata."""
        if audio_file.artists:
            audio_file.artists = [MetadataHandler._clean_artist_name(a) for a in audio_file.artists if a]
            audio_file.artists = list(set(audio_file.artists))

        if audio_file.primary_artist:
            audio_file.primary_artist = MetadataHandler._clean_artist_name(audio_file.primary_artist)

        if audio_file.album:
            audio_file.album = MetadataHandler._clean_title(audio_file.album)

        if audio_file.title:
            audio_file.title = MetadataHandler._clean_title(audio_file.title)

        if audio_file.genre:
            audio_file.genre = MetadataHandler._standardize_genre(audio_file.genre)

        return audio_file

    @staticmethod
    def _clean_artist_name(name: str) -> str:
        """Clean up artist name."""
        if not name:
            return ""

        name = re.sub(r'^(the |The )', '', name)
        name = re.sub(r'\s+$', '', name)
        name = re.sub(r'\s+feat\.\s*', ' feat. ', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+featuring\s+', ' featuring ', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+with\s+', ' with ', name, flags=re.IGNORECASE)

        return name.strip()

    @staticmethod
    def _clean_title(title: str) -> str:
        """Clean up album or track title."""
        if not title:
            return ""

        title = re.sub(r'\s+', ' ', title)
        title = re.sub(r'\[(\d{4})\]', r'(\1)', title)

        return title.strip()

    @staticmethod
    def _standardize_genre(genre: str) -> str:
        """Standardize genre names."""
        if not genre:
            return ""

        genre_map = {
            'rock & roll': 'Rock', 'r&b': 'R&B',
            'hip-hop': 'Hip Hop', 'electronica': 'Electronic',
            'new age': 'New Age',
        }

        genre_lower = genre.lower()
        if genre_lower in genre_map:
            return genre_map[genre_lower]

        return ' '.join(word.capitalize() for word in genre.split())

    @staticmethod
    def find_cover_art(directory: Path) -> List[Path]:
        """Find cover art files in a directory."""
        if not directory.is_dir():
            return []

        cover_patterns = [
            '*cover*', '*front*', '*folder*', '*album*',
            '*.jpg', '*.jpeg', '*.png',
        ]

        cover_files = []
        for pattern in cover_patterns:
            for file_path in directory.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    cover_files.append(file_path)

        seen = set()
        unique_files = []
        for f in cover_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)

        return unique_files


# =============================================================================
# CONTENT CLASSIFIER
# =============================================================================

class ContentClassifier:
    """Classify audio content type based on metadata and patterns."""

    LIVE_PATTERNS = [
        r'\blive\b', r'live at', r'live in', r'bootleg',
        r'recording', r'concert', r'performance', r'show', r'festival',
    ]

    DATE_PATTERNS = [
        r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{1,2}\.\d{1,2}\.\d{4}',
    ]

    COMPILATION_PATTERNS = [
        r'\bgreatest hits\b', r'\bbest of\b', r'\bthe best\b',
        r'\bessential\b', r'\bcollection\b', r'\banthology\b',
        r'\bgold\b', r'\bplatinum\b', r'\bultimate\b', r'\bhits\b',
        r'\bsingles\b', r'\bthe very best\b',
    ]

    RARITY_PATTERNS = [
        r'\bdemo\b', r'\brare\b', r'\bunreleased\b',
        r'\bbonus\b', r'\bextra\b', r'\bspecial edition\b',
        r'\blimited edition\b', r'\banniversary\b', r'\bdeluxe\b',
        r'\bremastered\b', r'\bexpanded\b', r'\bbootleg\b',
    ]

    @classmethod
    def classify(cls, audio_file: AudioFile) -> Tuple[ContentType, float]:
        """Classify audio file content type."""
        collab_type, collab_score = cls._classify_collaboration(audio_file)
        if collab_type == ContentType.COLLABORATION and collab_score > 0.6:
            return collab_type, collab_score

        live_type, live_score = cls._classify_live(audio_file)
        if live_type == ContentType.LIVE and live_score > 0.5:
            return live_type, live_score

        comp_type, comp_score = cls._classify_compilation(audio_file)
        if comp_type == ContentType.COMPILATION and comp_score > 0.6:
            return comp_type, comp_score

        rarity_type, rarity_score = cls._classify_rarity(audio_file)
        if rarity_type == ContentType.RARITY and rarity_score > 0.6:
            return rarity_type, rarity_score

        return ContentType.STUDIO, 0.5

    @classmethod
    def _classify_live(cls, audio_file: AudioFile) -> Tuple[ContentType, float]:
        """Classify if content is a live recording."""
        score = 0.0

        text_to_check = []
        if audio_file.album:
            text_to_check.append(audio_file.album.lower())
        if audio_file.title:
            text_to_check.append(audio_file.title.lower())
        if audio_file.path:
            text_to_check.append(audio_file.path.name.lower())

        for text in text_to_check:
            for pattern in cls.LIVE_PATTERNS:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    score += 0.3

            for pattern in cls.DATE_PATTERNS:
                if re.search(pattern, text):
                    score += 0.2
                    match = re.search(pattern, text)
                    if match and not audio_file.date:
                        audio_file.date = match.group()

        if audio_file.location:
            score += 0.4

        if audio_file.path:
            path_parts = audio_file.path.parts
            for part in path_parts:
                if re.search(r'\d{4}\s+-\s+.+', part):
                    score += 0.2
                    if not audio_file.location:
                        audio_file.location = part
                    break

        if audio_file.genre and 'live' in audio_file.genre.lower():
            score += 0.1

        score = min(score, 1.0)

        if score > 0.5:
            return ContentType.LIVE, score

        return ContentType.STUDIO, score

    @classmethod
    def _classify_collaboration(cls, audio_file: AudioFile) -> Tuple[ContentType, float]:
        """Classify if content is a collaboration."""
        score = 0.0

        if audio_file.primary_artist:
            if ',' in audio_file.primary_artist:
                primary_artists = [a.strip() for a in audio_file.primary_artist.split(',') if a.strip()]

                if len(primary_artists) == 2:
                    artist_str = ' & '.join(primary_artists).lower()
                    if any(term in audio_file.album.lower() for term in [' with ', ' and ', ' & ']) if audio_file.album else False:
                        score += 0.5
                    elif audio_file.path and any(artist.lower() in str(audio_file.path).lower() for artist in primary_artists):
                        score += 0.6
                elif len(primary_artists) == 3:
                    if not any(prefix in audio_file.album.lower() for prefix in ['trio', 'quartet', 'quintet']) if audio_file.album else False:
                        score += 0.4

        if len(audio_file.artists) > 1:
            if len(audio_file.artists) > 5:
                score -= 0.2
            elif len(audio_file.artists) == 2:
                score += 0.7
            elif len(audio_file.artists) <= 3:
                score += 0.5

        text_to_check = [audio_file.album] if audio_file.album else []
        collab_terms = [' with ', ' and ', ' & ', ' featuring ', ' feat ', 'featuring ', 'presents']

        for text in text_to_check:
            if not text:
                continue
            for term in collab_terms:
                if term in text.lower():
                    score += 0.4
                    break

        if audio_file.album:
            if ' vs ' in audio_file.album.lower():
                score += 0.5
            if ' duets' in audio_file.album.lower():
                score += 0.3
            if ' duo' in audio_file.album.lower() or ' trio' in audio_file.album.lower():
                score += 0.3

        if audio_file.path:
            path_str = str(audio_file.path).lower()
            if ' & ' in path_str or ' and ' in path_str:
                folder = Path(path_str).name
                if ' & ' in folder or ' and ' in folder:
                    score += 0.4

        if audio_file.genre and 'jazz' in audio_file.genre.lower():
            score -= 0.2

        score = max(0, min(score, 1.0))

        if score > 0.6:
            return ContentType.COLLABORATION, score

        return ContentType.STUDIO, score

    @classmethod
    def _classify_compilation(cls, audio_file: AudioFile) -> Tuple[ContentType, float]:
        """Classify if content is a compilation."""
        score = 0.0

        text_to_check = []
        if audio_file.album:
            text_to_check.append(audio_file.album.lower())
        if audio_file.title:
            text_to_check.append(audio_file.title.lower())

        for text in text_to_check:
            for pattern in cls.COMPILATION_PATTERNS:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    score += 0.4

        if audio_file.primary_artist and audio_file.primary_artist.lower() in ['various artists', 'va', 'various']:
            score += 0.6

        if audio_file.primary_artist:
            if 'dj' in audio_file.primary_artist.lower() and audio_file.album and 'mix' in audio_file.album.lower():
                score += 0.3

        score = min(score, 1.0)

        if score > 0.6:
            return ContentType.COMPILATION, score

        return ContentType.STUDIO, score

    @classmethod
    def _classify_rarity(cls, audio_file: AudioFile) -> Tuple[ContentType, float]:
        """Classify if content is a rarity or special edition."""
        score = 0.0

        text_to_check = []
        if audio_file.album:
            text_to_check.append(audio_file.album.lower())
        if audio_file.title:
            text_to_check.append(audio_file.title.lower())

        for text in text_to_check:
            for pattern in cls.RARITY_PATTERNS:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    score += 0.3

        if audio_file.album:
            anniversary_match = re.search(r'\(\d+(?:st|nd|rd|th)\s+anniversary\)', audio_file.album.lower())
            if anniversary_match:
                score += 0.5

            if re.search(r'\[(?:remastered|deluxe|expanded|bonus)\]', audio_file.album, flags=re.IGNORECASE):
                score += 0.4

        if audio_file.path:
            path_str = str(audio_file.path).lower()
            for pattern in cls.RARITY_PATTERNS:
                if re.search(pattern, path_str):
                    score += 0.2

        score = min(score, 1.0)

        if score > 0.6:
            return ContentType.RARITY, score

        return ContentType.STUDIO, score


# =============================================================================
# FILE MOVER
# =============================================================================

class FileMover:
    """Handle safe file operations with backup and rollback support."""

    def __init__(self, backup_enabled: bool = True, backup_dir: Optional[Path] = None):
        self.backup_enabled = backup_enabled
        self.backup_dir = backup_dir
        self.operations: List[Dict] = []
        self.started = False

    def start_operation(self, source_root: Path) -> None:
        """Start a new operation session with optional backup."""
        if self.started:
            raise FileOperationError("Operation already in progress")

        self.started = True
        self.operations = []

        if self.backup_enabled:
            if not self.backup_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.backup_dir = source_root.parent / f"backup_{timestamp}"

            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self._create_backup_manifest(source_root)

    def finish_operation(self) -> None:
        """Finish current operation session."""
        if not self.started:
            raise FileOperationError("No operation in progress")

        if self.backup_enabled and self.backup_dir:
            self._save_operation_log()

        self.started = False

    def move_file(self, audio_file: AudioFile, target_path: Path) -> Path:
        """Move an audio file to its target location."""
        if not self.started:
            raise FileOperationError("Must start operation before moving files")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        final_target = self._resolve_duplicate(target_path)

        try:
            if self.backup_enabled and self.backup_dir:
                self._backup_file(audio_file.path)

            shutil.move(str(audio_file.path), str(final_target))
            audio_file.path = final_target

            self.operations.append({
                'type': 'move',
                'original': str(audio_file.path),
                'target': str(final_target),
                'timestamp': datetime.now().isoformat()
            })

            return final_target

        except Exception as e:
            raise FileOperationError(f"Failed to move {audio_file.path}: {e}")

    def move_cover_art(self, cover_art: CoverArt, target_dir: Path) -> Optional[Path]:
        """Move cover art to target directory."""
        if not cover_art or not cover_art.path.exists():
            return None

        target_filename = self._get_cover_art_filename(cover_art)
        target_path = target_dir / target_filename
        target_path = self._resolve_duplicate(target_path)

        try:
            if self.backup_enabled and self.backup_dir:
                self._backup_file(cover_art.path)

            shutil.move(str(cover_art.path), str(target_path))
            cover_art.path = target_path

            self.operations.append({
                'type': 'move_cover',
                'original': str(cover_art.path),
                'target': str(target_path),
                'timestamp': datetime.now().isoformat()
            })

            return target_path

        except Exception as e:
            raise FileOperationError(f"Failed to move cover art {cover_art.path}: {e}")

    def rollback(self) -> None:
        """Rollback all performed operations."""
        if not self.operations:
            return

        for op in reversed(self.operations):
            try:
                if op['type'] in ['move', 'move_cover']:
                    original = Path(op['original'])
                    target = Path(op['target'])

                    if target.exists() and not original.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(target), str(original))

            except Exception as e:
                print(f"Warning: Failed to rollback {op['target']}: {e}")

        self.operations = []

    def get_operation_summary(self) -> Dict:
        """Get summary of performed operations."""
        summary = {
            'total_files': 0,
            'total_cover_art': 0,
            'directories_created': set(),
        }

        for op in self.operations:
            if op['type'] == 'move':
                summary['total_files'] += 1
            elif op['type'] == 'move_cover':
                summary['total_cover_art'] += 1

            target = Path(op['target'])
            summary['directories_created'].add(str(target.parent))

        summary['directories_created'] = len(summary['directories_created'])
        return summary

    def _resolve_duplicate(self, target_path: Path) -> Path:
        """Resolve duplicate filenames by adding a number."""
        if not target_path.exists():
            return target_path

        base = target_path.stem
        ext = target_path.suffix
        parent = target_path.parent
        counter = 1

        while True:
            new_name = f"{base} ({counter}){ext}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1

    def _backup_file(self, file_path: Path) -> None:
        """Create a backup of the file."""
        if not self.backup_dir:
            return

        try:
            relative_path = file_path.relative_to(file_path.anchor)
            backup_path = self.backup_dir / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            if not backup_path.exists():
                shutil.copy2(str(file_path), str(backup_path))

        except Exception as e:
            print(f"Warning: Failed to backup {file_path}: {e}")

    def _create_backup_manifest(self, source_root: Path) -> None:
        """Create a manifest of all files before starting operations."""
        if not self.backup_dir:
            return

        manifest = {
            'source_root': str(source_root),
            'timestamp': datetime.now().isoformat(),
            'files': []
        }

        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}
        cover_extensions = {'.jpg', '.jpeg', '.png', '.gif'}

        for file_path in source_root.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in audio_extensions or ext in cover_extensions:
                    try:
                        stat = file_path.stat()
                        manifest['files'].append({
                            'path': str(file_path.relative_to(source_root)),
                            'size': stat.st_size,
                            'mtime': stat.st_mtime,
                            'type': 'audio' if ext in audio_extensions else 'cover'
                        })
                    except (OSError, ValueError):
                        pass

        manifest_path = self.backup_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _save_operation_log(self) -> None:
        """Save operation log to backup directory."""
        if not self.backup_dir or not self.operations:
            return

        log_path = self.backup_dir / 'operations.json'
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'operations': self.operations,
            'summary': self.get_operation_summary()
        }

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def _get_cover_art_filename(self, cover_art: CoverArt) -> str:
        """Get standardized filename for cover art."""
        if cover_art.type == 'front':
            return f'folder.{cover_art.format}'
        elif cover_art.type == 'back':
            return f'back.{cover_art.format}'
        elif cover_art.type == 'disc':
            return f'disc.{cover_art.format}'
        else:
            return f'cover.{cover_art.format}'


class DirectoryOrganizer:
    """Helper class for creating and validating directory structures."""

    DIRECTORIES = ["Albums", "Live", "Collaborations", "Compilations", "Rarities"]

    @staticmethod
    def create_directory_structure(base_path: Path) -> None:
        """Create the standard music directory structure."""
        for dir_name in DirectoryOrganizer.DIRECTORIES:
            dir_path = base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_structure(base_path: Path) -> Dict[str, bool]:
        """Validate that required directories exist."""
        validation = {}
        for dir_name in DirectoryOrganizer.DIRECTORIES:
            validation[dir_name] = (base_path / dir_name).exists()
        return validation

    @staticmethod
    def get_empty_directories(base_path: Path) -> List[Path]:
        """Find empty directories that can be cleaned up."""
        empty_dirs = []

        for root, dirs, files in os.walk(base_path):
            audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}
            cover_extensions = {'.jpg', '.jpeg', '.png', '.gif'}

            has_media = False
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in audio_extensions or ext in cover_extensions:
                    has_media = True
                    break

            if not has_media and not dirs:
                empty_dirs.append(Path(root))

        return empty_dirs


# =============================================================================
# MUSIC ORGANIZER
# =============================================================================

class MusicOrganizer:
    """Main orchestrator for music library organization."""

    def __init__(self, config: Config, dry_run: bool = False, interactive: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.interactive = interactive
        self.metadata_handler = MetadataHandler()
        self.classifier = ContentClassifier()
        self.file_mover = FileMover(
            backup_enabled=config.file_operations.backup,
            backup_dir=config.target_directory.parent / "backup" if config.file_operations.backup else None
        )
        self.user_decisions = {}

    def scan_directory(self, directory: Path) -> List[Path]:
        """Scan directory for audio files."""
        if not directory.exists():
            raise MusicOrganizerError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise MusicOrganizerError(f"Path is not a directory: {directory}")

        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}
        files = []

        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                files.append(file_path)

        return files

    def organize_files(self, files: List[Path], progress=None, task_id=None) -> Dict[str, Any]:
        """Organize a list of audio files."""
        results = {
            'processed': 0,
            'moved': 0,
            'skipped': 0,
            'by_category': {
                'Albums': 0, 'Live': 0, 'Collaborations': 0,
                'Compilations': 0, 'Rarities': 0, 'Unknown': 0
            },
            'errors': []
        }

        if not self.dry_run:
            self.file_mover.start_operation(self.config.source_directory)

        try:
            if not self.dry_run:
                DirectoryOrganizer.create_directory_structure(self.config.target_directory)

            file_groups = self._group_files(files)

            for group_path, group_files in file_groups.items():
                for file_path in group_files:
                    try:
                        moved = self._process_file(file_path)
                        if moved:
                            results['moved'] += 1
                        else:
                            results['skipped'] += 1

                        if hasattr(moved, 'content_type'):
                            category = self._get_category_name(moved.content_type)
                            results['by_category'][category] += 1

                    except Exception as e:
                        error_msg = f"Failed to process {file_path.name}: {e}"
                        results['errors'].append(error_msg)

                    results['processed'] += 1

                    if progress and task_id is not None:
                        progress.advance(task_id)

        finally:
            if not self.dry_run:
                self.file_mover.finish_operation()

        return results

    def _process_file(self, file_path: Path) -> Optional[AudioFile]:
        """Process a single audio file."""
        audio_file = self.metadata_handler.extract_metadata(file_path)
        content_type, confidence = self.classifier.classify(audio_file)
        audio_file.content_type = content_type

        if self.dry_run:
            target_path = audio_file.get_target_path(self.config.target_directory)
            target_filename = audio_file.get_target_filename()
            full_target = target_path / target_filename
            print(f"Would move: {file_path.name} -> {full_target.relative_to(self.config.target_directory)}")
            return audio_file

        target_dir = audio_file.get_target_path(self.config.target_directory)
        target_filename = audio_file.get_target_filename()
        target_path = target_dir / target_filename

        self.file_mover.move_file(audio_file, target_path)
        self._process_cover_art(file_path, target_dir)

        return audio_file

    def _group_files(self, files: List[Path]) -> Dict[Path, List[Path]]:
        """Group files by their containing directory for batch processing."""
        groups = {}
        for file_path in files:
            group_path = file_path.parent
            if group_path not in groups:
                groups[group_path] = []
            groups[group_path].append(file_path)
        return groups

    def _process_cover_art(self, audio_file_path: Path, target_dir: Path) -> None:
        """Find and move cover art for an audio file."""
        cover_files = self.metadata_handler.find_cover_art(audio_file_path.parent)

        for cover_path in cover_files:
            cover_art = CoverArt.from_file(cover_path)
            if cover_art:
                if self.dry_run:
                    print(f"Would move cover art: {cover_path.name} -> {target_dir}")
                else:
                    self.file_mover.move_cover_art(cover_art, target_dir)

    def _get_category_name(self, content_type) -> str:
        """Map content type to category name for results."""
        category_map = {
            ContentType.STUDIO: 'Albums',
            ContentType.LIVE: 'Live',
            ContentType.COLLABORATION: 'Collaborations',
            ContentType.COMPILATION: 'Compilations',
            ContentType.RARITY: 'Rarities',
            ContentType.UNKNOWN: 'Unknown'
        }
        return category_map.get(content_type, 'Unknown')


# =============================================================================
# CLI
# =============================================================================

console = SimpleConsole()


async def organize_command_async(args):
    """Handle the organize command with async support."""
    try:
        cfg = Config(
            source_directory=args.source,
            target_directory=args.target,
            file_operations=type('FileOps', (), {'backup': args.backup})()
        )

        organizer = MusicOrganizer(
            cfg,
            dry_run=args.dry_run,
            interactive=args.interactive
        )

        console.print("\nMusic Organization Plan", 'bold')
        console.print(f"Source: {args.source}")
        console.print(f"Target: {args.target}")
        console.print(f"Strategy: Move files")
        console.print(f"Backup: {'Enabled' if args.backup else 'Disabled'}")
        console.print(f"Interactive: {'Enabled' if args.interactive else 'Disabled'}")
        console.print(f"Dry run: {'Yes' if args.dry_run else 'No'}")

        if not args.dry_run:
            if not console.confirm("\nProceed with organization?", default=True):
                console.print("Cancelled", 'yellow')
                return 0

        console.print("\nScanning music files...")
        files = organizer.scan_directory(args.source)

        if not files:
            console.print("No audio files found!", 'yellow')
            return 0

        console.print(f"\nFound {len(files)} audio files", 'green')

        progress = SimpleProgress(len(files), "Processing")
        results = organizer.organize_files(files, progress=progress)
        progress.finish()

        console.rule("Results")

        summary_data = [
            ["Processed", str(results['processed'])],
            ["Moved", str(results['moved'])],
            ["Skipped", str(results['skipped'])],
            ["Errors", str(len(results['errors']))]
        ]
        console.table(summary_data, ["Metric", "Count"])

        if results['moved'] > 0:
            console.print("\nFiles by category:")
            for category, count in results['by_category'].items():
                if count > 0:
                    console.print(f"  {category}: {count}", 'green')

        if results['errors']:
            console.print("\nErrors encountered:", 'red')
            for error in results['errors'][:10]:
                console.print(f"  {error}", 'red')
            if len(results['errors']) > 10:
                console.print(f"  ... and {len(results['errors']) - 10} more errors", 'red')

        return 0

    except MusicOrganizerError as e:
        console.print(f"\nError: {e}", 'red')
        return 1
    except Exception as e:
        console.print(f"\nUnexpected error: {e}", 'red')
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


def organize_command(args):
    """Wrapper to run the async organize command."""
    return asyncio.run(organize_command_async(args))


def scan_command(args):
    """Handle the scan command."""
    try:
        handler = MetadataHandler()
        classifier = ContentClassifier()

        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}
        files = []

        if args.recursive:
            files = [f for f in args.directory.rglob('*') if f.suffix.lower() in audio_extensions]
        else:
            files = [f for f in args.directory.glob('*') if f.suffix.lower() in audio_extensions]

        if not files:
            console.print("No audio files found!", 'yellow')
            return 0

        analysis = {
            'total_files': len(files),
            'file_types': {},
            'content_types': {},
            'has_metadata': 0,
            'total_size_mb': 0,
            'sample_files': []
        }

        console.print(f"\nAnalyzing {len(files)} files...")

        progress = SimpleProgress(len(files), "Analyzing")

        for file_path in files:
            try:
                audio_file = handler.extract_metadata(file_path)
                content_type, _ = classifier.classify(audio_file)

                ext = file_path.suffix.lower()
                analysis['file_types'][ext] = analysis['file_types'].get(ext, 0) + 1
                analysis['content_types'][content_type.value] = \
                    analysis['content_types'].get(content_type.value, 0) + 1

                if audio_file.album or audio_file.title or audio_file.artists:
                    analysis['has_metadata'] += 1

                analysis['total_size_mb'] += audio_file.size_mb

                if len(analysis['sample_files']) < 5:
                    analysis['sample_files'].append({
                        'path': str(file_path.relative_to(args.directory)),
                        'type': content_type.value,
                        'artists': audio_file.artists,
                        'album': audio_file.album
                    })

            except Exception:
                pass

            progress.update()

        progress.finish()

        console.rule("Directory Analysis")

        info_data = [
            ["Total files", str(analysis['total_files'])],
            ["Total size", f"{analysis['total_size_mb']:.1f} MB"],
            ["Files with metadata", f"{analysis['has_metadata']} ({analysis['has_metadata']/analysis['total_files']*100:.1f}%)"]
        ]
        console.table(info_data, ["Metric", "Value"])

        if analysis['file_types']:
            console.print("\nFile Types:")
            for ext, count in sorted(analysis['file_types'].items()):
                console.print(f"  {ext}: {count}")

        if analysis['content_types']:
            console.print("\nContent Types:")
            for content_type, count in sorted(analysis['content_types'].items()):
                console.print(f"  {content_type}: {count}")

        if analysis['sample_files']:
            console.print("\nSample Files:")
            for sample in analysis['sample_files']:
                console.print(f"\n  {sample['path']}", 'dim')
                console.print(f"    Type: {sample['type']}")
                if sample['artists']:
                    console.print(f"    Artists: {', '.join(sample['artists'][:3])}")
                if sample['album']:
                    console.print(f"    Album: {sample['album']}")

        return 0

    except Exception as e:
        console.print(f"\nError: {e}", 'red')
        return 1


def inspect_command(args):
    """Handle the inspect command."""
    try:
        handler = MetadataHandler()
        classifier = ContentClassifier()

        if args.file_path.suffix.lower() not in {'.flac', '.mp3', '.wav', '.m4a', '.aac'}:
            console.print("Not a supported audio file", 'red')
            return 0

        audio_file = handler.extract_metadata(args.file_path)
        content_type, confidence = classifier.classify(audio_file)

        console.print(f"\nFile: {args.file_path.name}", 'bold')

        info_data = [
            ["File type", audio_file.file_type],
            ["Size", f"{audio_file.size_mb:.1f} MB"],
            ["Content type", f"{content_type.value} (confidence: {confidence:.2f})"]
        ]

        if audio_file.artists:
            info_data.append(["Artists", ", ".join(audio_file.artists)])
        if audio_file.primary_artist:
            info_data.append(["Primary artist", audio_file.primary_artist])
        if audio_file.album:
            info_data.append(["Album", audio_file.album])
        if audio_file.title:
            info_data.append(["Title", audio_file.title])
        if audio_file.year:
            info_data.append(["Year", str(audio_file.year)])
        if audio_file.date:
            info_data.append(["Date", audio_file.date])
        if audio_file.location:
            info_data.append(["Location", audio_file.location])
        if audio_file.track_number:
            info_data.append(["Track", str(audio_file.track_number)])
        if audio_file.genre:
            info_data.append(["Genre", audio_file.genre])
        info_data.append(["Has cover art", "Yes" if audio_file.has_cover_art else "No"])

        console.table(info_data, ["Property", "Value"])

        target_path = audio_file.get_target_path(Path("/tmp/test"))
        console.print(f"\nTarget directory: {target_path}", 'cyan')

        if audio_file.metadata:
            console.print("\nRaw Metadata:")
            for key, value in list(audio_file.metadata.items())[:20]:
                console.print(f"  {key}: {value}")

        return 0

    except Exception as e:
        console.print(f"\nError: {e}", 'red')
        return 1


def validate_command(args):
    """Handle the validate command."""
    try:
        console.print(f"\nValidating organization in: {args.directory}")

        validation = DirectoryOrganizer.validate_structure(args.directory)

        structure_data = []
        for dir_name, exists in validation.items():
            status = "" if exists else ""
            color = 'green' if exists else 'red'
            structure_data.append([dir_name, f"{color}{status}{SimpleConsole.COLORS['reset']}"])

        console.table(structure_data, ["Directory", "Status"], "Directory Structure")

        all_good = all(validation.values())

        empty_dirs = DirectoryOrganizer.get_empty_directories(args.directory)
        if empty_dirs:
            console.print(f"\nFound {len(empty_dirs)} empty directories", 'yellow')
            if console.confirm("List empty directories?"):
                for empty_dir in empty_dirs[:10]:
                    console.print(f"  {empty_dir}")
                if len(empty_dirs) > 10:
                    console.print(f"  ... and {len(empty_dirs) - 10} more")

        misplaced = []
        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}

        for item in args.directory.iterdir():
            if item.is_file() and item.suffix.lower() in audio_extensions:
                misplaced.append(item)

        if misplaced:
            console.print(f"\nFound {len(misplaced)} audio files in root directory", 'yellow')
            for file_path in misplaced[:5]:
                console.print(f"  {file_path.name}")
            if len(misplaced) > 5:
                console.print(f"  ... and {len(misplaced) - 5} more")

        if all_good and not misplaced and not empty_dirs:
            console.print("\nDirectory is properly organized!", 'green')

        return 0

    except Exception as e:
        console.print(f"\nError: {e}", 'red')
        return 1


def main():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Organize your music library with smart metadata-based categorization",
        epilog="Example: music_organize.py organize /path/to/music /path/to/organized --dry-run"
    )

    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Organize command
    org_parser = subparsers.add_parser('organize', help='Organize music files')
    org_parser.add_argument('source', type=Path, help='Source directory containing music files')
    org_parser.add_argument('target', type=Path, help='Target directory for organized files')
    org_parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    org_parser.add_argument('--interactive', action='store_true', help='Prompt for ambiguous categorizations')
    org_parser.add_argument('--backup/--no-backup', default=True, help='Create backup before reorganization (default: enabled)')
    org_parser.add_argument('--verbose', action='store_true', help='Verbose output')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Analyze music library')
    scan_parser.add_argument('directory', type=Path, help='Directory to scan')
    scan_parser.add_argument('--recursive', action='store_true', default=True, help='Scan subdirectories recursively')

    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect metadata of a single audio file')
    inspect_parser.add_argument('file_path', type=Path, help='Path to audio file')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate music directory organization')
    validate_parser.add_argument('directory', type=Path, help='Directory to validate')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'organize':
        return organize_command(args)
    elif args.command == 'scan':
        return scan_command(args)
    elif args.command == 'inspect':
        return inspect_command(args)
    elif args.command == 'validate':
        return validate_command(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
