"""Metadata handling for audio files using mutagen."""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import re

from mutagen import File as MutagenFile
from mutagen.flac import FLAC
from mutagen.id3 import ID3NoHeaderError, ID3v1SaveOptions, ID3
from mutagen.mp4 import MP4
from mutagen.wave import WAVE
from mutagen.aiff import AIFF

# Try to import optional format modules
try:
    from mutagen.oggvorbis import OggVorbis
except ImportError:
    OggVorbis = None

try:
    from mutagen.oggopus import OggOpus
except ImportError:
    OggOpus = None

try:
    from mutagen.asf import ASF
except ImportError:
    ASF = None

try:
    from mutagen.apev2 import APEv2File
except ImportError:
    APEv2File = None

from ..exceptions import MetadataError
from ..models.audio_file import AudioFile


class MetadataHandler:
    """Handle all metadata operations using mutagen."""

    @staticmethod
    def extract_metadata(file_path: Path) -> AudioFile:
        """Extract metadata from audio file."""
        try:
            # Create base AudioFile
            audio_file = AudioFile.from_path(file_path)

            # Load with mutagen
            mutagen_file = MutagenFile(file_path)
            if mutagen_file is None:
                raise MetadataError(f"Unsupported file format: {file_path}")

            # Extract metadata based on file type
            if isinstance(mutagen_file, FLAC):
                audio_file = MetadataHandler._extract_flac_metadata(audio_file, mutagen_file)
            elif isinstance(mutagen_file, ID3):
                audio_file = MetadataHandler._extract_id3_metadata(audio_file, mutagen_file)
            elif isinstance(mutagen_file, MP4):
                audio_file = MetadataHandler._extract_mp4_metadata(audio_file, mutagen_file)
            elif isinstance(mutagen_file, (WAVE, AIFF)):
                audio_file = MetadataHandler._extract_wave_metadata(audio_file, mutagen_file)
            elif OggVorbis and isinstance(mutagen_file, OggVorbis):
                audio_file = MetadataHandler._extract_ogg_metadata(audio_file, mutagen_file)
            elif OggOpus and isinstance(mutagen_file, OggOpus):
                audio_file = MetadataHandler._extract_ogg_metadata(audio_file, mutagen_file)
            elif ASF and isinstance(mutagen_file, ASF):
                audio_file = MetadataHandler._extract_wma_metadata(audio_file, mutagen_file)
            elif APEv2File and isinstance(mutagen_file, APEv2File):
                audio_file = MetadataHandler._extract_ape_metadata(audio_file, mutagen_file)

            # Post-process metadata
            audio_file = MetadataHandler._post_process_metadata(audio_file)

            return audio_file

        except Exception as e:
            raise MetadataError(f"Failed to extract metadata from {file_path}: {e}")

    @staticmethod
    def _extract_flac_metadata(audio_file: AudioFile, flac_file: FLAC) -> AudioFile:
        """Extract metadata from FLAC file (Vorbis comments)."""
        if not flac_file.tags:
            return audio_file

        tags = flac_file.tags

        # Basic fields
        raw_artists = MetadataHandler._get_list_field(tags, ['ARTIST'])

        # Split artists if comma-separated
        audio_file.artists = []
        for artist in raw_artists:
            if ',' in artist:
                # Split and clean up
                split_artists = [a.strip() for a in artist.split(',') if a.strip()]
                audio_file.artists.extend(split_artists)
            else:
                audio_file.artists.append(artist)

        # Remove duplicates while preserving order
        seen = set()
        unique_artists = []
        for artist in audio_file.artists:
            if artist not in seen:
                seen.add(artist)
                unique_artists.append(artist)
        audio_file.artists = unique_artists

        # Check for ALBUMARTIST first (this is the primary artist of the album)
        albumartist = MetadataHandler._get_single_field(tags, ['ALBUMARTIST'])
        if albumartist:
            audio_file.primary_artist = albumartist
        else:
            # If no album artist, determine if this is a collaboration
            # by checking if the first artist is likely the primary
            if audio_file.artists:
                # For now, use the first artist as primary
                # The classifier will determine if it's actually a collaboration
                audio_file.primary_artist = audio_file.artists[0]
        audio_file.album = MetadataHandler._get_single_field(tags, ['ALBUM'])
        audio_file.title = MetadataHandler._get_single_field(tags, ['TITLE'])
        audio_file.genre = MetadataHandler._get_single_field(tags, ['GENRE'])

        # Date fields
        date = MetadataHandler._get_single_field(tags, ['DATE'])
        if date:
            # Try to extract year
            year_match = re.match(r'(\d{4})', date)
            if year_match:
                audio_file.year = int(year_match.group(1))
            audio_file.date = date

        # Track number
        tracknumber = MetadataHandler._get_single_field(tags, ['TRACKNUMBER'])
        if tracknumber:
            # Handle "total" format like "5/12"
            track_match = re.match(r'(\d+)', tracknumber)
            if track_match:
                audio_file.track_number = int(track_match.group(1))

        # Live recording specific fields
        audio_file.location = MetadataHandler._get_single_field(tags, ['LOCATION', 'VENUE'])

        # Check for cover art
        if hasattr(flac_file, 'pictures') and flac_file.pictures:
            audio_file.has_cover_art = True

        # Store all tags for reference
        audio_file.metadata = dict(tags)

        return audio_file

    @staticmethod
    def _extract_id3_metadata(audio_file: AudioFile, id3_file: ID3) -> AudioFile:
        """Extract metadata from MP3 file (ID3 tags)."""
        if not id3_file.tags:
            return audio_file

        tags = id3_file.tags

        # Basic fields
        audio_file.artists = MetadataHandler._get_id3_text(tags, ['TPE1'])  # Artist

        primary_artist_list = MetadataHandler._get_id3_text(tags, ['TPE2'])  # Album Artist
        audio_file.primary_artist = primary_artist_list[0] if primary_artist_list else \
                                   (audio_file.artists[0] if audio_file.artists else None)

        # Single-value fields - get first element
        album_list = MetadataHandler._get_id3_text(tags, ['TALB'])
        audio_file.album = album_list[0] if album_list else None

        title_list = MetadataHandler._get_id3_text(tags, ['TIT2'])
        audio_file.title = title_list[0] if title_list else None

        genre_list = MetadataHandler._get_id3_text(tags, ['TCON'])
        audio_file.genre = genre_list[0] if genre_list else None

        # Date fields
        year = MetadataHandler._get_id3_text(tags, ['TDRC', 'TYER'])
        if year:
            year_match = re.match(r'(\d{4})', str(year[0]) if isinstance(year, list) else str(year))
            if year_match:
                audio_file.year = int(year_match.group(1))

        # Track number
        track = MetadataHandler._get_id3_text(tags, ['TRCK'])
        if track:
            track_match = re.match(r'(\d+)', str(track[0]) if isinstance(track, list) else str(track))
            if track_match:
                audio_file.track_number = int(track_match.group(1))

        # Check for cover art
        if 'APIC:' in tags:
            audio_file.has_cover_art = True

        # Store key metadata
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
    def _extract_mp4_metadata(audio_file: AudioFile, mp4_file: MP4) -> AudioFile:
        """Extract metadata from M4A/MP4 file."""
        if not mp4_file.tags:
            return audio_file

        tags = mp4_file.tags

        # MP4 tags use different keys
        audio_file.artists = MetadataHandler._get_mp4_field(tags, ['\xa9ART'])
        audio_file.primary_artist = MetadataHandler._get_mp4_field(tags, ['aART']) or \
                                   (audio_file.artists[0] if audio_file.artists else None)
        audio_file.album = MetadataHandler._get_mp4_field(tags, ['\xa9alb'])
        audio_file.title = MetadataHandler._get_mp4_field(tags, ['\xa9nam'])
        audio_file.genre = MetadataHandler._get_mp4_field(tags, ['\xa9gen'])

        # Date
        date = MetadataHandler._get_mp4_field(tags, ['\xa9day'])
        if date:
            year_match = re.match(r'(\d{4})', str(date))
            if year_match:
                audio_file.year = int(year_match.group(1))

        # Track number
        track = MetadataHandler._get_mp4_field(tags, ['trkn'])
        if track and isinstance(track, tuple) and len(track) >= 1:
            audio_file.track_number = track[0]

        # Check for cover art
        if 'covr' in tags:
            audio_file.has_cover_art = True

        # Store metadata
        audio_file.metadata = dict(tags)

        return audio_file

    @staticmethod
    def _extract_wave_metadata(audio_file: AudioFile, wave_file) -> AudioFile:
        """Extract metadata from WAV/AIFF file."""
        # These formats often have minimal metadata
        if hasattr(wave_file, 'tags') and wave_file.tags:
            tags = wave_file.tags
            audio_file.title = tags.get('TIT2', [None])[0]
            audio_file.artists = tags.get('TPE1', [])
            audio_file.album = tags.get('TALB', [None])[0]

            if audio_file.artists:
                audio_file.primary_artist = audio_file.artists[0]

        return audio_file

    @staticmethod
    def _extract_ogg_metadata(audio_file: AudioFile, ogg_file) -> AudioFile:
        """Extract metadata from OGG Vorbis/Opus file."""
        if not ogg_file.tags:
            return audio_file

        tags = ogg_file.tags

        # OGG files use Vorbis comments similar to FLAC
        # Basic fields
        raw_artists = MetadataHandler._get_list_field(tags, ['ARTIST'])

        # Split artists if comma-separated
        audio_file.artists = []
        for artist in raw_artists:
            if ',' in artist:
                # Split and clean up
                split_artists = [a.strip() for a in artist.split(',') if a.strip()]
                audio_file.artists.extend(split_artists)
            else:
                audio_file.artists.append(artist)

        # Remove duplicates while preserving order
        seen = set()
        unique_artists = []
        for artist in audio_file.artists:
            if artist not in seen:
                seen.add(artist)
                unique_artists.append(artist)
        audio_file.artists = unique_artists

        # Check for ALBUMARTIST first
        albumartist = MetadataHandler._get_single_field(tags, ['ALBUMARTIST'])
        if albumartist:
            audio_file.primary_artist = albumartist
        else:
            if audio_file.artists:
                audio_file.primary_artist = audio_file.artists[0]

        audio_file.album = MetadataHandler._get_single_field(tags, ['ALBUM'])
        audio_file.title = MetadataHandler._get_single_field(tags, ['TITLE'])
        audio_file.genre = MetadataHandler._get_single_field(tags, ['GENRE'])

        # Date fields
        date = MetadataHandler._get_single_field(tags, ['DATE'])
        if date:
            # Try to extract year
            year_match = re.match(r'(\d{4})', date)
            if year_match:
                audio_file.year = int(year_match.group(1))
            audio_file.date = date

        # Track number
        tracknumber = MetadataHandler._get_single_field(tags, ['TRACKNUMBER'])
        if tracknumber:
            # Handle "total" format like "5/12"
            track_match = re.match(r'(\d+)', tracknumber)
            if track_match:
                audio_file.track_number = int(track_match.group(1))

        # Live recording specific fields
        audio_file.location = MetadataHandler._get_single_field(tags, ['LOCATION', 'VENUE'])

        # Store all tags for reference
        audio_file.metadata = dict(tags)

        return audio_file

    @staticmethod
    def _extract_wma_metadata(audio_file: AudioFile, wma_file) -> AudioFile:
        """Extract metadata from WMA file."""
        if not wma_file.tags:
            return audio_file

        tags = wma_file.tags

        # WMA uses different tag names
        # Basic fields
        audio_file.artists = MetadataHandler._get_wma_field(tags, ['Author', 'Artist'])

        primary_artist_list = MetadataHandler._get_wma_field(tags, ['AlbumArtist'])
        audio_file.primary_artist = primary_artist_list[0] if primary_artist_list else \
                                   (audio_file.artists[0] if audio_file.artists else None)

        # Single-value fields - get first element
        album_list = MetadataHandler._get_wma_field(tags, ['AlbumTitle', 'Album'])
        audio_file.album = album_list[0] if album_list else None

        title_list = MetadataHandler._get_wma_field(tags, ['Title'])
        audio_file.title = title_list[0] if title_list else None

        genre_list = MetadataHandler._get_wma_field(tags, ['Genre'])
        audio_file.genre = genre_list[0] if genre_list else None

        # Date fields
        year = MetadataHandler._get_wma_field(tags, ['Year', 'ReleaseDate'])
        if year:
            year_match = re.match(r'(\d{4})', str(year[0]) if isinstance(year, list) else str(year))
            if year_match:
                audio_file.year = int(year_match.group(1))

        # Track number
        track = MetadataHandler._get_wma_field(tags, ['TrackNumber'])
        if track:
            try:
                audio_file.track_number = int(track[0] if isinstance(track, list) else track)
            except (ValueError, TypeError):
                pass

        # Store metadata
        audio_file.metadata = dict(tags)

        return audio_file

    @staticmethod
    def _extract_ape_metadata(audio_file: AudioFile, ape_file) -> AudioFile:
        """Extract metadata from APE file."""
        if not ape_file.tags:
            return audio_file

        tags = ape_file.tags

        # APE uses Vorbis-like comments with specific tag names
        # Basic fields
        raw_artists = MetadataHandler._get_ape_field(tags, ['Artist', 'Author'])

        # Split artists if comma-separated
        audio_file.artists = []
        for artist in raw_artists:
            if ',' in artist:
                split_artists = [a.strip() for a in artist.split(',') if a.strip()]
                audio_file.artists.extend(split_artists)
            else:
                audio_file.artists.append(artist)

        # Remove duplicates while preserving order
        seen = set()
        unique_artists = []
        for artist in audio_file.artists:
            if artist not in seen:
                seen.add(artist)
                unique_artists.append(artist)
        audio_file.artists = unique_artists

        # Check for Album Artist
        albumartist = MetadataHandler._get_ape_single_field(tags, ['Album Artist', 'AlbumArtist'])
        if albumartist:
            audio_file.primary_artist = albumartist
        else:
            if audio_file.artists:
                audio_file.primary_artist = audio_file.artists[0]

        audio_file.album = MetadataHandler._get_ape_single_field(tags, ['Album'])
        audio_file.title = MetadataHandler._get_ape_single_field(tags, ['Title'])
        audio_file.genre = MetadataHandler._get_ape_single_field(tags, ['Genre'])

        # Date fields
        year = MetadataHandler._get_ape_field(tags, ['Year', 'Release Date'])
        if year:
            year_match = re.match(r'(\d{4})', str(year[0]) if isinstance(year, list) else str(year))
            if year_match:
                audio_file.year = int(year_match.group(1))

        # Track number
        track = MetadataHandler._get_ape_field(tags, ['Track'])
        if track:
            # Handle "total" format like "5/12"
            track_str = track[0] if isinstance(track, list) else str(track)
            track_match = re.match(r'(\d+)', track_str)
            if track_match:
                audio_file.track_number = int(track_match.group(1))

        # Store metadata
        audio_file.metadata = dict(tags)

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
    def _get_wma_field(tags: Dict, keys: List[str]) -> List[str]:
        """Get field from WMA tags."""
        for key in keys:
            if key in tags:
                value = tags[key]
                if isinstance(value, list):
                    return [str(v) for v in value]
                return [str(value)]
        return []

    @staticmethod
    def _get_ape_field(tags: Dict, keys: List[str]) -> List[str]:
        """Get field from APE tags."""
        for key in keys:
            if key in tags:
                value = tags[key]
                if isinstance(value, list):
                    return [str(v) for v in value]
                return [str(value)]
        return []

    @staticmethod
    def _get_ape_single_field(tags: Dict, keys: List[str]) -> Optional[str]:
        """Get a single-value field from APE tags."""
        for key in keys:
            if key in tags:
                value = tags[key]
                if isinstance(value, list) and value:
                    return str(value[0])
                return str(value) if value else None
        return None

    @staticmethod
    def _post_process_metadata(audio_file: AudioFile) -> AudioFile:
        """Clean up and standardize extracted metadata."""
        # Clean up artist names
        if audio_file.artists:
            audio_file.artists = [MetadataHandler._clean_artist_name(a) for a in audio_file.artists if a]
            audio_file.artists = list(set(audio_file.artists))  # Remove duplicates

        if audio_file.primary_artist:
            audio_file.primary_artist = MetadataHandler._clean_artist_name(audio_file.primary_artist)

        # Clean up album and title
        if audio_file.album:
            audio_file.album = MetadataHandler._clean_title(audio_file.album)

        if audio_file.title:
            audio_file.title = MetadataHandler._clean_title(audio_file.title)

        # Standardize genre
        if audio_file.genre:
            audio_file.genre = MetadataHandler._standardize_genre(audio_file.genre)

        return audio_file

    @staticmethod
    def _clean_artist_name(name: str) -> str:
        """Clean up artist name."""
        if not name:
            return ""

        # Remove common prefixes/suffixes
        name = re.sub(r'^(the |The )', '', name)
        name = re.sub(r'\s+$', '', name)

        # Standardize separators
        name = re.sub(r'\s+feat\.\s*', ' feat. ', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+featuring\s+', ' featuring ', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+with\s+', ' with ', name, flags=re.IGNORECASE)

        return name.strip()

    @staticmethod
    def _clean_title(title: str) -> str:
        """Clean up album or track title."""
        if not title:
            return ""

        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title)

        # Clean up common patterns
        title = re.sub(r'\[(\d{4})\]', r'(\1)', title)  # [2001] -> (2001)

        return title.strip()

    @staticmethod
    def _standardize_genre(genre: str) -> str:
        """Standardize genre names."""
        if not genre:
            return ""

        # Common genre mappings
        genre_map = {
            'rock & roll': 'Rock',
            'r&b': 'R&B',
            'hip-hop': 'Hip Hop',
            'electronica': 'Electronic',
            'new age': 'New Age',
        }

        genre_lower = genre.lower()
        if genre_lower in genre_map:
            return genre_map[genre_lower]

        # Capitalize properly
        return ' '.join(word.capitalize() for word in genre.split())

    @staticmethod
    def parse_artists(artist_string: str) -> Tuple[List[str], Optional[str]]:
        """Parse artist string into individual artists and primary artist."""
        if not artist_string:
            return [], None

        # Split on common separators
        separators = [
            r'\s+feat\.\s+',
            r'\s+featuring\s+',
            r'\s+with\s+',
            r'\s+&\s+',
            r'\s+x\s+',
            r'\s+and\s+',
        ]

        artists = [artist_string.strip()]
        primary = artist_string.strip()

        for sep in separators:
            if re.search(sep, artist_string, flags=re.IGNORECASE):
                parts = re.split(sep, artist_string, flags=re.IGNORECASE)
                artists = [p.strip() for p in parts]
                primary = artists[0]
                break

        return artists, primary

    @staticmethod
    def find_cover_art(directory: Path) -> List[Path]:
        """Find cover art files in a directory."""
        if not directory.is_dir():
            return []

        cover_patterns = [
            '*cover*',
            '*front*',
            '*folder*',
            '*album*',
            '*.jpg',
            '*.jpeg',
            '*.png',
        ]

        cover_files = []
        for pattern in cover_patterns:
            for file_path in directory.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    cover_files.append(file_path)

        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in cover_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)

        return unique_files