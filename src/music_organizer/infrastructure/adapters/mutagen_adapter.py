"""
Mutagen Adapter - Anti-Corruption Layer for mutagen library.

This adapter isolates the domain from the mutagen library dependency,
providing a clean interface for metadata operations.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from mutagen import File as MutagenFile
    from mutagen.id3 import ID3NoHeaderError
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

from ...domain.catalog import Metadata, ArtistName, TrackNumber, AudioPath, FileFormat


class MutagenMetadataAdapter:
    """
    Adapter for reading/writing metadata using mutagen library.

    This adapter protects the domain from mutagen-specific details and
    provides a clean abstraction layer.
    """

    def __init__(self):
        if not MUTAGEN_AVAILABLE:
            raise ImportError("mutagen library is required for MutagenMetadataAdapter")

    def read_metadata(self, file_path: Path) -> Optional[Metadata]:
        """
        Read metadata from an audio file using mutagen.

        Returns None if the file cannot be read or has no metadata.
        """
        try:
            audio_file = MutagenFile(file_path)
            if audio_file is None:
                return None

            # Extract metadata using mutagen's abstract interface
            metadata_dict = {}

            # Title
            if hasattr(audio_file, 'get'):
                title = audio_file.get('TITLE', audio_file.get('\xa9nam', []))
                if title:
                    metadata_dict['title'] = str(title[0])

                # Artists
                artists = audio_file.get('ARTIST', audio_file.get('\xa9ART', []))
                if artists:
                    metadata_dict['artists'] = [str(a) for a in artists]

                # Album
                album = audio_file.get('ALBUM', audio_file.get('\xa9alb', []))
                if album:
                    metadata_dict['album'] = str(album[0])

                # Year
                date = audio_file.get('DATE', audio_file.get('\xa9day', []))
                if date:
                    # Extract year from date string
                    date_str = str(date[0])
                    year = self._extract_year(date_str)
                    if year:
                        metadata_dict['year'] = year

                # Genre
                genre = audio_file.get('GENRE', audio_file.get('\xa9gen', []))
                if genre:
                    metadata_dict['genre'] = str(genre[0])

                # Track number
                track = audio_file.get('TRACKNUMBER', audio_file.get('trkn', []))
                if track:
                    track_num = self._parse_track_number(str(track[0]))
                    if track_num:
                        metadata_dict['track_number'] = TrackNumber(track_num)

                # Total tracks
                if track and isinstance(track[0], tuple) and len(track[0]) > 1:
                    metadata_dict['total_tracks'] = track[0][1]

                # Disc number
                disc = audio_file.get('DISCNUMBER', audio_file.get('disk', []))
                if disc and isinstance(disc[0], tuple):
                    metadata_dict['disc_number'] = disc[0][0]
                    metadata_dict['total_discs'] = disc[0][1]

                # Album artist
                albumartist = audio_file.get('ALBUMARTIST', audio_file.get('aART', []))
                if albumartist:
                    metadata_dict['albumartist'] = ArtistName(str(albumartist[0]))

                # Composer
                composer = audio_file.get('COMPOSER', audio_file.get('\xa9wrt', []))
                if composer:
                    metadata_dict['composer'] = str(composer[0])

                # Technical metadata
                if hasattr(audio_file, 'info'):
                    info = audio_file.info
                    metadata_dict['duration_seconds'] = getattr(info, 'length', 0)
                    metadata_dict['bitrate'] = getattr(info, 'bitrate', 0)
                    metadata_dict['sample_rate'] = getattr(info, 'sample_rate', 0)
                    metadata_dict['channels'] = getattr(info, 'channels', 0)

            # Create domain Metadata object
            return self._dict_to_metadata(metadata_dict, file_path)

        except (ID3NoHeaderError, Exception) as e:
            # Log error but don't crash
            print(f"Error reading metadata from {file_path}: {e}")
            return None

    def write_metadata(self, file_path: Path, metadata: Metadata) -> bool:
        """
        Write metadata to an audio file using mutagen.

        Returns True if successful, False otherwise.
        """
        try:
            audio_file = MutagenFile(file_path)
            if audio_file is None:
                return False

            # Convert domain metadata to mutagen format
            mutagen_tags = self._metadata_to_mutagen_tags(metadata)

            # Apply tags
            for tag, value in mutagen_tags.items():
                audio_file[tag] = value

            # Save to file
            audio_file.save()
            return True

        except Exception as e:
            print(f"Error writing metadata to {file_path}: {e}")
            return False

    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate MD5 hash of file content.
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def generate_fingerprint(self, file_path: Path) -> str:
        """
        Generate a simple fingerprint based on file properties.

        This is a lightweight fingerprinting approach. For more accurate
        acoustic fingerprinting, integrate with libraries like chromaprint.
        """
        try:
            audio_file = MutagenFile(file_path)
            if audio_file is None or not hasattr(audio_file, 'info'):
                # Fallback to file properties
                stat = file_path.stat()
                content = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
                return hashlib.sha256(content.encode()).hexdigest()

            # Use audio properties for fingerprint
            info = audio_file.info
            fingerprint_data = {
                "duration": getattr(info, 'length', 0),
                "bitrate": getattr(info, 'bitrate', 0),
                "sample_rate": getattr(info, 'sample_rate', 0),
                "channels": getattr(info, 'channels', 0),
                "file_size": file_path.stat().st_size,
            }

            # Include metadata fingerprint
            metadata = self.read_metadata(file_path)
            if metadata:
                metadata_str = f"{metadata.title}_{metadata.primary_artist}_{metadata.album}"
                fingerprint_data["metadata"] = metadata_str

            # Create hash from fingerprint data
            content = str(fingerprint_data).encode()
            return hashlib.sha256(content).hexdigest()

        except Exception as e:
            print(f"Error generating fingerprint for {file_path}: {e}")
            # Fallback to file hash
            return self.calculate_file_hash(file_path)

    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string."""
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return int(year_match.group())
        return None

    def _parse_track_number(self, track_str: str) -> Optional[str]:
        """Parse track number from various formats."""
        import re
        # Extract track number from formats like "5/12", "5", "05"
        track_match = re.match(r'^(\d+)', track_str.strip())
        if track_match:
            return track_match.group(1)
        return None

    def _dict_to_metadata(self, metadata_dict: Dict[str, Any], file_path: Path) -> Metadata:
        """Convert metadata dictionary to domain Metadata object."""
        # Convert artists
        artists = []
        if 'artists' in metadata_dict:
            artists = [ArtistName(a) for a in metadata_dict['artists']]

        # Convert album artist
        albumartist = None
        if 'albumartist' in metadata_dict:
            albumartist = ArtistName(metadata_dict['albumartist'])

        return Metadata(
            title=metadata_dict.get('title'),
            artists=artists,
            album=metadata_dict.get('album'),
            year=metadata_dict.get('year'),
            genre=metadata_dict.get('genre'),
            track_number=metadata_dict.get('track_number'),
            total_tracks=metadata_dict.get('total_tracks'),
            disc_number=metadata_dict.get('disc_number'),
            total_discs=metadata_dict.get('total_discs'),
            albumartist=albumartist,
            composer=metadata_dict.get('composer'),
            duration_seconds=metadata_dict.get('duration_seconds'),
            bitrate=metadata_dict.get('bitrate'),
            sample_rate=metadata_dict.get('sample_rate'),
            channels=metadata_dict.get('channels'),
            file_hash=self.calculate_file_hash(file_path),
            acoustic_fingerprint=self.generate_fingerprint(file_path),
        )

    def _metadata_to_mutagen_tags(self, metadata: Metadata) -> Dict[str, Any]:
        """Convert domain Metadata to mutagen tag format."""
        tags = {}

        if metadata.title:
            tags['TITLE'] = metadata.title

        if metadata.artists:
            tags['ARTIST'] = [str(a) for a in metadata.artists]

        if metadata.album:
            tags['ALBUM'] = metadata.album

        if metadata.year:
            tags['DATE'] = str(metadata.year)

        if metadata.genre:
            tags['GENRE'] = metadata.genre

        if metadata.track_number:
            if metadata.total_tracks:
                tags['TRACKNUMBER'] = f"{metadata.track_number.number}/{metadata.total_tracks}"
            else:
                tags['TRACKNUMBER'] = str(metadata.track_number.number)

        if metadata.disc_number:
            if metadata.total_discs:
                tags['DISCNUMBER'] = f"{metadata.disc_number}/{metadata.total_discs}"
            else:
                tags['DISCNUMBER'] = str(metadata.disc_number)

        if metadata.albumartist:
            tags['ALBUMARTIST'] = str(metadata.albumartist)

        if metadata.composer:
            tags['COMPOSER'] = metadata.composer

        return tags