"""
Kodi/Jellyfin NFO export plugin.

Generates NFO metadata files for Kodi, Jellyfin, and other media centers.
The plugin hooks into the organization process and generates NFO files
alongside organized music files.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..base import OutputPlugin, PluginInfo
from ...models.audio_file import AudioFile
from ...infrastructure.nfo.nfo_generator import (
    NfoGenerator,
    NfoConfig,
    ArtistInfo,
    AlbumInfo
)
from ...infrastructure.external.musicbrainz_mbid import MusicBrainzMbidAdapter

logger = logging.getLogger(__name__)


class KodiNfoExporterPlugin(OutputPlugin):
    """Plugin to generate Kodi/Jellyfin NFO files during organization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin.

        Args:
            config: Plugin configuration dictionary
        """
        super().__init__(config)
        self.nfo_generator: Optional[NfoGenerator] = None
        self.mbid_adapter: Optional[MusicBrainzMbidAdapter] = None
        self._artist_cache: Dict[str, ArtistInfo] = {}
        self._album_cache: Dict[str, AlbumInfo] = {}
        self._generated_nfos: set = set()

    @property
    def info(self) -> PluginInfo:
        """Return plugin information."""
        return PluginInfo(
            name="kodi_nfo_exporter",
            version="1.0.0",
            description="Generates NFO files for Kodi/Jellyfin compatibility",
            author="Music Organizer Team",
            dependencies=[],
        )

    def initialize(self) -> None:
        """Initialize the plugin."""
        config = NfoConfig(
            include_bios=self.config.get("include_bios", True),
            include_reviews=self.config.get("include_reviews", True),
            include_ratings=self.config.get("include_ratings", True),
            include_moods=self.config.get("include_moods", True),
            indent_xml=self.config.get("indent_xml", True)
        )
        self.nfo_generator = NfoGenerator(config)
        self.mbid_adapter = None
        self._artist_cache = {}
        self._album_cache = {}
        self._generated_nfos = set()
        logger.info("Initialized Kodi NFO exporter plugin")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.mbid_adapter:
            self.mbid_adapter.clear_cache()
        self._artist_cache.clear()
        self._album_cache.clear()
        self._generated_nfos.clear()
        logger.info("Cleaned up Kodi NFO exporter plugin")

    async def export(self, audio_files: List[AudioFile], output_path: Path) -> bool:
        """Export NFO files for organized audio files.

        This method generates NFO files in the appropriate directories
        based on the audio files' locations.

        Args:
            audio_files: List of organized audio files
            output_path: Base output path (not directly used for NFO generation)

        Returns:
            True if export was successful
        """
        try:
            if not audio_files:
                logger.warning("No audio files to export NFOs for")
                return True

            generated_count = 0
            for audio_file in audio_files:
                if not audio_file.path.exists():
                    continue

                # Generate album NFO
                if self.config.get("generate_album_nfo", True):
                    if await self._generate_album_nfo(audio_file):
                        generated_count += 1

                # Generate artist NFO (if using Kodi structure)
                if self.config.get("generate_artist_nfo", False):
                    if await self._generate_artist_nfo(audio_file):
                        generated_count += 1

            logger.info(f"Generated {generated_count} NFO files")
            return True

        except Exception as e:
            logger.error(f"Failed to export NFO files: {e}")
            return False

    async def _generate_album_nfo(self, audio_file: AudioFile) -> bool:
        """Generate album.nfo for an audio file's album.

        Args:
            audio_file: The audio file

        Returns:
            True if NFO was generated
        """
        if not audio_file.album or not audio_file.artists:
            return False

        target_dir = audio_file.path.parent
        nfo_path = target_dir / "album.nfo"

        # Skip if already generated
        nfo_key = str(nfo_path)
        if nfo_key in self._generated_nfos:
            return False
        if nfo_path.exists():
            self._generated_nfos.add(nfo_key)
            return False

        artist = audio_file.artists[0]
        album_key = f"album:{artist}:{audio_file.album}"

        # Get or fetch album metadata
        if album_key not in self._album_cache:
            metadata = await self._get_album_metadata(audio_file.album, artist)
            album_info = AlbumInfo(
                title=audio_file.album,
                artist=artist,
                year=audio_file.year,
                genre=audio_file.genre,
                musicbrainz_release_id=metadata.get("musicbrainz_release_id"),
                musicbrainz_releasegroup_id=metadata.get("musicbrainz_releasegroup_id"),
                tracks=metadata.get("tracks", [])
            )
            self._album_cache[album_key] = album_info
        else:
            album_info = self._album_cache[album_key]

        # Write NFO file
        if self.nfo_generator:
            self.nfo_generator.write_album_nfo(nfo_path, album_info)
            self._generated_nfos.add(nfo_key)
            logger.debug(f"Generated album.nfo for {audio_file.album}")
            return True

        return False

    async def _generate_artist_nfo(self, audio_file: AudioFile) -> bool:
        """Generate artist.nfo for an audio file's artist.

        Args:
            audio_file: The audio file

        Returns:
            True if NFO was generated
        """
        if not audio_file.artists:
            return False

        artist = audio_file.artists[0]
        target_dir = audio_file.path.parent

        # Try to find artist directory (parent of album dir)
        # Standard structure: Albums/Artist/Album (Year)/
        # So artist dir would be target_dir.parent
        if target_dir.parent.name != "Albums":
            artist_dir = target_dir.parent
        else:
            artist_dir = target_dir.parent.parent

        nfo_path = artist_dir / "artist.nfo"

        # Skip if already generated or exists
        nfo_key = str(nfo_path)
        if nfo_key in self._generated_nfos or nfo_path.exists():
            self._generated_nfos.add(nfo_key)
            return False

        # Get or fetch artist metadata
        artist_key = f"artist:{artist}"
        if artist_key not in self._artist_cache:
            metadata = await self._get_artist_metadata(artist)
            artist_info = ArtistInfo(
                name=artist,
                musicbrainz_artist_id=metadata.get("musicbrainz_artist_id"),
                genre=metadata.get("genre") or audio_file.genre,
                formed=metadata.get("formed"),
                disbanded=metadata.get("disbanded"),
                biography=metadata.get("biography"),
                albums=metadata.get("albums", [])
            )
            self._artist_cache[artist_key] = artist_info
        else:
            artist_info = self._artist_cache[artist_key]

        # Write NFO file
        if self.nfo_generator:
            self.nfo_generator.write_artist_nfo(nfo_path, artist_info)
            self._generated_nfos.add(nfo_key)
            logger.debug(f"Generated artist.nfo for {artist}")
            return True

        return False

    async def _get_artist_metadata(self, artist: str) -> Dict[str, Any]:
        """Get metadata for artist NFO.

        Args:
            artist: Artist name

        Returns:
            Metadata dictionary
        """
        metadata = {
            "genre": None,
            "formed": None,
            "disbanded": None,
            "biography": None,
            "albums": []
        }

        if self.config.get("fetch_mbid", False):
            if self.mbid_adapter is None:
                self.mbid_adapter = MusicBrainzMbidAdapter()

            mb_data = await self.mbid_adapter.get_artist_with_mbid(artist)
            if mb_data:
                metadata.update({
                    "musicbrainz_artist_id": mb_data.get("musicbrainz_artist_id"),
                    "type": mb_data.get("type"),
                    "country": mb_data.get("country"),
                    "formed": mb_data.get("formed"),
                    "disbanded": mb_data.get("disbanded"),
                    "genre": mb_data.get("genre")
                })

                # Fetch albums if requested
                if self.config.get("include_artist_albums", False):
                    albums = await self.mbid_adapter.get_artist_albums(artist)
                    metadata["albums"] = albums

        return metadata

    async def _get_album_metadata(self, album: str, artist: str) -> Dict[str, Any]:
        """Get metadata for album NFO.

        Args:
            album: Album title
            artist: Artist name

        Returns:
            Metadata dictionary
        """
        metadata = {
            "musicbrainz_release_id": None,
            "musicbrainz_releasegroup_id": None,
            "tracks": []
        }

        if self.config.get("fetch_mbid", False):
            if self.mbid_adapter is None:
                self.mbid_adapter = MusicBrainzMbidAdapter()

            mb_data = await self.mbid_adapter.find_release_by_album_artist(
                album, artist
            )
            if mb_data:
                metadata.update({
                    "musicbrainz_release_id": mb_data.get("musicbrainz_release_id"),
                    "musicbrainz_releasegroup_id": mb_data.get("musicbrainz_releasegroup_id"),
                    "tracks": mb_data.get("tracks", [])
                })

        return metadata

    def get_supported_formats(self) -> List[str]:
        """Return supported export formats."""
        return ['nfo']

    def get_file_extension(self) -> str:
        """Return file extension for NFO format."""
        return 'nfo'


def create_plugin(config: Optional[Dict[str, Any]] = None) -> KodiNfoExporterPlugin:
    """Create an instance of the plugin.

    Args:
        config: Plugin configuration dictionary

    Returns:
        Plugin instance
    """
    return KodiNfoExporterPlugin(config)


__all__ = ["create_plugin", "KodiNfoExporterPlugin"]
