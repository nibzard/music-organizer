"""Spotify playlist export plugin."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import OutputPlugin, PluginInfo
from ...models.audio_file import AudioFile
from ...plugins.config import PluginConfigSchema, ConfigOption

logger = logging.getLogger(__name__)


class SpotifyPlaylistExportPlugin(OutputPlugin):
    """Plugin to export local playlists to Spotify."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="spotify_playlist_export",
            version="1.0.0",
            description="Export local playlists to Spotify",
            author="Music Organizer Team",
            dependencies=["aiohttp"],
        )

    def get_config_schema(self) -> PluginConfigSchema:
        """Get configuration schema for this plugin."""
        return PluginConfigSchema([
            ConfigOption(
                name="enabled",
                type=bool,
                default=True,
                description="Enable Spotify playlist export"
            ),
            ConfigOption(
                name="client_id",
                type=str,
                default="",
                description="Spotify client ID"
            ),
            ConfigOption(
                name="client_secret",
                type=str,
                default="",
                description="Spotify client secret"
            ),
            ConfigOption(
                name="access_token",
                type=str,
                default="",
                description="Spotify access token (required for playlist creation)"
            ),
            ConfigOption(
                name="refresh_token",
                type=str,
                default="",
                description="Spotify refresh token"
            ),
            ConfigOption(
                name="user_id",
                type=str,
                default="",
                description="Spotify user ID for playlist creation"
            ),
            ConfigOption(
                name="timeout",
                type=float,
                default=10.0,
                min_value=1.0,
                max_value=60.0,
                description="Request timeout in seconds"
            ),
            ConfigOption(
                name="rate_limit",
                type=float,
                default=10.0,
                min_value=1.0,
                max_value=50.0,
                description="Maximum requests per second"
            ),
            ConfigOption(
                name="min_confidence",
                type=float,
                default=0.7,
                min_value=0.0,
                max_value=1.0,
                description="Minimum confidence for track matching"
            ),
            ConfigOption(
                name="playlist_name",
                type=str,
                default="",
                description="Default playlist name (if not in source)"
            ),
            ConfigOption(
                name="public",
                type=bool,
                default=False,
                description="Make playlist public on Spotify"
            ),
        ])

    def initialize(self) -> None:
        """Initialize the plugin."""
        self._adapter: Optional[Any] = None
        self._match_cache: Dict[str, Optional[str]] = {}
        logger.info("Spotify playlist export plugin initialized")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._adapter:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._adapter.close())
            except RuntimeError:
                pass
            self._adapter = None
        self._match_cache.clear()
        logger.info("Spotify playlist export plugin cleaned up")

    def _get_adapter(self):
        """Get or create the Spotify adapter."""
        if not self.enabled:
            return None

        client_id = self.config.get("client_id")
        client_secret = self.config.get("client_secret")

        if not client_id or not client_secret:
            logger.warning("Spotify credentials not configured")
            return None

        access_token = self.config.get("access_token")
        if not access_token:
            logger.warning("Spotify access token required for playlist export")
            return None

        try:
            from ...infrastructure.external.spotify_adapter import SpotifyAdapter
        except ImportError:
            logger.warning("Spotify adapter not available")
            return None

        if self._adapter is None:
            self._adapter = SpotifyAdapter(
                client_id=client_id,
                client_secret=client_secret,
                access_token=access_token,
                refresh_token=self.config.get("refresh_token"),
                timeout=self.config.get("timeout", 10.0),
                rate_limit=self.config.get("rate_limit", 10.0)
            )

        return self._adapter

    def get_supported_formats(self) -> List[str]:
        """Return list of supported export formats."""
        return ["spotify", "spotify-uri", "spotify-id"]

    def get_file_extension(self) -> str:
        """Return the file extension for this export format."""
        return ".txt"

    async def export(self, audio_files: List[AudioFile], output_path: Path) -> bool:
        """Export local playlist to Spotify.

        Args:
            audio_files: List of local audio files to export
            output_path: Output path (can be file for URIs or special value "spotify:")

        Returns:
            True if export was successful
        """
        adapter = self._get_adapter()
        if not adapter:
            logger.error("Cannot get Spotify adapter (access token required)")
            return False

        output_str = str(output_path)

        if output_str == "spotify:" or output_str.startswith("spotify:new:"):
            return await self._create_spotify_playlist(
                adapter,
                audio_files,
                output_path
            )

        return self._write_spotify_uris(audio_files, output_path)

    async def _create_spotify_playlist(
        self,
        adapter: Any,
        audio_files: List[AudioFile],
        output_path: Path
    ) -> bool:
        """Create a new Spotify playlist from local files."""
        user_id = self.config.get("user_id")
        if not user_id:
            logger.error("Spotify user_id required for playlist creation")
            return False

        playlist_name = self.config.get("playlist_name", "Imported Playlist")
        if ":" in str(output_path):
            parts = str(output_path).split(":")
            if len(parts) > 2 and parts[2]:
                playlist_name = parts[2]

        matched_uris = await self._match_tracks_to_spotify(adapter, audio_files)

        if not matched_uris:
            logger.warning("No tracks matched to Spotify")
            return False

        spotify_playlist = await adapter.create_playlist(
            user_id=user_id,
            name=playlist_name,
            description=f"Created by Music Organizer with {len(matched_uris)} tracks",
            public=self.config.get("public", False)
        )

        if not spotify_playlist:
            logger.error("Failed to create Spotify playlist")
            return False

        success = await adapter.add_to_playlist(spotify_playlist.id, matched_uris)

        if success:
            logger.info(
                f"Created Spotify playlist '{playlist_name}' "
                f"({spotify_playlist.uri}) with {len(matched_uris)} tracks"
            )
        else:
            logger.error("Failed to add tracks to Spotify playlist")

        return success

    async def _match_tracks_to_spotify(
        self,
        adapter: Any,
        audio_files: List[AudioFile]
    ) -> List[str]:
        """Match local audio files to Spotify tracks."""
        min_confidence = self.config.get("min_confidence", 0.7)
        matched_uris = []

        for audio_file in audio_files:
            cache_key = f"{audio_file.path}:{audio_file.title}:{audio_file.album}"

            if cache_key in self._match_cache:
                uri = self._match_cache[cache_key]
                if uri:
                    matched_uris.append(uri)
                continue

            if not audio_file.title or not audio_file.artists:
                continue

            artist = audio_file.artists[0]
            title = audio_file.title
            album = audio_file.album or ""

            query = f"track:{title} artist:{artist}"
            if album:
                query += f" album:{album}"

            spotify_tracks = await adapter.search_track(query, limit=10)

            if not spotify_tracks:
                self._match_cache[cache_key] = None
                continue

            duration_ms = int(audio_file.metadata.get("duration_seconds", 0) * 1000)
            result = adapter.find_best_match(spotify_tracks, title, artist, duration_ms)

            if result and result[1] >= min_confidence:
                matched_uris.append(result[0].uri)
                self._match_cache[cache_key] = result[0].uri
                logger.debug(f"Matched: {title} -> {result[0].uri}")
            else:
                self._match_cache[cache_key] = None
                logger.debug(f"No match found: {title}")

        return matched_uris

    def _write_spotify_uris(
        self,
        audio_files: List[AudioFile],
        output_path: Path
    ) -> bool:
        """Write Spotify URIs to file (for matched tracks)."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Spotify URIs for matched tracks\n")
                f.write("# Use with Spotify API tools or for reference\n\n")

                for audio_file in audio_files:
                    uri = audio_file.metadata.get("spotify_uri")
                    if uri:
                        title = audio_file.title or "Unknown"
                        artist = audio_file.artists[0] if audio_file.artists else "Unknown"
                        f.write(f"{uri} # {artist} - {title}\n")

            matched_count = sum(
                1 for af in audio_files
                if af.metadata.get("spotify_uri")
            )
            logger.info(f"Wrote {matched_count} Spotify URIs to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error writing Spotify URIs: {e}")
            return False

    async def get_spotify_uris(
        self,
        audio_files: List[AudioFile]
    ) -> List[str]:
        """Get Spotify URIs for audio files, matching if needed.

        Args:
            audio_files: List of audio files

        Returns:
            List of Spotify URIs (None for unmatched tracks)
        """
        adapter = self._get_adapter()
        if not adapter:
            return []

        uris = []
        for audio_file in audio_files:
            existing_uri = audio_file.metadata.get("spotify_uri")
            if existing_uri:
                uris.append(existing_uri)
                continue

            if not audio_file.title or not audio_file.artists:
                uris.append(None)
                continue

            artist = audio_file.artists[0]
            title = audio_file.title
            query = f"track:{title} artist:{artist}"

            spotify_tracks = await adapter.search_track(query, limit=5)
            if spotify_tracks:
                uris.append(spotify_tracks[0].uri)
            else:
                uris.append(None)

        return uris
