"""Spotify metadata enhancement plugin."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..base import MetadataPlugin, PluginInfo
from ...models.audio_file import AudioFile
from ...plugins.config import PluginConfigSchema, ConfigOption

logger = logging.getLogger(__name__)


class SpotifyEnhancerPlugin(MetadataPlugin):
    """Plugin to enhance metadata using Spotify database."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="spotify_enhancer",
            version="1.0.0",
            description="Enhances metadata using Spotify database",
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
                description="Enable Spotify metadata enhancement"
            ),
            ConfigOption(
                name="client_id",
                type=str,
                default="",
                description="Spotify client ID (from https://developer.spotify.com/dashboard)"
            ),
            ConfigOption(
                name="client_secret",
                type=str,
                default="",
                description="Spotify client secret"
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
                name="add_audio_features",
                type=bool,
                default=False,
                description="Add Spotify audio features to metadata"
            ),
            ConfigOption(
                name="min_confidence",
                type=float,
                default=0.8,
                min_value=0.0,
                max_value=1.0,
                description="Minimum confidence for metadata enhancement"
            ),
            ConfigOption(
                name="cache_enabled",
                type=bool,
                default=True,
                description="Enable metadata caching"
            ),
        ])

    def initialize(self) -> None:
        """Initialize the plugin."""
        self._adapter: Optional[Any] = None
        self._cache: Dict[str, Any] = {}
        logger.info("Spotify enhancer plugin initialized")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._adapter:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._adapter.close())
            except RuntimeError:
                pass
            self._adapter = None
        self._cache.clear()
        logger.info("Spotify enhancer plugin cleaned up")

    def _get_adapter(self):
        """Get or create the Spotify adapter."""
        if not self.enabled:
            return None

        client_id = self.config.get("client_id")
        client_secret = self.config.get("client_secret")

        if not client_id or not client_secret:
            logger.warning("Spotify credentials not configured")
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
                timeout=self.config.get("timeout", 10.0),
                rate_limit=self.config.get("rate_limit", 10.0)
            )

        return self._adapter

    def _get_cache_key(self, entity_type: str, name: str, album: str = "") -> str:
        """Generate cache key for lookup."""
        key = f"{entity_type}:{name.lower()}"
        if album:
            key += f":{album.lower()}"
        return key

    async def _find_spotify_track(
        self,
        audio_file: AudioFile
    ) -> Optional[Any]:
        """Find matching Spotify track for an audio file."""
        if not audio_file.title or not audio_file.artists:
            return None

        adapter = self._get_adapter()
        if not adapter:
            return None

        artist = audio_file.artists[0]
        title = audio_file.title
        album = audio_file.album or ""

        cache_key = self._get_cache_key("track_match", f"{artist}:{title}", album)

        if self.config.get("cache_enabled", True) and cache_key in self._cache:
            return self._cache[cache_key]

        query = f"track:{title} artist:{artist}"
        if album:
            query += f" album:{album}"

        spotify_tracks = await adapter.search_track(query, limit=10)

        if not spotify_tracks:
            if self.config.get("cache_enabled", True):
                self._cache[cache_key] = None
            return None

        duration_ms = int(audio_file.metadata.get("duration_seconds", 0) * 1000)
        result = adapter.find_best_match(spotify_tracks, title, artist, duration_ms)

        if result and result[1] >= self.config.get("min_confidence", 0.8):
            if self.config.get("cache_enabled", True):
                self._cache[cache_key] = result[0]
            return result[0]

        if self.config.get("cache_enabled", True):
            self._cache[cache_key] = None
        return None

    async def enhance_metadata(self, audio_file: AudioFile) -> AudioFile:
        """Enhance metadata using Spotify.

        Args:
            audio_file: The audio file to enhance

        Returns:
            AudioFile with enhanced metadata
        """
        if not self.enabled:
            return audio_file

        spotify_track = await self._find_spotify_track(audio_file)

        if spotify_track:
            audio_file.metadata["spotify_uri"] = spotify_track.uri
            audio_file.metadata["spotify_id"] = spotify_track.id

            if spotify_track.isrc:
                audio_file.metadata["isrc"] = spotify_track.isrc

            if self.config.get("add_audio_features", False):
                adapter = self._get_adapter()
                if adapter:
                    features = await adapter.get_audio_features(spotify_track.id)
                    if features:
                        audio_file.metadata["spotify_audio_features"] = {
                            "danceability": features.danceability,
                            "energy": features.energy,
                            "valence": features.valence,
                            "tempo": features.tempo,
                            "key": features.key,
                            "mode": features.mode,
                            "acousticness": features.acousticness,
                            "instrumentalness": features.instrumentalness,
                            "liveness": features.liveness,
                            "speechiness": features.speechiness,
                        }

            logger.debug(f"Enhanced metadata with Spotify: {audio_file.title}")

        return audio_file

    async def batch_enhance(self, audio_files: List[AudioFile]) -> List[AudioFile]:
        """Enhance metadata for multiple audio files.

        Args:
            audio_files: List of audio files to enhance

        Returns:
            List of AudioFile objects with enhanced metadata
        """
        enhanced_files = []

        for audio_file in audio_files:
            if self.enabled:
                enhanced = await self.enhance_metadata(audio_file)
                enhanced_files.append(enhanced)
            else:
                enhanced_files.append(audio_file)

        return enhanced_files
