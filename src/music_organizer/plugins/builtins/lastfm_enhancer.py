"""Last.fm metadata enhancement plugin."""

import asyncio
import logging
from typing import Dict, Any, Optional, List

from ..base import MetadataPlugin, PluginInfo
from ...models.audio_file import AudioFile
from ...plugins.config import PluginConfigSchema, ConfigOption

logger = logging.getLogger(__name__)


class LastFmEnhancerPlugin(MetadataPlugin):
    """Plugin to enhance metadata using Last.fm database."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="lastfm_enhancer",
            version="1.0.0",
            description="Enhances metadata using Last.fm database",
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
                description="Enable Last.fm metadata enhancement"
            ),
            ConfigOption(
                name="api_key",
                type=str,
                default="",
                description="Last.fm API key (get from https://www.last.fm/api/account/create)"
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
                default=2.0,
                min_value=0.5,
                max_value=10.0,
                description="Maximum requests per second"
            ),
            ConfigOption(
                name="enhance_genres",
                type=bool,
                default=True,
                description="Enhance genre tags from Last.fm"
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
        self._cache: Dict[str, Dict[str, Any]] = {}
        logger.info("Last.fm enhancer plugin initialized")

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
        logger.info("Last.fm enhancer plugin cleaned up")

    def _get_adapter(self):
        """Get or create the Last.fm adapter."""
        if not self.enabled:
            return None

        api_key = self.config.get("api_key")
        if not api_key:
            logger.warning("Last.fm API key not configured")
            return None

        try:
            from ...infrastructure.external.lastfm_adapter import LastFmAdapter
        except ImportError:
            logger.warning("Last.fm adapter not available")
            return None

        if self._adapter is None:
            self._adapter = LastFmAdapter(
                api_key=api_key,
                api_secret="",  # Not needed for public API calls
                timeout=self.config.get("timeout", 10.0),
                rate_limit=self.config.get("rate_limit", 2.0)
            )

        return self._adapter

    def _get_cache_key(self, entity_type: str, name: str) -> str:
        """Generate cache key for lookup."""
        return f"{entity_type}:{name.lower()}"

    async def enhance_metadata(self, audio_file: AudioFile) -> AudioFile:
        """Enhance metadata using Last.fm.

        Args:
            audio_file: The audio file to enhance

        Returns:
            AudioFile with enhanced metadata
        """
        if not self.enabled or not audio_file.artists:
            return audio_file

        adapter = self._get_adapter()
        if not adapter:
            return audio_file

        artist = audio_file.artists[0]

        # Check if we need to enhance genre
        if (self.config.get("enhance_genres", True) and
            not audio_file.genre):
            cache_key = self._get_cache_key("artist", artist)

            # Check cache
            if self.config.get("cache_enabled", True) and cache_key in self._cache:
                artist_info = self._cache[cache_key]
            else:
                # Fetch from Last.fm
                artist_info = await adapter.get_artist_info(artist)
                if artist_info and self.config.get("cache_enabled", True):
                    self._cache[cache_key] = artist_info

            # Apply genre from tags
            if artist_info and artist_info.get("tags"):
                genres = artist_info["tags"]
                if genres:
                    audio_file.genre = genres[0]
                    logger.debug(f"Enhanced genre for {artist}: {genres[0]}")

        # Enhance track info if we have artist and title
        if audio_file.title:
            cache_key = self._get_cache_key("track", f"{artist}:{audio_file.title}")

            if self.config.get("cache_enabled", True) and cache_key in self._cache:
                track_info = self._cache[cache_key]
            else:
                track_info = await adapter.get_track_info(artist, audio_file.title)
                if track_info and self.config.get("cache_enabled", True):
                    self._cache[cache_key] = track_info

            # Apply duration if available and not in metadata
            if track_info:
                duration = track_info.get("duration")
                if duration and not audio_file.metadata.get("duration_seconds"):
                    audio_file.metadata["duration_seconds"] = duration

                # Apply album if missing
                album = track_info.get("album")
                if album and not audio_file.album:
                    audio_file.album = album

        return audio_file

    async def batch_enhance(self, audio_files: List[AudioFile]) -> List[AudioFile]:
        """Enhance metadata for multiple audio files.

        Args:
            audio_files: List of audio files to enhance

        Returns:
            List of AudioFile objects with enhanced metadata
        """
        enhanced_files = []

        # First pass: collect unique artists for batch lookup
        unique_artists = set()
        for audio_file in audio_files:
            if audio_file.artists:
                unique_artists.add(audio_file.artists[0])

        # Pre-fetch artist info
        adapter = self._get_adapter()
        if adapter and self.config.get("enhance_genres", True):
            for artist in unique_artists:
                cache_key = self._get_cache_key("artist", artist)
                if cache_key not in self._cache:
                    artist_info = await adapter.get_artist_info(artist)
                    if artist_info:
                        self._cache[cache_key] = artist_info

        # Second pass: enhance each file
        for audio_file in audio_files:
            if self.enabled:
                enhanced = await self.enhance_metadata(audio_file)
                enhanced_files.append(enhanced)
            else:
                enhanced_files.append(audio_file)

        return enhanced_files
