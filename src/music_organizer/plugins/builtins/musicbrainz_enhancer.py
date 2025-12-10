"""MusicBrainz metadata enhancement plugin."""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import quote
import xml.etree.ElementTree as ET

from ..base import MetadataPlugin, PluginInfo
from ...models.audio_file import AudioFile
from ...plugins.config import PluginConfigSchema, ConfigOption

logger = logging.getLogger(__name__)


class MusicBrainzEnhancerPlugin(MetadataPlugin):
    """Plugin to enhance metadata using MusicBrainz database."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="musicbrainz_enhancer",
            version="1.0.0",
            description="Enhances metadata using MusicBrainz database",
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
                description="Enable MusicBrainz metadata enhancement"
            ),
            ConfigOption(
                name="api_url",
                type=str,
                default="https://musicbrainz.org/ws/2",
                description="MusicBrainz API base URL"
            ),
            ConfigOption(
                name="user_agent",
                type=str,
                default="music-organizer/1.0.0 (https://github.com/nibzard/music-organizer)",
                description="User agent for API requests"
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
                default=1.0,
                min_value=0.5,
                max_value=5.0,
                description="Minimum delay between requests in seconds"
            ),
            ConfigOption(
                name="enhance_fields",
                type=list,
                default=["year", "genre", "track_number", "albumartist"],
                description="List of fields to enhance"
            ),
            ConfigOption(
                name="fallback_to_fuzzy",
                type=bool,
                default=True,
                description="Use fuzzy search when exact match fails"
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
        self._session = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_request_time = 0.0
        logger.info("MusicBrainz enhancer plugin initialized")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._session:
            asyncio.create_task(self._session.close())
            self._session = None
        self._cache.clear()
        logger.info("MusicBrainz enhancer plugin cleaned up")

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            try:
                import aiohttp
            except ImportError:
                logger.warning("aiohttp not available, MusicBrainz enhancement disabled")
                return None

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.get("timeout", 10.0)),
                headers={"User-Agent": self.config.get("user_agent", "")}
            )
        return self._session

    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        import time
        rate_limit = self.config.get("rate_limit", 1.0)
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < rate_limit:
            await asyncio.sleep(rate_limit - time_since_last)

        self._last_request_time = time.time()

    def _get_cache_key(self, artist: str, title: str, album: Optional[str] = None) -> str:
        """Generate cache key for lookup."""
        key = f"{artist.lower()}|{title.lower()}"
        if album:
            key += f"|{album.lower()}"
        return key

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata from cache."""
        if not self.config.get("cache_enabled", True):
            return None
        return self._cache.get(cache_key)

    def _add_to_cache(self, cache_key: str, metadata: Dict[str, Any]) -> None:
        """Add metadata to cache."""
        if self.config.get("cache_enabled", True):
            self._cache[cache_key] = metadata

    async def _search_musicbrainz(self, query: str, entity_type: str = "recording") -> Optional[ET.Element]:
        """Search MusicBrainz for a given query."""
        session = await self._get_session()
        if not session:
            return None

        await self._rate_limit()

        try:
            # Build search URL
            search_query = quote(query)
            url = f"{self.config.get('api_url')}/{entity_type}/?query={search_query}&fmt=json"

            # Make request
            async with session.get(url) as response:
                if response.status == 200:
                    import json
                    data = await response.json()
                    return data
                else:
                    logger.warning(f"MusicBrainz API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error searching MusicBrainz: {e}")
            return None

    async def _lookup_release(self, release_id: str) -> Optional[Dict[str, Any]]:
        """Lookup detailed release information."""
        session = await self._get_session()
        if not session:
            return None

        await self._rate_limit()

        try:
            url = f"{self.config.get('api_url')}/release/{release_id}?inc=artist-credits+recordings&fmt=json"

            async with session.get(url) as response:
                if response.status == 200:
                    import json
                    data = await response.json()
                    return data
                else:
                    logger.warning(f"MusicBrainz release lookup error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error looking up release: {e}")
            return None

    async def _extract_track_metadata(self, recording_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from recording data."""
        metadata = {}

        # Extract year from release
        if "releases" in recording_data and recording_data["releases"]:
            release = recording_data["releases"][0]
            if "date" in release:
                # Parse date to get year
                date_str = release["date"]
                try:
                    year = int(date_str.split("-")[0])
                    metadata["year"] = year
                except (ValueError, IndexError):
                    pass

            # Get album artist if available
            if "artist-credit" in release:
                artists = []
                for credit in release["artist-credit"]:
                    if "artist" in credit and "name" in credit["artist"]:
                        artists.append(credit["artist"]["name"])
                if artists:
                    metadata["albumartist"] = artists[0]

            # Get track number
            if "media" in release:
                for medium in release["media"]:
                    if "tracks" in medium:
                        for track in medium["tracks"]:
                            if track.get("recording", {}).get("id") == recording_data.get("id"):
                                # Use the track's position if available, otherwise fall back to enumeration
                                track_num = track.get("number")
                                if track_num:
                                    # Track numbers can be strings like "A1", "3", etc.
                                    try:
                                        metadata["track_number"] = int(track_num)
                                    except ValueError:
                                        # Skip if not a simple number
                                        pass
                                else:
                                    # Fallback: enumerate and find position
                                    for i, t in enumerate(medium["tracks"]):
                                        if t.get("recording", {}).get("id") == recording_data.get("id"):
                                            metadata["track_number"] = i + 1
                                            break
                                break

        # Extract genre from tags
        if "tags" in recording_data and recording_data["tags"]:
            genres = []
            for tag in recording_data["tags"][:5]:  # Take top 5 tags
                if isinstance(tag, dict) and "name" in tag:
                    genres.append(tag["name"])
            if genres:
                metadata["genre"] = ", ".join(genres)

        return metadata

    async def enhance_metadata(self, audio_file: AudioFile) -> AudioFile:
        """Enhance metadata using MusicBrainz.

        Args:
            audio_file: The audio file to enhance

        Returns:
            AudioFile with enhanced metadata
        """
        if not self.enabled or not audio_file.title or not audio_file.artists:
            return audio_file

        # Check cache first
        cache_key = self._get_cache_key(
            audio_file.artists[0],
            audio_file.title,
            audio_file.album
        )

        cached_metadata = self._get_from_cache(cache_key)
        if cached_metadata:
            # Apply cached metadata
            for key, value in cached_metadata.items():
                if key in self.config.get("enhance_fields", []):
                    if hasattr(audio_file, key):
                        setattr(audio_file, key, value)
                    else:
                        audio_file.metadata[key] = value
            return audio_file

        # Build search query
        artist = audio_file.artists[0]
        title = audio_file.title
        album = audio_file.album

        # Try exact match first
        query = f'recording:"{title}" AND artist:"{artist}"'
        if album:
            query += f' AND release:"{album}"'

        # Search for recording
        result = await self._search_musicbrainz(query, "recording")

        if not result and self.config.get("fallback_to_fuzzy", True):
            # Try fuzzy search
            query = f'"{title}" AND "{artist}"'
            if album:
                query += f' AND "{album}"'
            result = await self._search_musicbrainz(query, "recording")

        if result and "recordings" in result and result["recordings"]:
            # Extract metadata from first match
            recording = result["recordings"][0]
            metadata = await self._extract_track_metadata(recording)

            # Cache the results
            self._add_to_cache(cache_key, metadata)

            # Apply metadata enhancements
            enhanced_fields = self.config.get("enhance_fields", [])
            for field in enhanced_fields:
                if field in metadata:
                    if hasattr(audio_file, field):
                        current_value = getattr(audio_file, field)
                        # Only update if current value is missing/empty
                        if not current_value:
                            setattr(audio_file, field, metadata[field])
                    else:
                        audio_file.metadata[field] = metadata[field]

        return audio_file

    async def batch_enhance(self, audio_files: List[AudioFile]) -> List[AudioFile]:
        """Enhance metadata for multiple audio files.

        Overrides default implementation to add rate limiting awareness.

        Args:
            audio_files: List of audio files to enhance

        Returns:
            List of AudioFile objects with enhanced metadata
        """
        enhanced_files = []
        rate_limit = self.config.get("rate_limit", 1.0)

        # Process with awareness of rate limits
        for audio_file in audio_files:
            if self.enabled:
                enhanced = await self.enhance_metadata(audio_file)
                enhanced_files.append(enhanced)
            else:
                enhanced_files.append(audio_file)

            # Small delay to respect rate limits
            await asyncio.sleep(0.1)

        return enhanced_files