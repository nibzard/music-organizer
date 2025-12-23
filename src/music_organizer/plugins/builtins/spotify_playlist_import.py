"""Spotify playlist import plugin."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import OutputPlugin, PluginInfo
from ...models.audio_file import AudioFile
from ...plugins.config import PluginConfigSchema, ConfigOption

logger = logging.getLogger(__name__)


class SpotifyPlaylistImportPlugin(OutputPlugin):
    """Plugin to import Spotify playlists and match local files."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="spotify_playlist_import",
            version="1.0.0",
            description="Import Spotify playlists and match to local files",
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
                description="Enable Spotify playlist import"
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
                description="Spotify access token (for user playlists)"
            ),
            ConfigOption(
                name="refresh_token",
                type=str,
                default="",
                description="Spotify refresh token"
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
                name="export_m3u",
                type=bool,
                default=True,
                description="Export matched playlists as M3U"
            ),
        ])

    def initialize(self) -> None:
        """Initialize the plugin."""
        self._adapter: Optional[Any] = None
        self._match_cache: Dict[str, Optional[AudioFile]] = {}
        logger.info("Spotify playlist import plugin initialized")

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
        logger.info("Spotify playlist import plugin cleaned up")

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
                access_token=self.config.get("access_token"),
                refresh_token=self.config.get("refresh_token"),
                timeout=self.config.get("timeout", 10.0),
                rate_limit=self.config.get("rate_limit", 10.0)
            )

        return self._adapter

    def get_supported_formats(self) -> List[str]:
        """Return list of supported export formats."""
        return ["m3u", "m3u8"]

    def get_file_extension(self) -> str:
        """Return the file extension for this export format."""
        return ".m3u"

    async def export(self, audio_files: List[AudioFile], output_path: Path) -> bool:
        """Export Spotify playlist to local format.

        Args:
            audio_files: List of local audio files to match against
            output_path: Path to Spotify playlist (can be playlist ID or file with IDs)

        Returns:
            True if export was successful
        """
        adapter = self._get_adapter()
        if not adapter:
            logger.error("Cannot get Spotify adapter")
            return False

        playlist_id = self._extract_playlist_id(output_path)
        if not playlist_id:
            logger.error(f"Could not extract playlist ID from {output_path}")
            return False

        spotify_playlist = await adapter.get_playlist(playlist_id)
        if not spotify_playlist:
            logger.error(f"Could not fetch Spotify playlist {playlist_id}")
            return False

        matched_files = await self._match_playlist(spotify_playlist, audio_files)

        output_file = self._get_output_path(spotify_playlist.name, output_path)

        if self.config.get("export_m3u", True):
            return self._write_m3u_playlist(matched_files, spotify_playlist, output_file)

        return len(matched_files) > 0

    def _extract_playlist_id(self, source: Path) -> Optional[str]:
        """Extract playlist ID from various sources."""
        source_str = str(source)

        if source.is_file():
            try:
                return source.read_text().strip()
            except Exception:
                pass

        if "spotify.com" in source_str:
            for part in source_str.split("/"):
                if part.startswith("playlist/"):
                    return part.split("/")[1]
            for part in source_str.split("?")[0].split("/"):
                if part and part != "playlist" and not part.startswith("http"):
                    return part

        if source_str.startswith("spotify:playlist:"):
            return source_str.split(":")[-1]

        return source_str if source_str else None

    async def _match_playlist(
        self,
        spotify_playlist: Any,
        local_files: List[AudioFile]
    ) -> List[AudioFile]:
        """Match Spotify playlist tracks to local files."""
        adapter = self._get_adapter()
        min_confidence = self.config.get("min_confidence", 0.7)

        matched = []
        local_by_artist_title: Dict[str, List[AudioFile]] = {}

        for af in local_files:
            if af.artists and af.title:
                key = f"{af.artists[0].lower()}:{af.title.lower()}"
                if key not in local_by_artist_title:
                    local_by_artist_title[key] = []
                local_by_artist_title[key].append(af)

        for spotify_track in spotify_playlist.tracks:
            artist_key = spotify_track.artists[0].lower() if spotify_track.artists else ""
            title_key = spotify_track.name.lower()
            key = f"{artist_key}:{title_key}"

            cache_key = f"match:{spotify_track.id}"
            if cache_key in self._match_cache:
                matched_file = self._match_cache[cache_key]
                if matched_file:
                    matched.append(matched_file)
                continue

            best_match = None
            best_score = 0.0

            if key in local_by_artist_title:
                for local_file in local_by_artist_title[key]:
                    score = self._calculate_match_score(
                        spotify_track,
                        local_file
                    )
                    if score >= min_confidence and score > best_score:
                        best_score = score
                        best_match = local_file

            if not best_match:
                for local_file in local_files:
                    score = self._calculate_match_score(
                        spotify_track,
                        local_file
                    )
                    if score >= min_confidence and score > best_score:
                        best_score = score
                        best_match = local_file

            if best_match:
                best_match.metadata["spotify_uri"] = spotify_track.uri
                best_match.metadata["spotify_id"] = spotify_track.id
                matched.append(best_match)
                logger.debug(f"Matched: {spotify_track.name} -> {best_match.path}")
            else:
                logger.debug(f"No match found: {spotify_track.name}")

            self._match_cache[cache_key] = best_match

        return matched

    def _calculate_match_score(self, spotify_track: Any, local_file: AudioFile) -> float:
        """Calculate match score between Spotify track and local file."""
        if not local_file.artists or not local_file.title:
            return 0.0

        try:
            from ...utils.string_similarity import music_metadata_similarity
        except ImportError:
            def music_metadata_similarity(a: str, b: str) -> float:
                a_lower = a.lower().strip()
                b_lower = b.lower().strip()
                if a_lower == b_lower:
                    return 1.0
                if a_lower in b_lower or b_lower in a_lower:
                    return 0.8
                return 0.0

        title_score = music_metadata_similarity(spotify_track.name, local_file.title)

        artist_scores = [
            music_metadata_similarity(spotify_artist, local_artist)
            for spotify_artist in spotify_track.artists
            for local_artist in local_file.artists
        ]
        artist_score = max(artist_scores) if artist_scores else 0.0

        combined_score = (title_score * 0.6 + artist_score * 0.4)

        duration_sec = local_file.metadata.get("duration_seconds", 0)
        if duration_sec > 0:
            duration_diff = abs(spotify_track.duration_ms / 1000 - duration_sec)
            if duration_diff < 5:
                combined_score *= 1.1
            elif duration_diff > 30:
                combined_score *= 0.5

        return min(combined_score, 1.0)

    def _get_output_path(self, playlist_name: str, original_path: Path) -> Path:
        """Generate output path for M3U file."""
        if original_path.is_dir():
            safe_name = "".join(
                c for c in playlist_name
                if c.isalnum() or c in (' ', '-', '_', '.')
            ).strip()
            return original_path / f"{safe_name}.m3u"
        return original_path.with_suffix(".m3u")

    def _write_m3u_playlist(
        self,
        matched_files: List[AudioFile],
        spotify_playlist: Any,
        output_path: Path
    ) -> bool:
        """Write matched files as M3U playlist."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("#EXTM3U\n")
                f.write(f"#PLAYLIST:{spotify_playlist.name}\n")

                if spotify_playlist.description:
                    desc = spotify_playlist.description.replace('\n', ' ')
                    f.write(f"#EXTINF:{desc}\n")

                for audio_file in matched_files:
                    duration = int(audio_file.metadata.get("duration_seconds", 0))
                    artist = audio_file.artists[0] if audio_file.artists else "Unknown"
                    title = audio_file.title or "Unknown"

                    f.write(f"#EXTINF:{duration},{artist} - {title}\n")
                    f.write(f"{audio_file.path}\n")

            logger.info(f"Exported {len(matched_files)} tracks to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error writing M3U playlist: {e}")
            return False
