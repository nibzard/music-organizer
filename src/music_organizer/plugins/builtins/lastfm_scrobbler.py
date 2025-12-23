"""Last.fm scrobbling plugin - submits tracks when organized."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..base import OutputPlugin, PluginInfo
from ...models.audio_file import AudioFile
from ...plugins.config import PluginConfigSchema, ConfigOption

logger = logging.getLogger(__name__)


class LastFmScrobblerPlugin(OutputPlugin):
    """Plugin to scrobble tracks to Last.fm after organization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the scrobbler plugin."""
        super().__init__(config)
        self._adapter: Optional[Any] = None
        self._scrobble_queue: List[Dict[str, Any]] = []
        self._failed_scrobbles: List[Dict[str, Any]] = []

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="lastfm_scrobbler",
            version="1.0.0",
            description="Scrobbles tracks to Last.fm after organization",
            author="Music Organizer Team",
            dependencies=["aiohttp"],
        )

    def get_config_schema(self) -> PluginConfigSchema:
        """Get configuration schema for this plugin."""
        return PluginConfigSchema([
            ConfigOption(
                name="enabled",
                type=bool,
                default=False,
                description="Enable Last.fm scrobbling"
            ),
            ConfigOption(
                name="api_key",
                type=str,
                default="",
                description="Last.fm API key"
            ),
            ConfigOption(
                name="api_secret",
                type=str,
                default="",
                description="Last.fm API secret"
            ),
            ConfigOption(
                name="session_key",
                type=str,
                default="",
                description="Last.fm session key (authenticated)"
            ),
            ConfigOption(
                name="scrobble_on_export",
                type=bool,
                default=True,
                description="Scrobble when export() is called"
            ),
            ConfigOption(
                name="scrobble_timestamp",
                type=str,
                default="current",
                choices=["current", "file_mtime"],
                description="Timestamp to use: 'current' or 'file_mtime'"
            ),
            ConfigOption(
                name="batch_size",
                type=int,
                default=50,
                min_value=1,
                max_value=500,
                description="Number of scrobbles to batch together"
            ),
            ConfigOption(
                name="retry_failed",
                type=bool,
                default=True,
                description="Retry failed scrobbles"
            ),
            ConfigOption(
                name="max_retries",
                type=int,
                default=3,
                min_value=0,
                max_value=10,
                description="Maximum retry attempts for failed scrobbles"
            ),
        ])

    def initialize(self) -> None:
        """Initialize the plugin."""
        self._scrobble_queue = []
        self._failed_scrobbles = []
        logger.info("Last.fm scrobbler plugin initialized")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._adapter:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._adapter.close())
            except RuntimeError:
                pass
            self._adapter = None
        self._scrobble_queue.clear()
        self._failed_scrobbles.clear()
        logger.info("Last.fm scrobbler plugin cleaned up")

    def _get_adapter(self):
        """Get or create the Last.fm adapter."""
        if not self.enabled:
            return None

        api_key = self.config.get("api_key")
        api_secret = self.config.get("api_secret")
        session_key = self.config.get("session_key")

        if not api_key or not api_secret:
            logger.warning("Last.fm API key/secret not configured")
            return None

        if not session_key:
            logger.warning("Last.fm not authenticated (no session key)")
            return None

        try:
            from ...infrastructure.external.lastfm_adapter import LastFmAdapter
        except ImportError:
            logger.warning("Last.fm adapter not available")
            return None

        if self._adapter is None:
            self._adapter = LastFmAdapter(
                api_key=api_key,
                api_secret=api_secret,
                session_key=session_key
            )

        return self._adapter

    def _get_timestamp(self, audio_file: AudioFile) -> int:
        """Get timestamp for scrobble.

        Args:
            audio_file: Audio file to get timestamp for

        Returns:
            Unix timestamp
        """
        if self.config.get("scrobble_timestamp", "current") == "file_mtime":
            try:
                return int(audio_file.path.stat().st_mtime)
            except (OSError, FileNotFoundError):
                pass
        return int(time.time())

    async def _scrobble_track(self, audio_file: AudioFile) -> bool:
        """Scrobble a single track.

        Args:
            audio_file: Audio file to scrobble

        Returns:
            True if successful
        """
        adapter = self._get_adapter()
        if not adapter:
            return False

        artist = audio_file.artists[0] if audio_file.artists else "Unknown Artist"
        track = audio_file.title or "Unknown Track"

        # Get timestamp
        timestamp = self._get_timestamp(audio_file)

        # Get duration from metadata
        duration = audio_file.metadata.get("duration_seconds")

        # Submit scrobble
        success = await adapter.scrobble(
            artist=artist,
            track=track,
            timestamp=timestamp,
            album=audio_file.album,
            track_number=audio_file.track_number,
            duration=duration
        )

        if success:
            logger.info(f"Scrobbled: {audio_file.get_display_name()}")
        else:
            logger.warning(f"Failed to scrobble: {audio_file.get_display_name()}")

        return success

    async def _scrobble_batch(self, audio_files: List[AudioFile]) -> Dict[str, int]:
        """Scrobble multiple tracks in batch.

        Args:
            audio_files: List of audio files to scrobble

        Returns:
            Dict with success and failure counts
        """
        results = {"success": 0, "failed": 0}

        for audio_file in audio_files:
            success = await self._scrobble_track(audio_file)
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
                self._failed_scrobbles.append({
                    "audio_file": audio_file,
                    "retries": 0
                })

        return results

    async def export(self, audio_files: List[AudioFile], output_path: Path) -> bool:
        """Export (scrobble) audio files to Last.fm.

        This method implements the OutputPlugin interface but actually
        performs scrobbling rather than file export. The output_path
        parameter is ignored but kept for interface compatibility.

        Args:
            audio_files: List of audio files to scrobble
            output_path: Ignored (for interface compatibility)

        Returns:
            True if export was successful
        """
        if not self.config.get("scrobble_on_export", True):
            return True

        adapter = self._get_adapter()
        if not adapter:
            logger.warning("Last.fm adapter not available, skipping scrobbles")
            return False

        logger.info(f"Scrobbling {len(audio_files)} tracks to Last.fm...")

        # Process in batches
        batch_size = self.config.get("batch_size", 50)
        total_success = 0
        total_failed = 0

        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i + batch_size]
            results = await self._scrobble_batch(batch)
            total_success += results["success"]
            total_failed += results["failed"]

        logger.info(f"Scrobbling complete: {total_success} succeeded, {total_failed} failed")

        # Retry failed scrobbles if enabled
        if self.config.get("retry_failed", True) and self._failed_scrobbles:
            await self._retry_failed()

        return total_failed == 0

    async def _retry_failed(self) -> None:
        """Retry failed scrobbles."""
        max_retries = self.config.get("max_retries", 3)
        adapter = self._get_adapter()
        if not adapter:
            return

        logger.info(f"Retrying {len(self._failed_scrobbles)} failed scrobbles...")

        still_failed = []

        for item in self._failed_scrobbles:
            if item["retries"] >= max_retries:
                still_failed.append(item)
                continue

            item["retries"] += 1
            success = await self._scrobble_track(item["audio_file"])

            if not success:
                still_failed.append(item)

        self._failed_scrobbles = still_failed

        if self._failed_scrobbles:
            logger.warning(f"{len(self._failed_scrobbles)} scrobbles still failed after retries")

    def get_supported_formats(self) -> List[str]:
        """Return supported export formats (not applicable for scrobbler)."""
        return ["lastfm"]

    def get_file_extension(self) -> str:
        """Return file extension (not applicable for scrobbler)."""
        return ""

    async def scrobble_files(self, audio_files: List[AudioFile]) -> Dict[str, int]:
        """Direct method to scrobble files.

        This is a convenience method that can be called directly
        without going through the export interface.

        Args:
            audio_files: List of audio files to scrobble

        Returns:
            Dict with success and failure counts
        """
        return await self._scrobble_batch(audio_files)

    async def scrobble_single(self, audio_file: AudioFile) -> bool:
        """Scrobble a single audio file.

        Args:
            audio_file: Audio file to scrobble

        Returns:
            True if successful
        """
        return await self._scrobble_track(audio_file)

    def get_failed_scrobbles(self) -> List[Dict[str, Any]]:
        """Get list of failed scrobbles.

        Returns:
            List of failed scrobble info dicts
        """
        return self._failed_scrobbles.copy()

    def clear_failed_scrobbles(self) -> None:
        """Clear the failed scrobbles list."""
        self._failed_scrobbles.clear()
