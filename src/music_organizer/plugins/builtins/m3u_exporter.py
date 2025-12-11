"""M3U playlist export plugin."""

from pathlib import Path
from typing import Dict, Any, List
from ..base import OutputPlugin, PluginInfo
from ...models.audio_file import AudioFile


class M3UExporterPlugin(OutputPlugin):
    """Plugin to export audio files as M3U playlists."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="m3u_exporter",
            version="1.0.0",
            description="Exports audio files as M3U playlists",
            author="Music Organizer Team",
            dependencies=[],
        )

    def initialize(self) -> None:
        """Initialize the plugin."""
        print("M3U exporter plugin initialized")

    def cleanup(self) -> None:
        """Cleanup resources."""
        print("M3U exporter plugin cleaned up")

    async def export(self, audio_files: List[AudioFile], output_path: Path) -> bool:
        """Export audio files as M3U playlist.

        Args:
            audio_files: List of audio files to export
            output_path: Destination path for M3U file

        Returns:
            True if export was successful
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("#EXTM3U\n")

                for audio_file in audio_files:
                    # Write extended info if available
                    if audio_file.title and audio_file.artists:
                        duration = self._get_duration_estimate(audio_file)
                        artist = audio_file.artists[0]
                        f.write(f"#EXTINF:{duration},{artist} - {audio_file.title}\n")
                    elif audio_file.title:
                        duration = self._get_duration_estimate(audio_file)
                        f.write(f"#EXTINF:{duration},{audio_file.title}\n")

                    # Write file path (convert to relative if possible)
                    try:
                        # Try to make path relative to output file
                        rel_path = audio_file.path.relative_to(output_path.parent)
                        f.write(str(rel_path) + "\n")
                    except ValueError:
                        # Use absolute path if not relative
                        f.write(str(audio_file.path) + "\n")

            return True

        except Exception as e:
            print(f"Error exporting M3U playlist: {e}")
            return False

    def _get_duration_estimate(self, audio_file: AudioFile) -> int:
        """Get duration estimate for M3U extended info.

        Since we don't store duration, return a placeholder.
        Real implementation would extract this from metadata.
        """
        return -1  # Unknown duration

    def get_supported_formats(self) -> List[str]:
        """Return supported export formats."""
        return ['m3u']

    def get_file_extension(self) -> str:
        """Return file extension for M3U format."""
        return '.m3u'