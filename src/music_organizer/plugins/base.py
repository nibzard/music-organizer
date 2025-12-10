"""Base plugin interfaces for the music organizer."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Protocol
from dataclasses import dataclass, field
from pathlib import Path

from ..models.audio_file import AudioFile


@dataclass(slots=True)
class PluginInfo:
    """Information about a plugin."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    min_python_version: str = "3.9"

    def __post_init__(self) -> None:
        """Validate plugin info after creation."""
        if not self.name:
            raise ValueError("Plugin name is required")
        if not self.version:
            raise ValueError("Plugin version is required")


class Plugin(ABC):
    """Base class for all plugins."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize plugin with optional configuration."""
        self.config = config or {}
        self.enabled = True

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Return plugin information."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the plugin."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources when plugin is disabled."""
        pass

    def enable(self) -> None:
        """Enable the plugin."""
        self.enabled = True

    def disable(self) -> None:
        """Disable the plugin."""
        self.enabled = False
        self.cleanup()


class MetadataPlugin(Plugin):
    """Base class for metadata enhancement plugins."""

    @abstractmethod
    async def enhance_metadata(self, audio_file: AudioFile) -> AudioFile:
        """Enhance metadata for an audio file.

        Args:
            audio_file: The audio file to enhance

        Returns:
            AudioFile with enhanced metadata
        """
        pass

    async def batch_enhance(self, audio_files: List[AudioFile]) -> List[AudioFile]:
        """Enhance metadata for multiple audio files.

        Default implementation processes files sequentially.
        Override for batch processing optimizations.

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


class ClassificationPlugin(Plugin):
    """Base class for content classification plugins."""

    @abstractmethod
    async def classify(self, audio_file: AudioFile) -> Dict[str, Any]:
        """Classify an audio file.

        Args:
            audio_file: The audio file to classify

        Returns:
            Dictionary with classification results
        """
        pass

    @abstractmethod
    def get_supported_tags(self) -> List[str]:
        """Return list of classification tags this plugin provides.

        Examples: ['genre', 'mood', 'era', 'language']
        """
        pass

    async def batch_classify(self, audio_files: List[AudioFile]) -> List[Dict[str, Any]]:
        """Classify multiple audio files.

        Default implementation processes files sequentially.
        Override for batch processing optimizations.

        Args:
            audio_files: List of audio files to classify

        Returns:
            List of classification results
        """
        results = []
        for audio_file in audio_files:
            if self.enabled:
                result = await self.classify(audio_file)
                results.append(result)
            else:
                results.append({})
        return results


class OutputPlugin(Plugin):
    """Base class for output and export plugins."""

    @abstractmethod
    async def export(self, audio_files: List[AudioFile], output_path: Path) -> bool:
        """Export audio files to specified format/location.

        Args:
            audio_files: List of audio files to export
            output_path: Destination path for export

        Returns:
            True if export was successful
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported export formats.

        Examples: ['m3u', 'pls', 'csv', 'json']
        """
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """Return the file extension for this export format."""
        pass


class PathPlugin(Plugin):
    """Base class for custom path and naming pattern plugins."""

    @abstractmethod
    async def generate_target_path(self, audio_file: AudioFile, base_dir: Path) -> Path:
        """Generate target directory path for an audio file.

        Args:
            audio_file: The audio file to generate path for
            base_dir: Base directory for organized music

        Returns:
            Target directory path (not including filename)
        """
        pass

    @abstractmethod
    async def generate_filename(self, audio_file: AudioFile) -> str:
        """Generate filename for an audio file.

        Args:
            audio_file: The audio file to generate filename for

        Returns:
            Filename (including extension but not path)
        """
        pass

    def get_supported_variables(self) -> List[str]:
        """Return list of supported template variables.

        Examples: ['artist', 'album', 'year', 'track_number', 'title', 'genre']
        """
        return [
            'artist', 'artists', 'primary_artist',
            'album', 'year', 'track_number', 'title',
            'genre', 'duration', 'bitrate', 'format',
            'content_type', 'date', 'location'
        ]

    async def batch_generate_paths(self, audio_files: List[AudioFile], base_dir: Path) -> List[Path]:
        """Generate paths for multiple audio files.

        Default implementation processes files sequentially.
        Override for batch processing optimizations.

        Args:
            audio_files: List of audio files to generate paths for
            base_dir: Base directory for organized music

        Returns:
            List of target directory paths
        """
        paths = []
        for audio_file in audio_files:
            if self.enabled:
                path = await self.generate_target_path(audio_file, base_dir)
                paths.append(path)
            else:
                # Fall back to default behavior
                paths.append(audio_file.get_target_path(base_dir))
        return paths


# Protocol for plugin discovery
class PluginFactory(Protocol):
    """Protocol for plugin factory functions."""

    def __call__(self, config: Optional[Dict[str, Any]] = None) -> Plugin:
        """Create a plugin instance."""
        ...