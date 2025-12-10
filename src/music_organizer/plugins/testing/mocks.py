"""
Mock objects for plugin testing.

This module provides mock implementations of core objects to facilitate
isolated plugin testing without requiring actual audio files or external services.
"""

import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime

from ...models.audio_file import AudioFile
from ..base import (
    Plugin, MetadataPlugin, ClassificationPlugin,
    OutputPlugin, PathPlugin, PluginInfo
)
from ..manager import PluginManager


class MockAudioFile:
    """
    Mock implementation of AudioFile for testing.

    Provides realistic behavior without requiring actual audio files.
    """

    def __init__(
        self,
        path: Union[str, Path] = "/test/mock_file.mp3",
        metadata: Optional[Dict[str, Any]] = None,
        duration: float = 180.5,
        bitrate: int = 320,
        format: str = "mp3"
    ):
        """Initialize mock audio file."""
        self.path = Path(path)
        self.metadata = metadata or {}
        self.duration = duration
        self.bitrate = bitrate
        self.format = format
        self.created_at = datetime.now()

    def copy(self) -> 'MockAudioFile':
        """Create a copy of this mock audio file."""
        return MockAudioFile(
            path=self.path,
            metadata=self.metadata.copy(),
            duration=self.duration,
            bitrate=self.bitrate,
            format=self.format
        )

    def get_target_path(self, base_dir: Path) -> Path:
        """Get default target path for organization."""
        artist = self.metadata.get('artist', 'Unknown Artist')
        album = self.metadata.get('album', 'Unknown Album')
        return base_dir / artist / album / self.path.name

    def __eq__(self, other) -> bool:
        """Check equality with other audio file."""
        if not isinstance(other, (MockAudioFile, AudioFile)):
            return False
        return (
            self.path == other.path and
            self.metadata == other.metadata and
            abs(self.duration - other.duration) < 0.1
        )

    def __repr__(self) -> str:
        """String representation of mock audio file."""
        return f"MockAudioFile(path={self.path}, metadata={len(self.metadata)} fields)"


class MockPlugin:
    """
    Mock implementation of base Plugin for testing.

    Useful for testing PluginManager and hook systems.
    """

    def __init__(
        self,
        name: str = "mock-plugin",
        version: str = "1.0.0",
        plugin_type: str = "metadata"
    ):
        """Initialize mock plugin."""
        self.name = name
        self.version = version
        self.plugin_type = plugin_type
        self.enabled = True
        self.config = {}
        self.initialized = False
        self.cleaned_up = False

        # Track method calls for testing
        self.call_history = []

    @property
    def info(self) -> PluginInfo:
        """Return mock plugin info."""
        return PluginInfo(
            name=self.name,
            version=self.version,
            description=f"Mock {self.plugin_type} plugin for testing",
            author="Test Suite"
        )

    def initialize(self) -> None:
        """Initialize mock plugin."""
        self.initialized = True
        self.call_history.append("initialize")

    def cleanup(self) -> None:
        """Cleanup mock plugin."""
        self.cleaned_up = True
        self.call_history.append("cleanup")

    def enable(self) -> None:
        """Enable mock plugin."""
        self.enabled = True
        self.call_history.append("enable")

    def disable(self) -> None:
        """Disable mock plugin."""
        self.enabled = False
        self.call_history.append("disable")


class MockMetadataPlugin(MockPlugin, MetadataPlugin):
    """
    Mock implementation of MetadataPlugin for testing.

    Provides configurable behavior for testing metadata enhancement.
    """

    def __init__(
        self,
        name: str = "mock-metadata-plugin",
        enhance_result: Optional[Dict[str, Any]] = None,
        should_fail: bool = False,
        delay: float = 0.0
    ):
        """Initialize mock metadata plugin."""
        MockPlugin.__init__(self, name, "1.0.0", "metadata")
        MetadataPlugin.__init__(self)
        # Use the name parameter directly since self.name is set in MockPlugin.__init__
        self.enhance_result = enhance_result or {
            'enhanced_by': name,
            'enhanced_at': datetime.now().isoformat()
        }
        self.should_fail = should_fail
        self.delay = delay
        self.enhanced_files = []

    async def enhance_metadata(self, audio_file: AudioFile) -> AudioFile:
        """Mock metadata enhancement."""
        # Add delay if configured
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        # Simulate failure if configured
        if self.should_fail:
            raise Exception(f"Mock failure in {self.name}")

        # Create enhanced copy
        enhanced = audio_file.copy()

        # Always include plugin tracking fields
        tracking_fields = {
            'enhanced_by': self.name,
            'enhanced_at': datetime.now().isoformat()
        }

        # Apply custom result first, then tracking fields (so tracking is never overridden)
        enhanced.metadata.update(self.enhance_result)
        enhanced.metadata.update(tracking_fields)

        self.enhanced_files.append(enhanced)

        return enhanced

    async def batch_enhance(self, audio_files: List[AudioFile]) -> List[AudioFile]:
        """Mock batch enhancement."""
        results = []
        for audio_file in audio_files:
            enhanced = await self.enhance_metadata(audio_file)
            results.append(enhanced)
        return results


class MockClassificationPlugin(MockPlugin, ClassificationPlugin):
    """
    Mock implementation of ClassificationPlugin for testing.

    Provides configurable classification results.
    """

    def __init__(
        self,
        name: str = "mock-classification-plugin",
        classification_result: Optional[Dict[str, Any]] = None,
        supported_tags: Optional[List[str]] = None
    ):
        """Initialize mock classification plugin."""
        MockPlugin.__init__(self, name, "1.0.0", "classification")
        ClassificationPlugin.__init__(self)
        self.classification_result = classification_result or {
            'genre': 'mock-genre',
            'confidence': 0.85,
            'mood': 'neutral'
        }
        self.supported_tags = supported_tags or ['genre', 'confidence', 'mood']
        self.classified_files = []

    def get_supported_tags(self) -> List[str]:
        """Return supported classification tags."""
        return self.supported_tags

    async def classify(self, audio_file: AudioFile) -> Dict[str, Any]:
        """Mock classification."""
        result = self.classification_result.copy()
        result['classified_by'] = self.name
        self.classified_files.append(audio_file)
        return result

    async def batch_classify(self, audio_files: List[AudioFile]) -> List[Dict[str, Any]]:
        """Mock batch classification."""
        results = []
        for audio_file in audio_files:
            result = await self.classify(audio_file)
            results.append(result)
        return results


class MockOutputPlugin(MockPlugin, OutputPlugin):
    """
    Mock implementation of OutputPlugin for testing.

    Simulates file exports without actually writing files.
    """

    def __init__(
        self,
        name: str = "mock-output-plugin",
        supported_formats: Optional[List[str]] = None,
        file_extension: str = "mock",
        export_success: bool = True
    ):
        """Initialize mock output plugin."""
        MockPlugin.__init__(self, name, "1.0.0", "output")
        OutputPlugin.__init__(self)
        self.supported_formats = supported_formats or ['mock', 'test']
        self.file_extension = file_extension
        self.export_success = export_success
        self.exported_files = []

    def get_supported_formats(self) -> List[str]:
        """Return supported export formats."""
        return self.supported_formats

    def get_file_extension(self) -> str:
        """Return file extension."""
        return self.file_extension

    async def export(self, audio_files: List[AudioFile], output_path: Path) -> bool:
        """Mock export operation."""
        self.exported_files.append({
            'files': audio_files,
            'output_path': output_path,
            'timestamp': datetime.now()
        })
        return self.export_success


class MockPathPlugin(MockPlugin, PathPlugin):
    """
    Mock implementation of PathPlugin for testing.

    Provides predictable path and filename generation.
    """

    def __init__(
        self,
        name: str = "mock-path-plugin",
        path_pattern: str = "{artist}/{album}",
        filename_pattern: str = "{title}"
    ):
        """Initialize mock path plugin."""
        MockPlugin.__init__(self, name, "1.0.0", "path")
        PathPlugin.__init__(self)
        self.path_pattern = path_pattern
        self.filename_pattern = filename_pattern

    async def generate_target_path(self, audio_file: AudioFile, base_dir: Path) -> Path:
        """Mock path generation."""
        metadata = audio_file.metadata
        artist = metadata.get('artist', 'Unknown Artist')
        album = metadata.get('album', 'Unknown Album')
        year = metadata.get('year', 'Unknown Year')

        path_str = self.path_pattern.format(
            artist=artist,
            album=album,
            year=year
        )

        return base_dir / path_str

    async def generate_filename(self, audio_file: AudioFile) -> str:
        """Mock filename generation."""
        metadata = audio_file.metadata
        title = metadata.get('title', 'Unknown Title')
        extension = audio_file.path.suffix

        # Convert track_number to int for formatting, handling non-numeric values
        track_number = metadata.get('track_number', '')
        try:
            track_number_int = int(track_number)
        except (ValueError, TypeError):
            track_number_int = 0

        filename = self.filename_pattern.format(
            title=title,
            artist=metadata.get('artist', 'Unknown Artist'),
            track_number=track_number_int
        )

        return f"{filename}{extension}"


class MockPluginManager:
    """
    Mock implementation of PluginManager for testing.

    Provides controlled plugin loading and execution.
    """

    def __init__(self):
        """Initialize mock plugin manager."""
        self.plugins = {}
        self.plugin_order = []
        self.execution_history = []
        self.load_history = []

    def load_plugin(self, plugin: Plugin) -> None:
        """Load a plugin."""
        self.plugins[plugin.info.name] = plugin
        self.plugin_order.append(plugin.info.name)
        self.load_history.append(plugin.info.name)

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a loaded plugin by name."""
        return self.plugins.get(name)

    def get_plugins_by_type(self, plugin_type: str) -> List[Plugin]:
        """Get all plugins of a specific type."""
        # This is a simplified implementation
        return list(self.plugins.values())

    async def process_metadata(self, audio_file: AudioFile) -> AudioFile:
        """Process metadata through loaded metadata plugins."""
        result = audio_file
        for plugin_name in self.plugin_order:
            plugin = self.plugins[plugin_name]
            if isinstance(plugin, MetadataPlugin) and plugin.enabled:
                self.execution_history.append(f"metadata:{plugin_name}")
                result = await plugin.enhance_metadata(result)
        return result

    async def classify_file(self, audio_file: AudioFile) -> Dict[str, Any]:
        """Classify file through loaded classification plugins."""
        results = {}
        for plugin_name in self.plugin_order:
            plugin = self.plugins[plugin_name]
            if isinstance(plugin, ClassificationPlugin) and plugin.enabled:
                self.execution_history.append(f"classify:{plugin_name}")
                result = await plugin.classify(audio_file)
                results.update(result)
        return results


class MockExternalAPI:
    """
    Mock external API for testing plugins that make HTTP requests.

    Provides configurable responses and behaviors.
    """

    def __init__(self):
        """Initialize mock API."""
        self.requests = []
        self.responses = {}
        self.should_fail = False
        self.failure_message = "Mock API failure"
        self.delay = 0.0

    def set_response(self, endpoint: str, response: Dict[str, Any]):
        """Set response for a specific endpoint."""
        self.responses[endpoint] = response

    def set_failure(self, should_fail: bool, message: str = "Mock API failure"):
        """Configure whether API calls should fail."""
        self.should_fail = should_fail
        self.failure_message = message

    def set_delay(self, delay: float):
        """Set delay for API responses."""
        self.delay = delay

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock GET request."""
        self.requests.append({'method': 'GET', 'endpoint': endpoint, 'params': params})

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise Exception(self.failure_message)

        if endpoint in self.responses:
            return self.responses[endpoint]

        return {'status': 'success', 'data': {}}

    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock POST request."""
        self.requests.append({'method': 'POST', 'endpoint': endpoint, 'data': data})

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise Exception(self.failure_message)

        if endpoint in self.responses:
            return self.responses[endpoint]

        return {'status': 'success', 'data': {}}


def create_mock_audio_file(
    path: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> MockAudioFile:
    """
    Factory function to create mock audio files with sensible defaults.

    Args:
        path: File path for the mock audio file
        metadata: Metadata dictionary
        **kwargs: Additional arguments for MockAudioFile

    Returns:
        MockAudioFile instance
    """
    if path is None:
        path = f"/test/mock_file_{id(kwargs)}.mp3"

    if metadata is None:
        metadata = {
            'title': 'Mock Song',
            'artist': 'Mock Artist',
            'album': 'Mock Album',
            'year': '2023',
            'track_number': '1',
            'genre': 'Mock Genre'
        }

    return MockAudioFile(path=path, metadata=metadata, **kwargs)


def create_mock_plugin(
    plugin_type: str,
    name: Optional[str] = None,
    **kwargs
) -> Plugin:
    """
    Factory function to create mock plugins of different types.

    Args:
        plugin_type: Type of plugin to create ('metadata', 'classification', 'output', 'path')
        name: Optional name for the plugin
        **kwargs: Additional arguments for the specific plugin type

    Returns:
        Mock plugin instance
    """
    if name is None:
        name = f"mock-{plugin_type}-plugin"

    if plugin_type == 'metadata':
        return MockMetadataPlugin(name=name, **kwargs)
    elif plugin_type == 'classification':
        return MockClassificationPlugin(name=name, **kwargs)
    elif plugin_type == 'output':
        return MockOutputPlugin(name=name, **kwargs)
    elif plugin_type == 'path':
        return MockPathPlugin(name=name, **kwargs)
    else:
        return MockPlugin(name=name, plugin_type=plugin_type)