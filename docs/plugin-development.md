# Plugin Development Guide

Welcome to the music-organizer plugin development guide! This guide will help you create custom plugins to extend the functionality of the music organizer.

## Table of Contents

1. [Plugin Architecture Overview](#plugin-architecture-overview)
2. [Plugin Types](#plugin-types)
3. [Getting Started](#getting-started)
4. [Plugin Development Tutorial](#plugin-development-tutorial)
5. [Configuration System](#configuration-system)
6. [Best Practices](#best-practices)
7. [Testing Your Plugins](#testing-your-plugins)
8. [Plugin Examples](#plugin-examples)

## Plugin Architecture Overview

The music-organizer uses a plugin architecture that allows for extensibility through four main plugin types:

- **Metadata Plugins**: Enhance and enrich metadata for audio files
- **Classification Plugins**: Classify audio files with custom tags and categories
- **Output Plugins**: Export audio files to various formats and destinations
- **Path Plugins**: Generate custom file organization patterns and naming schemes

All plugins inherit from the base `Plugin` class and implement specific abstract methods based on their type.

## Plugin Types

### MetadataPlugin
Enhance metadata for audio files by fetching data from external sources or applying transformations.

**Key Methods:**
- `enhance_metadata(audio_file: AudioFile) -> AudioFile`: Enhance a single file
- `batch_enhance(audio_files: List[AudioFile]) -> List[AudioFile]`: Enhance multiple files

**Use Cases:**
- Fetching metadata from MusicBrainz, Discogs, or Spotify
- Adding custom metadata fields
- Normalizing or cleaning existing metadata

### ClassificationPlugin
Classify audio files with custom tags and categories.

**Key Methods:**
- `classify(audio_file: AudioFile) -> Dict[str, Any]`: Classify a single file
- `get_supported_tags() -> List[str]`: Return supported classification tags
- `batch_classify(audio_files: List[AudioFile]) -> List[Dict[str, Any]]`: Classify multiple files

**Use Cases:**
- Genre detection and classification
- Mood and energy analysis
- Language detection
- Era identification

### OutputPlugin
Export audio files to various formats and destinations.

**Key Methods:**
- `export(audio_files: List[AudioFile], output_path: Path) -> bool`: Export files
- `get_supported_formats() -> List[str]`: Return supported export formats
- `get_file_extension() -> str`: Return file extension for export format

**Use Cases:**
- Playlist export (M3U, PLS, etc.)
- CSV/JSON data export
- Cloud storage integration
- Database export

### PathPlugin
Generate custom file organization patterns and naming schemes.

**Key Methods:**
- `generate_target_path(audio_file: AudioFile, base_dir: Path) -> Path`: Generate directory path
- `generate_filename(audio_file: AudioFile) -> str`: Generate filename
- `get_supported_variables() -> List[str]`: Return supported template variables

**Use Cases:**
- Custom directory structures
- Filename templates with metadata variables
- Complex organization patterns

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Understanding of async/await patterns
- Basic knowledge of audio file formats and metadata

### Project Structure

Create your plugin in the appropriate location:

```
your-plugin/
├── __init__.py
├── plugin.py          # Main plugin implementation
├── config.py          # Configuration schema (optional)
└── tests/             # Plugin tests
    └── test_plugin.py
```

### Basic Plugin Template

All plugins follow this basic structure:

```python
from typing import Dict, Any, Optional, List
from pathlib import Path

from music_organizer.plugins.base import PluginType, PluginInfo
from music_organizer.plugins.base import [PluginType]Plugin  # e.g., MetadataPlugin
from music_organizer.models.audio_file import AudioFile

class YourPlugin([PluginType]Plugin):
    """Your plugin description."""

    @property
    def info(self) -> PluginInfo:
        """Return plugin information."""
        return PluginInfo(
            name="your-plugin",
            version="1.0.0",
            description="Description of your plugin",
            author="Your Name",
            dependencies=["requests"],  # Optional external dependencies
        )

    def initialize(self) -> None:
        """Initialize the plugin."""
        # Setup resources, connections, etc.
        pass

    def cleanup(self) -> None:
        """Cleanup resources when plugin is disabled."""
        # Clean up resources, close connections, etc.
        pass

    # Implement type-specific abstract methods
    async def [type_specific_method](self, ...):
        """Implement your plugin's core functionality."""
        pass
```

## Plugin Development Tutorial

Let's create a simple metadata plugin that adds a "processed" tag to audio files.

### Step 1: Create the Plugin File

```python
# simple_tagger.py
from typing import Dict, Any, Optional
from datetime import datetime

from music_organizer.plugins.base import MetadataPlugin, PluginInfo
from music_organizer.models.audio_file import AudioFile

class SimpleTaggerPlugin(MetadataPlugin):
    """A simple plugin that adds a processed timestamp to audio files."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="simple-tagger",
            version="1.0.0",
            description="Adds a processed timestamp to audio files",
            author="Your Name"
        )

    def initialize(self) -> None:
        """Initialize the plugin."""
        self.processed_count = 0
        print(f"Initialized {self.info.name} plugin")

    def cleanup(self) -> None:
        """Cleanup resources."""
        print(f"Processed {self.processed_count} files")

    async def enhance_metadata(self, audio_file: AudioFile) -> AudioFile:
        """Add processed timestamp to audio file."""
        # Create a copy to avoid modifying the original
        enhanced_file = audio_file.copy()

        # Add custom metadata
        enhanced_file.metadata['processed'] = datetime.now().isoformat()
        enhanced_file.metadata['processed_by'] = self.info.name

        self.processed_count += 1
        return enhanced_file
```

### Step 2: Create Configuration Schema (Optional)

```python
# config.py
from typing import Dict, Any

def get_config_schema() -> Dict[str, Any]:
    """Return the configuration schema for this plugin."""
    return {
        "type": "object",
        "properties": {
            "tag_name": {
                "type": "string",
                "default": "processed",
                "description": "Name of the tag to add"
            },
            "include_timestamp": {
                "type": "boolean",
                "default": True,
                "description": "Whether to include timestamp"
            }
        }
    }

def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "tag_name": "processed",
        "include_timestamp": True
    }
```

### Step 3: Register the Plugin

Create an `__init__.py` file that exports your plugin:

```python
# __init__.py
from .simple_tagger import SimpleTaggerPlugin
from .config import get_config_schema, get_default_config

# Export the plugin factory
def create_plugin(config=None):
    return SimpleTaggerPlugin(config)

__all__ = ['create_plugin', 'get_config_schema', 'get_default_config']
```

## Configuration System

Plugins can accept configuration through the `config` parameter in their constructor. The configuration system supports:

### Configuration Schema

Define your configuration schema using JSON Schema format:

```python
def get_config_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "api_key": {
                "type": "string",
                "description": "API key for external service"
            },
            "timeout": {
                "type": "integer",
                "minimum": 1,
                "maximum": 300,
                "default": 30,
                "description": "Request timeout in seconds"
            },
            "enabled_features": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["basic"],
                "description": "Features to enable"
            }
        },
        "required": ["api_key"]
    }
```

### Accessing Configuration

```python
class YourPlugin(MetadataPlugin):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = self.config.get('api_key')
        self.timeout = self.config.get('timeout', 30)
        self.enabled_features = self.config.get('enabled_features', ['basic'])
```

## Best Practices

### 1. Error Handling

Always handle errors gracefully and provide meaningful error messages:

```python
import logging

logger = logging.getLogger(__name__)

async def enhance_metadata(self, audio_file: AudioFile) -> AudioFile:
    try:
        # Your enhancement logic here
        enhanced_file = await self._fetch_metadata(audio_file)
        return enhanced_file
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch metadata for {audio_file.path}: {e}")
        return audio_file  # Return original file on error
    except Exception as e:
        logger.error(f"Unexpected error in {self.info.name}: {e}")
        raise
```

### 2. Performance Optimization

- Implement batch processing when possible
- Use async/await for I/O operations
- Cache results when appropriate

```python
from functools import lru_cache
import asyncio

class OptimizedPlugin(MetadataPlugin):
    def __init__(self, config=None):
        super().__init__(config)
        self._cache = {}

    async def batch_enhance(self, audio_files: List[AudioFile]) -> List[AudioFile]:
        """Optimized batch processing."""
        # Group files by artist for batch API calls
        by_artist = {}
        for file in audio_files:
            artist = file.metadata.get('artist', 'Unknown')
            by_artist.setdefault(artist, []).append(file)

        # Process in parallel
        tasks = [self._process_artist_group(files) for files in by_artist.values()]
        results = await asyncio.gather(*tasks)

        # Flatten results
        return [file for group in results for file in group]
```

### 3. Resource Management

Properly initialize and cleanup resources:

```python
class ResourceAwarePlugin(MetadataPlugin):
    def initialize(self) -> None:
        """Initialize resources."""
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': f'{self.info.name}/{self.info.version}'})

    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'session'):
            self.session.close()
```

### 4. Logging

Use appropriate logging levels:

```python
import logging

logger = logging.getLogger(__name__)

async def process_file(self, audio_file: AudioFile) -> AudioFile:
    logger.debug(f"Processing file: {audio_file.path}")

    if not self._should_process(audio_file):
        logger.info(f"Skipping file: {audio_file.path}")
        return audio_file

    try:
        result = await self._do_processing(audio_file)
        logger.info(f"Successfully processed: {audio_file.path}")
        return result
    except Exception as e:
        logger.error(f"Failed to process {audio_file.path}: {e}")
        raise
```

### 5. Type Hints

Always include type hints for better IDE support and documentation:

```python
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

def process_with_retry(
    self,
    audio_file: AudioFile,
    max_retries: int = 3
) -> Union[AudioFile, None]:
    """Process file with retry logic."""
    pass
```

## Testing Your Plugins

### Unit Testing

Create comprehensive tests for your plugin:

```python
import pytest
from unittest.mock import Mock, patch

from your_plugin import YourPlugin
from music_organizer.models.audio_file import AudioFile

@pytest.fixture
def plugin():
    config = {"api_key": "test_key"}
    return YourPlugin(config)

@pytest.fixture
def sample_audio_file():
    return AudioFile(
        path=Path("/test/song.mp3"),
        metadata={
            "title": "Test Song",
            "artist": "Test Artist",
            "album": "Test Album"
        }
    )

@pytest.mark.asyncio
async def test_enhance_metadata(plugin, sample_audio_file):
    """Test metadata enhancement."""
    result = await plugin.enhance_metadata(sample_audio_file)

    assert result.metadata.get("enhanced") is True
    assert result.metadata.get("enhanced_by") == plugin.info.name

@pytest.mark.asyncio
async def test_batch_processing(plugin, sample_audio_file):
    """Test batch processing."""
    files = [sample_audio_file, sample_audio_file.copy()]
    results = await plugin.batch_enhance(files)

    assert len(results) == 2
    for result in results:
        assert result.metadata.get("enhanced") is True

def test_plugin_info(plugin):
    """Test plugin information."""
    info = plugin.info
    assert info.name == "your-plugin"
    assert info.version == "1.0.0"
    assert "description" in info.description

def test_initialization(plugin):
    """Test plugin initialization."""
    plugin.initialize()
    # Add assertions for initialization state

def test_cleanup(plugin):
    """Test plugin cleanup."""
    plugin.initialize()
    plugin.cleanup()
    # Add assertions for cleanup state
```

### Integration Testing

Test your plugin with the PluginManager:

```python
import pytest
from music_organizer.plugins.manager import PluginManager

@pytest.mark.asyncio
async def test_plugin_integration():
    """Test plugin integration with PluginManager."""
    manager = PluginManager()

    # Load your plugin
    await manager.load_plugin("your_plugin_module")

    # Test plugin discovery and execution
    plugins = manager.get_plugins_by_type("metadata")
    assert len(plugins) > 0

    # Test actual processing
    test_file = AudioFile(Path("/test/song.mp3"), {})
    enhanced = await manager.process_metadata(test_file)

    assert enhanced.metadata != test_file.metadata
```

### Mock External Services

When testing plugins that use external APIs, mock the services:

```python
import pytest
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
@patch('your_plugin.requests.get')
async def test_external_api_integration(mock_get, plugin, sample_audio_file):
    """Test integration with external API."""
    # Mock API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "metadata": {"genre": "Rock", "year": 2023}
    }
    mock_get.return_value = mock_response

    result = await plugin.enhance_metadata(sample_audio_file)

    assert result.metadata.get("genre") == "Rock"
    mock_get.assert_called_once()
```

## Plugin Examples

### Example 1: Genre Classification Plugin

```python
import asyncio
from typing import Dict, Any, List
from music_organizer.plugins.base import ClassificationPlugin, PluginInfo
from music_organizer.models.audio_file import AudioFile

class GenreClassifierPlugin(ClassificationPlugin):
    """Classifies music by genre using keyword analysis."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="genre-classifier",
            version="1.0.0",
            description="Classifies music by genre using keyword analysis",
            author="Your Name"
        )

    def initialize(self) -> None:
        self.genre_keywords = {
            "rock": ["guitar", "drum", "electric", "heavy"],
            "jazz": ["saxophone", "trumpet", "swing", "improv"],
            "classical": ["orchestra", "symphony", "violin", "piano"],
            "electronic": ["synth", "beat", "digital", "loop"]
        }

    def cleanup(self) -> None:
        pass

    def get_supported_tags(self) -> List[str]:
        return ["genre", "confidence"]

    async def classify(self, audio_file: AudioFile) -> Dict[str, Any]:
        metadata = audio_file.metadata
        text_to_analyze = " ".join([
            metadata.get("title", ""),
            metadata.get("artist", ""),
            metadata.get("album", ""),
            metadata.get("comment", "")
        ]).lower()

        genre_scores = {}
        for genre, keywords in self.genre_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_to_analyze)
            if score > 0:
                genre_scores[genre] = score / len(keywords)

        if genre_scores:
            best_genre = max(genre_scores, key=genre_scores.get)
            confidence = genre_scores[best_genre]
            return {
                "genre": best_genre,
                "confidence": confidence,
                "all_scores": genre_scores
            }

        return {"genre": "unknown", "confidence": 0.0}
```

### Example 2: CSV Export Plugin

```python
import csv
from typing import List
from pathlib import Path
from music_organizer.plugins.base import OutputPlugin, PluginInfo
from music_organizer.models.audio_file import AudioFile

class CSVExporterPlugin(OutputPlugin):
    """Exports audio file metadata to CSV format."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="csv-exporter",
            version="1.0.0",
            description="Exports audio file metadata to CSV format",
            author="Your Name"
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def get_supported_formats(self) -> List[str]:
        return ["csv"]

    def get_file_extension(self) -> str:
        return "csv"

    async def export(self, audio_files: List[AudioFile], output_path: Path) -> bool:
        try:
            output_path = output_path.with_suffix(".csv")

            # Define CSV headers
            headers = [
                "path", "title", "artist", "album", "year", "genre",
                "duration", "bitrate", "format"
            ]

            # Add any custom metadata fields
            all_metadata = set()
            for file in audio_files:
                all_metadata.update(file.metadata.keys())
            headers.extend(sorted(all_metadata - set(headers)))

            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()

                for file in audio_files:
                    row = {
                        "path": str(file.path),
                        "title": file.metadata.get("title", ""),
                        "artist": file.metadata.get("artist", ""),
                        "album": file.metadata.get("album", ""),
                        "year": file.metadata.get("year", ""),
                        "genre": file.metadata.get("genre", ""),
                        "duration": file.duration,
                        "bitrate": file.bitrate,
                        "format": file.format
                    }

                    # Add custom metadata
                    for key in all_metadata:
                        if key not in row:
                            row[key] = file.metadata.get(key, "")

                    writer.writerow(row)

            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
```

### Example 3: Custom Path Plugin

```python
import re
from typing import List
from pathlib import Path
from music_organizer.plugins.base import PathPlugin, PluginInfo
from music_organizer.models.audio_file import AudioFile

class SmartPathPlugin(PathPlugin):
    """Generates smart organization paths based on metadata analysis."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="smart-path",
            version="1.0.0",
            description="Generates smart organization paths based on metadata",
            author="Your Name"
        )

    def initialize(self) -> None:
        self.special_chars_pattern = re.compile(r'[<>:"/\\|?*]')

    def cleanup(self) -> None:
        pass

    def _sanitize_name(self, name: str) -> str:
        """Remove or replace special characters in names."""
        return self.special_chars_pattern.sub('_', name).strip()

    async def generate_target_path(self, audio_file: AudioFile, base_dir: Path) -> Path:
        """Generate directory path based on metadata."""
        metadata = audio_file.metadata

        # Extract metadata values
        artist = self._sanitize_name(metadata.get("artist", "Unknown Artist"))
        album = self._sanitize_name(metadata.get("album", "Unknown Album"))
        year = metadata.get("year", "")

        # Determine if it's a compilation
        is_compilation = metadata.get("compilation", False)

        if is_compilation:
            # For compilations, organize by genre or category
            genre = self._sanitize_name(metadata.get("genre", "Compilations"))
            target_path = base_dir / "Compilations" / genre
        elif year and year.isdigit():
            # Organize by decade
            decade = f"{year[:3]}0s"
            target_path = base_dir / decade / artist / album
        else:
            # Default organization
            target_path = base_dir / artist / album

        return target_path

    async def generate_filename(self, audio_file: AudioFile) -> str:
        """Generate filename with track number and title."""
        metadata = audio_file.metadata
        track_num = metadata.get("track_number", "")
        title = self._sanitize_name(metadata.get("title", "Unknown Title"))

        if track_num:
            # Pad track number to 2 digits
            track_str = f"{int(track_num):02d} - "
        else:
            track_str = ""

        filename = f"{track_str}{title}{audio_file.path.suffix}"
        return filename

    def get_supported_variables(self) -> List[str]:
        variables = super().get_supported_variables()
        variables.extend([
            "is_compilation", "decade", "sanitized_artist",
            "sanitized_album", "sanitized_title"
        ])
        return variables
```

## Distribution and Sharing

### Packaging Your Plugin

Create a `setup.py` or `pyproject.toml` for your plugin:

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "music-organizer-your-plugin"
version = "1.0.0"
description="Your plugin description"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
dependencies = [
    "music-organizer>=1.0.0",
]

[project.entry-points."music_organizer.plugins"]
your_plugin = "your_plugin:create_plugin"
```

### Publishing

Publish your plugin to PyPI:

```bash
python -m build
python -m twine upload dist/*
```

## Getting Help

- Check the [music-organizer documentation](https://github.com/your-repo/music-organizer)
- Look at existing plugins in the `src/music_organizer/plugins/builtins/` directory
- Join our community discussions for plugin development support

Happy plugin development! 🎵