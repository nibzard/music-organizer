# API Reference

Complete API documentation for the music-organizer library.

## Table of Contents

1. [Core Organizer API](#core-organizer-api)
2. [AudioFile Model](#audiofile-model)
3. [CLI Commands](#cli-commands)
4. [CQRS API](#cqrs-api)
5. [Plugin API](#plugin-api)
6. [Configuration API](#configuration-api)

---

## Core Organizer API

### EnhancedAsyncMusicOrganizer

Main orchestration class for music organization with operation history tracking.

**Location:** `src/music_organizer/core/enhanced_async_organizer.py`

#### Constructor

```python
EnhancedAsyncMusicOrganizer(
    config: Config,
    dry_run: bool = False,
    interactive: bool = False,
    max_workers: int = 4,
    use_cache: bool = True,
    cache_ttl: Optional[int] = None,
    enable_parallel_extraction: bool = True,
    use_processes: bool = False,
    use_smart_cache: bool = False,
    session_id: Optional[str] = None,
    history_tracker: Optional[OperationHistoryTracker] = None,
    enable_operation_history: bool = True
)
```

**Parameters:**
- `config` - Configuration object (see [Configuration API](#configuration-api))
- `dry_run` - Simulate operations without making changes
- `interactive` - Enable interactive prompts for ambiguous cases
- `max_workers` - Maximum number of worker threads (default: 4)
- `use_cache` - Enable metadata caching (default: True)
- `cache_ttl` - Cache TTL in days (default: 30)
- `enable_parallel_extraction` - Enable parallel metadata extraction (default: True)
- `use_processes` - Use process pool instead of thread pool (default: False)
- `use_smart_cache` - Use smart caching with adaptive TTL (default: False)
- `session_id` - Session ID for operation tracking (auto-generated if None)
- `history_tracker` - Custom history tracker
- `enable_operation_history` - Enable operation history tracking (default: True)

#### Methods

##### organize_files

```python
async def organize_files(
    source_dir: Path,
    target_dir: Path
) -> Result[Dict, MusicOrganizerError]
```

Organize files with comprehensive operation tracking.

**Returns:** `Result` containing:
- `session_id` - Unique session identifier
- `total_files` - Total files processed
- `organized_files` - Successfully organized count
- `failed_files` - Failed file count
- `metadata_extraction_failures` - Metadata extraction failures
- `organization_failures` - Organization failures
- `skipped_files` - Skipped file count
- `dry_run` - Whether this was a dry run
- `operation_history_enabled` - Whether history tracking was enabled

**Example:**
```python
from pathlib import Path
from music_organizer.core.enhanced_async_organizer import EnhancedAsyncMusicOrganizer
from music_organizer.models.config import Config

config = Config(
    source_directory=Path("~/Music/unorganized"),
    target_directory=Path("~/Music/organized")
)

organizer = EnhancedAsyncMusicOrganizer(config)
result = await organizer.organize_files(
    source_dir=Path("~/Music/unorganized"),
    target_dir=Path("~/Music/organized")
)

if result.is_success():
    stats = result.value()
    print(f"Organized {stats['organized_files']} files")
else:
    print(f"Error: {result.error()}")
```

##### get_operation_history

```python
async def get_operation_history() -> Result[List[Dict], MusicOrganizerError]
```

Get operation history for the current session.

**Returns:** List of operation records as dictionaries

##### rollback_session

```python
async def rollback_session(dry_run: bool = False) -> Result[Dict, MusicOrganizerError]
```

Rollback the current session's operations.

**Parameters:**
- `dry_run` - Simulate rollback without making changes

**Example:**
```python
# Rollback a failed organization
result = await organizer.rollback_session(dry_run=False)
```

##### list_recent_sessions

```python
async def list_recent_sessions(limit: int = 10) -> Result[List[Dict], MusicOrganizerError]
```

List recent operation sessions.

**Parameters:**
- `limit` - Maximum number of sessions to return (default: 10)

##### organize_files_bulk

```python
async def organize_files_bulk(
    source_dir: Path,
    target_dir: Path,
    bulk_config: Optional[BulkOperationConfig] = None
) -> Result[Dict, MusicOrganizerError]
```

Organize files using bulk operations with optimized performance.

**Parameters:**
- `bulk_config` - Optional bulk operation configuration

#### Context Manager Support

```python
async with EnhancedAsyncMusicOrganizer(config) as organizer:
    result = await organizer.organize_files(source_dir, target_dir)
    # Automatically cleanup resources on exit
```

---

## AudioFile Model

Core data model representing audio files with metadata.

**Location:** `src/music_organizer/models/audio_file.py`

### ContentType Enum

```python
class ContentType(Enum):
    STUDIO = "studio"
    LIVE = "live"
    COLLABORATION = "collaboration"
    COMPILATION = "compilation"
    RARITY = "rarity"
    UNKNOWN = "unknown"
```

### AudioFile Dataclass

```python
@dataclass(slots=True)
class AudioFile:
    path: Path
    file_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_type: ContentType = ContentType.UNKNOWN
    artists: List[str] = field(default_factory=list)
    primary_artist: Optional[str] = None
    album: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    date: Optional[str] = None
    location: Optional[str] = None
    track_number: Optional[int] = None
    genre: Optional[str] = None
    has_cover_art: bool = False
```

#### Properties

##### filename
```python
@property
def filename(self) -> str
```
Get the filename without path.

##### extension
```python
@property
def extension(self) -> str
```
Get the file extension (lowercase, without dot).

##### size_mb
```python
@property
def size_mb(self) -> float
```
Get file size in megabytes.

#### Methods

##### get_display_name
```python
def get_display_name(self) -> str
```
Get human-readable display name (e.g., "Artist - Title").

##### get_target_path
```python
def get_target_path(base_dir: Path) -> Path
```
Generate target path based on content type and metadata.

**Example:**
```python
from music_organizer.models.audio_file import AudioFile, ContentType

audio_file = AudioFile(
    path=Path("/music/song.flac"),
    file_type="FLAC",
    artists=["Artist Name"],
    album="Album Name",
    year=2023,
    title="Song Title",
    content_type=ContentType.STUDIO
)

target_path = audio_file.get_target_path(Path("/organized"))
# Result: /organized/Albums/Artist Name/Album Name (2023)
```

##### get_target_filename
```python
def get_target_filename(self) -> str
```
Generate target filename with track number padding.

##### from_path
```python
@classmethod
def from_path(cls, path: Path) -> "AudioFile"
```
Create AudioFile instance from file path.

**Example:**
```python
from pathlib import Path
from music_organizer.models.audio_file import AudioFile

audio_file = AudioFile.from_path(Path("/music/song.flac"))
```

### CoverArt Dataclass

```python
@dataclass(slots=True)
class CoverArt:
    path: Path
    type: str  # 'front', 'back', 'disc', etc.
    format: str  # 'jpg', 'png', etc.
    size: int  # Size in bytes
```

#### from_path
```python
@classmethod
def from_file(cls, path: Path) -> Optional["CoverArt"]
```
Create CoverArt instance from image file.

---

## CLI Commands

Command-line interface for music organization operations.

**Location:** `src/music_organizer/cli.py`

### Main Commands

#### organize

Organize music files from source to target directory.

```bash
music-organize organize SOURCE TARGET [OPTIONS]
```

**Options:**
- `--config PATH` - Configuration file path
- `--dry-run` - Show what would be done without making changes
- `--interactive` - Prompt for ambiguous categorizations
- `--backup/--no-backup` - Create backup before reorganization (default: enabled)
- `--verbose` - Verbose output
- `--incremental` - Only process new or modified files
- `--force-full-scan` - Force full scan instead of incremental
- `--workers N` - Number of worker threads (default: 4)

**Example:**
```bash
# Dry run to preview changes
music-organize organize ~/Music/unorganized ~/Music/organized --dry-run

# Incremental organization with 8 workers
music-organize organize ~/Music/unorganized ~/Music/organized --incremental --workers 8

# Interactive mode for ambiguous cases
music-organize organize ~/Music/unorganized ~/Music/organized --interactive
```

#### scan

Analyze music library statistics.

```bash
music-organize scan DIRECTORY [OPTIONS]
```

**Options:**
- `--recursive` - Scan subdirectories recursively (default: True)

**Example:**
```bash
music-organize scan ~/Music --recursive
```

#### inspect

Inspect metadata of a single audio file.

```bash
music-organize inspect FILE_PATH
```

**Example:**
```bash
music-organize inspect ~/Music/song.flac
```

#### validate

Validate music directory organization structure.

```bash
music-organize validate DIRECTORY
```

**Example:**
```bash
music-organize validate ~/Music/organized
```

### Programmatic CLI Usage

```python
from music_organizer.cli import organize_command_async
import argparse

args = argparse.Namespace(
    source=Path("~/Music/unorganized"),
    target=Path("~/Music/organized"),
    dry_run=True,
    interactive=False,
    backup=True,
    workers=4,
    incremental=True,
    force_full_scan=False,
    config=None,
    verbose=False
)

exit_code = await organize_command_async(args)
```

---

## CQRS API

Command-Query Responsibility Segregation pattern implementation.

### Commands API

**Location:** `src/music_organizer/application/commands/`

#### Base Command Class

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    command_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### CommandHandler

```python
class CommandHandler(ABC, Generic[C, R]):
    @abstractmethod
    async def handle(self, command: C) -> R:
        """Handle the command and return a result."""
        pass

    @abstractmethod
    def can_handle(self, command_type: type) -> bool:
        """Check if this handler can handle the given command type."""
        pass
```

#### CommandBus

```python
class CommandBus:
    def register(self, command_type: type, handler: CommandHandler) -> None:
        """Register a handler for a command type."""

    def register_middleware(self, middleware: Callable) -> None:
        """Register middleware for command processing pipeline."""

    async def dispatch(self, command: Command) -> CommandResult:
        """Dispatch a command to its registered handler."""
```

**Example:**
```python
from music_organizer.application.commands.base import Command, CommandBus, CommandHandler, CommandResult

class OrganizeFileCommand(Command):
    source_path: Path
    target_path: Path
    conflict_strategy: str = "number"

class OrganizeFileHandler(CommandHandler):
    async def handle(self, command: OrganizeFileCommand) -> CommandResult:
        # Handle organization logic
        return CommandResult(success=True, command_id=command.command_id)

    def can_handle(self, command_type: type) -> bool:
        return command_type == OrganizeFileCommand

# Usage
bus = CommandBus()
bus.register(OrganizeFileCommand, OrganizeFileHandler())
result = await bus.dispatch(OrganizeFileCommand(source_path=..., target_path=...))
```

#### CommandResult

```python
@dataclass(slots=True)
class CommandResult:
    success: bool
    command_id: str
    message: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    events: List[DomainEvent] = field(default_factory=list)
    result_data: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None
```

### Queries API

**Location:** `src/music_organizer/application/queries/`

#### Base Query Class

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Query:
    query_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    include_metadata: bool = False
    cache_key: Optional[str] = None
    cache_ttl_seconds: int = 300
```

#### QueryHandler

```python
class QueryHandler(ABC, Generic[Q, R]):
    @abstractmethod
    async def handle(self, query: Q) -> R:
        """Handle the query and return results."""
        pass

    @abstractmethod
    def can_handle(self, query_type: type) -> bool:
        """Check if this handler can handle the given query type."""
        pass
```

#### QueryBus

```python
class QueryBus:
    def register(self, query_type: type, handler: QueryHandler) -> None:
        """Register a handler for a query type."""

    def register_middleware(self, middleware: Callable) -> None:
        """Register middleware for query processing pipeline."""

    def set_cache(self, cache: QueryCache) -> None:
        """Set the query cache implementation."""

    async def dispatch(self, query: Query) -> QueryResult:
        """Dispatch a query to its registered handler."""
```

**Example:**
```python
from music_organizer.application.queries.base import Query, QueryBus, QueryHandler, QueryResult

class GetRecordingsQuery(Query):
    artist_name: str
    include_collaborations: bool = True

class GetRecordingsHandler(QueryHandler):
    async def handle(self, query: GetRecordingsQuery) -> QueryResult:
        # Query database/service
        recordings = await self.repository.find_by_artist(query.artist_name)
        return QueryResult(data=recordings, success=True, query_id=query.query_id)

    def can_handle(self, query_type: type) -> bool:
        return query_type == GetRecordingsQuery

# Usage
bus = QueryBus()
bus.register(GetRecordingsQuery, GetRecordingsHandler())
result = await bus.dispatch(GetRecordingsQuery(artist_name="Artist Name"))
```

#### QueryResult

```python
@dataclass(frozen=True, slots=True)
class QueryResult(Generic[R]):
    data: Optional[R] = None
    success: bool = True
    query_id: str = ""
    message: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    from_cache: bool = False
    execution_time_ms: Optional[float] = None
    cached_at: Optional[datetime] = None
    total_count: Optional[int] = None
```

---

## Plugin API

Extend functionality through plugins.

**Location:** `src/music_organizer/plugins/base.py`

### Base Plugin Classes

#### Plugin

```python
class Plugin(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize plugin with optional configuration."""

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Return plugin information."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the plugin."""

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources when plugin is disabled."""

    def enable(self) -> None:
        """Enable the plugin."""

    def disable(self) -> None:
        """Disable the plugin."""
```

#### PluginInfo

```python
@dataclass(slots=True)
class PluginInfo:
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    min_python_version: str = "3.9"
```

### Plugin Types

#### MetadataPlugin

Enhance metadata for audio files.

```python
class MetadataPlugin(Plugin):
    @abstractmethod
    async def enhance_metadata(self, audio_file: AudioFile) -> AudioFile:
        """Enhance metadata for an audio file."""

    async def batch_enhance(self, audio_files: List[AudioFile]) -> List[AudioFile]:
        """Enhance metadata for multiple audio files."""
```

**Example:**
```python
from music_organizer.plugins.base import MetadataPlugin, PluginInfo

class MusicBrainzEnricher(MetadataPlugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="musicbrainz-enricher",
            version="1.0.0",
            description="Fetch metadata from MusicBrainz",
            author="Your Name"
        )

    def initialize(self) -> None:
        self.api_key = self.config.get("api_key")

    def cleanup(self) -> None:
        pass

    async def enhance_metadata(self, audio_file: AudioFile) -> AudioFile:
        # Fetch from MusicBrainz API
        enhanced = audio_file.copy()
        enhanced.metadata["musicbrainz_id"] = await self._fetch_id(audio_file)
        return enhanced
```

#### ClassificationPlugin

Classify audio files with custom tags.

```python
class ClassificationPlugin(Plugin):
    @abstractmethod
    async def classify(self, audio_file: AudioFile) -> Dict[str, Any]:
        """Classify an audio file."""

    @abstractmethod
    def get_supported_tags(self) -> List[str]:
        """Return list of classification tags."""

    async def batch_classify(self, audio_files: List[AudioFile]) -> List[Dict[str, Any]]:
        """Classify multiple audio files."""
```

**Example:**
```python
from music_organizer.plugins.base import ClassificationPlugin, PluginInfo

class GenreClassifier(ClassificationPlugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="genre-classifier",
            version="1.0.0",
            description="Classify music by genre",
            author="Your Name"
        )

    def initialize(self) -> None:
        self.genres = self.config.get("genres", ["rock", "jazz", "electronic"])

    def cleanup(self) -> None:
        pass

    async def classify(self, audio_file: AudioFile) -> Dict[str, Any]:
        return {"genre": self._detect_genre(audio_file)}

    def get_supported_tags(self) -> List[str]:
        return ["genre", "confidence"]
```

#### OutputPlugin

Export audio files to various formats.

```python
class OutputPlugin(Plugin):
    @abstractmethod
    async def export(self, audio_files: List[AudioFile], output_path: Path) -> bool:
        """Export audio files to specified format/location."""

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported export formats."""

    @abstractmethod
    def get_file_extension(self) -> str:
        """Return the file extension for this export format."""
```

**Example:**
```python
from music_organizer.plugins.base import OutputPlugin, PluginInfo

class JSONExporter(OutputPlugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="json-exporter",
            version="1.0.0",
            description="Export metadata to JSON",
            author="Your Name"
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    async def export(self, audio_files: List[AudioFile], output_path: Path) -> bool:
        import json
        data = [af.to_dict() for af in audio_files]
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(data, f, indent=2)
        return True

    def get_supported_formats(self) -> List[str]:
        return ["json"]

    def get_file_extension(self) -> str:
        return "json"
```

#### PathPlugin

Generate custom file organization patterns.

```python
class PathPlugin(Plugin):
    @abstractmethod
    async def generate_target_path(self, audio_file: AudioFile, base_dir: Path) -> Path:
        """Generate target directory path for an audio file."""

    @abstractmethod
    async def generate_filename(self, audio_file: AudioFile) -> str:
        """Generate filename for an audio file."""

    def get_supported_variables(self) -> List[str]:
        """Return list of supported template variables."""

    async def batch_generate_paths(self, audio_files: List[AudioFile], base_dir: Path) -> List[Path]:
        """Generate paths for multiple audio files."""
```

**Example:**
```python
from music_organizer.plugins.base import PathPlugin, PluginInfo

class DecadeOrganizer(PathPlugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="decade-organizer",
            version="1.0.0",
            description="Organize by decade",
            author="Your Name"
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    async def generate_target_path(self, audio_file: AudioFile, base_dir: Path) -> Path:
        year = audio_file.year or 0
        decade = f"{year // 10 * 10}s"
        artist = audio_file.primary_artist or "Unknown"
        return base_dir / decade / artist

    async def generate_filename(self, audio_file: AudioFile) -> str:
        return audio_file.get_target_filename()
```

---

## Configuration API

Configuration management for the music organizer.

**Location:** `src/music_organizer/models/config.py`

### Config Dataclass

```python
@dataclass
class Config:
    source_directory: Path
    target_directory: Path
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    naming: NamingConfig = field(default_factory=NamingConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    file_operations: FileOperationsConfig = field(default_factory=FileOperationsConfig)
```

### Sub-Configurations

#### DirectoryConfig

```python
@dataclass
class DirectoryConfig:
    albums: str = "Albums"
    live: str = "Live"
    collaborations: str = "Collaborations"
    compilations: str = "Compilations"
    rarities: str = "Rarities"
```

#### NamingConfig

```python
@dataclass
class NamingConfig:
    album_format: str = "{artist}/{album} ({year})"
    live_format: str = "{artist}/{date} - {location}"
    collab_format: str = "{album} ({year}) - {artists}"
    compilation_format: str = "{artist}/{album} ({year})"
    rarity_format: str = "{artist}/{album} ({edition})"
```

#### MetadataConfig

```python
@dataclass
class MetadataConfig:
    enhance: bool = True
    musicbrainz: bool = True
    fix_capitalization: bool = True
    standardize_genres: bool = True
```

#### FileOperationsConfig

```python
@dataclass
class FileOperationsConfig:
    strategy: str = "move"  # "copy" or "move"
    backup: bool = True
    handle_duplicates: str = "number"  # "number", "skip", or "overwrite"
```

### Configuration Functions

#### load_config

```python
def load_config(config_path: Path) -> Config
```

Load configuration from JSON file.

**Example:**
```python
from music_organizer.models.config import load_config

config = load_config(Path("~/.config/music-organizer/config.json"))
```

#### save_config

```python
def save_config(config: Config, config_path: Path) -> None
```

Save configuration to JSON file.

**Example:**
```python
from music_organizer.models.config import Config, save_config

config = Config(
    source_directory=Path("~/Music/unorganized"),
    target_directory=Path("~/Music/organized")
)
save_config(config, Path("~/.config/music-organizer/config.json"))
```

#### create_default_config

```python
def create_default_config(config_path: Path) -> None
```

Create a default configuration file.

**Example:**
```python
from music_organizer.models.config import create_default_config

create_default_config(Path("~/.config/music-organizer/config.json"))
```

### Configuration JSON Example

```json
{
  "source_directory": "/path/to/source",
  "target_directory": "/path/to/target",
  "directories": {
    "albums": "Albums",
    "live": "Live",
    "collaborations": "Collaborations",
    "compilations": "Compilations",
    "rarities": "Rarities"
  },
  "naming": {
    "album_format": "{artist}/{album} ({year})",
    "live_format": "{artist}/{date} - {location}",
    "collab_format": "{album} ({year}) - {artists}",
    "compilation_format": "{artist}/{album} ({year})",
    "rarity_format": "{artist}/{album} ({edition})"
  },
  "metadata": {
    "enhance": true,
    "musicbrainz": true,
    "fix_capitalization": true,
    "standardize_genres": true
  },
  "file_operations": {
    "strategy": "move",
    "backup": true,
    "handle_duplicates": "number"
  }
}
```

---

## Type Reference

### Result Type

Used throughout the API for error handling.

```python
from music_organizer.domain.result import Result, Success, Failure

# Success case
result = Success({"files": 100, "moved": 95})

# Failure case
result = Failure("Failed to process files")

# Usage
if result.is_success():
    data = result.value()
else:
    error = result.error()
```

---

## Additional Resources

- [Plugin Development Guide](plugin-development.md) - Complete plugin development tutorial
- [CQRS Implementation](cqrs-implementation.md) - CQRS architecture details
- [Rollback System](rollback-system.md) - Operation history and rollback
- [Interactive Duplicate Resolution](interactive-duplicate-resolution.md) - Duplicate handling
