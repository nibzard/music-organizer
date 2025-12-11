# 🎯 Music Organizer Development Roadmap

> **Guiding Principles**: Ultra-fast core, minimal dependencies, Pythonic simplicity, magical UX

## 📋 Phase 1: Foundation & Performance (Week 1-2)

### Core Optimizations
- [x] ✅ Implement async I/O with ThreadPoolExecutor for file operations (scan_directory, move_files)
- [x] ✅ Add SQLite metadata caching system with TTL (cache unchanged files for 30 days)
- [x] ✅ Create streaming pipeline architecture to process files in batches (memory-efficient)
- [x] ✅ Implement zero-copy data structures using @dataclass(slots=True) for AudioFile
- [x] ✅ Add intelligent progress tracking with real-time metrics (files/sec, ETA)

### Minimal Dependency Approach
- [x] ✅ Refactor to use only mutagen as external dependency (remove click, rich, pydantic)
- [x] ✅ Implement custom CLI using argparse with rich progress simulation
- [x] ✅ Create built-in progress bars and tables using only standard library
- [x] ✅ Add manual configuration handling (YAML/JSON) without external libraries

### Performance Metrics
- [x] ✅ Implement performance benchmarks and CI integration
- [x] ✅ Add memory usage monitoring and optimization targets
- [x] ✅ Create performance regression tests

## 📋 Phase 2: Plugin Architecture (Week 3-4)

### Plugin System Foundation
- [x] ✅ Design Plugin interface and base classes (MetadataPlugin, ClassificationPlugin, OutputPlugin)
- [x] ✅ Implement PluginManager with automatic discovery from plugins/ directory
- [x] ✅ Create hook system for plugin integration (pre/post processing events)
- [x] ✅ Add plugin configuration system with validation

### Core Plugins
- [x] ✅ Develop MusicBrainz metadata enhancement plugin
- [x] ✅ Create duplicate detection plugin with audio fingerprinting
- [x] ✅ Implement playlist export plugin (M3U, PLS formats)
- [x] ✅ Add custom naming pattern plugin for user-defined organization rules

### Plugin Examples
- [x] ✅ Create example plugin demonstrating classification rules
- [x] ✅ Write plugin development guide and templates
- [x] ✅ Add plugin testing utilities and mock framework

**Plugin Development Framework Completed:**
- ✅ Comprehensive plugin development documentation (docs/plugin-development.md)
- ✅ Plugin templates for all 4 plugin types (metadata, classification, output, path)
- ✅ Complete testing framework with mocks, fixtures, validators, and performance profiling
- ✅ 28+ unit tests covering all aspects of plugin development
- ✅ Ready-to-use plugin factory functions and validation utilities

## 📋 Phase 3: Domain-Driven Refactoring (Week 5-6)

### Domain Models
- [x] ✅ Define core domain entities (AudioLibrary, Collection, Release, Recording)
- [x] ✅ Implement bounded contexts (Catalog, Organization, Classification)
- [x] ✅ Create domain services for business logic
- [x] ✅ Add value objects for domain primitives (AudioPath, Metadata)

### Architecture Improvements
- [x] ✅ Implement anti-corruption layers for external dependencies
- [x] ✅ Add event-driven architecture for loose coupling
- [x] ✅ Create repository pattern for data access
- [x] ✅ Implement command/query separation (CQRS) where appropriate

### Code Organization
- [x] ✅ Refactor package structure to reflect domain boundaries
- [x] ✅ Add comprehensive type hints with Protocol for duck typing
- [ ] 🟡 Implement proper error handling with Result pattern
- [x] ✅ Create domain-specific exception hierarchy

## 📋 Phase 4: Advanced Features (Week 7-8)

### Performance Enhancements
- [ ] 🟡 Implement incremental scanning (only process new/modified files)
- [ ] 🟡 Add parallel metadata extraction with worker pools
- [ ] 🟡 Optimize file operations with bulk moves/copies
- [ ] 🟢 Add smart caching strategy based on file modification times

### User Experience
- [ ] 🟡 Implement "Magic Mode" - zero configuration organization with AI suggestions
- [ ] 🟡 Add interactive duplicate resolution with side-by-side comparison
- [ ] 🟡 Create organization preview with dry-run visualization
- [ ] 🟢 Implement rollback system with operation history

### Additional Features
- [ ] 🟡 Add support for additional audio formats (OGG, OPUS, WMA)
- [ ] 🟡 Implement organization rules engine with regex patterns
- [ ] 🟡 Create statistics dashboard with library insights
- [ ] 🟢 Add batch operations (bulk tagging, metadata updates)

## 📋 Phase 5: Polish & Production (Week 9-10)

### Testing & Quality
- [ ] 🔴 Achieve 95% test coverage with unit and integration tests
- [ ] 🟡 Add property-based testing for edge cases
- [ ] 🟡 Implement performance benchmarks in CI/CD
- [ ] 🟢 Add security audit for file operations

### Documentation
- [ ] 🟡 Create comprehensive API documentation with examples
- [x] ✅ Write plugin development guide with tutorials
- [ ] 🟡 Add performance tuning guide
- [ ] 🟢 Create troubleshooting FAQ

### Distribution
- [ ] 🟡 Create single-file distribution with minimal dependencies
- [ ] 🟡 Add GitHub Actions for automated releases
- [ ] 🟡 Implement auto-update mechanism
- [ ] 🟢 Prepare PyPI package with proper metadata

## 📋 Phase 6: Future Enhancements (Optional)

### Experimental Features
- [ ] 🟢 Explore Rust/Cython extensions for hot paths
- [ ] 🟢 Investigate WebAssembly for browser-based organization
- [ ] 🟢 Research ML-based classification improvements
- [ ] 🟢 Prototype cloud storage integration

### Integration
- [ ] 🟢 Add MusicBrainz integration for metadata enrichment
- [ ] 🟢 Implement Last.fm scrobbling integration
- [ ] 🟢 Create Kodi/Jellyfin compatibility mode
- [ ] 🟢 Add Spotify playlist import/export

## 📝 Recent Implementations

### 🏗️ Core Domain Entities (2024-12-11)
Implemented comprehensive domain entities following Domain-Driven Design principles:

**Domain Entities Created**:
- **Recording**: Represents individual audio tracks with identity and lifecycle
  - Core identity through AudioPath and Metadata value objects
  - Comprehensive duplicate detection with multiple strategies (hash, fingerprint, metadata similarity)
  - Genre classification and content type tracking
  - Move history and error tracking for audit trails
  - Processing status management and workflow integration

- **Release**: Groups recordings that belong to the same album/EP/single
  - Aggregate root that manages Recording lifecycle
  - Automatic track sorting by track number and title
  - Duplicate group detection with configurable similarity thresholds
  - Release metadata (type, genre, total tracks, disc information)
  - Merge operations for combining duplicate releases

- **Collection**: Curated groupings of releases representing user-defined categories
  - Hierarchical support with parent-child relationships
  - Powerful filtering capabilities (by genre, year, artist)
  - Aggregated statistics and metrics computation
  - Support for genre patterns and year ranges for automated classification

- **AudioLibrary**: Root aggregate representing the complete music library
  - Global duplicate detection across all recordings and collections
  - Comprehensive library statistics and insights
  - Recently added tracking with configurable time windows
  - Multiple duplicate resolution strategies (skip, rename, replace, keep both)

**Key Features**:
- **Rich Domain Logic**: Business rules embedded within entities
- **Identity & Lifecycle**: Proper entity lifecycle with clear aggregate boundaries
- **Value Objects Integration**: Seamless use of existing AudioPath, ArtistName, etc.
- **Performance Optimized**: Efficient O(n²) duplicate detection with early termination
- **Type Safety**: Complete type hints with Optional and Union types

**Duplicate Detection Algorithm**:
```python
# Multiple strategies with fallbacks
1. Exact file hash match (100% similarity)
2. Acoustic fingerprint match (95% similarity)
3. Metadata-based similarity (weighted scoring):
   - Title similarity (40% weight)
   - Artist similarity (35% weight)
   - Album similarity (15% weight)
   - Duration similarity (10% weight)
```

**Usage Examples**:
```python
# Create library with collection
library = AudioLibrary(name="My Music", root_path=Path("/music"))

# Add recordings to releases
release = Release(title="Abbey Road", primary_artist=ArtistName("The Beatles"))
recording = Recording(path=audio_path, metadata=metadata)
release.add_recording(recording)

# Create collections and organize
collection = Collection(name="1960s Rock")
collection.add_release(release)
library.add_collection(collection)

# Find duplicates library-wide
duplicates = library.find_duplicates(similarity_threshold=0.85)

# Get library statistics
stats = library.get_statistics()
# Returns format distribution, top artists, decades, etc.
```

**Key Files**:
- `src/music_organizer/domain/entities.py` - All domain entity implementations
- `tests/test_domain_entities.py` - Comprehensive test suite (70+ tests)
- `src/music_organizer/domain/__init__.py` - Updated module exports

### 🏗️ Domain Value Objects (2024-12-11)
Implemented comprehensive domain value objects following Domain-Driven Design principles:

**Value Objects Created**:
- **AudioPath**: Immutable representation of audio file paths with domain-specific operations
  - Validates file paths and formats
  - Supports all common audio formats (FLAC, MP3, MP4, M4A, WAV, AIFF, OGG, OPUS, WMA)
  - Provides size information, path operations (with_name, with_suffix, relative_to)
  - Normalizes paths for consistent comparison

- **ArtistName**: Normalized artist name with sorting support
  - Whitespace normalization and validation
  - Smart handling of articles ("The", "A", "An") for alphabetical sorting
  - Case-insensitive comparison and searching
  - First-letter extraction for categorization

- **TrackNumber**: Flexible track number parsing and formatting
  - Handles multiple formats: "5", "5/12", "05", "5 of 12"
  - Zero-padded formatting with custom width
  - Validates and normalizes track information

- **Metadata**: Immutable collection of audio metadata
  - Comprehensive metadata fields (title, artists, album, year, genre, etc.)
  - Smart inference (is_live, is_compilation, has_multiple_artists)
  - Formatted string representations
  - Hash generation for duplicate detection
  - Validation for all fields (year ranges, technical metadata constraints)

- **ContentPattern**: Pattern matching for content classification
  - Case-insensitive pattern matching
  - Priority-based classification
  - Support for multiple patterns per content type

**Key Features**:
- **Immutability**: All value objects are frozen dataclasses with slots for performance
- **Rich Domain Logic**: Domain-specific operations embedded in value objects
- **Validation**: Comprehensive validation with meaningful error messages
- **Type Safety**: Full type hints with Optional and Union types
- **Performance**: Optimized for speed with __slots__ and frozen dataclasses

**Usage Examples**:
```python
# Audio path with validation
path = AudioPath("/music/artist/album/track.flac")
assert path.format == FileFormat.FLAC
assert path.size_mb > 0

# Artist name with smart sorting
artist = ArtistName("The Beatles")
assert artist.sortable == "Beatles, The"

# Track number from various formats
track = TrackNumber("5/12")
assert track.formatted_with_total() == "05/012"

# Rich metadata with validation
meta = Metadata(
    title="Song Title",
    artists=[ArtistName("Artist Name")],
    year=2023,
    track_number=TrackNumber(5)
)
assert meta.formatted_title() == "05 Song Title"

# Pattern matching
pattern = ContentPattern("Live", {"live", "concert", "tour"})
assert pattern.matches("Live at Budokan")
```

**Key Files**:
- `src/music_organizer/domain/value_objects.py` - All value object implementations
- `src/music_organizer/domain/__init__.py` - Domain module exports
- `tests/test_domain_value_objects.py` - Comprehensive test suite (43 tests)

### 🚀 Minimal Dependency Refactor (2024-12-11)
Successfully removed all external dependencies except mutagen:
- **Replaced Pydantic**: Switched to dataclasses for configuration models
- **Replaced Rich**: Created custom SimpleConsole and SimpleProgress implementations
- **Replaced Click**: Migrated to argparse for CLI parsing
- **Removed YAML**: Switched to JSON for configuration files
- **Zero external dependencies**: Now only requires mutagen for audio metadata

Key files:
- `src/music_organizer/console_utils.py` - Custom console utilities
- `src/music_organizer/cli.py` - Refactored CLI using argparse
- `src/music_organizer/models/config.py` - Dataclass-based configuration

### 🎵 MusicBrainz Metadata Enhancement (2024-12-11)
Implemented comprehensive MusicBrainz integration for automatic metadata enhancement:

**Plugin Features**:
- **API Integration**: Full MusicBrainz web service API integration with configurable endpoints
- **Rate Limiting**: Respects MusicBrainz usage policies with configurable rate limiting (default 1 request/second)
- **Intelligent Caching**: In-memory caching to avoid redundant API calls for identical tracks
- **Field Enhancement**: Configurable enhancement fields including year, genre, track_number, and albumartist
- **Fuzzy Search**: Automatic fallback to fuzzy search when exact matches fail
- **Error Handling**: Graceful degradation when aiohttp is unavailable or API is unreachable
- **Batch Processing**: Optimized batch processing with rate limiting awareness

**Configuration**:
- JSON-based configuration with validation
- Customizable API endpoint, timeout, and user agent
- Selective field enhancement to control what metadata is updated
- Toggle for cache enablement and fuzzy search fallback

**Key Files**:
- `src/music_organizer/plugins/builtins/musicbrainz_enhancer.py` - Main plugin implementation
- `tests/test_musicbrainz_plugin.py` - Comprehensive test suite (11 tests)
- `config/plugins/musicbrainz_enhancer.json` - Plugin configuration

### 🏷️ Custom Naming Pattern Plugin (2024-12-11)
Implemented flexible custom naming pattern plugin for user-defined organization rules:

**Plugin Features**:
- **Template System**: Advanced template engine with variable substitution for paths and filenames
- **Variable Support**: Comprehensive set of variables including artist, album, year, track number, genre, decade, and custom computed fields
- **Conditional Sections**: Support for conditional blocks like `{if:year} ({year}){endif}` for optional metadata
- **Filesystem Safety**: Automatic cleaning of filesystem-incompatible characters
- **Pattern Validation**: Built-in validation with detailed error messages for malformed templates
- **Multiple Organization Strategies**: Support for genre-based, decade-based, artist-first-letter, and custom organization schemes
- **Content Type Awareness**: Different patterns for studio albums, live recordings, compilations, and collaborations

**Template Variables**:
- Basic: `{artist}`, `{album}`, `{year}`, `{track_number}`, `{title}`, `{genre}`
- Advanced: `{decade}`, `{first_letter}`, `{albumartist}`, `{disc_number}`, `{content_type}`
- Conditional: `{if:variable}content{endif}` blocks

**Configuration**:
- JSON-based configuration with default patterns
- Content-type specific patterns (studio, live, compilation, etc.)
- Genre and artist-specific pattern overrides
- Additional organization options (date directories, genre subfolders, decade grouping)

**Pattern Examples**:
```json
{
  "path_patterns": {
    "studio": "Albums/{artist}/{album} ({year})",
    "live": "Live/{artist}/{date} - {location}",
    "compilation": "Compilations/{albumartist}/{album} ({year})"
  },
  "filename_pattern": "{track_number} {title}{file_extension}"
}
```

**Key Files**:
- `src/music_organizer/plugins/base.py` - Added PathPlugin base class
- `src/music_organizer/plugins/builtins/custom_naming_pattern.py` - Main plugin implementation
- `config/plugins/custom_naming_pattern.json` - Plugin configuration
- `tests/test_custom_naming_pattern_plugin.py` - Comprehensive test suite (40+ tests)

### 🔄 Duplicate Detection Plugin (2024-12-11)
Implemented comprehensive duplicate detection plugin with multiple detection strategies:

**Plugin Features**:
- **Multiple Detection Strategies**: Metadata-based, exact file hash, and audio fingerprinting
- **Zero External Dependencies**: Implements fingerprinting using file properties and metadata
- **Configurable Thresholds**: Adjustable similarity thresholds for acoustic matching
- **Flexible Filtering**: Filter duplicates by type (exact, metadata, acoustic) and confidence level
- **Batch Processing**: Optimized for processing large libraries efficiently
- **Duplicate Groups**: Reports all duplicate relationships with detailed information

**Detection Strategies**:
1. **Metadata Matching**: Normalized comparison of artist, title, album, and track number
2. **File Hashing**: MD5-based detection of exact bit-for-bit duplicates
3. **Audio Fingerprinting**: SHA256-based fingerprint using file size, duration, bitrate, and metadata

**Configuration**:
- JSON-based configuration with comprehensive validation
- Selectable detection strategies
- Configurable similarity thresholds (default 85% for acoustic)
- Filtering by minimum confidence and duplicate types

**Key Files**:
- `src/music_organizer/plugins/builtins/duplicate_detector.py` - Main plugin implementation
- `tests/test_duplicate_detector_plugin.py` - Comprehensive test suite (20 tests)
- `config/plugins/duplicate_detector.json` - Plugin configuration

**Usage**:
The plugin is automatically discovered and can be configured via:
```json
{
  "enabled": true,
  "strategies": ["metadata", "file_hash", "audio_fingerprint"],
  "similarity_threshold": 0.85,
  "min_confidence": 0.5,
  "allowed_types": ["exact", "metadata", "acoustic"]
}
```

### ✨ Intelligent Progress Tracking (2024-12-10)
```json
{
  "enabled": true,
  "enhance_fields": ["year", "genre", "track_number", "albumartist"],
  "rate_limit": 1.0,
  "cache_enabled": true,
  "fallback_to_fuzzy": true
}
```

### ✨ Intelligent Progress Tracking (2024-12-10)
Implemented comprehensive progress tracking with:
- **Real-time metrics**: Files/second processing rate, ETA calculations
- **Stage tracking**: Scanning, metadata extraction, classification, moving stages
- **Dual rendering**: Rich UI for sync CLI, terminal-based for async CLI
- **Error tracking**: Count and display errors in progress
- **Byte tracking**: Show total data processed (useful for large libraries)
- **Intelligent rate calculation**: Rolling window for accurate instant rates

Key files:
- `src/music_organizer/progress_tracker.py` - Core tracking logic
- `src/music_organizer/rich_progress_renderer.py` - Rich UI renderer
- `src/music_organizer/async_progress_renderer.py` - Terminal renderer

### 🎯 Performance Benchmarks & CI (2024-12-11)
Implemented comprehensive performance monitoring and regression testing:

**Benchmark System**:
- **Automated benchmarks**: Tests all performance targets from TODO.md
- **Realistic test data**: Creates realistic music library structures for testing
- **Memory profiling**: Tracks RSS, Python memory, and tracemalloc usage
- **Performance regression detection**: Compares against baseline and previous runs

**CI/CD Integration**:
- **GitHub Actions workflow**: Automated benchmarks on every push/PR
- **Multi-Python support**: Tests across Python 3.9-3.12
- **Performance comparison**: Shows deltas between PR and base branch
- **Automated alerts**: Fails builds if performance targets aren't met

**Memory Monitoring**:
- **MemoryProfiler class**: Context manager for profiling code blocks
- **Global monitoring**: Track memory usage across the application
- **Memory pressure detection**: Automatically switch to streaming for large libraries
- **Profiling decorators**: Easy annotation for profiling functions

**Key Files**:
- `benchmarks/run_benchmarks.py` - Main benchmark runner
- `.github/workflows/benchmarks.yml` - CI/CD workflow
- `src/music_organizer/utils/memory_monitor.py` - Memory monitoring utilities
- `tests/test_performance_regression.py` - Performance regression tests

**Usage**:
```bash
# Run benchmarks locally
cd benchmarks && python run_benchmarks.py

# Run with memory profiling
python -m music_organizer --profile-memory

# Performance regression tests
python -m pytest tests/test_performance_regression.py -v
```

### 🔌 Plugin Architecture (2024-12-11)
Implemented comprehensive plugin system for extensibility:

**Core Architecture**:
- **Plugin interfaces**: Abstract base classes for MetadataPlugin, ClassificationPlugin, and OutputPlugin
- **PluginManager**: Automatic discovery and lifecycle management from plugins/ directory
- **Hook system**: Event-driven integration points (PRE/POST for scan, metadata, classify, move operations)
- **Configuration system**: Schema-based validation with type checking and defaults

**Key Features**:
- **Zero configuration required**: Plugins work out-of-the-box with sensible defaults
- **Async support**: All plugin operations support async/await for performance
- **Batch processing**: Built-in batch operations for handling multiple files efficiently
- **Validation**: Comprehensive configuration validation with custom validators
- **Discovery**: Automatic plugin discovery from files and directories
- **Isolation**: Each plugin runs in isolation with proper cleanup

**Example Plugins**:
- **ExampleClassifier**: Classifies music by decade, energy level, and language
- **M3UExporter**: Exports playlists in M3U format with extended metadata

**Key Files**:
- `src/music_organizer/plugins/base.py` - Core plugin interfaces
- `src/music_organizer/plugins/manager.py` - Plugin discovery and lifecycle management
- `src/music_organizer/plugins/hooks.py` - Event hook system
- `src/music_organizer/plugins/config.py` - Configuration validation
- `src/music_organizer/plugins/builtins/` - Example plugin implementations

## 🔥 Priority Semaphore

- 🔴 **Critical**: Must-have for MVP (core functionality, performance)
- 🟡 **Important**: Significant value add (plugins, UX, features)
- 🟢 **Nice-to-have**: Enhancement for polish (optimizations, extras)

## 📊 Success Metrics

### Performance Targets
- Process 10,000 files in < 10 seconds
- Memory usage < 100MB for large libraries
- 90% speed improvement on cached runs
- Startup time < 100ms

### Quality Metrics
- Zero dependency conflicts
- 95%+ test coverage
- No breaking changes in minor releases
- Plugin ecosystem with 5+ community plugins

## 🚀 Quick Start for Dev Team

### Setup
```bash
# Clone and setup
git clone https://github.com/nibzard/music-organizer.git
cd music-organizer

# Install dependencies
pip install -e .
# OR with uv (recommended)
uv install

# Run tests
python -m pytest tests/

# Run with async CLI (high performance for large libraries)
music-organize-async organize /source /target --workers 8

# Run with original CLI
music-organize organize /source /target
```

### Development Workflow
1. Pick a task from TODO.md
2. Create feature branch
3. Write tests first (TDD)
4. Implement with minimal code
5. Ensure performance targets
6. Update documentation

### Code Style
- Single file distribution when possible
- Use only standard library + mutagen
- Async/await for I/O operations
- Type hints everywhere
- No unnecessary abstractions

### 🏗️ Bounded Contexts Architecture (2024-12-11)
Successfully implemented Domain-Driven Design bounded contexts architecture with three distinct contexts:

**Catalog Context** - Managing music catalog and metadata:
- **Entities**: Recording, Release, Artist, Catalog
  - Rich domain logic embedded within entities
  - Proper aggregate boundaries and lifecycle management
  - Duplicate detection algorithms with multiple strategies
- **Value Objects**: AudioPath, ArtistName, TrackNumber, Metadata, FileFormat
  - Immutable with validation and domain-specific operations
- **Services**: CatalogService, MetadataService
  - Cross-entity business operations
  - Batch metadata enhancement and normalization
- **Repositories**: RecordingRepository, ReleaseRepository, ArtistRepository, CatalogRepository
  - Abstract data access with in-memory and file-based implementations

**Organization Context** - Managing physical file organization:
- **Entities**: OrganizationRule, FolderStructure, MovedFile, ConflictResolution, OrganizationSession
  - File move operations with conflict resolution
  - Organization rule evaluation and application
  - Session tracking for batch operations
- **Value Objects**: TargetPath, OrganizationPattern, ConflictStrategy, PathTemplate
  - Flexible pattern matching with conditional blocks
  - Filesystem-safe path generation
- **Services**: OrganizationService, PathGenerationService
  - Parallel file organization with conflict handling
  - Template-based path generation with validation

**Classification Context** - Content classification and duplicate detection:
- **Entities**: Classifier, DuplicateGroup, ContentType, ClassificationRule, SimilarityScore
  - Machine learning-ready classification framework
  - Multi-strategy duplicate detection
  - Classification rules with priority-based execution
- **Value Objects**: ContentTypeEnum, ClassificationPattern, SimilarityThreshold, AudioFeatures
  - Rich content type signatures with pattern matching
  - Audio feature extraction results
- **Services**: ClassificationService, DuplicateService, ContentAnalysisService
  - Batch classification with confidence scoring
  - Duplicate group management and resolution
  - Audio content analysis pipeline

**Cross-Cutting Infrastructure**:
- **Event System**: Event-driven architecture with domain events
  - Loose coupling between contexts through events
  - Async event handling with priority support
  - Event middleware and filtering capabilities
- **Anti-Corruption Layers**:
  - MutagenAdapter for metadata reading/writing
  - MusicBrainzAdapter for external metadata enrichment
  - AcoustIdAdapter for acoustic fingerprinting
  - FilesystemAdapter for file operations
- **Domain Services**:
  - MusicLibraryOrchestrator for cross-context workflows
  - ContextIntegrationService for bounded context communication

**Architecture Benefits**:
- **Clear Separation of Concerns**: Each context has focused responsibilities
- **Independent Evolution**: Contexts can develop independently
- **Better Testability**: Smaller, focused units are easier to test
- **Reduced Coupling**: Contexts communicate through well-defined interfaces
- **Domain Clarity**: Business rules captured in appropriate contexts

**Key Files**:
- `src/music_organizer/domain/catalog/` - Catalog context implementation
- `src/music_organizer/domain/organization/` - Organization context implementation
- `src/music_organizer/domain/classification/` - Classification context implementation
- `src/music_organizer/events/` - Event system implementation
- `src/music_organizer/infrastructure/adapters/` - Anti-corruption layers
- `src/music_organizer/infrastructure/repositories/` - Repository implementations

### 🏗️ Command Query Responsibility Segregation (CQRS) Implementation (2024-12-11)
Successfully implemented CQRS pattern to separate read and write operations for improved scalability and maintainability:

**Architecture Components Implemented**:
- **Command Side (Write Model)**: Handles state-changing operations
  - Base command classes with metadata and correlation support
  - Command handlers for catalog operations (AddRecording, UpdateMetadata, RemoveRecording)
  - Command handlers for organization operations (OrganizeFile, MoveFile, CreateDirectoryStructure)
  - CommandBus for mediating commands to handlers with middleware support
  - Command results with execution metrics and event publishing

- **Query Side (Read Model)**: Optimized for data retrieval
  - Base query classes with caching support
  - Query handlers for catalog queries (by ID, artist, genre, search)
  - Statistics queries for library analytics
  - QueryBus with built-in caching and middleware support
  - Materialized view support for complex aggregations

- **Event System**: Drives consistency between models
  - Domain events for all state changes
  - EventBus for publishing events to multiple handlers
  - EventStore for persistence and replay capabilities
  - Read model projectors for updating denormalized views

**Key Features**:
- **Separation of Concerns**: Clear distinction between commands (intent) and queries (information)
- **Performance Optimization**: Query result caching with configurable TTL
- **Scalability**: Read and write operations can scale independently
- **Extensibility**: Easy to add new commands and queries without affecting existing code
- **Event Sourcing Ready**: All state changes captured as domain events
- **Middleware Support**: Cross-cutting concerns (logging, validation, metrics)
- **Type Safety**: Full type hints with generic base classes

**Usage Examples**:
```python
# Command side - Add a recording
command = AddRecordingCommand(
    file_path=Path("/music/artist/album/track.flac"),
    metadata={"title": "Song", "artists": ["Artist"], "year": 2023}
)
result = await command_bus.dispatch(command)

# Query side - Search recordings
query = SearchRecordingsQuery(search_term="Rock", limit=10)
result = await query_bus.dispatch(query)
# Result may come from cache for better performance

# Event handling - Update read models
@event_handler(RecordingAddedEvent)
async def update_search_index(event):
    await search_index.add_recording(event.aggregate_id)
```

**Performance Benefits**:
- Query results cached automatically with invalidation on updates
- Commands processed without blocking read operations
- Optimized read models for specific query patterns
- No transaction conflicts between reads and writes

**Key Files**:
- `src/music_organizer/application/commands/` - Command implementations
- `src/music_organizer/application/queries/` - Query implementations
- `src/music_organizer/application/events/` - Event system
- `src/music_organizer/application/read_models/` - Read model projections
- `tests/test_cqrs_integration.py` - Integration tests
- `examples/cqrs_example.py` - Complete usage example
- `docs/cqrs-implementation.md` - Detailed documentation

---

**Remember**: Simplicity is our superpower. Every feature should justify its complexity. If in doubt, leave it out.