# 🎵 Music Organizer

A Python-based music library organizer that uses metadata-aware categorization to structure your music collection. Built with astral uv for package management and mutagen for audio metadata handling.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

## ✨ Features

- 🤖 **Smart Classification**: Automatically categorizes music into Albums, Live Recordings, Collaborations, Compilations, and Rarities
- 📊 **Metadata Enhancement**: Uses MusicBrainz lookup to enhance and complete missing metadata
- 🔒 **Safe Operations**: Optional backup creation with rollback capability
- 🎯 **Interactive Mode**: Prompts for ambiguous categorizations
- 🎧 **Multiple Formats**: Support for FLAC, MP3, WAV, M4A, and AAC
- 💎 **Preserve Quality**: Maintains all high-quality audio files without transcoding
- 📈 **Beautiful Output**: Rich terminal formatting with progress bars and tables
- ⚡ **Fast Processing**: Efficiently handles large music libraries

## 📦 Installation

### Prerequisites

- Python 3.11+
- [astral uv](https://github.com/astral-sh/uv) (modern Python package manager)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/music-organizer.git
cd music-organizer

# Install dependencies with uv
uv install

# Install the package in development mode
uv pip install -e .
```

### Installing uv

If you don't have uv installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 🚀 Quick Start

### Basic Usage

```bash
# From the music-organizer directory, run:
./mo organize /path/to/unorganized/music /path/to/organized/music

# Or run with uv directly:
uv run python music-organizer organize /path/to/unorganized/music /path/to/organized/music

# Create an alias for global use:
echo 'alias music-organizer="/path/to/music-organizer/mo"' >> ~/.zshrc
source ~/.zshrc

# Then use from anywhere:
music-organize organize /path/to/unorganized/music /path/to/organized/music
```

### Recommended Workflow

```bash
# 1. First, scan your music library to understand what you have
./mo scan /path/to/music

# 2. Do a dry run to see what would happen (no changes made!)
./mo organize /path/to/music /path/to/organized --dry-run

# 3. Review the output, then run the actual organization
./mo organize /path/to/music /path/to/organized

# 4. Use interactive mode for ambiguous cases
./mo organize /path/to/music /path/to/organized --interactive
```

## 📋 Command Reference

### Main Commands

```bash
# Organize music from SOURCE to TARGET directory
./mo organize [OPTIONS] SOURCE TARGET

Options:
  --dry-run                 Show what would be done without making changes
  --interactive             Prompt for ambiguous categorizations
  --backup / --no-backup    Create backup before reorganization (default: enabled)
  --config PATH             Use custom configuration file
  --verbose                 Verbose output

# Scan and analyze music files
./mo scan [OPTIONS] DIRECTORY

Options:
  --recursive              Scan subdirectories recursively (default: enabled)

# Inspect metadata of a single file
./mo inspect FILE_PATH

# Validate existing organization
./mo validate DIRECTORY
```

### Configuration File

Create a custom configuration file:

```bash
./mo organize source target --config my-config.yaml
```

Example configuration (`config/default.yaml`):

```yaml
# Source and target directories
source_directory: "/Users/nikola/Music/Unorganized"
target_directory: "/Users/nikola/Music/Organized"

# Directory names (customizable)
directories:
  albums: "Albums"
  live: "Live Recordings"
  collaborations: "Collaborations"
  compilations: "Compilations"
  rarities: "Rarities & Special Editions"

# Naming patterns (customize how files/folders are named)
naming:
  album_format: "{artist}/{album} ({year})"
  live_format: "{artist}/{date} - {location}"
  collab_format: "{album} ({year}) - {artists}"
  compilation_format: "{artist}/{album} ({year})"
  rarity_format: "{artist}/{album} ({edition})"

# Metadata handling
metadata:
  enhance: true              # Enable metadata enhancement
  musicbrainz: true          # Use MusicBrainz lookup
  fix_capitalization: true   # Fix capitalization in tags
  standardize_genres: true   # Standardize genre names

# File operations
file_operations:
  strategy: "move"            # or "copy"
  backup: true               # Create backup before changes
  handle_duplicates: "number" # or "skip" or "overwrite"
```

## 📁 Directory Structure

The organizer creates the following clean structure:

```
Music/
├── 📂 Albums/                           # Studio albums
│   └── 📂 Artist Name/
│       └── 📂 Album Name (Year)/
│           ├── 01. Song Title.flac
│           ├── 02. Song Title.flac
│           └── folder.jpg
│
├── 📂 Live Recordings/                  # Live concerts & performances
│   └── 📂 Artist Name/
│       └── 📂 YYYY-MM-DD - Venue Name/
│           ├── 01. Song Title.flac
│           └── cover.jpg
│
├── 📂 Collaborations/                   # Multi-artist works
│   └── 📂 Album Name (Year) - Artist1, Artist2/
│       ├── 01. Song Title.flac
│       └── folder.jpg
│
├── 📂 Compilations/                     # Greatest hits, collections
│   └── 📂 Artist Name/
│       └── 📂 Album Name (Year)/
│           ├── 01. Song Title.flac
│           └── folder.jpg
│
└── 📂 Rarities & Special Editions/     # Demos, unreleased, special editions
    └── 📂 Artist Name/
        └── 📂 Album Name (Edition Info)/
            ├── 01. Demo.flac
            └── disc.jpg
```

## 🧠 Classification Logic

### 📀 Albums (Studio Recordings)
- **Pattern**: Single primary artist, standard album structure
- **Examples**: Regular studio albums, EPs, singles
- **Naming**: `Artist/Album (Year)/`

### 🎤 Live Recordings
- **Indicators**:
  - Contains "live", "concert", "show", "performance"
  - Date information (YYYY-MM-DD)
  - Venue/location in metadata or filename
- **Examples**: "Live at Madison Square Garden", "2024-09-12 - San Jose"
- **Naming**: `Artist/YYYY-MM-DD - Location/`

### 🤝 Collaborations
- **Indicators**:
  - Multiple primary artists (comma-separated)
  - "feat.", "featuring", "with", "&", "x" patterns
  - Known collaboration projects
- **Examples**: Santana & McLaughlin, Various Artist collaborations
- **Naming**: `Album (Year) - Artist1, Artist2/`

### 📚 Compilations
- **Indicators**:
  - "Greatest Hits", "Best Of", "Essential", "Collection"
  - "Anthology", "The Very Best"
  - Various Artists albums
- **Examples**: "Greatest Hits", "Best of 90s"
- **Naming**: `Artist/Album (Year)/`

### 💎 Rarities & Special Editions
- **Indicators**:
  - "Demo", "Unreleased", "Bootleg"
  - "Special Edition", "Limited Edition"
  - "Anniversary Edition", "Deluxe", "Expanded"
- **Examples**: "Demo Tracks", "25th Anniversary Edition"
- **Naming**: `Artist/Album (Edition)/`

## 🔧 Development

### Project Structure

```
music-organizer/
├── src/music_organizer/
│   ├── cli.py              # Command-line interface
│   ├── core/
│   │   ├── organizer.py    # Main orchestration logic
│   │   ├── metadata.py     # Audio metadata handling
│   │   ├── classifier.py   # Content classification
│   │   └── mover.py        # File operations
│   ├── models/
│   │   ├── audio_file.py   # Audio file data model
│   │   └── config.py       # Configuration model
│   └── utils/              # Utility functions
├── tests/                  # Test suite
├── config/                 # Default configuration
├── pyproject.toml         # Project configuration
├── music-organizer        # Entry point script
└── mo                     # Convenient wrapper script
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=music_organizer --cov-report=html

# Run specific test file
uv run pytest tests/test_metadata.py

# Run tests with verbose output
uv run pytest -v

# Run tests matching a pattern
uv run pytest -k "test_classify"
```

### Code Quality Tools

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type checking
uv run mypy src/

# All at once
uv run black src/ && uv run ruff check src/ && uv run mypy src/
```

## 🔒 Safety Features

### Backup & Recovery

```bash
# Backups are automatically created
# Location: /path/to/target/backup_YYYYMMDD_HHMMSS/
├── manifest.json          # File inventory before changes
├── operations.json        # Detailed operation log
└── [original files]       # Backup of all moved files
```

### Safety Checks

- ✅ **File Integrity Verification**: Checks file sizes and modification times
- ✅ **Duplicate Handling**: Automatically numbers duplicate files
- ✅ **Permission Validation**: Ensures write access before operations
- ✅ **Dry Run Mode**: Preview all changes before executing
- ✅ **Rollback Capability**: Undo all changes if something goes wrong
- ✅ **Progress Tracking**: Real-time progress bars and status updates

## 🐛 Troubleshooting

### Common Issues

#### Permission Denied
```bash
# Ensure you have write permissions
chmod +w /path/to/target/directory

# Or run with sudo if necessary (not recommended)
sudo ./mo organize source target
```

#### Unicode/Metadata Issues
```bash
# The tool handles most encoding issues automatically
# If you see encoding errors, try:
LC_ALL=C ./mo organize source target
```

#### Large Libraries
```bash
# For very large libraries (>10,000 files):
./mo organize source target --verbose  # Shows detailed progress
```

#### Corrupted Files
```bash
# The tool automatically skips unreadable files
# Check the error output for specific file names
```

### Debug Mode

```bash
# Enable verbose logging
./mo organize source target --verbose

# Check what files would be affected
./mo scan source --recursive
```

## 📊 Example Output

### Scan Command
```
🔍 Analyzing 199 files...
✅ Processing files... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

📊 Directory Analysis
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Metric              ┃        Value ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Total files         │          199 │
│ Total size          │   12783.8 MB │
│ Files with metadata │ 199 (100.0%) │
│ FLAC files          │          191 │
│ MP3 files           │            8 │
└─────────────────────┴──────────────┘

📂 Content Distribution
┏━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Category        ┃ Count ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Studio Albums   │   171 │
│ Collaborations  │    16 │
│ Compilations    │    21 │
│ Live Recordings │     5 │
│ Rarities        │     2 │
└─────────────────┴───────┘
```

### Organization Results
```
🎵 Organization Complete!
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Category              ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Albums                │   171 │
│ Live Recordings       │     5 │
│ Collaborations        │    16 │
│ Compilations          │    21 │
│ Rarities              │     2 │
│ Directories Created   │    42 │
│ Files Moved           │   199 │
│ Total Size Organized  │ 12.8GB │
└───────────────────────┴───────┘
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Ensure all tests pass**
   ```bash
   uv run pytest
   uv run black src/ tests/
   uv run ruff check src/ tests/
   uv run mypy src/
   ```
6. **Commit your changes**
   ```bash
   git commit -m "feat: add amazing feature"
   ```
7. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```
8. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write descriptive commit messages
- Include tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [**mutagen**](https://mutagen.readthedocs.io/) - Robust audio metadata handling
- [**MusicBrainz**](https://musicbrainz.org/) - Comprehensive music database
- [**astral uv**](https://github.com/astral-sh/uv) - Fast Python package manager
- [**rich**](https://rich.readthedocs.io/) - Beautiful terminal output
- [**click**](https://click.palletsprojects.com/) - Elegant CLI framework
- [**pydantic**](https://pydantic-docs.helpmanual.io/) - Data validation

## 📞 Support

- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/music-organizer/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/music-organizer/discussions)

---

**Made with ❤️ for music lovers who value organization and quality.**