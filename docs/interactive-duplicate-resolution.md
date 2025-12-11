# Interactive Duplicate Resolution

The Music Organizer now includes an interactive duplicate resolution system that helps you identify and manage duplicate audio files in your music library with intelligent quality comparison and user-friendly decision making.

## Features

- **Side-by-side comparison**: View duplicate files side by side with detailed metadata and quality metrics
- **Intelligent quality scoring**: Automatically determines the best version based on format, bitrate, sample rate, and metadata completeness
- **Multiple resolution strategies**: Choose from interactive, automatic, and smart resolution strategies
- **Safe operations**: Dry-run mode and confirmation prompts to prevent accidental data loss
- **Batch processing**: Handle multiple duplicate groups efficiently
- **Detailed reports**: Generate comprehensive reports of all actions taken

## Quick Start

### Command Line Usage

```bash
# Interactive duplicate resolution (asks for each duplicate)
music-organize-duplicates resolve /path/to/music/library

# Preview duplicates without resolving
music-organize-duplicates preview /path/to/music/library

# Automatically keep best quality version
music-organize-duplicates resolve /path/to/music/library --strategy auto_best

# Move duplicates to separate directory instead of deleting
music-organize-duplicates resolve /path/to/music/library --move-duplicates-to /path/to/duplicates

# Dry run to see what would happen
music-organize-duplicates resolve /path/to/music/library --dry-run

# Organize music with duplicate resolution
music-organize-duplicates organize /source /target --strategy auto_smart
```

### Python API Usage

```python
from music_organizer import (
    InteractiveDuplicateResolver,
    ResolutionStrategy,
    quick_duplicate_resolution
)
from pathlib import Path

# Quick duplicate resolution
resolution_summary = await quick_duplicate_resolution(
    source_dir=Path("/music/library"),
    strategy=ResolutionStrategy.AUTO_KEEP_BEST,
    duplicate_dir=Path("/duplicates"),
    dry_run=False
)

print(f"Resolved {resolution_summary.resolved_groups} duplicate groups")
print(f"Space saved: {resolution_summary.space_saved_mb:.2f} MB")
```

## Resolution Strategies

### Interactive (Default)
- Shows each duplicate group with side-by-side comparison
- Lets you choose which file to keep for each group
- Provides quality metrics and file information
- Best for small to medium libraries or when you want full control

### Auto Keep Best
- Automatically keeps the highest quality version in each duplicate group
- Uses intelligent quality scoring based on:
  - Audio format (FLAC > WAV > MP3 > etc.)
  - Bitrate and sample rate
  - Metadata completeness
  - File size (as quality indicator)
- Best for large libraries where manual review isn't practical

### Auto First
- Always keeps the first file found in each duplicate group
- Moves or deletes all other duplicates
- Fastest option when you don't care about quality differences
- Useful when duplicates are exact copies

### Auto Smart
- Makes intelligent decisions based on file characteristics:
  - For exact duplicates: keeps file with cleanest path
  - For metadata duplicates: keeps file with better metadata
  - For acoustic duplicates: keeps highest quality version
- Good balance between automation and smart decision-making

## Duplicate Detection

The system detects duplicates using multiple strategies:

1. **Exact Duplicates**: Bit-for-bit identical files
2. **Metadata Duplicates**: Files with matching artist, title, album, and track number
3. **Acoustic Duplicates**: Files with similar audio fingerprints (similar sounding but different encodings)

## Quality Scoring

The quality scorer evaluates files based on:

- **Format Quality** (40% weight): Lossless formats (FLAC, WAV) score higher than lossy (MP3, AAC)
- **Bitrate** (25% weight): Higher bitrate gets higher score
- **Sample Rate** (15% weight): Higher sample rate indicates better quality
- **Metadata Completeness** (10% weight): More complete metadata gets bonus points
- **File Size** (10% weight): Larger files often have better quality

## UI Features

### Side-by-Side Comparison
```
============================================================
DUPLICATE GROUP: METADATA
Confidence: 95%
Reason: Same artist, title, album, and duration
Files in group: 2
============================================================

★ BEST
────────────────────────────────────────────────────────────
Path: /music/Artist/Album/01 Song.flac
Size: 45.2 MB | Format: FLAC
Quality: 1000 kbps | 96000 Hz

Metadata:
  Title: Song Title
  Artist: Artist Name
  Album: Album Name (2023)
  Year: 2023
  Genre: Rock
  Track: 1/12

[2]
────────────────────────────────────────────────────────────
Path: /music/Artist/Album/01 Song.mp3
Size: 8.5 MB | Format: MP3
Quality: 320 kbps | 44100 Hz

Metadata:
  Title: Song Title
  Artist: Artist Name
  Album: Album Name (2023)
  Year: 2023
  Genre: Rock
  Track: 1/12

Yellow Side-by-Side Comparison:
────────────────────────────────────────────────────────────────────────
Format      | ✓ FLAC               | ✗ MP3
Size        | ✓ 45.2 MB            | ✗ 8.5 MB
Bitrate     | ✓ 1000 kbps          | ✗ 320 kbps
Sample Rate | ✓ 96000 Hz           | ✗ 44100 Hz

Quality Scores: File 1: 9.5 | File 2: 6.2

Choose action:
  1) Keep first file
  2) Keep second file
  3) Keep best quality
  4) Keep both files
  5) Move second file
  6) Delete second file

Enter choice [1-6]:
```

### Progress Tracking
- Real-time progress indicator during scanning
- Shows current file being processed
- Displays files per second and estimated time remaining

### Resolution Summary
After processing all duplicates, you'll see a summary:
```
============================================================
RESOLUTION SUMMARY
============================================================

Groups Processed:
  Total groups: 25
  Resolved groups: 25
  Skipped groups: 0

File Actions:
  Files kept: 25
  Files moved: 20
  Files deleted: 5
  Files skipped: 0

Space Saved: 850.75 MB
```

## Configuration Options

### Command Line Options
- `--strategy`: Resolution strategy (interactive, auto_best, auto_first, auto_smart)
- `--move-duplicates-to`: Directory to move duplicates (instead of deleting)
- `--dry-run`: Preview actions without making changes
- `--no-duplicates`: Skip duplicate detection (for organize command)
- `--resolve-first`: Resolve duplicates before organizing (default: true)

### Duplicate Resolution Directory
When using `--move-duplicates-to`, duplicates are organized as:
```
/duplicates/
  ├── exact_duplicates/      # Bit-for-bit duplicates
  ├── metadata_duplicates/   # Same metadata, different files
  └── acoustic_duplicates/   # Similar sounding files
```

Files are renamed if conflicts occur (e.g., `song_1.mp3`, `song_2.mp3`).

## API Reference

### Classes

#### InteractiveDuplicateResolver
Main class for resolving duplicates.

```python
resolver = InteractiveDuplicateResolver(
    strategy=ResolutionStrategy.AUTO_KEEP_BEST,
    duplicate_dir=Path("/duplicates"),
    dry_run=False
)
```

#### DuplicateQualityScorer
Scores files to determine quality.

```python
scorer = DuplicateQualityScorer()
score = scorer.score_file(audio_file)
best_file, best_score = scorer.choose_best([file1, file2])
```

#### DuplicateResolverUI
Terminal UI for interactive resolution.

```python
ui = DuplicateResolverUI(dry_run=False)
decision = await ui.show_duplicate_group(duplicate_group)
ui.show_summary(resolution_summary)
```

### Enums

#### ResolutionStrategy
- `INTERACTIVE`: Ask user for each duplicate
- `AUTO_KEEP_BEST`: Automatically keep best quality
- `AUTO_FIRST`: Always keep first file
- `AUTO_SMART`: Make smart decisions

#### DuplicateAction
- `KEEP_FIRST`: Keep the first file in group
- `KEEP_SECOND`: Keep the second file
- `KEEP_BOTH`: Keep all files
- `KEEP_BEST`: Keep highest quality file
- `MOVE_DUPLICATE`: Move duplicate to separate directory
- `DELETE_DUPLICATE`: Delete duplicate file

## Integration with Magic Mode

The duplicate resolution integrates seamlessly with Magic Mode:

1. Magic Mode detects high likelihood of duplicates during library analysis
2. Suggests running duplicate resolution as a preprocessing step
3. Provides quick wins like "Remove duplicates to clean up the library"

```python
from music_organizer import MagicModeOrchestrator

# Analyze library
orchestrator = MagicModeOrchestrator()
suggestion = await orchestrator.analyze_and_suggest(audio_files)

# Check if duplicate resolution is recommended
if "Remove duplicates" in suggestion.quick_wins:
    # Run duplicate resolution
    await quick_duplicate_resolution(source_dir)
```

## Best Practices

1. **Always use dry-run first**: Test with `--dry-run` to see what would happen
2. **Back up important files**: Although the system is safe, always have backups
3. **Start with preview**: Use `preview` command to see duplicate count
4. **Use move instead of delete**: Move duplicates to separate directory first
5. **Review reports**: Save and review resolution reports for audit trail

## Troubleshooting

### Common Issues

**No duplicates found but you expect some?**
- Check if files have different metadata (even slightly different titles/artists)
- Try different duplicate detection strategies
- Ensure files are in supported formats

**Wrong file kept as "best"?**
- The quality scoring prioritizes format and bitrate
- Use interactive mode for manual control
- Custom scoring can be implemented if needed

**Running out of memory with large libraries?**
- Use auto strategies instead of interactive
- Process library in chunks
- Increase system memory or use a machine with more RAM

### Error Messages

- `"Directory does not exist"`: Check that the source directory path is correct
- `"Permission denied"`: Ensure you have read/write permissions
- `"No duplicates found"`: The library might not have duplicates based on current detection criteria

## Examples

### Example 1: Clean up a messy library
```bash
# First preview duplicates
music-organize-duplicates preview /messy/music/library

# Then resolve with safe strategy
music-organize-duplicates resolve /messy/music/library \
  --strategy auto_best \
  --move-duplicates-to /messy/music/duplicates \
  --dry-run

# If satisfied, run without dry-run
music-organize-duplicates resolve /messy/music/library \
  --strategy auto_best \
  --move-duplicates-to /messy/music/duplicates
```

### Example 2: Prepare library for organization
```bash
# Resolve duplicates before organizing
music-organize-duplicates organize /source/music /target/organized \
  --strategy auto_smart \
  --resolve-first
```

### Example 3: Python script for batch processing
```python
import asyncio
from pathlib import Path
from music_organizer import quick_duplicate_resolution, ResolutionStrategy

async def clean_music_libraries():
    libraries = [
        Path("/music/rock"),
        Path("/music/jazz"),
        Path("/music/classical")
    ]

    for library in libraries:
        print(f"\nProcessing {library}...")
        summary = await quick_duplicate_resolution(
            source_dir=library,
            strategy=ResolutionStrategy.AUTO_KEEP_BEST,
            duplicate_dir=library / "duplicates",
            dry_run=False
        )

        print(f"Resolved {summary.resolved_groups} groups")
        print(f"Freed {summary.space_saved_mb:.2f} MB")

asyncio.run(clean_music_libraries())
```

## Contributing

To contribute to the duplicate resolution system:

1. Add new quality metrics to `DuplicateQualityScorer`
2. Implement additional resolution strategies
3. Enhance the UI with better visualizations
4. Add more duplicate detection algorithms
5. Improve performance for large libraries

## License

This feature is part of the Music Organizer project and follows the same license terms.