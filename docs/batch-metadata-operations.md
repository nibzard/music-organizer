# Batch Metadata Operations

The music organizer now includes comprehensive batch metadata operations for bulk tagging and metadata updates across your music library.

## Features

- **Bulk metadata updates** across multiple files
- **Pattern-based transformations** using regex
- **Conditional operations** based on existing metadata
- **Conflict resolution strategies** (skip, replace, merge)
- **Preview mode** for safe testing
- **High performance** with parallel processing
- **Backup and undo support** for metadata changes
- **Comprehensive validation** before applying changes

## Installation

The batch metadata CLI is included with the music organizer. After installation, you'll have the `music-batch-metadata` command available.

## Quick Start

### Basic Operations

```bash
# Set genre for all files in a directory
music-batch-metadata /path/to/music --set-genre "Rock"

# Set year for all files
music-batch-metadata /path/to/music --set-year 2023

# Add an additional artist to all files
music-batch-metadata /path/to/music --add-artist "Featured Artist"

# Preview changes without applying them
music-batch-metadata /path/to/music --set-genre "Jazz" --dry-run
```

### Advanced Operations

```bash
# Standardize genre names
music-batch-metadata /path/to/music --standardize-genres

# Capitalize titles and album names
music-batch-metadata /path/to/music --capitalize-titles

# Fix track number formatting
music-batch-metadata /path/to/music --fix-track-numbers

# Remove featuring artists from titles
music-batch-metadata /path/to/music --remove-feat-artists
```

### Using Operation Files

For complex operations, use a JSON configuration file:

```bash
# Apply operations from a file
music-batch-metadata /path/to/music --operations operations.json

# Example operations.json file structure:
{
  "operations": [
    {
      "field": "genre",
      "operation": "set",
      "value": "Rock",
      "condition": {"genre": null},
      "conflict_strategy": "skip"
    }
  ]
}
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `directory` | Directory containing music files to process |
| `--operations` | JSON file containing metadata operations |
| `--filter` | Filter files by pattern (e.g., "*.flac") |
| `--workers` | Number of parallel workers (default: 4) |
| `--batch-size` | Files per batch (default: 100) |
| `--dry-run` | Preview changes without applying them |
| `--no-backup` | Skip creating backup before updates |
| `--continue-on-error` | Continue processing even if some files fail |
| `--preserve-time` | Preserve file modification timestamps |
| `--quiet` | Suppress progress output |

### Quick Operations

| Option | Description |
|--------|-------------|
| `--set-genre` | Set genre for all files |
| `--set-year` | Set year for all files |
| `--add-artist` | Add artist to all files |
| `--standardize-genres` | Standardize genre names |
| `--capitalize-titles` | Capitalize title and album names |
| `--fix-track-numbers` | Fix track number formatting |
| `--remove-feat-artists` | Remove featuring artists from title field |

## Operation Types

### SET
Set a field to a specific value.

```json
{
  "field": "genre",
  "operation": "set",
  "value": "Rock",
  "conflict_strategy": "replace"
}
```

### ADD
Add a value to multi-value fields (artists, genres).

```json
{
  "field": "artists",
  "operation": "add",
  "value": "Featured Artist"
}
```

### REMOVE
Remove a value from a field.

```json
{
  "field": "artists",
  "operation": "remove",
  "value": "Old Artist Name"
}
```

### TRANSFORM
Transform field values using regex patterns.

```json
{
  "field": "title",
  "operation": "transform",
  "pattern": "s/^(\\w)/\\U$1/"
}
```

### COPY
Copy value from one field to another.

```json
{
  "field": "albumartist",
  "operation": "copy",
  "value": "artists"
}
```

### CLEAR
Clear a field value.

```json
{
  "field": "comment",
  "operation": "clear"
}
```

## Conflict Strategies

- **SKIP**: Skip operation if field has a value
- **REPLACE**: Replace existing value (default)
- **MERGE**: Merge with existing value (for multi-value fields)
- **ASK**: Ask user (interactive mode only)

## Conditions

Operations can be conditional based on existing metadata:

```json
{
  "field": "genre",
  "operation": "set",
  "value": "Classical",
  "condition": {
    "artists": {
      "regex": "(?i)(bach|beethoven|mozart)"
    }
  }
}
```

### Condition Operators

- **equals**: Exact match
- **contains**: Value contains substring
- **regex**: Regular expression match

## Pattern Transformations

Transform patterns use sed-like syntax:

- `s/find/replace/` - Basic substitution
- `s/find/replace/g` - Global substitution
- `s/find/replace/i` - Case insensitive
- Multiple patterns with `|` separator

Examples:
- `s/^(\\w)/\\U$1/` - Capitalize first letter
- `s/\\s*\\(feat\\..*\\)//g` - Remove featuring artists
- `s/^0*(\\d+)$/$1/g` - Remove leading zeros

## Examples

### Standardize Genres

```json
{
  "operations": [
    {
      "name": "Standardize Genres",
      "field": "genre",
      "operation": "transform",
      "pattern": "s/^rock$/Rock/g|s/^pop$/Pop/g|s/^jazz$/Jazz/g"
    }
  ]
}
```

### Extract Year from Album Name

```json
{
  "operations": [
    {
      "name": "Extract Year from Album",
      "field": "year",
      "operation": "transform",
      "pattern": "s/.*\\((19|20)\\d{2}\\).*/$1/g",
      "condition": {
        "album": {
          "regex": "\\((19|20)\\d{2}\\)"
        }
      }
    }
  ]
}
```

### Set Album Artist

```json
{
  "operations": [
    {
      "name": "Set Album Artist",
      "field": "albumartist",
      "operation": "copy",
      "value": "artists",
      "condition": {
        "albumartist": null
      },
      "conflict_strategy": "skip"
    }
  ]
}
```

### Fix Track Numbers

```json
{
  "operations": [
    {
      "name": "Extract Track Number",
      "field": "track_number",
      "operation": "transform",
      "pattern": "s/^(\\d+)\\s*/$1/g"
    },
    {
      "name": "Extract Total Tracks",
      "field": "total_tracks",
      "operation": "transform",
      "pattern": "s/^\\d+\\s*\\/\\s*(\\d+)$/$1/g",
      "condition": {
        "track_number": {
          "regex": "^\\d+\\s*/\\s*\\d+$"
        }
      }
    }
  ]
}
```

## Performance Tips

1. **Increase workers** for faster processing:
   ```bash
   music-batch-metadata /path/to/music --workers 8
   ```

2. **Use filters** to process only relevant files:
   ```bash
   music-batch-metadata /path/to/music --filter "*.flac"
   ```

3. **Batch size** tuning:
   - Smaller batches (50-100): Less memory usage
   - Larger batches (200-500): Better throughput

4. **Preview first**:
   ```bash
   music-batch-metadata /path/to/music --dry-run --operations ops.json
   ```

## Safety Features

- **Automatic backups** before changes (unless `--no-backup`)
- **Dry run mode** to preview changes
- **Metadata validation** before writing
- **Error handling** with detailed reports
- **Continue on error** option

## Integration with Other Tools

The batch metadata operations integrate seamlessly with other music organizer features:

```bash
# Organize with batch metadata updates
music-batch-metadata /music/source --set-genre "Rock"
music-organize-async organize /music/source /music/organized

# Use with incremental scanning
music-batch-metadata /music/library --standardize-genres
music-organize-async organize /music/library /organized --incremental
```

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure write permissions on files
2. **Locked files**: Close media players before processing
3. **Invalid metadata**: Check validation errors in output
4. **Pattern errors**: Test regex patterns with --dry-run

### Debug Mode

Enable verbose output for debugging:

```bash
music-batch-metadata /path/to/music --operations ops.json --dry-run
```

This will show you exactly which operations would be applied to each file without making any changes.

## API Usage

You can also use the batch metadata operations programmatically:

```python
from music_organizer.core.batch_metadata import (
    BatchMetadataProcessor,
    BatchMetadataConfig,
    MetadataOperation,
    OperationType,
    MetadataOperationBuilder
)

# Create processor
config = BatchMetadataConfig(dry_run=True)
processor = BatchMetadataProcessor(config)

# Create operations
operations = [
    MetadataOperationBuilder.set_genre("Rock"),
    MetadataOperationBuilder.set_year(2023)
]

# Process files
result = await processor.apply_operations(files, operations)
print(f"Processed {result.successful} files")
```