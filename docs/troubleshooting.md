# Troubleshooting FAQ

Frequently asked questions and solutions for common issues with music-organizer.

## Table of Contents

1. [Installation & Setup Issues](#installation--setup-issues)
2. [File Operation Errors](#file-operation-errors)
3. [Metadata Extraction Issues](#metadata-extraction-issues)
4. [Classification Problems](#classification-problems)
5. [Performance Issues](#performance-issues)
6. [Plugin Issues](#plugin-issues)
7. [CLI Usage Problems](#cli-usage-problems)
8. [Error Messages Reference](#error-messages-reference)

---

## Installation & Setup Issues

### "command not found: music-organize"

**Problem**: After installation, the command is not recognized.

**Solutions**:
1. **Local installation** - Use the wrapper script:
   ```bash
   ./mo organize source target
   ```

2. **uv run** - Run via uv:
   ```bash
   uv run python -m music_organizer.cli organize source target
   ```

3. **Create alias** - Add to your shell config:
   ```bash
   echo 'alias music-organizer="/path/to/music-organizer/mo"' >> ~/.zshrc
   source ~/.zshrc
   ```

4. **Install in editable mode**:
   ```bash
   uv pip install -e .
   ```

### "ModuleNotFoundError: No module named 'mutagen'"

**Problem**: Required dependencies are missing.

**Solution**:
```bash
# Reinstall dependencies
uv install

# Or install mutagen directly
uv pip install mutagen
```

### Permission Denied Errors

**Problem**: `PermissionError: [Errno 13] Permission denied`

**Solutions**:
1. Check write permissions on target directory:
   ```bash
   ls -la /path/to/target
   ```

2. Add write permissions:
   ```bash
   chmod +w /path/to/target/directory
   ```

3. Ensure source files are readable:
   ```bash
   chmod +r /path/to/source/files/*
   ```

4. Run without sudo if possible (security risk with sudo)

---

## File Operation Errors

### "FileOperationError: Failed to move file"

**Problem**: File moves fail during organization.

**Common Causes & Solutions**:

1. **Destination file exists and cannot be overwritten**:
   ```bash
   # Use different duplicate handling strategy
   ./mo organize source target --config config.yaml
   # In config: file_operations.handle_duplicates: "number"
   ```

2. **Insufficient disk space**:
   ```bash
   # Check disk space
   df -h /path/to/target

   # Clean up or use different target
   ```

3. **Network storage timeout**:
   ```bash
   # Reduce workers for network storage
   ./mo organize source target --workers 2
   ```

### Files Skipped During Organization

**Problem**: Some files are not being moved.

**Possible Causes**:

1. **Unsupported format** - Only FLAC, MP3, WAV, M4A, AAC, OGG, Opus, WMA supported

2. **Corrupted file** - File cannot be read:
   ```bash
   # Test file integrity
   ffmpeg -v error -i file.flac -f null -
   ```

3. **No metadata** - Files with no metadata may be skipped:
   ```bash
   # Inspect file to see what metadata exists
   ./mo inspect /path/to/file.flac
   ```

### Backup Creation Fails

**Problem**: `FileOperationError: Failed to create backup`

**Solutions**:
1. **Disable backup** (not recommended):
   ```bash
   ./mo organize source target --no-backup
   ```

2. **Check disk space**:
   ```bash
   df -h
   ```

3. **Specify custom backup location** in config:
   ```yaml
   file_operations:
     backup: true
     backup_dir: "/path/to/backup"
   ```

---

## Metadata Extraction Issues

### "MetadataError: Failed to extract metadata"

**Problem**: Cannot read metadata from file.

**Solutions**:

1. **Check if file is corrupted**:
   ```bash
   # Test with metaflac (for FLAC)
   metaflac --list file.flac

   # Test with mutagen-inspect
   mutagen-inspect file.flac
   ```

2. **Verify file format**:
   ```bash
   file /path/to/audiofile
   ```

3. **Re-encode/fix file**:
   ```bash
   # For FLAC files
   ffmpeg -i corrupted.flac -c copy fixed.flac
   ```

### Missing or Incorrect Metadata

**Problem**: Files have wrong artist, album, or title information.

**Solutions**:

1. **Inspect current metadata**:
   ```bash
   ./mo inspect /path/to/file.flac
   ```

2. **Fix metadata with external tools**:
   ```bash
   # Using metaflac for FLAC
   metaflac --set-tag=ARTIST="Correct Artist" file.flac
   metaflac --set-tag=ALBUM="Correct Album" file.flac
   metaflac --set-tag=TITLE="Correct Title" file.flac
   ```

3. **Use MusicBrainz Picard** for bulk metadata fixing
4. **Enable metadata enhancement**:
   ```yaml
   metadata:
     enhance: true
     musicbrainz: true
   ```

### Unicode/Encoding Issues in Tags

**Problem**: Strange characters in artist/album names.

**Symptoms**: `Artist Name` displays as `Artist NÃ me`

**Solutions**:
1. **Check file encoding**:
   ```bash
   mutagen-inspect file.flac | grep -i artist
   ```

2. **Convert to UTF-8**:
   ```bash
   # For FLAC files
   metaflac --remove-all-tags --import-tags-from=- file.flac < tags.txt
   ```

3. **Use music-organizer with locale**:
   ```bash
   LC_ALL=en_US.UTF-8 ./mo organize source target
   ```

---

## Classification Problems

### Wrong Category Assignment

**Problem**: Live recording classified as Album, or vice versa.

**Why it happens**:
- Classification is based on metadata patterns and keywords
- Files without clear metadata default to "Album"

**Solutions**:

1. **Check current classification**:
   ```bash
   ./mo inspect /path/to/file.flac
   # Look for "Content type" in output
   ```

2. **Add identifying metadata**:
   ```bash
   # For live recordings
   metaflac --set-tag=LOCATION="Madison Square Garden" file.flac
   metaflac --set-tag=DATE="2024-09-15" file.flac
   ```

3. **Use custom naming plugin** for specific patterns
4. **Interactive mode** - Prompt for ambiguous files:
   ```bash
   ./mo organize source target --interactive
   ```

### Collaboration Not Detected

**Problem**: Multi-artist files go to "Albums" instead of "Collaborations"

**Solutions**:
1. **Ensure proper artist tag format**:
   - Use multiple artist tags: `ARTIST=Artist1`, `ARTIST=Artist2`
   - Or comma-separated: `ARTIST=Artist1, Artist2`
   - Or use album artist tag: `ALBUMARTIST=Various Artists`

2. **Check metadata**:
   ```bash
   ./mo inspect /path/to/file.flac
   ```

3. **Manually fix tags**:
   ```bash
   metaflac --set-tag=ARTIST="Artist1" file.flac
   metaflac --set-tag=ARTIST="Artist2" file.flac
   ```

---

## Performance Issues

### Slow Processing Speed

**Problem**: Organization takes too long.

**Solutions**:

1. **Increase worker count**:
   ```bash
   ./mo organize source target --workers 8
   ```

2. **Enable caching** (for subsequent runs):
   ```bash
   # In Python API
   organizer = EnhancedAsyncMusicOrganizer(
       config=config,
       use_cache=True,
       use_smart_cache=True
   )
   ```

3. **Use incremental scanning**:
   ```bash
   ./mo organize source target --incremental
   ```

4. **Check for bottlenecks**:
   ```bash
   # Disk I/O monitoring
   iostat -x 1

   # Network storage may be slow
   # Consider copying to local storage first
   ```

### High Memory Usage

**Problem**: Process uses too much RAM or crashes with OOM.

**Solutions**:

1. **Reduce workers**:
   ```bash
   ./mo organize source target --workers 2
   ```

2. **Process in smaller batches**:
   ```bash
   # Organize one album at a time
   ./mo organize source/Album-A target
   ./mo organize source/Album-B target
   ```

3. **Disable parallel extraction**:
   ```python
   organizer = EnhancedAsyncMusicOrganizer(
       config=config,
       enable_parallel_extraction=False
   )
   ```

### Cache Not Working

**Problem**: Subsequent runs aren't faster.

**Solutions**:

1. **Check cache exists**:
   ```bash
   ls -la ~/.cache/music-organizer/
   ```

2. **Verify cache TTL**:
   ```python
   organizer = EnhancedAsyncMusicOrganizer(
       config=config,
       use_cache=True,
       cache_ttl=30  # days
   )
   ```

3. **Clear and rebuild cache**:
   ```bash
   rm -rf ~/.cache/music-organizer/
   ./mo organize source target
   ```

---

## Plugin Issues

### Plugin Not Found

**Problem**: `PluginError: Plugin 'xyz' not found`

**Solutions**:

1. **List available plugins**:
   ```bash
   # In Python
   from music_organizer.plugins import PluginManager
   pm = PluginManager()
   pm.discover_plugins()
   print(pm.list_plugins())
   ```

2. **Check plugin directory**:
   ```bash
   ls -la src/music_organizer/plugins/builtins/
   ```

3. **Verify plugin name** (no .py extension in name)

### Plugin Initialization Fails

**Problem**: `PluginError: Failed to initialize plugin`

**Solutions**:

1. **Check plugin dependencies**:
   ```bash
   cat src/music_organizer/plugins/builtins/example.py | grep import
   ```

2. **Install missing dependencies**:
   ```bash
   uv pip install missing-dependency
   ```

3. **Check plugin config**:
   ```bash
   cat ~/.config/music-organizer/plugins.json
   ```

4. **Enable verbose logging**:
   ```bash
   LOG_LEVEL=DEBUG ./mo organize source target
   ```

### Plugin Produces Wrong Results

**Problem**: Custom classifier/naming plugin misbehaves.

**Solutions**:

1. **Test plugin independently**:
   ```python
   from music_organizer.plugins import PluginManager
   pm = PluginManager()
   plugin = pm.load_plugin("your-plugin")

   # Test with sample file
   result = plugin.classify(audio_file)
   print(result)
   ```

2. **Check plugin config schema**:
   ```python
   schema = pm.get_plugin_schema("your-plugin")
   print(schema)
   ```

3. **Disable problematic plugin**:
   ```bash
   # Remove from config or disable
   ```

---

## CLI Usage Problems

### "Source directory not found"

**Problem**: CLI can't find the source directory.

**Solutions**:

1. **Use absolute paths**:
   ```bash
   ./mo organize /full/path/to/source /full/path/to/target
   ```

2. **Escape spaces in paths**:
   ```bash
   ./mo organize "/path/to/Music Files" "/path/to/Organized Music"
   ```

3. **Check directory exists**:
   ```bash
   ls -la /path/to/source
   ```

### Dry Run Shows No Changes

**Problem**: `--dry-run` output shows "No files to organize"

**Possible Causes**:

1. **No audio files found** - Check supported formats
2. **All files already organized** - Files may be in correct location
3. **Incremental scan skipped files**:
   ```bash
   # Force full scan
   ./mo organize source target --incremental --force-full-scan
   ```

### Help Command Not Working

**Problem**: Cannot see command options.

**Solutions**:
```bash
# General help
./mo --help

# Command-specific help
./mo organize --help
./mo scan --help
./mo inspect --help
```

---

## Error Messages Reference

### Exception Types

| Exception | Meaning | Solution |
|-----------|---------|----------|
| `MusicOrganizerError` | Base error for all issues | Check specific error message |
| `MetadataError` | Failed to read/write tags | Check file integrity, permissions |
| `FileOperationError` | Move/copy failed | Check disk space, permissions |
| `ClassificationError` | Content classification failed | Check metadata quality |
| `ConfigurationError` | Invalid config file | Validate JSON/YAML syntax |
| `MagicModeError` | Magic mode operation failed | Check regex patterns |

### Common Error Messages

| Message | Cause | Fix |
|---------|-------|-----|
| `Unsupported file format` | File type not supported | Convert to FLAC/MP3 |
| `No metadata found` | File has no tags | Add metadata with Picard/metaflac |
| `Destination already exists` | Duplicate file | Change `handle_duplicates` config |
| `Permission denied` | No write access | chmod or run with appropriate user |
| `Disk full` | No space left | Clean up disk or use different target |
| `Connection timeout` | Network storage issue | Increase timeout, reduce workers |
| `Cache locked` | Another process running | Wait or remove lock file |

### Getting Debug Information

```bash
# Enable verbose output
./mo organize source target --verbose

# Enable debug logging
LOG_LEVEL=DEBUG ./mo organize source target

# Save output to file
./mo organize source target --verbose 2>&1 | tee organize.log

# Check operation history
./mo rollback list-sessions
```

---

## Getting Help

If none of these solutions work:

1. **Check GitHub Issues** - Search for similar problems
2. **Create minimal test case** - Isolate the problematic file
3. **Gather diagnostic info**:
   ```bash
   # System info
   uname -a
   python3 --version

   # Package versions
   uv pip list | grep music-organizer
   uv pip list | grep mutagen

   # File info
   ls -la /path/to/problematic/file
   file /path/to/problematic/file
   ```

4. **File a bug report** with:
   - Error message (full traceback)
   - Command used
   - Sample file (if possible, or detailed description)
   - System information

---

## Additional Resources

- [API Reference](api-reference.md) - Complete API documentation
- [Performance Tuning Guide](performance-tuning.md) - Optimization tips
- [Plugin Development Guide](plugin-development.md) - Writing custom plugins
- [README](../README.md) - Project overview and quick start
