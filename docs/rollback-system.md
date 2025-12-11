# Rollback System Documentation

The Music Organizer now includes a comprehensive rollback system that tracks all file operations and allows you to undo them when needed. This provides safety and confidence when organizing large music libraries.

## Overview

The rollback system tracks:
- File moves and copies
- Cover art operations
- Directory creation
- Checksums for integrity verification
- Operation timestamps and status
- Error messages for failed operations

All operations are stored in a persistent SQLite database, allowing rollback even after the program has closed.

## Key Components

### 1. Operation History Tracker
- **Purpose**: Tracks all file operations in a persistent database
- **Location**: `~/.cache/music-organizer/operation_history.db`
- **Features**:
  - Session-based tracking
  - Operation timestamps
  - Checksum verification
  - Status tracking (pending, in_progress, completed, failed, rolled_back)

### 2. Operation Rollback Service
- **Purpose**: Executes rollback operations
- **Features**:
  - Full session rollback
  - Partial rollback (specific operations)
  - Dry-run mode
  - Conflict resolution
  - Integrity verification

### 3. Enhanced File Mover
- **Purpose**: Performs file operations with tracking
- **Features**:
  - Automatic operation recording
  - Checksum calculation
  - Backup integration
  - Progress tracking

## Usage

### Basic Organization with Tracking

```bash
# Organize with operation history enabled (default)
music-organize-async organize /music/unsorted /music/organized

# Organize with custom session ID
music-organize-async organize /music/unsorted /music/organized --session-id "my_organization"

# Organize without operation history (legacy mode)
music-organize-async organize-legacy /music/unsorted /music/organized
```

### Viewing Operation History

```bash
# List recent sessions
music-organize-async sessions

# View operations for a specific session
music-organize-async history --session-id SESSION_ID

# View only failed operations
music-organize-async history --session-id SESSION_ID --status failed

# Export history as JSON
music-organize-async history --session-id SESSION_ID --format json
```

### Rolling Back Operations

```bash
# Preview rollback (dry run)
music-organize-async rollback SESSION_ID --dry-run

# Rollback entire session
music-organize-async rollback SESSION_ID

# Rollback with confirmation
music-organize-async rollback SESSION_ID --force

# Rollback specific operations
music-organize-async rollback SESSION_ID --operation-ids op1 op2 op3
```

### Restoring from Backups

```bash
# Restore from backup directory
music-organize-async restore /path/to/backup /music/organized

# Preview restore
music-organize-async restore /path/to/backup /music/organized --dry-run
```

## Configuration Options

### Command Line Options

```bash
# Disable operation history tracking
music-organize-async organize /source /target --disable-history

# Disable backup creation
music-organize-async organize /source /target --no-backup

# Enable smart caching
music-organize-async organize /source /target --smart-cache

# Custom session ID
music-organize-async organize /source /target --session-id "custom_id"
```

### Programmatic Usage

```python
from music_organizer import (
    EnhancedAsyncMusicOrganizer,
    OperationHistoryTracker,
    OperationRollbackService
)
from music_organizer.models.config import Config

# Create organizer with operation history
config = Config()
organizer = EnhancedAsyncMusicOrganizer(
    config=config,
    enable_operation_history=True,
    session_id="my_session"
)

# Organize files
result = await organizer.organize_files(source_dir, target_dir)
if result.is_success():
    stats = result.value()
    session_id = stats["session_id"]

    # Get operation history
    history = await organizer.get_operation_history()
    print(f"Tracked {len(history.value())} operations")

    # Rollback if needed
    rollback_result = await organizer.rollback_session(dry_run=True)
    if rollback_result.is_success():
        print("Rollback preview:", rollback_result.value())
```

## Database Schema

### operation_sessions Table
- `session_id`: Unique identifier for the session
- `start_time`: When the session started
- `end_time`: When the session ended (NULL if running)
- `source_root`: Source directory path
- `target_root`: Target directory path
- `total_operations`: Total number of operations
- `completed_operations`: Number of successful operations
- `failed_operations`: Number of failed operations
- `status`: Session status (running, completed, failed, rolled_back)
- `metadata`: JSON metadata for the session

### operation_records Table
- `id`: Unique operation identifier
- `session_id`: Reference to session
- `timestamp`: When the operation occurred
- `operation_type`: Type of operation (move, copy, delete, etc.)
- `source_path`: Original file path
- `target_path`: Target file path
- `backup_path`: Backup file path (if any)
- `status`: Operation status
- `error_message`: Error message if failed
- `checksum_before`: SHA-256 checksum before operation
- `checksum_after`: SHA-256 checksum after operation
- `file_size`: File size in bytes
- `metadata`: Additional operation metadata

## Safety Features

### 1. Checksum Verification
- SHA-256 checksums calculated before and after operations
- Automatic rollback on checksum mismatch
- Verification option for copy operations

### 2. Conflict Resolution
- Automatic detection of existing files
- Multiple strategies: skip, rename, replace, keep_both
- Safe rollback with conflict checking

### 3. Dry Run Mode
- Preview operations before execution
- See what would be rolled back
- No actual file modifications

### 4. Backup Integration
- Automatic backup creation
- Backup directory with operation logs
- Restore functionality from backups

## Performance Considerations

### Database Operations
- SQLite with WAL mode for concurrent access
- Indexed queries for fast retrieval
- Batch operations for bulk updates

### Memory Usage
- Streaming operation records
- Configurable batch sizes
- Automatic cleanup of old sessions

### I/O Optimization
- Async database operations
- Minimal filesystem calls
- Batch directory operations

## Troubleshooting

### Common Issues

1. **Database Locked Error**
   - Another process is using the database
   - Wait and try again
   - Check for hanging music-organizer processes

2. **Rollback Fails**
   - Files may have been modified after organization
   - Check if target files still exist
   - Use dry-run to preview what will happen

3. **Large Database Size**
   - Old sessions accumulate over time
   - Consider cleaning up old sessions
   - Use sessions command to list and delete old ones

### Debugging

```bash
# Enable verbose logging
music-organize-async organize /source /target --verbose

# Check database integrity
sqlite3 ~/.cache/music-organizer/operation_history.db "PRAGMA integrity_check;"

# List all sessions with details
music-organize-async sessions --limit 50 --format json | jq '.'
```

## API Reference

### OperationHistoryTracker

```python
class OperationHistoryTracker:
    async def start_session(self, session_id: str, source_root: Path,
                          target_root: Path, metadata: Optional[Dict] = None) -> Result[OperationSession]
    async def record_operation(self, operation: OperationRecord) -> Result[None]
    async def end_session(self, session_id: str, status: str = "completed") -> Result[OperationSession]
    async def get_session(self, session_id: str) -> Optional[OperationSession]
    async def get_session_operations(self, session_id: str,
                                   status_filter: Optional[OperationStatus] = None) -> List[OperationRecord]
    async def list_sessions(self, limit: int = 50) -> List[OperationSession]
    async def delete_session(self, session_id: str) -> Result[None]
```

### OperationRollbackService

```python
class OperationRollbackService:
    async def rollback_session(self, session_id: str, dry_run: bool = False) -> Result[Dict]
    async def rollback_partial(self, session_id: str, operation_ids: List[str],
                             dry_run: bool = False) -> Result[Dict]
```

### EnhancedAsyncFileMover

```python
class EnhancedAsyncFileMover:
    async def start_operation(self, source_root: Path, target_root: Path,
                            metadata: Optional[Dict] = None) -> Result[OperationSession]
    async def move_file(self, audio_file: AudioFile, target_path: Path,
                       verify_checksum: bool = False) -> Result[Path]
    async def move_cover_art(self, cover_art: CoverArt, target_dir: Path,
                           verify_checksum: bool = False) -> Result[Optional[Path]]
    async def rollback_session(self, dry_run: bool = False) -> Result[Dict]
    async def get_operation_history(self) -> Result[List[OperationRecord]]
    async def get_session_summary(self) -> Result[Dict]
```

## Best Practices

1. **Always Use Session IDs**
   - Provide meaningful session IDs
   - Record session IDs for later reference
   - Use timestamps for automatic IDs

2. **Enable Checksum Verification**
   - Especially for important libraries
   - Helps detect corruption
   - Automatic rollback on mismatch

3. **Test with Dry Run**
   - Always preview large operations
   - Check rollback preview before committing
   - Verify operation count looks correct

4. **Regular Cleanup**
   - Delete old sessions periodically
   - Keep database size manageable
   - Export important sessions if needed

5. **Monitor Performance**
   - Use sessions command to track activity
   - Watch for failed operations
   - Check rollback success rates