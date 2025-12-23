# Cloud Storage Integration Research

**Research Date**: 2025-12-23
**Status**: Technical Investigation Complete - Implementation Plan Ready

## Executive Summary

The music-organizer codebase has excellent foundations for cloud storage integration. The existing async architecture, adapter pattern, and bulk operation framework make cloud integration feasible without major restructuring.

**Recommendation**: Implement a **storage abstraction layer** with provider-specific adapters (S3, GCS, Azure) that extends the existing `FilesystemAdapter`. This enables hybrid local/cloud organization while maintaining backward compatibility.

**Estimated Effort**: 3-4 weeks for MVP (single provider), 6-8 weeks for multi-provider support

## Current Architecture Analysis

### Existing File Operations

**AsyncFileMover** (`src/music_organizer/core/async_mover.py`):
- Uses `shutil.move()` and `shutil.copy2()` for file operations
- Thread-safe via `ThreadPoolExecutor`
- Backup and rollback support
- Progress tracking and conflict resolution

**FilesystemAdapter** (`src/music_organizer/infrastructure/adapters/filesystem_adapter.py`):
- Anti-corruption layer for filesystem operations
- Async wrapper around blocking operations
- Security validation via `PathValidationUtils`

**AsyncMusicOrganizer** (`src/music_organizer/core/async_organizer.py`):
- High-level orchestration with parallel processing
- Incremental scanning and memory-efficient streaming
- Bulk operations for large libraries

### Extensibility Points

1. **Adapter Pattern**: `FilesystemAdapter` can be extended for cloud storage
2. **Configuration System**: `Config` model supports new storage backends
3. **Bulk Operations**: Optimized for batch processing (extends to cloud uploads)
4. **Progress Tracking**: `IntelligentProgressTracker` works with any async operation

## Proposed Cloud Architecture

### Storage Abstraction Layer

```
┌─────────────────────────────────────────────────────────────┐
│                    Storage Interface                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  StorageBackend (ABC)                              │    │
│  │  - read(file) -> bytes                             │    │
│  │  - write(file, data) -> bool                       │    │
│  │  - move(src, dst) -> bool                          │    │
│  │  - delete(file) -> bool                            │    │
│  │  - list(path) -> List[str]                         │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
           │                      │                   │
           ▼                      ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  LocalStorage    │  │   S3Storage      │  │   GCSStorage     │
│  (FilesystemAdapter) │  │   (boto3)       │  │   (google-cloud)│
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

### Provider Adapters

| Provider | Library | License | Notes |
|----------|---------|---------|-------|
| Amazon S3 | boto3 | Apache 2.0 | Industry standard |
| Google Cloud | google-cloud-storage | Apache 2.0 | First-party SDK |
| Azure Blob | azure-storage-blob | MIT | Microsoft Azure |
| Dropbox | dropbox | MIT | Personal cloud |
| Google Drive | google-api-python-client | Apache 2.0 | Requires OAuth |

### Configuration Model

```python
@dataclass
class CloudStorageConfig:
    provider: Literal["s3", "gcs", "azure", "dropbox", "gdrive"]
    bucket_name: str
    region: Optional[str] = None
    credentials_path: Optional[Path] = None
    credentials_env: Optional[str] = None
    remote_path: str = "/music"

    # Upload settings
    chunk_size: int = 8 * 1024 * 1024  # 8MB chunks
    max_concurrent_uploads: int = 4
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Sync settings
    sync_mode: Literal["upload", "download", "bidirectional"] = "bidirectional"
    delete_after_sync: bool = False
    bandwidth_limit: Optional[int] = None  # bytes/sec

@dataclass
class Config:
    # Existing fields...
    cloud_storage: Optional[CloudStorageConfig] = None
```

## Implementation Plan

### Phase 1: Core Abstraction (1 week)

**Task 1.1: Storage Backend Interface**
```python
# src/music_organizer/infrastructure/storage/backend.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Optional

class StorageBackend(ABC):
    @abstractmethod
    async def read(self, path: str) -> bytes: ...

    @abstractmethod
    async def write(self, path: str, data: bytes) -> bool: ...

    @abstractmethod
    async def move(self, src: str, dst: str) -> bool: ...

    @abstractmethod
    async def delete(self, path: str) -> bool: ...

    @abstractmethod
    async def list(self, path: str, pattern: str = "*") -> AsyncIterator[str]: ...

    @abstractmethod
    async def exists(self, path: str) -> bool: ...

    @abstractmethod
    async def get_size(self, path: str) -> int: ...

    @abstractmethod
    async def get_mtime(self, path: str) -> float: ...
```

**Task 1.2: Local Storage Adapter**
```python
# src/music_organizer/infrastructure/storage/local.py
class LocalStorage(StorageBackend):
    """Wraps existing FilesystemAdapter in StorageBackend interface"""
    def __init__(self, root_path: Path):
        self.root = root_path
        self.adapter = FilesystemAdapter()

    async def read(self, path: str) -> bytes:
        return await self.adapter.read_file(self.root / path)
    # ... implement other methods
```

**Deliverables**:
- `StorageBackend` abstract interface
- `LocalStorage` adapter (wraps existing code)
- Unit tests for interface
- No breaking changes to existing code

### Phase 2: S3 Provider (1 week)

**Task 2.1: S3 Storage Adapter**
```python
# src/music_organizer/infrastructure/storage/s3.py
import boto3
from botocore.exceptions import ClientError

class S3Storage(StorageBackend):
    def __init__(self, config: CloudStorageConfig):
        self.config = config
        session = boto3.Session(
            aws_access_key_id=config.credentials.get("access_key"),
            aws_secret_access_key=config.credentials.get("secret_key"),
            region_name=config.region
        )
        self.s3 = session.client("s3")
        self.bucket = config.bucket_name

    async def read(self, path: str) -> bytes:
        response = await asyncio.to_thread(
            self.s3.get_object, Bucket=self.bucket, Key=path
        )
        return response["Body"].read()

    async def write(self, path: str, data: bytes) -> bool:
        try:
            await asyncio.to_thread(
                self.s3.put_object,
                Bucket=self.bucket,
                Key=path,
                Body=data,
                **self._upload_args()
            )
            return True
        except ClientError:
            return False

    async def write_chunked(self, path: str, data: AsyncIterator[bytes]) -> bool:
        """Multipart upload for large files"""
        mp = await asyncio.to_thread(
            self.s3.create_multipart_upload,
            Bucket=self.bucket, Key=path
        )
        parts = []
        part_number = 1
        async for chunk in data:
            part = await asyncio.to_thread(
                self.s3.upload_part,
                Bucket=self.bucket, Key=path, PartNumber=part_number,
                UploadId=mp["UploadId"], Body=chunk
            )
            parts.append({"PartNumber": part_number, "ETag": part["ETag"]})
            part_number += 1

        await asyncio.to_thread(
            self.s3.complete_multipart_upload,
            Bucket=self.bucket, Key=path, UploadId=mp["UploadId"],
            MultipartUpload={"Parts": parts}
        )
        return True
```

**Task 2.2: Configuration**
```python
# config/cloud/s3.json
{
    "provider": "s3",
    "bucket_name": "my-music-library",
    "region": "us-east-1",
    "credentials_env": "AWS_CREDENTIALS",
    "remote_path": "/music",
    "chunk_size": 8388608,
    "max_concurrent_uploads": 4,
    "retry_attempts": 3
}
```

**Deliverables**:
- `S3Storage` adapter with full interface
- Multipart upload support for large files
- Retry logic with exponential backoff
- Integration tests (with LocalStack or moto)

### Phase 3: Hybrid File Operations (1 week)

**Task 3.1: Hybrid File Mover**
```python
# src/music_organizer/core/hybrid_mover.py
class HybridFileMover:
    def __init__(self, config: Config):
        self.local = AsyncFileMover(config)
        if config.cloud_storage:
            self.cloud = self._create_cloud_adapter(config.cloud_storage)
        else:
            self.cloud = None

    async def move_file(
        self,
        audio_file: AudioFile,
        target_path: Path,
        cloud_target: Optional[str] = None
    ) -> Result[MoveOperation]:
        """Move file to local, cloud, or both"""
        results = []

        # Local move
        if target_path:
            result = await self.local.move_file(audio_file, target_path)
            results.append(result)

        # Cloud upload
        if self.cloud and cloud_target:
            data = await self._read_file(audio_file.path)
            success = await self.cloud.write(cloud_target, data)
            results.append(MoveOperation(
                source=str(audio_file.path),
                target=cloud_target,
                success=success,
                operation_type="cloud_upload"
            ))

        return combine_results(results)

    async def sync_to_cloud(self, local_path: Path, remote_path: str) -> List[Result]:
        """Sync local directory to cloud storage"""
        tasks = []
        async for file_path in self._scan_directory(local_path):
            relative = file_path.relative_to(local_path)
            remote = f"{remote_path}/{relative}"
            tasks.append(self._upload_file(file_path, remote))

        return await asyncio.gather(*tasks)
```

**Task 3.2: Sync Engine**
```python
# src/music_organizer/core/sync_engine.py
class SyncEngine:
    async def sync_bidirectional(self, local: Path, remote: str) -> SyncReport:
        """Sync local and cloud storage bidirectionally"""

        # 1. List both sides
        local_files = await self._list_local(local)
        remote_files = await self._list_remote(remote)

        # 2. Detect changes
        changes = self._detect_changes(local_files, remote_files)

        # 3. Resolve conflicts
        resolved = await self._resolve_conflicts(changes)

        # 4. Apply changes
        results = await self._apply_changes(resolved)

        return SyncReport(
            local_added=results.local_added,
            remote_added=results.remote_added,
            conflicts_resolved=results.conflicts_resolved
        )
```

**Deliverables**:
- `HybridFileMover` with local/cloud support
- `SyncEngine` for bidirectional sync
- CLI commands for sync operations
- Tests for hybrid operations

### Phase 4: Additional Providers (Optional)

**Priority Order** (by user demand):
1. Google Cloud Storage (1 week)
2. Azure Blob Storage (1 week)
3. Dropbox (2 weeks - OAuth complexity)
4. Google Drive (2 weeks - OAuth complexity)

## CLI Integration

### New Commands

```bash
# Sync local library to cloud
music-organize-async sync /local/music --to-cloud s3://my-bucket/music

# Sync from cloud to local
music-organize-async sync s3://my-bucket/music --to-local /local/music

# Bidirectional sync
music-organize-async sync /local/music s3://my-bucket/music --bidirectional

# Organize directly in cloud (no local download)
music-organize-async organize s3://source/music s3://target/music --cloud-only

# Preview sync changes
music-organize-async sync /local/music s3://bucket/music --dry-run

# Set bandwidth limit
music-organize-async sync /local/music s3://bucket/music --bandwidth 10M
```

### Configuration

```bash
# Set cloud credentials
music-organize-async config set cloud.provider s3
music-organize-async config set cloud.bucket my-music-library
music-organize-async config set cloud.region us-east-1

# Use environment variables
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
music-organize-async sync /music s3://bucket/music
```

## Performance Considerations

### Upload Performance

| Operation | Local | Cloud (S3) |
|-----------|-------|------------|
| File scan | ~1000 files/sec | ~100 files/sec (API limits) |
| Small file (<1MB) | <1ms | 50-200ms |
| Large file (>100MB) | 100-500ms | 5-30s |
| Metadata | 10-50ms | 50-200ms |

### Optimization Strategies

1. **Multipart Upload**: Upload large files in parallel chunks
2. **Transfer Acceleration**: Enable S3 Transfer Acceleration
3. **Concurrent Uploads**: 4-8 parallel uploads (configurable)
4. **Compression**: Compress metadata before upload
5. **Caching**: Cache cloud file listings locally
6. **Incremental Sync**: Only upload changed files (hash-based)

### Cost Considerations

**S3 Pricing (us-east-1)**:
- Storage: $0.023/GB/month
- Upload: Free
- Download: $0.09/GB
- List requests: $0.005 per 1,000 requests
- Data transfer out: $0.09/GB

**Example**: 10,000 songs (50GB, avg 5MB/song)
- Storage: ~$1.15/month
- Initial upload: Free (PUT requests)
- Full download: ~$4.50 (one-time)
- 1,000 list operations: ~$0.005

## Security and Credentials

### Credential Management

**Option 1: Environment Variables** (Recommended)
```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

**Option 2: Configuration File** (Not recommended for production)
```json
{
    "credentials": {
        "access_key": "...",
        "secret_key": "..."
    }
}
```

**Option 3: IAM Roles** (Best for EC2/Lambda)
- No credentials needed
- Automatic rotation
- Least privilege access

### Encryption

**In Transit**:
- HTTPS/TLS enforced by default
- Signature version 4 (SigV4) for S3

**At Rest**:
- S3: Server-side encryption (SSE-S3, SSE-KMS)
- GCS: Server-side encryption enabled by default
- Client-side encryption: Optional (encrypt before upload)

### Access Control

**S3 Bucket Policy**:
```json
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"AWS": "arn:aws:iam::USER_ID:user/music-organizer"},
        "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
        "Resource": "arn:aws:s3:::my-music-library/*"
    }]
}
```

## Error Handling and Resilience

### Retry Strategy

```python
@backoff.on_exception(
    backoff.expo,
    ClientError,
    max_tries=3,
    jitter=backoff.full_jitter
)
async def write_with_retry(self, path: str, data: bytes) -> bool:
    return await self.cloud.write(path, data)
```

### Transient Error Handling

| Error Type | Retry | Action |
|------------|-------|--------|
| Connection timeout | Yes | Exponential backoff |
| 5xx errors | Yes | Exponential backoff |
| 429 (rate limit) | Yes | Add delay |
| 404 (not found) | No | Fail fast |
| 403 (forbidden) | No | Fail fast, notify user |

### Offline Mode

```python
class OfflineQueue:
    def __init__(self, db_path: Path):
        self.db = sqlite3.connect(db_path)

    async def enqueue_upload(self, local_path: Path, remote_path: str):
        """Queue file for upload when connection restored"""
        self.db.execute(
            "INSERT INTO upload_queue (local, remote, status) VALUES (?, ?, ?)",
            (str(local_path), remote_path, "pending")
        )

    async def process_queue(self):
        """Process pending uploads when online"""
        for task in self.get_pending():
            try:
                await self.upload(task.local, task.remote)
                self.mark_complete(task.id)
            except Exception:
                self.mark_failed(task.id)
```

## Testing Strategy

### Unit Tests

```python
# Use moto for AWS service mocking
import moto
@moto.mock_s3
def test_s3_write():
    storage = S3Storage(config)
    await storage.write("test.mp3", b"audio data")
    assert await storage.exists("test.mp3")
```

### Integration Tests

```python
# Use LocalStack for local S3 emulation
@pytest.fixture
def local_s3():
    with LocalStack() as stack:
        yield stack.s3

def test_sync_to_localstack(local_s3):
    # Test full sync against local S3
    ...
```

### Mock Providers

```python
class MockStorage(StorageBackend):
    """In-memory storage for testing"""
    def __init__(self):
        self.files = {}

    async def write(self, path: str, data: bytes) -> bool:
        self.files[path] = data
        return True

    async def read(self, path: str) -> bytes:
        return self.files.get(path, b"")
```

## Dependency Impact

### New Dependencies

| Package | Version | Size | License |
|---------|---------|------|--------|
| boto3 | >=1.28.0 | ~200KB | Apache 2.0 |
| google-cloud-storage | >=2.10.0 | ~500KB | Apache 2.0 |
| azure-storage-blob | >=12.17.0 | ~400KB | MIT |
| dropbox | >=11.36.0 | ~200KB | MIT |

**Installation**:
```bash
# Install with cloud support
pip install music-organizer[cloud]

# Or specific provider
pip install music-organizer[s3]
pip install music-organizer[gcs]
```

### pyproject.toml

```toml
[project.optional-dependencies]
cloud = ["boto3>=1.28.0", "google-cloud-storage>=2.10.0", "azure-storage-blob>=12.17.0"]
s3 = ["boto3>=1.28.0"]
gcs = ["google-cloud-storage>=2.10.0"]
azure = ["azure-storage-blob>=12.17.0"]
```

## Cost-Benefit Analysis

### Benefits

| Feature | Value | Users Affected |
|---------|-------|----------------|
| Cloud backup | High | All |
| Remote access | Medium | Users with large libraries |
| Cross-device sync | Medium | Multi-device users |
| Reduced local storage | Low | Users with limited disk space |

### Costs

| Cost | Impact | Mitigation |
|------|--------|------------|
| AWS/GCP costs | Low ($1-5/month) | User pays for their storage |
| Complexity | Medium | Optional feature, well-isolated |
| Testing burden | Medium | Comprehensive test suite |
| Dependency bloat | Low | Optional dependencies |

### Comparison

| Alternative | Effort | Value |
|-------------|--------|-------|
| Cloud integration | 6-8 weeks | Medium-High |
| Web UI | 5-8 weeks | High |
| Desktop GUI | 8-12 weeks | Medium |
| Mobile app | 12+ weeks | Low |

## Recommendation

**Proceed with S3 integration as MVP** (3-4 weeks):
1. Core abstraction layer (1 week)
2. S3 provider with multipart upload (1 week)
3. Hybrid file operations (1 week)
4. CLI sync commands (1 week)

**Phase 2**: Add GCS and Azure (2 weeks)

**Future Considerations**:
- Dropbox/Google Drive for personal cloud
- Web UI for cloud library management
- Mobile app for cloud music access

## Conclusion

Cloud storage integration is technically feasible with minimal architectural changes. The existing async patterns and adapter abstraction make this a natural extension of the current codebase.

**Key Success Factors**:
1. Keep it optional - core functionality unchanged
2. Start with S3 - industry standard, well-documented
3. Comprehensive testing - mock services for unit tests
4. Security-first - proper credential management
5. User control - clear sync modes and dry-run support

**Estimated MVP Effort**: 3-4 weeks (S3 only)
**Full Multi-Provider Effort**: 6-8 weeks

---

*Research by task-master agent on 2025-12-23*
