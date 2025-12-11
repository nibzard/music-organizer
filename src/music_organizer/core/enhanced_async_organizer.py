"""Enhanced async orchestration logic with operation history tracking."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple, Union
from datetime import timedelta
import logging
from datetime import datetime

from ..models.audio_file import AudioFile, CoverArt, ContentType
from ..models.config import Config
from ..exceptions import MusicOrganizerError
from ..domain.result import Result, Success, Failure
from .metadata import MetadataHandler
from .cached_metadata import CachedMetadataHandler
from .smart_cached_metadata import SmartCachedMetadataHandler
from .classifier import ContentClassifier
from .async_mover import AsyncFileMover, AsyncDirectoryOrganizer
from .enhanced_file_mover import EnhancedAsyncFileMover, file_operation_session
from .operation_history import OperationHistoryTracker, OperationRollbackService
from .incremental_scanner import IncrementalScanner
from .parallel_metadata import ParallelMetadataExtractor, ExtractionResult
from ..progress_tracker import IntelligentProgressTracker, ProgressStage
from .bulk_operations import BulkMoveOperator, BulkOperationConfig, ConflictStrategy
from .bulk_organizer import BulkAsyncOrganizer
from .bulk_progress_tracker import BulkProgressTracker

logger = logging.getLogger(__name__)


class EnhancedAsyncMusicOrganizer:
    """Enhanced async orchestrator with comprehensive operation history tracking."""

    def __init__(self,
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
                 enable_operation_history: bool = True):
        """
        Initialize the enhanced async music organizer.

        Args:
            config: Configuration object
            dry_run: Whether to perform a dry run (default: False)
            interactive: Whether to enable interactive mode (default: False)
            max_workers: Maximum number of worker threads (default: 4)
            use_cache: Whether to use metadata caching (default: True)
            cache_ttl: Cache TTL in days (default: 30)
            enable_parallel_extraction: Enable parallel metadata extraction (default: True)
            use_processes: Use process pool instead of thread pool (default: False)
            use_smart_cache: Use smart caching with adaptive TTL (default: False)
            session_id: Session ID for operation tracking (auto-generated if None)
            history_tracker: Custom history tracker (creates default if None)
            enable_operation_history: Enable operation history tracking (default: True)
        """
        self.config = config
        self.dry_run = dry_run
        self.interactive = interactive
        self.max_workers = max_workers
        self.use_cache = use_cache
        self.enable_parallel_extraction = enable_parallel_extraction
        self.use_smart_cache = use_smart_cache
        self.enable_operation_history = enable_operation_history

        # Session management
        self.session_id = session_id
        self.current_session = None

        # Initialize progress tracker first
        self.progress_tracker = IntelligentProgressTracker()

        # Initialize operation history tracking
        if self.enable_operation_history:
            self.history_tracker = history_tracker or OperationHistoryTracker()
            self.rollback_service = OperationRollbackService(self.history_tracker)
        else:
            self.history_tracker = None
            self.rollback_service = None

        # Initialize metadata handler
        if self.use_smart_cache:
            self.metadata_handler = SmartCachedMetadataHandler(
                ttl=timedelta(days=cache_ttl or 30),
                enable_warming=True,
                enable_optimization=True
            )
        elif self.use_cache:
            self.metadata_handler = CachedMetadataHandler(ttl=timedelta(days=cache_ttl or 30))
        else:
            self.metadata_handler = MetadataHandler()

        # Initialize parallel metadata extractor if enabled
        if self.enable_parallel_extraction:
            self.parallel_extractor = ParallelMetadataExtractor(
                max_workers=max_workers,
                use_processes=use_processes,
                memory_threshold=80.0,
                batch_size=50,
                enable_memory_monitoring=True,
                progress_tracker=self.progress_tracker
            )
        else:
            self.parallel_extractor = None

        self.classifier = ContentClassifier()

        # Initialize enhanced file mover if operation history is enabled
        if self.enable_operation_history:
            self.file_mover = EnhancedAsyncFileMover(
                backup_enabled=config.file_operations.backup,
                backup_dir=config.target_directory.parent / "backup" if config.file_operations.backup else None,
                max_workers=max_workers,
                history_tracker=self.history_tracker,
                session_id=self.session_id
            )
        else:
            # Fall back to original AsyncFileMover for backward compatibility
            self.file_mover = AsyncFileMover(
                backup_enabled=config.file_operations.backup,
                backup_dir=config.target_directory.parent / "backup" if config.file_operations.backup else None,
                max_workers=max_workers
            )

        self.user_decisions = {}  # Cache user decisions for similar cases

        # Initialize incremental scanner
        self.incremental_scanner = IncrementalScanner()

        # Initialize bulk organizer for large operations
        self.bulk_organizer = BulkAsyncOrganizer(
            config=config,
            max_workers=max_workers,
            enable_parallel_extraction=enable_parallel_extraction,
            history_tracker=self.history_tracker if self.enable_operation_history else None
        )

    async def organize_files(self, source_dir: Path, target_dir: Path) -> Result[Dict]:
        """
        Organize files with comprehensive operation tracking.

        Returns:
            Result with organization statistics
        """
        # Generate session ID if not provided
        if not self.session_id:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Prepare metadata for session
        metadata = {
            "source_directory": str(source_dir),
            "target_directory": str(target_dir),
            "dry_run": self.dry_run,
            "interactive": self.interactive,
            "max_workers": self.max_workers,
            "use_cache": self.use_cache,
            "enable_parallel_extraction": self.enable_parallel_extraction,
            "config": {
                "file_operations": {
                    "backup": self.config.file_operations.backup,
                    "create_subdirs": self.config.file_operations.create_subdirs,
                    "conflict_strategy": self.config.file_operations.conflict_strategy.value
                }
            }
        }

        try:
            # Use operation session context manager
            async with file_operation_session(
                source_root=source_dir,
                target_root=target_dir,
                backup_enabled=self.config.file_operations.backup,
                session_id=self.session_id,
                metadata=metadata
            ) as session:
                self.current_session = session

                # Update progress tracker
                self.progress_tracker.start_stage(
                    ProgressStage.SCAN,
                    "Scanning source directory",
                    estimated_items=None
                )

                # Scan for files
                if hasattr(self, 'enable_incremental') and self.enable_incremental:
                    # Use incremental scanning if enabled
                    file_paths = []
                    async for file_path, is_modified in self.incremental_scanner.scan_directory_async(source_dir):
                        file_paths.append(file_path)
                        self.progress_tracker.update_progress()
                else:
                    # Regular scanning
                    file_paths = []
                    async for file_path in self._scan_directory_async(source_dir):
                        file_paths.append(file_path)
                        self.progress_tracker.update_progress()

                total_files = len(file_paths)
                if total_files == 0:
                    return Success({
                        "session_id": self.session_id,
                        "total_files": 0,
                        "organized_files": 0,
                        "failed_files": 0,
                        "skipped_files": 0,
                        "message": "No audio files found to organize"
                    })

                # Update progress for extraction stage
                self.progress_tracker.start_stage(
                    ProgressStage.EXTRACTION,
                    "Extracting metadata",
                    estimated_items=total_files
                )

                # Extract metadata with or without parallel processing
                if self.enable_parallel_extraction and self.parallel_extractor:
                    extraction_result = await self._extract_metadata_parallel(file_paths)
                    audio_files = extraction_result.successful
                    failed_files = extraction_result.failed
                else:
                    audio_files = []
                    failed_files = []
                    for file_path in file_paths:
                        result = await self._extract_metadata_single(file_path)
                        if result.is_success():
                            audio_files.append(result.value())
                        else:
                            failed_files.append((file_path, str(result.error())))

                # Update progress for classification stage
                self.progress_tracker.start_stage(
                    ProgressStage.CLASSIFY,
                    "Classifying content",
                    estimated_items=len(audio_files)
                )

                # Classify files
                for audio_file in audio_files:
                    try:
                        audio_file.content_type = await self._classify_content_async(audio_file)
                    except Exception as e:
                        logger.warning(f"Failed to classify {audio_file.path}: {e}")
                        audio_file.content_type = ContentType.UNKNOWN
                    self.progress_tracker.update_progress()

                # Update progress for organization stage
                self.progress_tracker.start_stage(
                    ProgressStage.ORGANIZE,
                    "Organizing files",
                    estimated_items=len(audio_files)
                )

                # Organize files
                organized_files = []
                organization_failures = []

                if not self.dry_run:
                    for audio_file in audio_files:
                        try:
                            result = await self._organize_single_file(audio_file, target_dir)
                            if result.is_success():
                                organized_files.append(audio_file)
                            else:
                                organization_failures.append((audio_file.path, str(result.error())))
                        except Exception as e:
                            organization_failures.append((audio_file.path, str(e)))
                        self.progress_tracker.update_progress()
                else:
                    # Dry run - just simulate organization
                    for audio_file in audio_files:
                        try:
                            target_path = await self._get_target_path(audio_file, target_dir)
                            organized_files.append(audio_file)
                            self.progress_tracker.update_progress()
                        except Exception as e:
                            organization_failures.append((audio_file.path, str(e)))

                # Complete organization
                stats = {
                    "session_id": self.session_id,
                    "total_files": total_files,
                    "organized_files": len(organized_files),
                    "failed_files": len(failed_files) + len(organization_failures),
                    "metadata_extraction_failures": len(failed_files),
                    "organization_failures": len(organization_failures),
                    "skipped_files": 0,
                    "dry_run": self.dry_run,
                    "operation_history_enabled": self.enable_operation_history
                }

                # Add session summary if operation history is enabled
                if self.enable_operation_history and self.history_tracker:
                    summary_result = await self.history_tracker.get_session(self.session_id)
                    if summary_result:
                        session = await self.history_tracker.get_session(self.session_id)
                        if session:
                            stats["session_summary"] = {
                                "start_time": session.start_time.isoformat(),
                                "end_time": session.end_time.isoformat() if session.end_time else None,
                                "status": session.status,
                                "completed_operations": session.completed_operations,
                                "failed_operations": session.failed_operations
                            }

                return Success(stats)

        except Exception as e:
            # End session with error status
            if self.enable_operation_history and self.history_tracker and self.session_id:
                await self.history_tracker.end_session(self.session_id, "failed")
            return Failure(f"Organization failed: {str(e)}")

    async def get_operation_history(self) -> Result[List[Dict]]:
        """Get operation history for the current session."""
        if not self.enable_operation_history or not self.history_tracker:
            return Failure("Operation history is not enabled")

        try:
            if not self.session_id:
                return Failure("No active session")

            operations = await self.history_tracker.get_session_operations(self.session_id)
            return Success([op.to_dict() for op in operations])
        except Exception as e:
            return Failure(f"Failed to get operation history: {str(e)}")

    async def rollback_session(self, dry_run: bool = False) -> Result[Dict]:
        """Rollback the current session."""
        if not self.enable_operation_history or not self.rollback_service:
            return Failure("Operation rollback is not enabled")

        if not self.session_id:
            return Failure("No session to rollback")

        return await self.rollback_service.rollback_session(self.session_id, dry_run)

    async def list_recent_sessions(self, limit: int = 10) -> Result[List[Dict]]:
        """List recent operation sessions."""
        if not self.enable_operation_history or not self.history_tracker:
            return Failure("Operation history is not enabled")

        try:
            sessions = await self.history_tracker.list_sessions(limit)
            return Success([session.to_dict() for session in sessions])
        except Exception as e:
            return Failure(f"Failed to list sessions: {str(e)}")

    async def organize_files_bulk(self, source_dir: Path, target_dir: Path,
                                 bulk_config: Optional[BulkOperationConfig] = None) -> Result[Dict]:
        """
        Organize files using bulk operations with operation tracking.

        Args:
            source_dir: Source directory containing music files
            target_dir: Target directory for organized files
            bulk_config: Optional bulk operation configuration

        Returns:
            Result with bulk organization statistics
        """
        if not bulk_config:
            bulk_config = BulkOperationConfig(
                max_workers=self.max_workers,
                conflict_strategy=self.config.file_operations.conflict_strategy
            )

        # Set session ID in bulk organizer
        if self.enable_operation_history:
            self.bulk_organizer.session_id = self.session_id

        return await self.bulk_organizer.organize_bulk(source_dir, target_dir, bulk_config)

    # Private helper methods

    async def _scan_directory_async(self, directory: Path) -> AsyncGenerator[Path, None]:
        """Async scan directory for audio files using a generator."""
        if not directory.exists():
            raise MusicOrganizerError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise MusicOrganizerError(f"Path is not a directory: {directory}")

        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}

        def _scan_sync():
            """Synchronous scan to be run in thread pool."""
            for file_path in directory.rglob('*'):
                if file_path.suffix.lower() in audio_extensions:
                    yield file_path

        # Run scanning in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            async for file_path in loop.run_in_executor(executor, lambda: _scan_sync()):
                yield file_path

    async def _extract_metadata_parallel(self, file_paths: List[Path]) -> ExtractionResult:
        """Extract metadata using parallel processing."""
        if not self.parallel_extractor:
            raise RuntimeError("Parallel extractor not initialized")

        return await self.parallel_extractor.extract_metadata_batch(file_paths)

    async def _extract_metadata_single(self, file_path: Path) -> Result[AudioFile]:
        """Extract metadata for a single file."""
        try:
            # Check cache first if enabled
            if hasattr(self.metadata_handler, 'get_cached'):
                cached_result = await self.metadata_handler.get_cached(file_path)
                if cached_result:
                    return Success(cached_result)

            # Extract metadata
            metadata = await asyncio.get_event_loop().run_in_executor(
                None, self.metadata_handler.extract_metadata, file_path
            )

            if metadata is None:
                return Failure(f"Could not extract metadata from {file_path}")

            return Success(metadata)

        except Exception as e:
            return Failure(f"Failed to extract metadata from {file_path}: {str(e)}")

    async def _classify_content_async(self, audio_file: AudioFile) -> ContentType:
        """Classify content type asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.classifier.classify, audio_file
        )

    async def _get_target_path(self, audio_file: AudioFile, target_dir: Path) -> Path:
        """Get target path for an audio file."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.config.get_target_path, audio_file, target_dir
        )

    async def _organize_single_file(self, audio_file: AudioFile, target_dir: Path) -> Result[Path]:
        """Organize a single audio file."""
        try:
            # Get target path
            target_path = await self._get_target_path(audio_file, target_dir)

            # Move file using enhanced file mover if available
            if isinstance(self.file_mover, EnhancedAsyncFileMover):
                result = await self.file_mover.move_file(audio_file, target_path)
                if result.is_failure():
                    return result
                return result
            else:
                # Fallback to original AsyncFileMover
                moved_path = await self.file_mover.move_file(audio_file, target_path)
                return Success(moved_path)

        except Exception as e:
            return Failure(f"Failed to organize {audio_file.path}: {str(e)}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources
        if hasattr(self, 'file_mover') and hasattr(self.file_mover, '__aexit__'):
            await self.file_mover.__aexit__(exc_type, exc_val, exc_tb)

        # End session if error occurred
        if exc_type and self.enable_operation_history and self.history_tracker and self.session_id:
            await self.history_tracker.end_session(self.session_id, "failed")