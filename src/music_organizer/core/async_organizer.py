"""Async orchestration logic for music organization."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from datetime import timedelta
import logging

from ..models.audio_file import AudioFile, CoverArt, ContentType
from ..models.config import Config
from ..exceptions import MusicOrganizerError
from .metadata import MetadataHandler
from .cached_metadata import CachedMetadataHandler
from .classifier import ContentClassifier
from .async_mover import AsyncFileMover, AsyncDirectoryOrganizer

logger = logging.getLogger(__name__)


class AsyncMusicOrganizer:
    """Async orchestrator for music library organization."""

    def __init__(self,
                 config: Config,
                 dry_run: bool = False,
                 interactive: bool = False,
                 max_workers: int = 4,
                 use_cache: bool = True,
                 cache_ttl: Optional[int] = None):
        """
        Initialize the async music organizer.

        Args:
            config: Configuration object
            dry_run: Whether to perform a dry run (default: False)
            interactive: Whether to enable interactive mode (default: False)
            max_workers: Maximum number of worker threads (default: 4)
            use_cache: Whether to use metadata caching (default: True)
            cache_ttl: Cache TTL in days (default: 30)
        """
        self.config = config
        self.dry_run = dry_run
        self.interactive = interactive
        self.max_workers = max_workers
        self.use_cache = use_cache

        # Use cached metadata handler if enabled
        if self.use_cache:
            self.metadata_handler = CachedMetadataHandler(ttl=timedelta(days=cache_ttl or 30))
        else:
            self.metadata_handler = MetadataHandler()

        self.classifier = ContentClassifier()
        self.file_mover = AsyncFileMover(
            backup_enabled=config.file_operations.backup,
            backup_dir=config.target_directory.parent / "backup" if config.file_operations.backup else None,
            max_workers=max_workers
        )
        self.user_decisions = {}  # Cache user decisions for similar cases

    async def scan_directory(self, directory: Path) -> AsyncGenerator[Path, None]:
        """Async scan directory for audio files using a generator."""
        if not directory.exists():
            raise MusicOrganizerError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise MusicOrganizerError(f"Path is not a directory: {directory}")

        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}

        def _scan_files():
            files = []
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    files.append(file_path)
            return files

        # Run the file system scan in a thread pool
        files = await asyncio.get_event_loop().run_in_executor(
            None, _scan_files
        )

        logger.info(f"Found {len(files)} audio files in {directory}")

        # Yield files as they're found
        for file_path in files:
            yield file_path

    async def scan_directory_batch(self, directory: Path, batch_size: int = 100) -> AsyncGenerator[List[Path], None]:
        """Async scan directory yielding files in batches."""
        batch = []
        async for file_path in self.scan_directory(directory):
            batch.append(file_path)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:  # Yield the last batch if it's not empty
            yield batch

    async def organize_files(self, files: List[Path], progress=None, task_id=None) -> Dict[str, Any]:
        """Organize a list of audio files asynchronously."""
        results = {
            'processed': 0,
            'moved': 0,
            'skipped': 0,
            'by_category': {
                'Albums': 0,
                'Live': 0,
                'Collaborations': 0,
                'Compilations': 0,
                'Rarities': 0,
                'Unknown': 0
            },
            'errors': []
        }

        # Start file mover session
        if not self.dry_run:
            await self.file_mover.start_operation(self.config.source_directory)

        try:
            # Create target directory structure
            if not self.dry_run:
                await AsyncDirectoryOrganizer.create_directory_structure(
                    self.config.target_directory
                )

            # Process files in parallel batches
            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_with_semaphore(file_path):
                async with semaphore:
                    return await self._process_file(file_path)

            # Process all files concurrently
            tasks = [process_with_semaphore(file_path) for file_path in files]
            processed_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in processed_results:
                if isinstance(result, Exception):
                    error_msg = f"Failed to process file: {result}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
                    results['skipped'] += 1
                else:
                    if result:
                        results['moved'] += 1
                        # Update category count
                        if hasattr(result, 'content_type'):
                            category = self._get_category_name(result.content_type)
                            results['by_category'][category] += 1
                    else:
                        results['skipped'] += 1

                results['processed'] += 1

                # Update progress
                if progress and task_id is not None:
                    progress.advance(task_id)

        finally:
            # Finish file mover session
            if not self.dry_run:
                await self.file_mover.finish_operation()

        return results

    async def organize_files_streaming(self,
                                     file_generator: AsyncGenerator[Path, None],
                                     batch_size: int = 50,
                                     progress=None) -> AsyncGenerator[Tuple[Path, bool, Optional[str]], None]:
        """Organize files using streaming approach for memory efficiency."""
        # Start file mover session
        if not self.dry_run:
            await self.file_mover.start_operation(self.config.source_directory)

        try:
            # Create target directory structure
            if not self.dry_run:
                await AsyncDirectoryOrganizer.create_directory_structure(
                    self.config.target_directory
                )

            # Process files in batches
            batch = []
            async for file_path in file_generator:
                batch.append(file_path)

                if len(batch) >= batch_size:
                    # Process the batch
                    async for result in self._process_batch(batch):
                        yield result
                    batch = []

            # Process the last batch
            if batch:
                async for result in self._process_batch(batch):
                    yield result

        finally:
            # Finish file mover session
            if not self.dry_run:
                await self.file_mover.finish_operation()

    async def _process_batch(self, batch: List[Path]) -> AsyncGenerator[Tuple[Path, bool, Optional[str]], None]:
        """Process a batch of files concurrently."""
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(file_path):
            async with semaphore:
                try:
                    result = await self._process_file(file_path)
                    return file_path, True, None
                except Exception as e:
                    return file_path, False, str(e)

        # Process all files in the batch concurrently
        tasks = [process_with_semaphore(file_path) for file_path in batch]
        for result in asyncio.as_completed(tasks):
            yield await result

    async def _process_file(self, file_path: Path) -> Optional[AudioFile]:
        """Process a single audio file asynchronously."""
        # Extract metadata (with cache if enabled)
        def _extract_metadata():
            if self.use_cache:
                return self.metadata_handler.extract_metadata(file_path, use_cache=True)
            else:
                return self.metadata_handler.extract_metadata(file_path)

        audio_file = await asyncio.get_event_loop().run_in_executor(
            None, _extract_metadata
        )

        # Classify content
        def _classify():
            return self.classifier.classify(audio_file)

        content_type, confidence = await asyncio.get_event_loop().run_in_executor(
            None, _classify
        )

        # Interactive mode for ambiguous cases
        if self.interactive and self.classifier.is_ambiguous(audio_file):
            content_type = await self._get_user_classification(audio_file, content_type, confidence)

        audio_file.content_type = content_type

        if self.dry_run:
            # Just show what would happen
            target_path = audio_file.get_target_path(self.config.target_directory)
            target_filename = audio_file.get_target_filename()
            full_target = target_path / target_filename
            print(f"Would move: {file_path.name} -> {full_target.relative_to(self.config.target_directory)}")
            return audio_file

        # Move the file
        target_dir = audio_file.get_target_path(self.config.target_directory)
        target_filename = audio_file.get_target_filename()
        target_path = target_dir / target_filename

        await self.file_mover.move_file(audio_file, target_path)

        # Handle cover art
        await self._process_cover_art(file_path, target_dir)

        return audio_file

    async def _process_cover_art(self, audio_file_path: Path, target_dir: Path) -> None:
        """Find and move cover art for an audio file asynchronously."""
        def _find_cover_art():
            return self.metadata_handler.find_cover_art(audio_file_path.parent)

        cover_files = await asyncio.get_event_loop().run_in_executor(
            None, _find_cover_art
        )

        for cover_path in cover_files:
            def _load_cover_art():
                return CoverArt.from_file(cover_path)

            cover_art = await asyncio.get_event_loop().run_in_executor(
                None, _load_cover_art
            )

            if cover_art:
                if self.dry_run:
                    print(f"Would move cover art: {cover_path.name} -> {target_dir}")
                else:
                    await self.file_mover.move_cover_art(cover_art, target_dir)

    async def _get_user_classification(self,
                                      audio_file: AudioFile,
                                      suggested_type,
                                      confidence: float):
        """Get user input for ambiguous classifications."""
        # Check if we have a similar decision cached
        cache_key = self._get_cache_key(audio_file)
        if cache_key in self.user_decisions:
            return self.user_decisions[cache_key]

        from ..console_utils import SimpleConsole

        console = SimpleConsole()

        console.print(f"\nAmbiguous classification for:", 'yellow')
        console.print(f"  File: {audio_file.path.name}")
        console.print(f"  Artists: {', '.join(audio_file.artists[:3]) if audio_file.artists else 'Unknown'}")
        console.print(f"  Album: {audio_file.album or 'Unknown'}")
        console.print(f"  Suggested: {suggested_type.value} (confidence: {confidence:.2f})")

        # Show available options
        options = {
            '1': 'Album',
            '2': 'Live Recording',
            '3': 'Collaboration',
            '4': 'Compilation',
            '5': 'Rarity/Special Edition',
            '6': 'Skip this file'
        }

        console.print("\nSelect category:", 'cyan')
        for key, label in options.items():
            console.print(f"  {key}. {label}")

        choice = console.prompt("Your choice", default='1')
        while choice not in options:
            console.print("Invalid choice. Please try again.", 'red')
            choice = console.prompt("Your choice", default='1')

        # Map choice to content type
        type_map = {
            '1': 'studio',
            '2': 'live',
            '3': 'collaboration',
            '4': 'compilation',
            '5': 'rarity',
            '6': None  # Skip
        }

        selected_type = type_map.get(choice)

        if selected_type:
            result = ContentType(selected_type)

            # Cache the decision
            self.user_decisions[cache_key] = result

            return result
        else:
            raise MusicOrganizerError("User chose to skip file")

    def _get_cache_key(self, audio_file: AudioFile) -> str:
        """Create a cache key for user decisions."""
        # Use a combination of artist and album for caching similar decisions
        artist_key = "_".join(audio_file.artists[:2]) if audio_file.artists else "unknown"
        album_key = audio_file.album or "unknown"
        return f"{artist_key}:{album_key}"

    def _get_category_name(self, content_type: ContentType) -> str:
        """Map content type to category name for results."""
        category_map = {
            ContentType.STUDIO: 'Albums',
            ContentType.LIVE: 'Live',
            ContentType.COLLABORATION: 'Collaborations',
            ContentType.COMPILATION: 'Compilations',
            ContentType.RARITY: 'Rarities',
            ContentType.UNKNOWN: 'Unknown'
        }

        return category_map.get(content_type, 'Unknown')

    async def rollback(self) -> None:
        """Rollback all changes made in the current session."""
        await self.file_mover.rollback()

    async def get_operation_summary(self) -> Dict[str, Any]:
        """Get a summary of the operations performed."""
        summary = await self.file_mover.get_operation_summary()

        # Add cache stats if caching is enabled
        if self.use_cache:
            cache_stats = self.metadata_handler.get_cache_stats()
            summary['cache'] = cache_stats

        return summary

    def cleanup_cache(self) -> int:
        """Clean up expired cache entries."""
        if self.use_cache:
            return self.metadata_handler.cleanup_expired()
        return 0

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.use_cache:
            return self.metadata_handler.get_cache_stats()
        return None

    def clear_cache(self) -> None:
        """Clear all cache entries."""
        if self.use_cache:
            self.metadata_handler.clear_cache()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.file_mover.__aexit__(exc_type, exc_val, exc_tb)


# Utility function for running async operations from sync code
async def run_async_organize(config: Config,
                           dry_run: bool = False,
                           interactive: bool = False) -> Dict[str, Any]:
    """Run the async organizer and return results."""
    async with AsyncMusicOrganizer(config, dry_run, interactive) as organizer:
        # Scan files in batches
        file_batches = []
        async for batch in organizer.scan_directory_batch(config.source_directory, batch_size=100):
            file_batches.extend(batch)

        # Organize all files
        return await organizer.organize_files(file_batches)


def organize_files_async(config: Config,
                        dry_run: bool = False,
                        interactive: bool = False) -> Dict[str, Any]:
    """Convenience function to run async organization from sync code."""
    return asyncio.run(run_async_organize(config, dry_run, interactive))