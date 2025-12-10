"""Main orchestration logic for music organization."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..models.audio_file import AudioFile, CoverArt
from ..models.config import Config
from ..exceptions import MusicOrganizerError
from .metadata import MetadataHandler
from .classifier import ContentClassifier
from .mover import FileMover, DirectoryOrganizer

logger = logging.getLogger(__name__)


class MusicOrganizer:
    """Main orchestrator for music library organization."""

    def __init__(self, config: Config, dry_run: bool = False, interactive: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.interactive = interactive
        self.metadata_handler = MetadataHandler()
        self.classifier = ContentClassifier()
        self.file_mover = FileMover(
            backup_enabled=config.file_operations.backup,
            backup_dir=config.target_directory.parent / "backup" if config.file_operations.backup else None
        )
        self.user_decisions = {}  # Cache user decisions for similar cases

    def scan_directory(self, directory: Path) -> List[Path]:
        """Scan directory for audio files."""
        if not directory.exists():
            raise MusicOrganizerError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise MusicOrganizerError(f"Path is not a directory: {directory}")

        # Find audio files
        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}
        files = []

        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                files.append(file_path)

        logger.info(f"Found {len(files)} audio files in {directory}")
        return files

    def organize_files(self, files: List[Path], progress=None, task_id=None) -> Dict[str, Any]:
        """Organize a list of audio files."""
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
            self.file_mover.start_operation(self.config.source_directory)

        try:
            # Create target directory structure
            if not self.dry_run:
                DirectoryOrganizer.create_directory_structure(self.config.target_directory)

            # Group files by album/cover art
            file_groups = self._group_files(files)

            # Process each group
            for group_path, group_files in file_groups.items():
                for file_path in group_files:
                    try:
                        # Process individual file
                        moved = self._process_file(file_path)
                        if moved:
                            results['moved'] += 1
                        else:
                            results['skipped'] += 1

                        # Update category count
                        if hasattr(moved, 'content_type'):
                            category = self._get_category_name(moved.content_type)
                            results['by_category'][category] += 1

                    except Exception as e:
                        error_msg = f"Failed to process {file_path.name}: {e}"
                        results['errors'].append(error_msg)
                        logger.error(error_msg)

                    results['processed'] += 1

                    # Update progress
                    if progress and task_id is not None:
                        progress.advance(task_id)

        finally:
            # Finish file mover session
            if not self.dry_run:
                self.file_mover.finish_operation()

        return results

    def _process_file(self, file_path: Path) -> Optional[AudioFile]:
        """Process a single audio file."""
        # Extract metadata
        audio_file = self.metadata_handler.extract_metadata(file_path)

        # Classify content
        content_type, confidence = self.classifier.classify(audio_file)

        # Interactive mode for ambiguous cases
        if self.interactive and self.classifier.is_ambiguous(audio_file):
            content_type = self._get_user_classification(audio_file, content_type, confidence)

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

        self.file_mover.move_file(audio_file, target_path)

        # Handle cover art
        self._process_cover_art(file_path, target_dir)

        return audio_file

    def _group_files(self, files: List[Path]) -> Dict[Path, List[Path]]:
        """Group files by their containing directory for batch processing."""
        groups = {}
        for file_path in files:
            group_path = file_path.parent
            if group_path not in groups:
                groups[group_path] = []
            groups[group_path].append(file_path)
        return groups

    def _process_cover_art(self, audio_file_path: Path, target_dir: Path) -> None:
        """Find and move cover art for an audio file."""
        # Look for cover art in the same directory
        cover_files = self.metadata_handler.find_cover_art(audio_file_path.parent)

        for cover_path in cover_files:
            cover_art = CoverArt.from_file(cover_path)
            if cover_art:
                if self.dry_run:
                    print(f"Would move cover art: {cover_path.name} -> {target_dir}")
                else:
                    self.file_mover.move_cover_art(cover_art, target_dir)

    def _get_user_classification(self, audio_file: AudioFile, suggested_type, confidence: float):
        """Get user input for ambiguous classifications."""
        # Check if we have a similar decision cached
        cache_key = self._get_cache_key(audio_file)
        if cache_key in self.user_decisions:
            return self.user_decisions[cache_key]

        from rich.console import Console
        from rich.prompt import Prompt

        console = Console()

        console.print(f"\n[yellow]Ambiguous classification for:[/yellow]")
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

        console.print("\n[cyan]Select category:[/cyan]")
        for key, label in options.items():
            console.print(f"  {key}. {label}")

        choice = Prompt.ask("Your choice", choices=list(options.keys()), default='1')

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
            from ..models.audio_file import ContentType
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

    def _get_category_name(self, content_type) -> str:
        """Map content type to category name for results."""
        from ..models.audio_file import ContentType

        category_map = {
            ContentType.STUDIO: 'Albums',
            ContentType.LIVE: 'Live',
            ContentType.COLLABORATION: 'Collaborations',
            ContentType.COMPILATION: 'Compilations',
            ContentType.RARITY: 'Rarities',
            ContentType.UNKNOWN: 'Unknown'
        }

        return category_map.get(content_type, 'Unknown')

    def rollback(self) -> None:
        """Rollback all changes made in the current session."""
        self.file_mover.rollback()

    def get_operation_summary(self) -> Dict[str, Any]:
        """Get a summary of the operations performed."""
        return self.file_mover.get_operation_summary()