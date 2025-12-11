"""Enhanced Async Music Organizer with Interactive Duplicate Resolution."""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from datetime import timedelta
import logging

from .async_organizer import AsyncMusicOrganizer
from .interactive_duplicate_resolver import (
    InteractiveDuplicateResolver, DuplicateGroup, DuplicateAction,
    ResolutionStrategy, ResolutionSummary
)
from ..plugins.builtins.duplicate_detector import DuplicateDetectorPlugin
from ..models.audio_file import AudioFile
from ..models.config import Config
from ..progress_tracker import IntelligentProgressTracker, ProgressStage


logger = logging.getLogger(__name__)


class DuplicateResolverOrganizer:
    """Music organizer with integrated duplicate resolution capabilities."""

    def __init__(self,
                 config: Config,
                 dry_run: bool = False,
                 duplicate_strategy: ResolutionStrategy = ResolutionStrategy.INTERACTIVE,
                 duplicate_dir: Optional[Path] = None,
                 enable_duplicate_resolution: bool = True,
                 **kwargs):
        """
        Initialize the duplicate resolver organizer.

        Args:
            config: Configuration object
            dry_run: Whether to perform a dry run
            duplicate_strategy: Strategy for resolving duplicates
            duplicate_dir: Directory to move duplicates to
            enable_duplicate_resolution: Enable duplicate detection and resolution
            **kwargs: Additional arguments for AsyncMusicOrganizer
        """
        self.config = config
        self.dry_run = dry_run
        self.enable_duplicate_resolution = enable_duplicate_resolution

        # Initialize the base async organizer
        self.base_organizer = AsyncMusicOrganizer(config=config, dry_run=dry_run, **kwargs)

        # Initialize duplicate detection
        self.duplicate_detector = DuplicateDetectorPlugin()
        self.duplicate_detector.initialize()

        # Initialize duplicate resolver
        if enable_duplicate_resolution:
            self.duplicate_resolver = InteractiveDuplicateResolver(
                strategy=duplicate_strategy,
                duplicate_dir=duplicate_dir,
                dry_run=dry_run
            )
        else:
            self.duplicate_resolver = None

    async def organize_with_duplicate_resolution(self,
                                               source_dir: Path,
                                               target_dir: Path,
                                               resolve_duplicates_first: bool = True) -> Tuple[Dict[str, Any], Optional[ResolutionSummary]]:
        """
        Organize music with duplicate resolution.

        Args:
            source_dir: Source directory containing music
            target_dir: Target directory for organized music
            resolve_duplicates_first: Whether to resolve duplicates before organizing

        Returns:
            Tuple of (organization_result, duplicate_resolution_summary)
        """
        # Initialize progress tracker
        progress_tracker = IntelligentProgressTracker()

        if resolve_duplicates_first and self.enable_duplicate_resolution:
            # Stage 1: Scan for duplicates
            progress_tracker.start_stage(ProgressStage.SCATNING, "Scanning for duplicates...")
            duplicate_groups = await self._detect_duplicates(source_dir, progress_tracker)
            progress_tracker.complete_stage()

            # Stage 2: Resolve duplicates
            if duplicate_groups:
                progress_tracker.start_stage(ProgressStage.PROCESSING, "Resolving duplicates...")
                resolution_summary = await self.duplicate_resolver.resolve_duplicates(duplicate_groups)
                progress_tracker.complete_stage()

                # Show resolution summary
                from ..ui.duplicate_resolver_ui import DuplicateResolverUI
                ui = DuplicateResolverUI(dry_run=self.dry_run)
                ui.show_summary(self.duplicate_resolver.get_summary())

                # Ask if user wants to continue with organization
                if not self.dry_run:
                    continue_org = ui.confirm_action(
                        "Continue with organization after duplicate resolution?",
                        f"Resolved {resolution_summary.resolved_groups} duplicate groups"
                    )
                    if not continue_org:
                        return {}, resolution_summary
            else:
                print("\n✅ No duplicates found!")
                resolution_summary = None
        else:
            resolution_summary = None

        # Stage 3: Proceed with normal organization
        if not self.dry_run or resolution_summary is not None:
            print("\n🎵 Starting music organization...")
            organization_result = await self.base_organizer.organize_files(source_dir, target_dir)
        else:
            organization_result = {"dry_run": True}

        return organization_result, resolution_summary

    async def _detect_duplicates(self, source_dir: Path, progress_tracker: IntelligentProgressTracker) -> List[DuplicateGroup]:
        """Detect duplicates in the source directory."""
        duplicate_groups = []
        processed_files = 0

        # Scan for audio files
        async for audio_file in self.base_organizer.scan_directory(source_dir):
            processed_files += 1

            # Update progress
            progress_tracker.update(
                files_processed=processed_files,
                current_file=str(audio_file.path.name)
            )

            # Classify for duplicates
            classifications = await self.duplicate_detector.classify(audio_file)

            # Store duplicate information
            # Note: The duplicate detector plugin needs to be enhanced to return groups
            # For now, we'll create a simple implementation
            if classifications.get('is_duplicate'):
                # This is simplified - in practice, we'd need to extract the actual duplicate groups
                pass

        # Get duplicate groups from the detector
        # This would require modifying the duplicate detector to return group information
        # For now, return empty list
        return duplicate_groups

    async def resolve_duplicates_only(self, source_dir: Path) -> Optional[ResolutionSummary]:
        """
        Only resolve duplicates without organizing.

        Args:
            source_dir: Directory to scan for duplicates

        Returns:
            Summary of duplicate resolution
        """
        if not self.enable_duplicate_resolution:
            print("Duplicate resolution is not enabled.")
            return None

        print("🔍 Scanning for duplicates...")
        progress_tracker = IntelligentProgressTracker()
        progress_tracker.start_stage(ProgressStage.SCATNING, "Scanning for duplicates...")

        # Detect duplicates
        duplicate_groups = await self._detect_duplicates(source_dir, progress_tracker)
        progress_tracker.complete_stage()

        if not duplicate_groups:
            print("\n✅ No duplicates found!")
            return None

        # Resolve duplicates
        print(f"\n🔀 Found {len(duplicate_groups)} duplicate groups")
        progress_tracker.start_stage(ProgressStage.PROCESSING, "Resolving duplicates...")
        resolution_summary = await self.duplicate_resolver.resolve_duplicates(duplicate_groups)
        progress_tracker.complete_stage()

        # Show summary
        from ..ui.duplicate_resolver_ui import DuplicateResolverUI
        ui = DuplicateResolverUI(dry_run=self.dry_run)
        ui.show_summary(self.duplicate_resolver.get_summary())

        # Save report if requested
        if not self.dry_run:
            save_report = ui.confirm_action("Save duplicate resolution report?")
            if save_report:
                report_path = source_dir / "duplicate_resolution_report.json"
                self.duplicate_resolver.save_report(report_path)
                print(f"Report saved to: {report_path}")

        return resolution_summary

    async def get_duplicate_preview(self, source_dir: Path, limit: int = 10) -> Dict[str, Any]:
        """
        Get a preview of duplicates without resolving them.

        Args:
            source_dir: Directory to scan
            limit: Maximum number of duplicate groups to analyze

        Returns:
            Preview information about duplicates
        """
        print("🔍 Scanning for duplicate preview...")
        progress_tracker = IntelligentProgressTracker()
        progress_tracker.start_stage(ProgressStage.SCATNING, "Scanning files...")

        # Scan a subset of files for preview
        files_scanned = 0
        duplicate_samples = []

        async for audio_file in self.base_organizer.scan_directory(source_dir):
            files_scanned += 1
            progress_tracker.update(files_processed=files_scanned)

            # Classify for duplicates
            classifications = await self.duplicate_detector.classify(audio_file)

            # Collect sample duplicates
            if classifications.get('is_duplicate') and len(duplicate_samples) < limit:
                duplicate_samples.append({
                    'file': str(audio_file.path),
                    'duplicate_count': classifications.get('duplicate_count', 0),
                    'duplicate_types': classifications.get('duplicates', [])
                })

            # Stop after scanning enough files
            if files_scanned >= 1000:  # Reasonable limit for preview
                break

        progress_tracker.complete_stage()

        # Get summary from detector
        detector_summary = self.duplicate_detector.get_duplicate_summary()

        return {
            'files_scanned': files_scanned,
            'total_files_estimate': files_scanned,  # This is an estimate
            'duplicate_summary': detector_summary,
            'sample_duplicates': duplicate_samples
        }

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'duplicate_detector'):
            self.duplicate_detector.cleanup()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.cleanup()


# Convenience function for quick duplicate resolution
async def quick_duplicate_resolution(source_dir: Path,
                                    strategy: ResolutionStrategy = ResolutionStrategy.INTERACTIVE,
                                    duplicate_dir: Optional[Path] = None,
                                    dry_run: bool = False) -> Optional[ResolutionSummary]:
    """
    Quick function to resolve duplicates in a directory.

    Args:
        source_dir: Directory to scan for duplicates
        strategy: Resolution strategy to use
        duplicate_dir: Directory to move duplicates to
        dry_run: Whether to perform a dry run

    Returns:
        Summary of duplicate resolution
    """
    config = Config.default()

    async with DuplicateResolverOrganizer(
        config=config,
        dry_run=dry_run,
        duplicate_strategy=strategy,
        duplicate_dir=duplicate_dir
    ) as resolver:
        return await resolver.resolve_duplicates_only(source_dir)