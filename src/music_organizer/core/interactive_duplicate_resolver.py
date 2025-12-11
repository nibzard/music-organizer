"""Interactive Duplicate Resolution System.

This module provides an interactive system for resolving duplicate audio files
with side-by-side comparison and user-friendly decision making.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, NamedTuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

from ..models.audio_file import AudioFile
from ..plugins.builtins.duplicate_detector import DuplicateDetectorPlugin
from ..exceptions import MusicOrganizerError


class DuplicateAction(Enum):
    """Actions that can be taken on duplicates."""
    KEEP_FIRST = "keep_first"
    KEEP_SECOND = "keep_second"
    KEEP_BOTH = "keep_both"
    KEEP_BEST = "keep_best_quality"
    MOVE_DUPLICATE = "move_duplicate"
    DELETE_DUPLICATE = "delete_duplicate"


class ResolutionStrategy(Enum):
    """Overall resolution strategies."""
    INTERACTIVE = "interactive"  # Ask user for each duplicate
    AUTO_KEEP_BEST = "auto_keep_best"  # Automatically keep best quality
    AUTO_FIRST = "auto_first"  # Always keep first file found
    AUTO_SMART = "auto_smart"  # Smart decisions based on metadata


@dataclass
class DuplicatePair:
    """Represents a pair of duplicate files."""
    file1: AudioFile
    file2: AudioFile
    duplicate_type: str  # 'exact', 'metadata', 'acoustic'
    confidence: float
    reason: str


@dataclass
class DuplicateGroup:
    """Represents a group of duplicate files (more than 2)."""
    files: List[AudioFile]
    duplicate_type: str
    confidence: float
    reason: str


@dataclass
class ResolutionDecision:
    """Represents a decision made for duplicates."""
    action: DuplicateAction
    target_file: Optional[Path] = None  # For actions that target a specific file
    destination: Optional[Path] = None  # For move actions
    reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResolutionSummary:
    """Summary of duplicate resolution session."""
    total_groups: int = 0
    resolved_groups: int = 0
    kept_files: int = 0
    moved_files: int = 0
    deleted_files: int = 0
    skipped_files: int = 0
    space_saved_mb: float = 0.0
    decisions: List[ResolutionDecision] = field(default_factory=list)


class DuplicateQualityScorer:
    """Scores duplicate files to determine the 'best' version."""

    def __init__(self):
        self.format_priority = {
            'flac': 10,
            'wav': 9,
            'aiff': 8,
            'alac': 7,
            'm4a': 6,
            'mp3': 5,
            'aac': 4,
            'ogg': 3,
            'opus': 3,
            'wma': 2
        }

    def score_file(self, audio_file: AudioFile) -> float:
        """Score an audio file based on quality metrics."""
        score = 0.0

        # Format quality (40% weight)
        format_score = self.format_priority.get(audio_file.format.value, 0)
        score += format_score * 0.4

        # Bitrate quality (25% weight)
        if hasattr(audio_file.metadata, 'bitrate') and audio_file.metadata.bitrate:
            # Normalize bitrate (assuming max 3200 kbps for high-quality FLAC)
            bitrate_score = min(audio_file.metadata.bitrate / 3200, 1.0) * 10
            score += bitrate_score * 0.25

        # Sample rate quality (15% weight)
        if hasattr(audio_file.metadata, 'sample_rate') and audio_file.metadata.sample_rate:
            # Normalize sample rate (assuming max 192000 Hz)
            sample_rate_score = min(audio_file.metadata.sample_rate / 192000, 1.0) * 10
            score += sample_rate_score * 0.15

        # Metadata completeness (10% weight)
        metadata_fields = ['title', 'artists', 'album', 'year', 'genre', 'track_number']
        metadata_score = sum(1 for field in metadata_fields
                           if getattr(audio_file.metadata, field, None))
        metadata_score = (metadata_score / len(metadata_fields)) * 10
        score += metadata_score * 0.10

        # File size preference (10% weight - larger files often have better quality)
        try:
            size_mb = audio_file.path.stat().st_size / (1024 * 1024)
            # Prefer larger files but cap the benefit
            size_score = min(size_mb / 100, 1.0) * 10
            score += size_score * 0.10
        except (OSError, FileNotFoundError):
            pass

        return score

    def choose_best(self, files: List[AudioFile]) -> Tuple[AudioFile, float]:
        """Choose the best file from a list of duplicates."""
        if not files:
            raise ValueError("No files to choose from")

        best_file = files[0]
        best_score = self.score_file(best_file)

        for file in files[1:]:
            score = self.score_file(file)
            if score > best_score:
                best_file = file
                best_score = score

        return best_file, best_score


class InteractiveDuplicateResolver:
    """Interactive duplicate resolution system."""

    def __init__(self,
                 strategy: ResolutionStrategy = ResolutionStrategy.INTERACTIVE,
                 duplicate_dir: Optional[Path] = None,
                 dry_run: bool = False):
        """
        Initialize the duplicate resolver.

        Args:
            strategy: Resolution strategy to use
            duplicate_dir: Directory to move duplicates to (if applicable)
            dry_run: Whether to perform a dry run
        """
        self.strategy = strategy
        self.duplicate_dir = duplicate_dir
        self.dry_run = dry_run
        self.quality_scorer = DuplicateQualityScorer()
        self.summary = ResolutionSummary()
        self.decisions_cache: Dict[str, ResolutionDecision] = {}

    async def resolve_duplicates(self,
                               duplicate_groups: List[DuplicateGroup]) -> ResolutionSummary:
        """
        Resolve duplicate groups based on the configured strategy.

        Args:
            duplicate_groups: List of duplicate groups to resolve

        Returns:
            Summary of resolution actions
        """
        self.summary = ResolutionSummary()
        self.summary.total_groups = len(duplicate_groups)

        if self.strategy == ResolutionStrategy.INTERACTIVE:
            return await self._resolve_interactive(duplicate_groups)
        elif self.strategy == ResolutionStrategy.AUTO_KEEP_BEST:
            return await self._resolve_auto_best(duplicate_groups)
        elif self.strategy == ResolutionStrategy.AUTO_FIRST:
            return await self._resolve_auto_first(duplicate_groups)
        elif self.strategy == ResolutionStrategy.AUTO_SMART:
            return await self._resolve_auto_smart(duplicate_groups)
        else:
            raise MusicOrganizerError(f"Unknown resolution strategy: {self.strategy}")

    async def _resolve_interactive(self, duplicate_groups: List[DuplicateGroup]) -> ResolutionSummary:
        """Resolve duplicates with interactive user input."""
        from ..ui.duplicate_resolver_ui import DuplicateResolverUI

        ui = DuplicateResolverUI(dry_run=self.dry_run)

        for i, group in enumerate(duplicate_groups, 1):
            print(f"\n{'='*60}")
            print(f"Duplicate Group {i}/{len(duplicate_groups)}")
            print(f"Type: {group.duplicate_type.upper()} | Confidence: {group.confidence:.1%}")
            print(f"Reason: {group.reason}")
            print(f"{'='*60}")

            # Show comparison
            decision = await ui.show_duplicate_group(group)

            if decision:
                await self._apply_decision(group, decision)
                self.summary.resolved_groups += 1
            else:
                # Skipped
                self.summary.skipped_files += len(group.files)

            # Clear screen for next group
            ui.clear_screen()

        return self.summary

    async def _resolve_auto_best(self, duplicate_groups: List[DuplicateGroup]) -> ResolutionSummary:
        """Automatically keep the best quality version in each group."""
        for group in duplicate_groups:
            if len(group.files) < 2:
                continue

            best_file, best_score = self.quality_scorer.choose_best(group.files)
            other_files = [f for f in group.files if f != best_file]

            # Create decision to keep the best file
            decision = ResolutionDecision(
                action=DuplicateAction.KEEP_BEST,
                target_file=best_file.path,
                reason=f"Best quality score: {best_score:.1f}"
            )

            await self._apply_decision(group, decision)
            self.summary.resolved_groups += 1

        return self.summary

    async def _resolve_auto_first(self, duplicate_groups: List[DuplicateGroup]) -> ResolutionSummary:
        """Automatically keep the first file in each group."""
        for group in duplicate_groups:
            if len(group.files) < 2:
                continue

            # Keep the first file, move/delete others
            first_file = group.files[0]
            other_files = group.files[1:]

            # Prefer moving to deleting if duplicate directory is specified
            action = (DuplicateAction.MOVE_DUPLICATE
                     if self.duplicate_dir else DuplicateAction.DELETE_DUPLICATE)

            decision = ResolutionDecision(
                action=action,
                target_file=first_file.path,
                destination=self.duplicate_dir,
                reason="Keep first file found"
            )

            await self._apply_decision(group, decision)
            self.summary.resolved_groups += 1

        return self.summary

    async def _resolve_auto_smart(self, duplicate_groups: List[DuplicateGroup]) -> ResolutionSummary:
        """Make smart decisions based on various heuristics."""
        for group in duplicate_groups:
            if len(group.files) < 2:
                continue

            # Smart decision logic
            decision = await self._make_smart_decision(group)
            await self._apply_decision(group, decision)
            self.summary.resolved_groups += 1

        return self.summary

    async def _make_smart_decision(self, group: DuplicateGroup) -> ResolutionDecision:
        """Make a smart decision for a duplicate group."""
        # If exact duplicates, keep the one with better path/naming
        if group.duplicate_type == 'exact':
            # Prefer files with cleaner paths
            best_file = min(group.files, key=lambda f: len(f.path.parts))
            return ResolutionDecision(
                action=DuplicateAction.KEEP_BEST,
                target_file=best_file.path,
                reason="Cleanest file path"
            )

        # For metadata duplicates, prefer the one with better metadata
        metadata_scores = []
        for file in group.files:
            fields = ['title', 'artists', 'album', 'year', 'genre']
            score = sum(1 for field in fields if getattr(file.metadata, field, None))
            metadata_scores.append((score, file))

        best_score, best_file = max(metadata_scores, key=lambda x: x[0])

        if best_score >= 4:  # Good metadata
            return ResolutionDecision(
                action=DuplicateAction.KEEP_BEST,
                target_file=best_file.path,
                reason=f"Best metadata completeness: {best_score}/5 fields"
            )
        else:
            # Poor metadata, prefer quality
            best_file, quality_score = self.quality_scorer.choose_best(group.files)
            return ResolutionDecision(
                action=DuplicateAction.KEEP_BEST,
                target_file=best_file.path,
                reason=f"Best audio quality score: {quality_score:.1f}"
            )

    async def _apply_decision(self, group: DuplicateGroup, decision: ResolutionDecision):
        """Apply a resolution decision to a duplicate group."""
        self.summary.decisions.append(decision)

        if self.dry_run:
            print(f"[DRY RUN] Would apply: {decision.action.value} - {decision.reason}")
            self.summary.kept_files += 1
            return

        # Actually apply the decision
        if decision.action in [DuplicateAction.KEEP_FIRST, DuplicateAction.KEEP_BEST]:
            # Keep one file, handle others
            if decision.target_file:
                keep_file = next(f for f in group.files if f.path == decision.target_file)
                duplicate_files = [f for f in group.files if f != keep_file]

                self.summary.kept_files += 1

                # Handle duplicates
                for dup_file in duplicate_files:
                    await self._handle_duplicate_file(dup_file, decision)

        elif decision.action == DuplicateAction.KEEP_BOTH:
            # Keep all files
            self.summary.kept_files += len(group.files)

        elif decision.action == DuplicateAction.MOVE_DUPLICATE:
            # Move specified file
            if decision.target_file:
                file_to_move = next(f for f in group.files if f.path == decision.target_file)
                await self._move_file(file_to_move, decision.destination)
                self.summary.moved_files += 1

        elif decision.action == DuplicateAction.DELETE_DUPLICATE:
            # Delete specified file
            if decision.target_file:
                file_to_delete = next(f for f in group.files if f.path == decision.target_file)
                await self._delete_file(file_to_delete)
                self.summary.deleted_files += 1

    async def _handle_duplicate_file(self, file: AudioFile, decision: ResolutionDecision):
        """Handle a duplicate file based on the decision."""
        if decision.action == DuplicateAction.KEEP_BEST and self.duplicate_dir:
            await self._move_file(file, self.duplicate_dir)
            self.summary.moved_files += 1
        else:
            await self._delete_file(file)
            self.summary.deleted_files += 1

    async def _move_file(self, file: AudioFile, destination: Optional[Path]):
        """Move a file to the destination directory."""
        if not destination:
            return

        try:
            destination.mkdir(parents=True, exist_ok=True)
            dest_path = destination / file.path.name

            # Handle name conflicts
            counter = 1
            while dest_path.exists():
                stem = file.path.stem
                suffix = file.path.suffix
                dest_path = destination / f"{stem}_{counter}{suffix}"
                counter += 1

            file.path.rename(dest_path)

            # Calculate space saved
            try:
                size_mb = file.path.stat().st_size / (1024 * 1024)
                self.summary.space_saved_mb += size_mb
            except OSError:
                pass

        except Exception as e:
            print(f"Error moving file {file.path}: {e}")

    async def _delete_file(self, file: AudioFile):
        """Delete a file."""
        try:
            # Calculate space saved before deletion
            try:
                size_mb = file.path.stat().st_size / (1024 * 1024)
                self.summary.space_saved_mb += size_mb
            except OSError:
                pass

            file.path.unlink()
        except Exception as e:
            print(f"Error deleting file {file.path}: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the resolution session."""
        return {
            "total_groups": self.summary.total_groups,
            "resolved_groups": self.summary.resolved_groups,
            "kept_files": self.summary.kept_files,
            "moved_files": self.summary.moved_files,
            "deleted_files": self.summary.deleted_files,
            "skipped_files": self.summary.skipped_files,
            "space_saved_mb": round(self.summary.space_saved_mb, 2),
            "decisions": [
                {
                    "action": d.action.value,
                    "target_file": str(d.target_file) if d.target_file else None,
                    "reason": d.reason,
                    "timestamp": d.timestamp.isoformat()
                }
                for d in self.summary.decisions
            ]
        }

    def save_report(self, output_path: Path):
        """Save a detailed report of the resolution session."""
        report = {
            "strategy": self.strategy.value,
            "duplicate_dir": str(self.duplicate_dir) if self.duplicate_dir else None,
            "dry_run": self.dry_run,
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


# Convenience function for quick duplicate resolution
async def resolve_duplicates_from_plugin(
    duplicate_detector: DuplicateDetectorPlugin,
    strategy: ResolutionStrategy = ResolutionStrategy.INTERACTIVE,
    duplicate_dir: Optional[Path] = None,
    dry_run: bool = False
) -> ResolutionSummary:
    """Resolve duplicates using the duplicate detector plugin results."""
    # Get duplicate summary from plugin
    duplicate_summary = duplicate_detector.get_duplicate_summary()

    if duplicate_summary['total_duplicate_groups'] == 0:
        print("No duplicates found!")
        return ResolutionSummary()

    # Convert plugin results to DuplicateGroup objects
    # This would need to be implemented based on the plugin's internal data structure
    # For now, we'll return an empty summary
    return ResolutionSummary()