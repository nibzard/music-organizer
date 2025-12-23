"""Magic Mode Organizer - Intelligent Zero-Configuration Music Organization.

This module provides the MagicMusicOrganizer class that extends the AsyncMusicOrganizer
with intelligent, zero-configuration organization powered by AI suggestions.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import json
from datetime import datetime

from .async_organizer import AsyncMusicOrganizer
from .magic_mode import (
    MagicModeOrchestrator,
    MagicSuggestion,
    MagicAnalyzer,
    MagicStrategyRecommender,
    analyze_music_library,
    create_magic_organization_config
)
from .smart_cached_metadata import get_smart_cached_metadata_handler
from .bulk_operations import BulkOperationConfig, ConflictStrategy
from .bulk_organizer import BulkAsyncOrganizer
from ..models.audio_file import AudioFile
from ..models.config import Config
from ..exceptions import MagicModeError, MusicOrganizerError
from ..progress_tracker import IntelligentProgressTracker, ProgressStage
from ..async_progress_renderer import AsyncProgressRenderer


class MagicMusicOrganizer(AsyncMusicOrganizer):
    """Enhanced organizer with Magic Mode for zero-configuration intelligent organization."""

    def __init__(
        self,
        config: Optional[Config] = None,
        enable_smart_cache: bool = True,
        enable_bulk_operations: bool = True,
        magic_mode_confidence_threshold: float = 0.6
    ):
        """Initialize Magic Music Organizer."""
        super().__init__(config)

        self.magic_mode_enabled = True
        self.confidence_threshold = magic_mode_confidence_threshold
        self.enable_smart_cache = enable_smart_cache
        self.enable_bulk_operations = enable_bulk_operations

        # Initialize magic components
        self.magic_orchestrator = MagicModeOrchestrator()
        self.magic_analyzer = MagicAnalyzer()
        self.magic_recommender = MagicStrategyRecommender()

        # Initialize enhanced components
        self.smart_cache_handler = None
        self.bulk_organizer = None

        # Cache for magic suggestions
        self._current_suggestion: Optional[MagicSuggestion] = None
        self._library_analysis = None

    async def initialize(self):
        """Initialize enhanced components."""
        await super().initialize()

        # Initialize smart cache if enabled
        if self.enable_smart_cache:
            self.smart_cache_handler = get_smart_cached_metadata_handler(
                enable_smart_cache=True,
                cache_warming_enabled=False,  # Enable on demand
                auto_optimize=True
            )

        # Initialize bulk organizer if enabled
        if self.enable_bulk_operations:
            self.bulk_organizer = BulkAsyncOrganizer(
                config=self.config,
                bulk_config=BulkOperationConfig(
                    chunk_size=200,
                    conflict_strategy=ConflictStrategy.RENAME,
                    verify_copies=False,
                    batch_directories=True,
                    memory_threshold_mb=512
                )
            )
            await self.bulk_organizer.initialize()

    async def analyze_library_for_magic_mode(
        self,
        source_dir: Path,
        sample_size: Optional[int] = None,
        force_analyze: bool = False
    ) -> MagicSuggestion:
        """Analyze library and generate Magic Mode suggestions."""
        if not force_analyze and self._current_suggestion:
            return self._current_suggestion

        # Scan audio files
        audio_files = []
        async for file_path in self.scan_directory(source_dir):
            try:
                audio_file = await self._process_file(file_path)
                if audio_file:
                    audio_files.append(audio_file)

                # Limit sample size for performance
                if sample_size and len(audio_files) >= sample_size:
                    break

            except Exception as e:
                # Skip files that can't be processed
                continue

        if not audio_files:
            raise MagicModeError("No valid audio files found for analysis")

        # Generate magic suggestions
        suggestion = await self.magic_orchestrator.analyze_and_suggest(audio_files)

        # Cache results
        self._current_suggestion = suggestion
        self._library_analysis = suggestion.analysis

        return suggestion

    async def organize_with_magic_mode(
        self,
        source_dir: Path,
        target_dir: Path,
        dry_run: bool = False,
        auto_accept: bool = False,
        sample_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Organize music using Magic Mode with intelligent suggestions."""

        # Analyze library if not already done
        suggestion = await self.analyze_library_for_magic_mode(
            source_dir,
            sample_size=sample_size,
            force_analyze=False
        )

        # Check if we're confident enough
        if not auto_accept and suggestion.strategy.confidence < self.confidence_threshold:
            # Show analysis and ask for confirmation
            await self._show_magic_analysis(suggestion)
            confirm = input(f"\nUse Magic Mode with {suggestion.strategy.confidence:.1%} confidence? (y/N): ")
            if confirm.lower() != 'y':
                raise MagicModeError("User rejected Magic Mode suggestion")

        # Generate magic configuration
        magic_config = await self.magic_orchestrator.generate_magic_config(suggestion)

        # Apply preprocessing steps if not dry run
        if not dry_run and suggestion.preprocessing_steps:
            await self._apply_preprocessing_steps(
                source_dir,
                suggestion.preprocessing_steps,
                kwargs
            )

        # Choose organization method based on library size and complexity
        if self.enable_bulk_operations and suggestion.analysis.total_files > 100:
            return await self._organize_with_bulk_magic(
                source_dir,
                target_dir,
                suggestion,
                magic_config,
                dry_run=dry_run,
                **kwargs
            )
        else:
            return await self._organize_with_standard_magic(
                source_dir,
                target_dir,
                suggestion,
                magic_config,
                dry_run=dry_run,
                **kwargs
            )

    async def _organize_with_standard_magic(
        self,
        source_dir: Path,
        target_dir: Path,
        suggestion: MagicSuggestion,
        magic_config: Dict[str, Any],
        dry_run: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Organize using standard methods with Magic Mode configuration."""

        # Configure organizer with magic settings
        original_config = self.config

        try:
            # Update config with magic path patterns
            magic_organization_config = magic_config.get("organization", {})

            # Create temporary config with magic patterns
            magic_org_config = Config(
                dry_run=dry_run,
                **kwargs
            )

            # Track results
            results = {
                "suggestion": suggestion,
                "config": magic_config,
                "organized_files": [],
                "errors": [],
                "stats": {}
            }

            # Initialize progress tracking
            progress_tracker = IntelligentProgressTracker()
            progress_renderer = AsyncProgressRenderer()
            progress_tracker.add_render_callback(progress_renderer.render)

            # Set total files
            total_files = suggestion.analysis.total_files
            progress_tracker.set_total_files(total_files)

            # Process files
            processed_count = 0
            error_count = 0

            async for file_path in self.scan_directory(source_dir):
                try:
                    # Extract metadata
                    metadata = await self.extract_metadata(file_path)

                    # Generate target path using magic pattern
                    target_path = self._generate_magic_target_path(
                        file_path,
                        metadata,
                        target_dir,
                        suggestion.strategy.path_pattern,
                        suggestion.strategy.filename_pattern
                    )

                    if not dry_run:
                        # Move file
                        await self.move_file(file_path, target_path)

                    results["organized_files"].append({
                        "source": str(file_path),
                        "target": str(target_path),
                        "strategy": suggestion.strategy.name
                    })

                    processed_count += 1
                    progress_tracker.update_progress(1)

                except Exception as e:
                    error_count += 1
                    results["errors"].append({
                        "file": str(file_path),
                        "error": str(e)
                    })
                    progress_tracker.update_progress(1, error=True)

            # Final statistics
            results["stats"] = {
                "total_files": total_files,
                "processed": processed_count,
                "errors": error_count,
                "success_rate": (processed_count / total_files) if total_files > 0 else 0,
                "strategy_used": suggestion.strategy.name,
                "confidence": suggestion.strategy.confidence,
                "magic_mode": True
            }

            return results

        finally:
            # Restore original config
            self.config = original_config

    async def _organize_with_bulk_magic(
        self,
        source_dir: Path,
        target_dir: Path,
        suggestion: MagicSuggestion,
        magic_config: Dict[str, Any],
        dry_run: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Organize using bulk operations with Magic Mode configuration."""

        if not self.bulk_organizer:
            raise MagicModeError("Bulk operations not available")

        # Configure bulk organizer with magic settings
        bulk_config = BulkOperationConfig(
            chunk_size=kwargs.get("chunk_size", 200),
            conflict_strategy=ConflictStrategy(kwargs.get("conflict_strategy", "rename")),
            verify_copies=kwargs.get("verify_copies", False),
            batch_directories=kwargs.get("batch_directories", True),
            memory_threshold_mb=kwargs.get("bulk_memory_threshold", 512)
        )

        self.bulk_organizer.bulk_config = bulk_config

        # Create bulk file operations with magic paths
        bulk_operations = []

        async for file_path in self.scan_directory(source_dir):
            try:
                metadata = await self.extract_metadata(file_path)

                target_path = self._generate_magic_target_path(
                    file_path,
                    metadata,
                    target_dir,
                    suggestion.strategy.path_pattern,
                    suggestion.strategy.filename_pattern
                )

                bulk_operations.append({
                    "source": file_path,
                    "target": target_path,
                    "type": "move",
                    "metadata": metadata
                })

            except Exception:
                continue  # Skip problematic files

        # Execute bulk organization
        if dry_run:
            # Preview mode
            preview = {
                "total_operations": len(bulk_operations),
                "estimated_size_mb": suggestion.analysis.total_size_mb,
                "strategy": suggestion.strategy.name,
                "confidence": suggestion.strategy.confidence,
                "operations": bulk_operations[:10]  # Show first 10 for preview
            }
            return {"preview": preview, "magic_mode": True}
        else:
            # Execute bulk operations
            bulk_result = await self.bulk_organizer.organize_files_bulk(bulk_operations)

            # Enhance result with magic information
            bulk_result["magic_mode"] = True
            bulk_result["suggestion"] = suggestion
            bulk_result["config"] = magic_config

            return bulk_result

    def _generate_magic_target_path(
        self,
        source_path: Path,
        metadata: Any,  # AudioFile or Metadata
        target_dir: Path,
        path_pattern: str,
        filename_pattern: str
    ) -> Path:
        """Generate target path using Magic Mode patterns."""

        # Extract metadata values
        title = getattr(metadata, 'title', 'Unknown Title')
        artists = getattr(metadata, 'artists', [])
        # Handle frozenset or list of artists
        if artists:
            if isinstance(artists, (frozenset, set)):
                artist = next(iter(artists), 'Unknown Artist')
            else:
                artist = artists[0] if artists else 'Unknown Artist'
        else:
            artist = 'Unknown Artist'
        album = getattr(metadata, 'album', 'Unknown Album')
        year = getattr(metadata, 'year', None)
        genre = getattr(metadata, 'genre', 'Unknown')
        track_number = getattr(metadata, 'track_number', None)

        # Format values
        artist_str = str(artist)
        title_str = str(title) if title else 'Unknown Title'
        album_str = str(album) if album else 'Unknown Album'
        year_str = str(year) if year else ''
        genre_str = str(genre) if genre else 'Unknown'
        # Handle TrackNumber object or int
        if track_number:
            if hasattr(track_number, 'number'):
                track_str = track_number.formatted()
            else:
                track_str = f"{int(track_number):02d}"
        else:
            track_str = ''

        # Calculate derived values
        decade = self._calculate_decade(year) if year else 'unknown'
        first_letter = artist_str[0].upper() if artist_str else 'A'

        # Format path
        path_vars = {
            'artist': artist_str,
            'album': album_str,
            'year': year_str,
            'genre': genre_str,
            'decade': decade,
            'first_letter': first_letter,
            'track_number': track_str,
            'title': title_str
        }

        # Replace path pattern variables
        formatted_path = path_pattern.format(**path_vars)

        # Replace filename pattern variables
        if filename_pattern:
            filename = filename_pattern.format(**path_vars)
            # Ensure file extension
            if not filename.endswith(source_path.suffix):
                filename += source_path.suffix
        else:
            filename = source_path.name

        # Create full path
        target_path = target_dir / formatted_path / filename

        # Clean path for filesystem safety
        target_path = self._clean_path_for_filesystem(target_path)

        return target_path

    def _clean_path_for_filesystem(self, path: Path) -> Path:
        """Clean path to be filesystem-safe."""
        # Replace problematic characters
        clean_name = path.name
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            clean_name = clean_name.replace(char, '_')

        # Handle reserved names (Windows)
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]

        name_without_ext = clean_name[:-len(path.suffix)] if path.suffix else clean_name
        if name_without_ext.upper() in reserved_names:
            clean_name = f"_{clean_name}"

        return path.parent / clean_name

    def _calculate_decade(self, year: Union[int, str]) -> str:
        """Calculate decade from year."""
        try:
            year_int = int(year)
            decade_start = (year_int // 10) * 10
            return f"{decade_start}s"
        except (ValueError, TypeError):
            return "unknown"

    async def _show_magic_analysis(self, suggestion: MagicSuggestion):
        """Display Magic Mode analysis to the user."""
        print("\n" + "="*60)
        print("🪄 MAGIC MODE ANALYSIS")
        print("="*60)

        # Library overview
        print(f"\n📊 LIBRARY OVERVIEW:")
        print(f"  Total files: {suggestion.analysis.total_files:,}")
        print(f"  Total size: {suggestion.analysis.total_size_mb:.1f} MB")
        print(f"  Artists: {suggestion.analysis.artist_count:,}")
        print(f"  Albums: {suggestion.analysis.album_count:,}")
        print(f"  Metadata completeness: {suggestion.analysis.metadata_completeness:.1%}")
        print(f"  Organization chaos score: {suggestion.analysis.organization_chaos_score:.1%}")

        # Recommended strategy
        print(f"\n🎯 RECOMMENDED STRATEGY:")
        print(f"  Strategy: {suggestion.strategy.name}")
        print(f"  Confidence: {suggestion.strategy.confidence:.1%}")
        print(f"  Complexity: {suggestion.strategy.complexity}")
        print(f"  Estimated time: {suggestion.strategy.estimated_time_minutes} minutes")
        print(f"  Path pattern: {suggestion.strategy.path_pattern}")
        print(f"  Filename pattern: {suggestion.strategy.filename_pattern}")

        # Reasons
        if suggestion.strategy.reasoning:
            print(f"\n💡 WHY THIS STRATEGY:")
            for reason in suggestion.strategy.reasoning:
                print(f"  • {reason}")

        # Pros and Cons
        if suggestion.strategy.pros:
            print(f"\n✅ PROS:")
            for pro in suggestion.strategy.pros:
                print(f"  • {pro}")

        if suggestion.strategy.cons:
            print(f"\n⚠️  CONS:")
            for con in suggestion.strategy.cons:
                print(f"  • {con}")

        # Quick wins
        if suggestion.quick_wins:
            print(f"\n🚀 QUICK WINS:")
            for win in suggestion.quick_wins:
                print(f"  • {win}")

        # Potential issues
        if suggestion.potential_issues:
            print(f"\n⚠️  POTENTIAL ISSUES:")
            for issue in suggestion.potential_issues:
                print(f"  • {issue}")

        print("="*60)

    async def _apply_preprocessing_steps(
        self,
        source_dir: Path,
        preprocessing_steps: List[str],
        kwargs: Dict[str, Any]
    ):
        """Apply preprocessing steps before organization."""
        if not preprocessing_steps:
            return

        print(f"\n🔧 APPLYING PREPROCESSING STEPS...")

        for step in preprocessing_steps:
            print(f"  • {step}")

            # Implement common preprocessing steps
            if "duplicate" in step.lower():
                await self._preprocessing_duplicate_detection(source_dir, kwargs)
            elif "metadata" in step.lower():
                await self._preprocessing_metadata_enhancement(source_dir, kwargs)
            elif "cleanup" in step.lower():
                await self._preprocessing_folder_cleanup(source_dir, kwargs)

        print("✅ Preprocessing complete\n")

    async def _preprocessing_duplicate_detection(self, source_dir: Path, kwargs: Dict[str, Any]):
        """Run duplicate detection as preprocessing."""
        # This would integrate with the duplicate detector plugin
        print("    Running duplicate detection...")
        # Implementation would go here
        pass

    async def _preprocessing_metadata_enhancement(self, source_dir: Path, kwargs: Dict[str, Any]):
        """Run metadata enhancement as preprocessing."""
        print("    Enhancing metadata...")
        # This would integrate with the MusicBrainz enhancer plugin
        pass

    async def _preprocessing_folder_cleanup(self, source_dir: Path, kwargs: Dict[str, Any]):
        """Run folder cleanup as preprocessing."""
        print("    Cleaning up folder structure...")
        # Basic cleanup implementation
        pass

    async def get_magic_mode_preview(
        self,
        source_dir: Path,
        target_dir: Path,
        sample_size: int = 50
    ) -> Dict[str, Any]:
        """Get a preview of Magic Mode organization."""

        suggestion = await self.analyze_library_for_magic_mode(
            source_dir,
            sample_size=sample_size
        )

        # Generate sample operations
        sample_operations = []
        count = 0

        async for file_path in self.scan_directory(source_dir):
            if count >= sample_size:
                break

            try:
                metadata = await self.extract_metadata(file_path)
                target_path = self._generate_magic_target_path(
                    file_path,
                    metadata,
                    target_dir,
                    suggestion.strategy.path_pattern,
                    suggestion.strategy.filename_pattern
                )

                sample_operations.append({
                    "source": str(file_path.relative_to(source_dir)),
                    "target": str(target_path.relative_to(target_dir)),
                    "size_mb": file_path.stat().st_size / (1024 * 1024)
                })

                count += 1

            except Exception:
                continue

        return {
            "suggestion": suggestion,
            "sample_operations": sample_operations,
            "total_estimated_operations": suggestion.analysis.total_files,
            "preview_type": "magic_mode"
        }

    async def save_magic_config(self, output_path: Path):
        """Save current Magic Mode configuration to file."""
        if not self._current_suggestion:
            raise MagicModeError("No Magic Mode analysis available")

        config = await self.magic_orchestrator.generate_magic_config(
            self._current_suggestion,
            output_path
        )

        return config

    def get_current_magic_suggestion(self) -> Optional[MagicSuggestion]:
        """Get the current Magic Mode suggestion."""
        return self._current_suggestion

    def get_library_analysis(self) -> Optional[Any]:
        """Get the current library analysis."""
        return self._library_analysis