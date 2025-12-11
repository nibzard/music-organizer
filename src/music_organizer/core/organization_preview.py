"""Organization Preview with Dry-Run Visualization.

This module provides comprehensive visualization of planned file operations
before execution, allowing users to review and understand exactly what
will happen during organization.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, NamedTuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
import json

from ..models.audio_file import AudioFile, ContentType
from ..models.config import Config
from ..console_utils import SimpleConsole


@dataclass
class PreviewOperation:
    """Represents a single file operation in the preview."""
    operation_type: str  # 'move', 'copy', 'create_dir', 'conflict'
    source_path: Path
    target_path: Path
    file_size: int
    audio_file: Optional[AudioFile] = None
    conflict_reason: Optional[str] = None
    resolution_strategy: Optional[str] = None
    confidence: float = 1.0  # For Magic Mode operations


@dataclass
class DirectoryPreview:
    """Preview of directory structure after organization."""
    path: Path
    file_count: int = 0
    total_size_mb: float = 0.0
    is_new: bool = False
    is_empty: bool = True
    content_types: Dict[str, int] = field(default_factory=dict)
    file_types: Dict[str, int] = field(default_factory=dict)
    subdirectories: List['DirectoryPreview'] = field(default_factory=list)


@dataclass
class PreviewStatistics:
    """Statistics for the organization preview."""
    total_files: int = 0
    total_operations: int = 0
    operations_by_type: Dict[str, int] = field(default_factory=dict)
    total_size_mb: float = 0.0
    space_saved_mb: float = 0.0  # From duplicates removed
    directories_created: int = 0
    conflicts_detected: int = 0
    confidence_avg: float = 0.0
    estimated_time_minutes: float = 0.0

    # Distribution statistics
    content_types: Dict[str, int] = field(default_factory=dict)
    file_types: Dict[str, int] = field(default_factory=dict)
    artists: Dict[str, int] = field(default_factory=dict)
    top_artists: List[Tuple[str, int]] = field(default_factory=list)

    # Organization quality metrics
    metadata_completeness: float = 0.0
    organization_score: float = 0.0  # 0-100
    duplicates_removed: int = 0


class OrganizationPreview:
    """Comprehensive organization preview system."""

    def __init__(self, config: Config):
        self.config = config
        self.console = SimpleConsole()
        self.operations: List[PreviewOperation] = []
        self.statistics = PreviewStatistics()
        self.directory_structure: Optional[DirectoryPreview] = None
        self.file_operations: Dict[Path, PreviewOperation] = {}
        self.conflicts: List[PreviewOperation] = []

    async def collect_operations(self,
                                source_files: List[AudioFile],
                                target_mapping: Dict[Path, Path],
                                conflict_resolutions: Dict[Path, str] = None) -> None:
        """Collect all planned operations for preview."""
        self.operations.clear()
        self.file_operations.clear()
        self.conflicts.clear()

        # Process file operations
        for audio_file in source_files:
            source_path = audio_file.path
            target_path = target_mapping.get(source_path)

            if not target_path:
                continue

            # Check for conflicts
            operation_type = 'move'
            conflict_reason = None
            resolution_strategy = None

            if target_path.exists() or target_path in [op.target_path for op in self.operations]:
                operation_type = 'conflict'
                conflict_reason = "Target file already exists"
                resolution_strategy = conflict_resolutions.get(source_path, 'skip') if conflict_resolutions else 'skip'
                self.conflicts.append(PreviewOperation(
                    operation_type='conflict',
                    source_path=source_path,
                    target_path=target_path,
                    file_size=int(audio_file.size_mb * 1024 * 1024),  # Convert to bytes
                    audio_file=audio_file,
                    conflict_reason=conflict_reason,
                    resolution_strategy=resolution_strategy
                ))

            # Add file operation
            op = PreviewOperation(
                operation_type=operation_type,
                source_path=source_path,
                target_path=target_path,
                file_size=int(audio_file.size_mb * 1024 * 1024),
                audio_file=audio_file,
                conflict_reason=conflict_reason,
                resolution_strategy=resolution_strategy
            )

            self.operations.append(op)
            self.file_operations[source_path] = op

        # Add directory creation operations
        directories_to_create = self._get_directories_to_create()
        for directory in directories_to_create:
            self.operations.append(PreviewOperation(
                operation_type='create_dir',
                source_path=Path(''),
                target_path=directory,
                file_size=0
            ))

        # Calculate statistics
        self._calculate_statistics(source_files)

        # Build directory structure preview
        self.directory_structure = await self._build_directory_preview()

    def _get_directories_to_create(self) -> Set[Path]:
        """Get all directories that need to be created."""
        directories = set()
        for op in self.operations:
            if op.target_path:
                # Add parent directories for file operations
                for parent in op.target_path.parents:
                    if not parent.exists():
                        directories.add(parent)
        return directories

    def _calculate_statistics(self, source_files: List[AudioFile]) -> None:
        """Calculate comprehensive statistics for the preview."""
        self.statistics = PreviewStatistics()

        if not self.operations:
            return

        # Basic counts
        self.statistics.total_files = len(source_files)
        self.statistics.total_operations = len(self.operations)

        # Operations by type
        op_counter = Counter(op.operation_type for op in self.operations)
        self.statistics.operations_by_type = dict(op_counter)
        self.statistics.directories_created = op_counter.get('create_dir', 0)

        # File sizes
        file_ops = [op for op in self.operations if op.operation_type in ['move', 'copy']]
        self.statistics.total_size_mb = sum(op.file_size for op in file_ops) / (1024 * 1024)

        # Conflicts
        self.statistics.conflicts_detected = len(self.conflicts)

        # Content type distribution
        content_counter = Counter()
        format_counter = Counter()
        artist_counter = Counter()

        for audio_file in source_files:
            if audio_file.content_type:
                content_counter[audio_file.content_type.value] += 1
            if audio_file.file_type:
                format_counter[audio_file.file_type] += 1
            if audio_file.artists:
                for artist in audio_file.artists:
                    artist_counter[str(artist)] += 1

        self.statistics.content_types = dict(content_counter)
        self.statistics.file_types = dict(format_counter)
        self.statistics.artists = dict(artist_counter.most_common(20))
        self.statistics.top_artists = artist_counter.most_common(10)

        # Quality metrics
        complete_metadata = sum(1 for f in source_files
                               if self._is_metadata_complete(f))
        self.statistics.metadata_completeness = complete_metadata / len(source_files) if source_files else 0

        # Organization score (0-100)
        self._calculate_organization_score()

        # Time estimation (rough estimate: 1 file per 2 seconds for small files, longer for large)
        avg_file_size_mb = self.statistics.total_size_mb / max(len(file_ops), 1)
        if avg_file_size_mb < 10:
            time_per_file = 2  # seconds
        elif avg_file_size_mb < 50:
            time_per_file = 5
        else:
            time_per_file = 10

        self.statistics.estimated_time_minutes = (len(file_ops) * time_per_file) / 60

    def _is_metadata_complete(self, audio_file: AudioFile) -> bool:
        """Check if an audio file has complete metadata."""
        required_fields = ['title', 'primary_artist', 'album']
        for field in required_fields:
            if not getattr(audio_file, field, None):
                return False
        return True

    def _calculate_organization_score(self) -> None:
        """Calculate overall organization quality score (0-100)."""
        score = 0

        # Metadata completeness (30 points)
        score += self.statistics.metadata_completeness * 30

        # Conflict rate (20 points, fewer conflicts = higher score)
        if self.statistics.total_files > 0:
            conflict_rate = self.statistics.conflicts_detected / self.statistics.total_files
            score += (1 - conflict_rate) * 20

        # Directory structure quality (20 points)
        # More organized structure (not too flat, not too deep)
        if self.directory_structure:
            max_depth = self._get_max_depth(self.directory_structure)
            if max_depth >= 2 and max_depth <= 4:
                score += 20
            elif max_depth >= 1 and max_depth <= 5:
                score += 10

        # Content type separation (20 points)
        if len(self.statistics.content_types) > 1:
            score += 20
        elif len(self.statistics.content_types) == 1:
            score += 10

        # Artist organization (10 points)
        unique_artists = len(self.statistics.artists)
        if unique_artists > 10:
            score += 10
        elif unique_artists > 5:
            score += 5

        self.statistics.organization_score = min(100, max(0, score))

    def _get_max_depth(self, directory: DirectoryPreview, current_depth: int = 0) -> int:
        """Get maximum depth of directory structure."""
        if not directory.subdirectories:
            return current_depth
        return max(self._get_max_depth(subdir, current_depth + 1) for subdir in directory.subdirectories)

    async def _build_directory_preview(self) -> DirectoryPreview:
        """Build preview of target directory structure."""
        # Group operations by target directory
        dir_structure = defaultdict(lambda: {
            'files': [],
            'size_mb': 0.0,
            'content_types': Counter(),
            'formats': Counter()
        })

        for op in self.operations:
            if op.operation_type in ['move', 'copy'] and op.target_path:
                parent = op.target_path.parent
                dir_structure[parent]['files'].append(op)
                dir_structure[parent]['size_mb'] += op.file_size / (1024 * 1024)

                if op.audio_file:
                    if op.audio_file.content_type:
                        dir_structure[parent]['content_types'][op.audio_file.content_type.value] += 1
                    if op.audio_file.file_type:
                        dir_structure[parent]['file_types'][op.audio_file.file_type] += 1

        # Build tree structure
        root_path = self.config.target_directory

        def create_directory_preview(path: Path) -> DirectoryPreview:
            # Check if this directory will have files
            files_data = dir_structure.get(path, {})
            is_new = not path.exists() if path != root_path else False
            is_empty = len(files_data.get('files', [])) == 0

            # Find subdirectories
            subdirs = []
            for dir_path in dir_structure.keys():
                if dir_path.parent == path:
                    subdirs.append(create_directory_preview(dir_path))

            return DirectoryPreview(
                path=path,
                file_count=len(files_data.get('files', [])),
                total_size_mb=files_data.get('size_mb', 0.0),
                is_new=is_new,
                is_empty=is_empty,
                content_types=dict(files_data.get('content_types', {})),
                file_types=dict(files_data.get('file_types', {})),
                subdirectories=subdirs
            )

        return create_directory_preview(root_path)

    def display_preview(self, detailed: bool = False) -> None:
        """Display the organization preview to the console."""
        self.console.print("\n" + "=" * 80)
        self.console.print("📋 ORGANIZATION PREVIEW", style="bold")
        self.console.print("=" * 80)

        # Display statistics
        self._display_statistics()

        # Display directory structure
        if self.directory_structure:
            self._display_directory_structure(self.directory_structure)

        # Display operations table
        if detailed:
            self._display_operations_table()
        else:
            self._display_operations_summary()

        # Display conflicts if any
        if self.conflicts:
            self._display_conflicts()

        self.console.print("=" * 80)

    def _display_statistics(self) -> None:
        """Display preview statistics."""
        self.console.print("\n📊 PREVIEW STATISTICS", style="bold")
        self.console.print("-" * 40)

        stats_table = [
            ["Total Files", f"{self.statistics.total_files:,}"],
            ["Total Operations", f"{self.statistics.total_operations:,}"],
            ["Total Size", f"{self.statistics.total_size_mb:.1f} MB"],
            ["Directories to Create", f"{self.statistics.directories_created:,}"],
            ["Conflicts Detected", f"{self.statistics.conflicts_detected:,}"],
            ["Organization Score", f"{self.statistics.organization_score:.0f}/100"],
            ["Est. Time", f"{self.statistics.estimated_time_minutes:.1f} min"]
        ]

        self.console.table(stats_table, headers=["Metric", "Value"])

        # Content distribution
        if self.statistics.content_types:
            self.console.print("\n📁 Content Type Distribution:")
            for content_type, count in sorted(self.statistics.content_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / self.statistics.total_files) * 100
                self.console.print(f"  • {content_type}: {count} ({percentage:.1f}%)")

    def _display_directory_structure(self, directory: DirectoryPreview, indent: int = 0) -> None:
        """Display directory structure tree."""
        prefix = "  " * indent

        if directory.path == self.config.target_directory:
            self.console.print(f"\n📂 Target Directory Structure:")
            self.console.print(f"📁 {directory.path.name}/", style="bold")
        else:
            icon = "📁" if not directory.is_empty else "📂"
            status = " (NEW)" if directory.is_new else ""
            if directory.file_count > 0:
                info = f" {directory.file_count} files, {directory.total_size_mb:.1f} MB"
            else:
                info = " (empty)"

            self.console.print(f"{prefix}{icon} {directory.path.name}{status}{info}")

        # Display subdirectories
        for subdir in directory.subdirectories:
            self._display_directory_structure(subdir, indent + 1)

    def _display_operations_summary(self) -> None:
        """Display summary of operations by type."""
        self.console.print("\n🔧 OPERATIONS SUMMARY")
        self.console.print("-" * 30)

        for op_type, count in self.statistics.operations_by_type.items():
            icon = {
                'move': '➡️',
                'copy': '📋',
                'create_dir': '📁',
                'conflict': '⚠️'
            }.get(op_type, '•')

            self.console.print(f"{icon} {op_type.title()}: {count} operations")

    def _display_operations_table(self) -> None:
        """Display detailed operations table."""
        self.console.print(f"\n📋 DETAILED OPERATIONS (showing first 20)")
        self.console.print("-" * 80)

        file_ops = [op for op in self.operations if op.operation_type in ['move', 'copy', 'conflict']]
        display_ops = file_ops[:20]  # Limit display

        headers = ["Operation", "Source", "Target", "Size", "Artist", "Title"]
        rows = []

        for op in display_ops:
            op_symbol = {
                'move': '➡️',
                'copy': '📋',
                'conflict': '⚠️'
            }.get(op.operation_type, '•')

            # Truncate paths for display
            source_name = op.source_path.name if op.source_path else ''
            source_parent = op.source_path.parent.name if op.source_path and op.source_path.parent.name else ''
            source_display = f"{source_parent}/{source_name}" if source_parent else source_name

            target_name = op.target_path.name
            target_parent = op.target_path.parent.name
            target_display = f"{target_parent}/{target_name}"

            size_mb = op.file_size / (1024 * 1024)
            size_display = f"{size_mb:.1f} MB" if size_mb > 0 else "-"

            artist = op.audio_file.primary_artist if op.audio_file and op.audio_file.primary_artist else "-"
            if not artist and op.audio_file and op.audio_file.artists:
                artist = op.audio_file.artists[0]
            title = op.audio_file.title if op.audio_file and op.audio_file.title else "-"

            rows.append([
                op_symbol,
                source_display[:30],
                target_display[:30],
                size_display,
                artist[:20],
                title[:25]
            ])

        self.console.table(rows, headers=headers)

        if len(file_ops) > 20:
            self.console.print(f"... and {len(file_ops) - 20} more operations")

    def _display_conflicts(self) -> None:
        """Display detected conflicts and resolutions."""
        self.console.print(f"\n⚠️  CONFLICTS DETECTED ({len(self.conflicts)})")
        self.console.print("-" * 40)

        for i, conflict in enumerate(self.conflicts[:10]):  # Limit display
            self.console.print(f"\n{i + 1}. {conflict.source_path.name}")
            self.console.print(f"   Conflict: {conflict.conflict_reason}")
            self.console.print(f"   Resolution: {conflict.resolution_strategy}")
            self.console.print(f"   Target: {conflict.target_path}")

        if len(self.conflicts) > 10:
            self.console.print(f"\n... and {len(self.conflicts) - 10} more conflicts")

    def export_preview(self, output_path: Path) -> None:
        """Export preview data to JSON file."""
        preview_data = {
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_files": self.statistics.total_files,
                "total_operations": self.statistics.total_operations,
                "operations_by_type": self.statistics.operations_by_type,
                "total_size_mb": self.statistics.total_size_mb,
                "directories_created": self.statistics.directories_created,
                "conflicts_detected": self.statistics.conflicts_detected,
                "organization_score": self.statistics.organization_score,
                "estimated_time_minutes": self.statistics.estimated_time_minutes,
                "content_types": self.statistics.content_types,
                "file_types": self.statistics.file_types,
                "top_artists": self.statistics.top_artists
            },
            "operations": [
                {
                    "operation_type": op.operation_type,
                    "source_path": str(op.source_path),
                    "target_path": str(op.target_path),
                    "file_size": op.file_size,
                    "conflict_reason": op.conflict_reason,
                    "resolution_strategy": op.resolution_strategy
                }
                for op in self.operations
            ],
            "conflicts": [
                {
                    "source_path": str(conflict.source_path),
                    "target_path": str(conflict.target_path),
                    "reason": conflict.conflict_reason,
                    "resolution": conflict.resolution_strategy
                }
                for conflict in self.conflicts
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(preview_data, f, indent=2)

        self.console.print(f"\n📄 Preview exported to: {output_path}")


class InteractivePreview:
    """Interactive preview system for reviewing operations."""

    def __init__(self, preview: OrganizationPreview):
        self.preview = preview
        self.console = SimpleConsole()
        self.filters = {
            'operation_type': None,
            'content_type': None,
            'artist': None,
            'show_only_conflicts': False
        }

    async def run_interactive_preview(self) -> bool:
        """Run interactive preview session.

        Returns:
            True if user confirms to proceed, False otherwise
        """
        while True:
            self._display_menu()
            choice = input("\nEnter your choice: ").strip().lower()

            if choice == '1':
                self.preview.display_preview(detailed=True)
            elif choice == '2':
                await self._filter_operations()
            elif choice == '3':
                self._export_preview()
            elif choice == '4':
                return True  # Proceed
            elif choice == '5':
                return False  # Cancel
            elif choice == 'q':
                return False
            else:
                self.console.print("Invalid choice. Please try again.")

    def _display_menu(self) -> None:
        """Display interactive preview menu."""
        self.console.print("\n" + "=" * 60)
        self.console.print("🎮 INTERACTIVE PREVIEW", style="bold")
        self.console.print("=" * 60)
        self.console.print("1. 📋 Show Detailed Preview")
        self.console.print("2. 🔍 Filter Operations")
        self.console.print("3. 💾 Export Preview")
        self.console.print("4. ✅ Proceed with Organization")
        self.console.print("5. ❌ Cancel")
        self.console.print("q. Quit")

    async def _filter_operations(self) -> None:
        """Filter operations for display."""
        self.console.print("\n🔍 Filter Operations")
        self.console.print("1. By operation type (move/copy/conflict)")
        self.console.print("2. By content type")
        self.console.print("3. By artist")
        self.console.print("4. Show only conflicts")
        self.console.print("5. Clear filters")

        choice = input("\nFilter by: ").strip()

        if choice == '1':
            op_type = input("Enter operation type (move/copy/conflict): ").strip().lower()
            if op_type in ['move', 'copy', 'conflict']:
                self.filters['operation_type'] = op_type
        elif choice == '2':
            content_type = input("Enter content type (studio/live/compilation/etc.): ").strip().lower()
            self.filters['content_type'] = content_type
        elif choice == '3':
            artist = input("Enter artist name: ").strip().lower()
            self.filters['artist'] = artist
        elif choice == '4':
            self.filters['show_only_conflicts'] = True
        elif choice == '5':
            self.filters = {k: None for k in self.filters}
            self.filters['show_only_conflicts'] = False

        self._display_filtered_operations()

    def _display_filtered_operations(self) -> None:
        """Display operations according to current filters."""
        filtered_ops = self.preview.operations

        if self.filters['operation_type']:
            filtered_ops = [op for op in filtered_ops
                          if op.operation_type == self.filters['operation_type']]

        if self.filters['show_only_conflicts']:
            filtered_ops = [op for op in filtered_ops if op.operation_type == 'conflict']

        self.console.print(f"\n🔍 Filtered Operations ({len(filtered_ops)} shown)")
        self.console.print("-" * 50)

        for i, op in enumerate(filtered_ops[:20]):
            op_symbol = {
                'move': '➡️',
                'copy': '📋',
                'conflict': '⚠️',
                'create_dir': '📁'
            }.get(op.operation_type, '•')

            self.console.print(f"{i+1:2d}. {op_symbol} {op.source_path.name}")
            if op.target_path:
                self.console.print(f"    → {op.target_path.parent.name}/{op.target_path.name}")
            if op.conflict_reason:
                self.console.print(f"    ⚠️ {op.conflict_reason}")

    def _export_preview(self) -> None:
        """Export preview to user-specified file."""
        filename = input("Enter export filename (default: preview.json): ").strip()
        if not filename:
            filename = "preview.json"

        output_path = Path(filename)
        try:
            self.preview.export_preview(output_path)
        except Exception as e:
            self.console.print(f"❌ Failed to export: {e}", style="error")