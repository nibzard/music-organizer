"""Command line interface for music organizer."""

import sys
import argparse
from pathlib import Path
from typing import Optional

from .core.organizer import MusicOrganizer
from .core.metadata import MetadataHandler
from .core.classifier import ContentClassifier
from .core.mover import FileMover, DirectoryOrganizer
from .models.config import Config, load_config
from .exceptions import MusicOrganizerError
from .progress_tracker import IntelligentProgressTracker, ProgressStage
from .console_utils import SimpleConsole, SimpleProgress, format_size

console = SimpleConsole()


def organize_command(args):
    """Handle the organize command."""
    try:
        # Load configuration
        if args.config:
            cfg = load_config(args.config)
        else:
            cfg = Config(
                source_directory=args.source,
                target_directory=args.target,
                file_operations=type('FileOps', (), {'backup': args.backup})()
            )

        # Show plan
        console.print("\n🎵 Music Organization Plan", 'bold')
        console.print(f"Source: {args.source}")
        console.print(f"Target: {args.target}")
        console.print(f"Strategy: Move files")
        console.print(f"Backup: {'Enabled' if args.backup else 'Disabled'}")
        console.print(f"Interactive: {'Enabled' if args.interactive else 'Disabled'}")
        console.print(f"Dry run: {'Yes' if args.dry_run else 'No'}")

        if not args.dry_run:
            if not console.confirm("\nProceed with organization?", default=True):
                console.print("Cancelled", 'yellow')
                return 0

        # Initialize organizer
        organizer = MusicOrganizer(cfg, dry_run=args.dry_run, interactive=args.interactive)

        # Scan files
        console.print("\n🔍 Scanning music files...")
        files = organizer.scan_directory(args.source)

        if not files:
            console.print("No audio files found!", 'yellow')
            return 0

        console.print(f"\nFound {len(files)} audio files", 'green')

        # Process files with progress
        progress = SimpleProgress(len(files), "Processing files")
        results = _organize_with_progress(organizer, files, progress)

        # Show results
        console.rule("📊 Results")

        # Summary table
        summary_data = [
            ["Processed", str(results['processed'])],
            ["Moved", str(results['moved'])],
            ["Skipped", str(results['skipped'])],
            ["Errors", str(len(results['errors']))]
        ]
        console.table(summary_data, ["Metric", "Count"])

        # Category breakdown
        if results['moved'] > 0:
            console.print("\n📂 Files by category:")
            for category, count in results['by_category'].items():
                if count > 0:
                    console.print(f"  {category}: {count}", 'green')

        # Show errors if any
        if results['errors']:
            console.print("\n❌ Errors encountered:", 'red')
            for error in results['errors'][:10]:  # Show first 10 errors
                console.print(f"  • {error}", 'red')
            if len(results['errors']) > 10:
                console.print(f"  ... and {len(results['errors']) - 10} more errors", 'red')

        return 0

    except MusicOrganizerError as e:
        console.print(f"\nError: {e}", 'red')
        return 1
    except Exception as e:
        console.print(f"\nUnexpected error: {e}", 'red')
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


def scan_command(args):
    """Handle the scan command."""
    try:
        handler = MetadataHandler()
        classifier = ContentClassifier()

        # Find audio files
        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}
        files = []

        if args.recursive:
            files = [f for f in args.directory.rglob('*') if f.suffix.lower() in audio_extensions]
        else:
            files = [f for f in args.directory.glob('*') if f.suffix.lower() in audio_extensions]

        if not files:
            console.print("No audio files found!", 'yellow')
            return 0

        # Analyze files
        analysis = {
            'total_files': len(files),
            'file_types': {},
            'content_types': {},
            'has_metadata': 0,
            'total_size_mb': 0,
            'sample_files': []
        }

        console.print(f"\nAnalyzing {len(files)} files...")

        progress = SimpleProgress(len(files), "Analyzing")

        for file_path in files:
            try:
                audio_file = handler.extract_metadata(file_path)
                content_type, _ = classifier.classify(audio_file)

                # File type
                ext = file_path.suffix.lower()
                analysis['file_types'][ext] = analysis['file_types'].get(ext, 0) + 1

                # Content type
                analysis['content_types'][content_type.value] = \
                    analysis['content_types'].get(content_type.value, 0) + 1

                # Metadata
                if audio_file.album or audio_file.title or audio_file.artists:
                    analysis['has_metadata'] += 1

                # Size
                analysis['total_size_mb'] += audio_file.size_mb

                # Sample files
                if len(analysis['sample_files']) < 5:
                    analysis['sample_files'].append({
                        'path': str(file_path.relative_to(args.directory)),
                        'type': content_type.value,
                        'artists': audio_file.artists,
                        'album': audio_file.album
                    })

            except Exception:
                pass  # Skip files that can't be processed

            progress.update()

        # Display results
        console.rule("📊 Directory Analysis")

        # Basic info
        info_data = [
            ["Total files", str(analysis['total_files'])],
            ["Total size", f"{analysis['total_size_mb']:.1f} MB"],
            ["Files with metadata", f"{analysis['has_metadata']} ({analysis['has_metadata']/analysis['total_files']*100:.1f}%)"]
        ]
        console.table(info_data, ["Metric", "Value"])

        # File types
        if analysis['file_types']:
            console.print("\n📄 File Types:")
            for ext, count in sorted(analysis['file_types'].items()):
                console.print(f"  {ext}: {count}")

        # Content types
        if analysis['content_types']:
            console.print("\n📂 Content Types:")
            for content_type, count in sorted(analysis['content_types'].items()):
                console.print(f"  {content_type}: {count}")

        # Sample files
        if analysis['sample_files']:
            console.print("\n🎵 Sample Files:")
            for sample in analysis['sample_files']:
                console.print(f"\n  {sample['path']}", 'dim')
                console.print(f"    Type: {sample['type']}")
                if sample['artists']:
                    console.print(f"    Artists: {', '.join(sample['artists'][:3])}")
                if sample['album']:
                    console.print(f"    Album: {sample['album']}")

        return 0

    except Exception as e:
        console.print(f"\nError: {e}", 'red')
        return 1


def inspect_command(args):
    """Handle the inspect command."""
    try:
        handler = MetadataHandler()
        classifier = ContentClassifier()

        if args.file_path.suffix.lower() not in {'.flac', '.mp3', '.wav', '.m4a', '.aac'}:
            console.print("Not a supported audio file", 'red')
            return 0

        # Extract metadata
        audio_file = handler.extract_metadata(args.file_path)
        content_type, confidence = classifier.classify(audio_file)

        # Display info
        console.print(f"\n🎵 File: {args.file_path.name}", 'bold')

        info_data = [
            ["File type", audio_file.file_type],
            ["Size", f"{audio_file.size_mb:.1f} MB"],
            ["Content type", f"{content_type.value} (confidence: {confidence:.2f})"]
        ]

        if audio_file.artists:
            info_data.append(["Artists", ", ".join(audio_file.artists)])

        if audio_file.primary_artist:
            info_data.append(["Primary artist", audio_file.primary_artist])

        if audio_file.album:
            info_data.append(["Album", audio_file.album])

        if audio_file.title:
            info_data.append(["Title", audio_file.title])

        if audio_file.year:
            info_data.append(["Year", str(audio_file.year)])

        if audio_file.date:
            info_data.append(["Date", audio_file.date])

        if audio_file.location:
            info_data.append(["Location", audio_file.location])

        if audio_file.track_number:
            info_data.append(["Track", str(audio_file.track_number)])

        if audio_file.genre:
            info_data.append(["Genre", audio_file.genre])

        info_data.append(["Has cover art", "Yes" if audio_file.has_cover_art else "No"])

        console.table(info_data, ["Property", "Value"])

        # Target path prediction
        target_path = audio_file.get_target_path(Path("/tmp/test"))
        console.print(f"\n📂 Target directory: {target_path}", 'cyan')

        # Show raw metadata if verbose
        if audio_file.metadata:
            console.print("\n📋 Raw Metadata:")
            for key, value in list(audio_file.metadata.items())[:20]:  # Limit to 20 items
                console.print(f"  {key}: {value}")

        return 0

    except Exception as e:
        console.print(f"\nError: {e}", 'red')
        return 1


def validate_command(args):
    """Handle the validate command."""
    try:
        console.print(f"\n🔍 Validating organization in: {args.directory}")

        # Check directory structure
        validation = DirectoryOrganizer.validate_structure(args.directory)

        structure_data = []
        for dir_name, exists in validation.items():
            status = "✓" if exists else "✗"
            color = 'green' if exists else 'red'
            structure_data.append([dir_name, f"{color}{status}{SimpleConsole.COLORS['reset']}"])

        console.table(structure_data, ["Directory", "Status"], "Directory Structure")

        all_good = all(validation.values())

        # Check for empty directories
        empty_dirs = DirectoryOrganizer.get_empty_directories(args.directory)
        if empty_dirs:
            console.print(f"\n⚠️ Found {len(empty_dirs)} empty directories", 'yellow')
            if console.confirm("List empty directories?"):
                for empty_dir in empty_dirs[:10]:  # Show first 10
                    console.print(f"  {empty_dir}")
                if len(empty_dirs) > 10:
                    console.print(f"  ... and {len(empty_dirs) - 10} more")

        # Check for misplaced files
        misplaced = []
        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}

        for item in args.directory.iterdir():
            if item.is_file() and item.suffix.lower() in audio_extensions:
                misplaced.append(item)

        if misplaced:
            console.print(f"\n⚠️ Found {len(misplaced)} audio files in root directory", 'yellow')
            for file_path in misplaced[:5]:
                console.print(f"  {file_path.name}")
            if len(misplaced) > 5:
                console.print(f"  ... and {len(misplaced) - 5} more")

        if all_good and not misplaced and not empty_dirs:
            console.print("\n✓ Directory is properly organized!", 'green')

        return 0

    except Exception as e:
        console.print(f"\nError: {e}", 'red')
        return 1


def _organize_with_progress(organizer, files, progress):
    """Organize files with progress tracking."""
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
    if not organizer.dry_run:
        organizer.file_mover.start_operation(organizer.config.source_directory)

    try:
        # Create target directory structure
        if not organizer.dry_run:
            from .core.mover import DirectoryOrganizer
            DirectoryOrganizer.create_directory_structure(organizer.config.target_directory)

        # Group files by album/cover art
        file_groups = organizer._group_files(files)

        # Process each group
        for group_path, group_files in file_groups.items():
            for file_path in group_files:
                results['processed'] += 1
                error_occurred = False

                progress.update()
                progress.set_description(f"Processing {file_path.name[:30]}...")

                try:
                    # Process individual file
                    moved = organizer._process_file(file_path)
                    if moved:
                        results['moved'] += 1
                    else:
                        results['skipped'] += 1

                    # Update category count
                    if hasattr(moved, 'content_type'):
                        category = _get_category_name(moved.content_type)
                        results['by_category'][category] += 1

                except Exception as e:
                    error_msg = f"Failed to process {file_path.name}: {e}"
                    results['errors'].append(error_msg)
                    error_occurred = True

    finally:
        # Finish file mover session
        if not organizer.dry_run:
            organizer.file_mover.finish_operation()

    progress.finish()
    return results


def _get_category_name(content_type):
    """Get category name from content type."""
    from .models.audio_file import ContentType
    mapping = {
        ContentType.ALBUM: 'Albums',
        ContentType.LIVE: 'Live',
        ContentType.COLLABORATION: 'Collaborations',
        ContentType.COMPILATION: 'Compilations',
        ContentType.RARITY: 'Rarities',
        ContentType.UNKNOWN: 'Unknown'
    }
    return mapping.get(content_type, 'Unknown')


def main():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Organize your music library with smart metadata-based categorization",
        epilog="Example: music-organize organize /path/to/music /path/to/organized --dry-run"
    )

    # Add version
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Organize command
    org_parser = subparsers.add_parser('organize', help='Organize music files')
    org_parser.add_argument('source', type=Path, help='Source directory containing music files')
    org_parser.add_argument('target', type=Path, help='Target directory for organized files')
    org_parser.add_argument('--config', type=Path, help='Configuration file path')
    org_parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    org_parser.add_argument('--interactive', action='store_true', help='Prompt for ambiguous categorizations')
    org_parser.add_argument('--backup/--no-backup', default=True, help='Create backup before reorganization (default: enabled)')
    org_parser.add_argument('--verbose', action='store_true', help='Verbose output')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Analyze music library')
    scan_parser.add_argument('directory', type=Path, help='Directory to scan')
    scan_parser.add_argument('--recursive', action='store_true', default=True, help='Scan subdirectories recursively')

    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect metadata of a single audio file')
    inspect_parser.add_argument('file_path', type=Path, help='Path to audio file')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate music directory organization')
    validate_parser.add_argument('directory', type=Path, help='Directory to validate')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run appropriate command
    if args.command == 'organize':
        return organize_command(args)
    elif args.command == 'scan':
        return scan_command(args)
    elif args.command == 'inspect':
        return inspect_command(args)
    elif args.command == 'validate':
        return validate_command(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())