"""Command line interface for music organizer."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from .core.organizer import MusicOrganizer
from .core.metadata import MetadataHandler
from .core.classifier import ContentClassifier
from .core.mover import FileMover, DirectoryOrganizer
from .models.config import Config, load_config
from .exceptions import MusicOrganizerError
from .progress_tracker import IntelligentProgressTracker, ProgressStage
from .rich_progress_renderer import RichProgressRenderer

console = Console()


@click.group()
@click.version_option()
def cli():
    """Organize your music library with smart metadata-based categorization."""
    pass


@cli.command()
@click.argument('source', type=click.Path(exists=True, path_type=Path))
@click.argument('target', type=click.Path(path_type=Path))
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration file path'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be done without making changes'
)
@click.option(
    '--interactive',
    is_flag=True,
    help='Prompt for ambiguous categorizations'
)
@click.option(
    '--backup/--no-backup',
    default=True,
    help='Create backup before reorganization'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Verbose output'
)
def organize(
    source: Path,
    target: Path,
    config: Optional[Path],
    dry_run: bool,
    interactive: bool,
    backup: bool,
    verbose: bool
):
    """Organize music from SOURCE directory to TARGET directory."""

    try:
        # Load configuration
        if config:
            cfg = load_config(config)
        else:
            cfg = Config(
                source_directory=source,
                target_directory=target,
                file_operations={'backup': backup}
            )

        # Show plan
        console.print("\n[bold cyan]Music Organization Plan[/bold cyan]")
        console.print(f"Source: {source}")
        console.print(f"Target: {target}")
        console.print(f"Strategy: Move files")
        console.print(f"Backup: {'Enabled' if backup else 'Disabled'}")
        console.print(f"Interactive: {'Enabled' if interactive else 'Disabled'}")
        console.print(f"Dry run: {'Yes' if dry_run else 'No'}")

        if not dry_run:
            if not Confirm.ask("\nProceed with organization?"):
                console.print("[yellow]Cancelled[/yellow]")
                return

        # Initialize intelligent progress tracker
        progress_tracker = IntelligentProgressTracker()
        progress_renderer = RichProgressRenderer(console)
        progress_tracker.add_render_callback(progress_renderer.render)

        # Initialize organizer
        organizer = MusicOrganizer(cfg, dry_run=dry_run, interactive=interactive)

        # Execute organization
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            # Scan files
            scan_task = progress.add_task("Scanning music files...", total=None)
            files = organizer.scan_directory(source)
            progress.update(scan_task, completed=True)

            console.print(f"\n[green]Found {len(files)} audio files[/green]")

            # Set up intelligent progress tracking
            progress_tracker.set_total_files(len(files))
            progress_tracker.start_stage(ProgressStage.SCANNING)
            progress_tracker.finish_stage(ProgressStage.SCANNING)
            progress_tracker.start_stage(ProgressStage.METADATA_EXTRACTION, total=len(files))

            # Process files with custom progress callback
            results = self._organize_with_intelligent_progress(
                organizer, files, progress_tracker
            )

        # Clean up progress tracker
        progress_tracker.finish_stage(ProgressStage.METADATA_EXTRACTION)
        progress_renderer.clear()

        # Show results with intelligent progress summary
        progress_renderer.finish(progress_tracker.metrics)

        results_table = Table(title="Results")
        results_table.add_column("Category", style="cyan")
        results_table.add_column("Count", justify="right")

        for category, count in results['by_category'].items():
            results_table.add_row(category, str(count))

        console.print(results_table)

        if results['errors']:
            console.print("\n[red]Errors encountered:[/red]")
            for error in results['errors'][:10]:  # Show first 10 errors
                console.print(f"  • {error}")
            if len(results['errors']) > 10:
                console.print(f"  ... and {len(results['errors']) - 10} more errors")

    except MusicOrganizerError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


def _organize_with_intelligent_progress(organizer, files, progress_tracker):
    """Organize files with intelligent progress tracking."""
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

                # Update progress tracker with file size
                try:
                    file_size = file_path.stat().st_size
                except:
                    file_size = 0

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

                progress_tracker.set_completed(
                    results['processed'],
                    bytes_processed=file_size,
                    error=error_occurred
                )

    finally:
        # Finish file mover session
        if not organizer.dry_run:
            organizer.file_mover.finish_operation()

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


@cli.command()
@click.argument('directory', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--recursive',
    is_flag=True,
    default=True,
    help='Scan subdirectories recursively'
)
def scan(directory: Path, recursive: bool):
    """Scan and analyze music files in a directory."""

    try:
        handler = MetadataHandler()
        classifier = ContentClassifier()

        # Find audio files
        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}
        files = []

        if recursive:
            files = [f for f in directory.rglob('*') if f.suffix.lower() in audio_extensions]
        else:
            files = [f for f in directory.glob('*') if f.suffix.lower() in audio_extensions]

        if not files:
            console.print("[yellow]No audio files found[/yellow]")
            return

        # Analyze files
        analysis = {
            'total_files': len(files),
            'file_types': {},
            'content_types': {},
            'has_metadata': 0,
            'total_size_mb': 0,
            'sample_files': []
        }

        console.print(f"\n[cyan]Analyzing {len(files)} files...[/cyan]")

        with Progress() as progress:
            task = progress.add_task("Processing files...", total=len(files))

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
                            'path': str(file_path.relative_to(directory)),
                            'type': content_type.value,
                            'artists': audio_file.artists,
                            'album': audio_file.album
                        })

                except Exception:
                    pass  # Skip files that can't be processed

                progress.advance(task)

        # Display results
        console.print("\n[bold]Directory Analysis[/bold]")

        # Basic info
        info_table = Table()
        info_table.add_column("Metric", style="cyan")
        info_table.add_column("Value", justify="right")

        info_table.add_row("Total files", str(analysis['total_files']))
        info_table.add_row("Total size", f"{analysis['total_size_mb']:.1f} MB")
        info_table.add_row("Files with metadata", f"{analysis['has_metadata']} ({analysis['has_metadata']/analysis['total_files']*100:.1f}%)")

        console.print(info_table)

        # File types
        if analysis['file_types']:
            console.print("\n[bold]File Types:[/bold]")
            for ext, count in sorted(analysis['file_types'].items()):
                console.print(f"  {ext}: {count}")

        # Content types
        if analysis['content_types']:
            console.print("\n[bold]Content Types:[/bold]")
            for content_type, count in sorted(analysis['content_types'].items()):
                console.print(f"  {content_type}: {count}")

        # Sample files
        if analysis['sample_files']:
            console.print("\n[bold]Sample Files:[/bold]")
            for sample in analysis['sample_files']:
                console.print(f"\n  [dim]{sample['path']}[/dim]")
                console.print(f"    Type: {sample['type']}")
                if sample['artists']:
                    console.print(f"    Artists: {', '.join(sample['artists'][:3])}")
                if sample['album']:
                    console.print(f"    Album: {sample['album']}")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True, path_type=Path))
def inspect(file_path: Path):
    """Inspect metadata of a single audio file."""

    try:
        handler = MetadataHandler()
        classifier = ContentClassifier()

        if file_path.suffix.lower() not in {'.flac', '.mp3', '.wav', '.m4a', '.aac'}:
            console.print("[red]Not a supported audio file[/red]")
            return

        # Extract metadata
        audio_file = handler.extract_metadata(file_path)
        content_type, confidence = classifier.classify(audio_file)

        # Display info
        console.print(f"\n[bold]File: {file_path.name}[/bold]")

        info_table = Table()
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value")

        info_table.add_row("File type", audio_file.file_type)
        info_table.add_row("Size", f"{audio_file.size_mb:.1f} MB")
        info_table.add_row("Content type", f"{content_type.value} (confidence: {confidence:.2f})")

        if audio_file.artists:
            info_table.add_row("Artists", ", ".join(audio_file.artists))

        if audio_file.primary_artist:
            info_table.add_row("Primary artist", audio_file.primary_artist)

        if audio_file.album:
            info_table.add_row("Album", audio_file.album)

        if audio_file.title:
            info_table.add_row("Title", audio_file.title)

        if audio_file.year:
            info_table.add_row("Year", str(audio_file.year))

        if audio_file.date:
            info_table.add_row("Date", audio_file.date)

        if audio_file.location:
            info_table.add_row("Location", audio_file.location)

        if audio_file.track_number:
            info_table.add_row("Track", str(audio_file.track_number))

        if audio_file.genre:
            info_table.add_row("Genre", audio_file.genre)

        info_table.add_row("Has cover art", "Yes" if audio_file.has_cover_art else "No")

        console.print(info_table)

        # Target path prediction
        target_path = audio_file.get_target_path(Path("/tmp/test"))
        console.print(f"\n[cyan]Target directory:[/cyan] {target_path}")

        # Show raw metadata if verbose
        if audio_file.metadata:
            console.print("\n[bold]Raw Metadata:[/bold]")
            for key, value in list(audio_file.metadata.items())[:20]:  # Limit to 20 items
                console.print(f"  {key}: {value}")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('directory', type=click.Path(exists=True, path_type=Path))
def validate(directory: Path):
    """Validate music directory organization."""

    try:
        console.print(f"\n[cyan]Validating organization in: {directory}[/cyan]")

        # Check directory structure
        validation = DirectoryOrganizer.validate_structure(directory)

        structure_table = Table(title="Directory Structure")
        structure_table.add_column("Directory", style="cyan")
        structure_table.add_column("Status", style="green")

        all_good = True
        for dir_name, exists in validation.items():
            status = "✓" if exists else "✗"
            color = "green" if exists else "red"
            structure_table.add_row(dir_name, f"[{color}]{status}[/{color}]")
            if not exists:
                all_good = False

        console.print(structure_table)

        # Check for empty directories
        empty_dirs = DirectoryOrganizer.get_empty_directories(directory)
        if empty_dirs:
            console.print(f"\n[yellow]Found {len(empty_dirs)} empty directories[/yellow]")
            if Confirm.ask("List empty directories?"):
                for empty_dir in empty_dirs[:10]:  # Show first 10
                    console.print(f"  {empty_dir}")
                if len(empty_dirs) > 10:
                    console.print(f"  ... and {len(empty_dirs) - 10} more")

        # Check for misplaced files
        misplaced = []
        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.aac'}

        for item in directory.iterdir():
            if item.is_file() and item.suffix.lower() in audio_extensions:
                misplaced.append(item)

        if misplaced:
            console.print(f"\n[yellow]Found {len(misplaced)} audio files in root directory[/yellow]")
            for file_path in misplaced[:5]:
                console.print(f"  {file_path.name}")
            if len(misplaced) > 5:
                console.print(f"  ... and {len(misplaced) - 5} more")

        if all_good and not misplaced and not empty_dirs:
            console.print("\n[green]✓ Directory is properly organized![/green]")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()