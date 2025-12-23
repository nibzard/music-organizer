"""Statistics dashboard for music library insights."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

from .core.async_organizer import AsyncMusicOrganizer
from .models.config import Config


class SimpleConsole:
    """Simple console output without rich dependency."""

    @staticmethod
    def print(text: str, style: Optional[str] = None):
        """Print text with optional style."""
        if style == 'bold':
            print(f"\033[1m{text}\033[0m")
        elif style == 'green':
            print(f"\033[92m{text}\033[0m")
        elif style == 'red':
            print(f"\033[91m{text}\033[0m")
        elif style == 'yellow':
            print(f"\033[93m{text}\033[0m")
        elif style == 'cyan':
            print(f"\033[96m{text}\033[0m")
        else:
            print(text)

    @staticmethod
    def rule(title: str):
        """Print a horizontal rule with title."""
        width = 80
        padding = (width - len(title) - 2) // 2
        print(f"{'-' * padding} {title} {'-' * padding}")

    @staticmethod
    def table(rows: List[List[str]], headers: List[str]):
        """Print a simple table."""
        if not rows:
            return

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))

        # Print rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            print(row_line)


@dataclass
class LibraryStatistics:
    """Simple library statistics data structure."""
    total_recordings: int = 0
    total_artists: int = 0
    total_releases: int = 0
    total_size_gb: float = 0.0
    total_duration_hours: float = 0.0
    format_distribution: Dict[str, int] = None
    genre_distribution: Dict[str, int] = None
    decade_distribution: Dict[str, int] = None
    top_artists: List[Tuple[str, int]] = None
    top_genres: List[Tuple[str, int]] = None
    recently_added: int = 0
    duplicates_count: int = 0
    average_bitrate: float = 0.0
    quality_distribution: Dict[str, int] = None

    def __post_init__(self):
        if self.format_distribution is None:
            self.format_distribution = {}
        if self.genre_distribution is None:
            self.genre_distribution = {}
        if self.decade_distribution is None:
            self.decade_distribution = {}
        if self.top_artists is None:
            self.top_artists = []
        if self.top_genres is None:
            self.top_genres = []
        if self.quality_distribution is None:
            self.quality_distribution = {}


@dataclass
class DashboardConfig:
    """Configuration for the statistics dashboard."""

    include_charts: bool = False  # ASCII charts
    max_top_items: int = 10
    show_file_details: bool = True
    show_quality_analysis: bool = True
    export_format: str = "json"  # json, csv, or txt


class StatisticsDashboard:
    """Interactive statistics dashboard for music library insights."""

    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.console = SimpleConsole()
        self.recordings = []

    async def initialize(self, music_library_path: Path):
        """Initialize the dashboard with a music library."""
        self.console.print("🔍 Scanning music library...", style="cyan")

        # Use AsyncMusicOrganizer to scan and extract metadata
        # Config requires source_directory and target_directory
        config = Config(
            source_directory=music_library_path,
            target_directory=music_library_path  # Dashboard doesn't move files, just reads
        )
        organizer = AsyncMusicOrganizer(config=config)

        progress = SimpleProgress(total=1, description="Loading library")

        try:
            # Scan directory for audio files
            audio_files = []
            for file_path in music_library_path.rglob("*"):
                if file_path.suffix.lower() in {'.mp3', '.flac', '.wav', '.m4a', '.aac',
                                               '.ogg', '.opus', '.wma', '.aiff', '.aif'}:
                    audio_files.append(file_path)

            progress.total = len(audio_files)

            # Extract metadata and create Recording entities
            for file_path in audio_files:
                try:
                    audio_file = await organizer.extract_metadata_async(file_path)
                    if audio_file:
                        self.recordings.append({
                            'path': file_path,
                            'metadata': audio_file.metadata
                        })
                except Exception:
                    # Skip files that can't be processed
                    pass

                progress.update()

            progress.current = progress.total
            print()  # New line

        except Exception as e:
            self.console.print(f"Error loading library: {e}", style="red")

    def _calculate_statistics(self) -> LibraryStatistics:
        """Calculate statistics from loaded recordings."""
        if not self.recordings:
            return LibraryStatistics()

        stats = LibraryStatistics()

        # Basic counts
        stats.total_recordings = len(self.recordings)

        # Calculate various statistics
        total_size_mb = 0
        total_duration_seconds = 0
        all_bitrates = []
        artists_set = set()
        albums_set = set()
        years = []

        for recording in self.recordings:
            metadata = recording['metadata']
            path = recording['path']

            # Size
            if path.exists():
                total_size_mb += path.stat().st_size / (1024 * 1024)

            # Duration
            duration = metadata.get('duration', 0)
            if isinstance(duration, (int, float)):
                total_duration_seconds += duration

            # Bitrate
            bitrate = metadata.get('bitrate')
            if bitrate and isinstance(bitrate, (int, float)):
                all_bitrates.append(bitrate)

            # Artists
            artist = metadata.get('artist', '')
            if artist:
                artists_set.add(artist)

            # Albums
            album = metadata.get('album', '')
            if album:
                albums_set.add(album)

            # Year
            year = metadata.get('year')
            if year and isinstance(year, int):
                years.append(year)

            # Format distribution
            format_name = path.suffix.upper().lstrip('.')
            stats.format_distribution[format_name] = stats.format_distribution.get(format_name, 0) + 1

            # Genre distribution
            genre = metadata.get('genre', '')
            if genre:
                for g in genre.split(','):
                    g = g.strip()
                    if g:
                        stats.genre_distribution[g] = stats.genre_distribution.get(g, 0) + 1

            # Decade distribution
            if year:
                decade = f"{(year // 10) * 10}s"
                stats.decade_distribution[decade] = stats.decade_distribution.get(decade, 0) + 1

        # Calculate derived statistics
        stats.total_artists = len(artists_set)
        stats.total_releases = len(albums_set)
        stats.total_size_gb = total_size_mb / 1024
        stats.total_duration_hours = total_duration_seconds / 3600

        if all_bitrates:
            stats.average_bitrate = sum(all_bitrates) / len(all_bitrates)

            # Quality distribution
            for bitrate in all_bitrates:
                if bitrate >= 320:
                    quality = "High (320+ kbps)"
                elif bitrate >= 256:
                    quality = "Good (256-319 kbps)"
                elif bitrate >= 192:
                    quality = "Standard (192-255 kbps)"
                elif bitrate >= 128:
                    quality = "Low (128-191 kbps)"
                else:
                    quality = "Very Low (<128 kbps)"
                stats.quality_distribution[quality] = stats.quality_distribution.get(quality, 0) + 1

        # Top artists
        artist_counts = {}
        for recording in self.recordings:
            artist = recording['metadata'].get('artist', '')
            if artist:
                artist_counts[artist] = artist_counts.get(artist, 0) + 1

        stats.top_artists = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:self.config.max_top_items]

        # Top genres
        stats.top_genres = sorted(stats.genre_distribution.items(), key=lambda x: x[1], reverse=True)[:self.config.max_top_items]

        return stats

    async def show_library_overview(self):
        """Show comprehensive library overview dashboard."""
        if not self.recordings:
            self.console.print("Dashboard not initialized. Call initialize() first.", style="red")
            return

        self.console.rule("📊 MUSIC LIBRARY DASHBOARD")
        self.console.print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="cyan")
        print()

        # Calculate statistics
        stats = self._calculate_statistics()

        # Overview Section
        self._print_overview_section(stats)

        # Format Distribution
        self._print_format_section(stats)

        # Genre Analysis
        self._print_genre_section(stats)

        # Quality Analysis
        if self.config.show_quality_analysis:
            self._print_quality_section(stats)

        # Top Artists
        self._print_artists_section(stats)

        # Temporal Analysis
        self._print_temporal_section(stats)

        # File Details
        if self.config.show_file_details:
            self._print_file_details_section(stats)

    def _print_overview_section(self, stats: LibraryStatistics):
        """Print overview statistics."""
        self.console.print("📈 OVERVIEW", style="bold")
        print()

        overview_data = [
            ["Total Tracks", f"{stats.total_recordings:,}"],
            ["Total Artists", f"{stats.total_artists:,}"],
            ["Total Albums", f"{stats.total_releases:,}"],
            ["Library Size", f"{stats.total_size_gb:.1f} GB"],
            ["Total Duration", f"{stats.total_duration_hours:.1f} hours"],
            ["Average Bitrate", f"{stats.average_bitrate:.0f} kbps"],
            ["Recently Added", f"{stats.recently_added} tracks"],
            ["Duplicate Groups", f"{stats.duplicates_count}"]
        ]

        self.console.table(overview_data, headers=["Metric", "Value"])
        print()

    def _print_format_section(self, stats: LibraryStatistics):
        """Print audio format distribution."""
        self.console.print("🎵 AUDIO FORMATS", style="bold")
        print()

        if stats.format_distribution:
            # Calculate percentages
            total = sum(stats.format_distribution.values())
            format_data = []
            for format_name, count in sorted(stats.format_distribution.items(),
                                           key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                format_data.append([format_name, f"{count:,}", f"{percentage:.1f}%"])

            self.console.table(format_data, headers=["Format", "Count", "Percentage"])

            # Simple ASCII bar chart
            if self.config.include_charts:
                print()
                max_count = max(stats.format_distribution.values())
                for format_name, count in sorted(stats.format_distribution.items(),
                                               key=lambda x: x[1], reverse=True):
                    bar_length = int((count / max_count) * 30)
                    bar = "█" * bar_length
                    print(f"{format_name:12} {bar} {count:,}")
        else:
            self.console.print("No format data available", style="yellow")
        print()

    def _print_genre_section(self, stats: LibraryStatistics):
        """Print genre distribution and analysis."""
        self.console.print("🎸 GENRE DISTRIBUTION", style="bold")
        print()

        if stats.genre_distribution:
            # Show top genres
            top_genres = list(stats.top_genres)[:self.config.max_top_items]
            genre_data = []
            total = sum(stats.genre_distribution.values())

            for genre, count in top_genres:
                percentage = (count / total) * 100
                genre_data.append([genre, f"{count:,}", f"{percentage:.1f}%"])

            self.console.table(genre_data, headers=["Genre", "Count", "Percentage"])

            # Genre diversity analysis
            total_genres = len(stats.genre_distribution)
            if total_genres > 0:
                genre_concentration = (len(top_genres) / total_genres) * 100
                print(f"Genre Diversity: {total_genres} unique genres")
                print(f"Top {len(top_genres)} genres represent {genre_concentration:.1f}% of library")
        else:
            self.console.print("No genre data available", style="yellow")
        print()

    def _print_quality_section(self, stats: LibraryStatistics):
        """Print audio quality analysis."""
        self.console.print("🎯 QUALITY ANALYSIS", style="bold")
        print()

        if stats.quality_distribution:
            quality_data = []
            total = sum(stats.quality_distribution.values())

            for quality, count in stats.quality_distribution.items():
                percentage = (count / total) * 100
                quality_data.append([quality, f"{count:,}", f"{percentage:.1f}%"])

            self.console.table(quality_data, headers=["Quality", "Count", "Percentage"])

            # Quality assessment
            high_quality = stats.quality_distribution.get("High (320+ kbps)", 0)
            good_quality = stats.quality_distribution.get("Good (256-319 kbps)", 0)
            premium_percentage = ((high_quality + good_quality) / total) * 100 if total > 0 else 0

            print(f"Premium Quality (256+ kbps): {premium_percentage:.1f}%")
            print(f"Average Bitrate: {stats.average_bitrate:.0f} kbps")
        else:
            self.console.print("No quality data available", style="yellow")
        print()

    def _print_artists_section(self, stats: LibraryStatistics):
        """Print top artists in the library."""
        self.console.print("👥 TOP ARTISTS", style="bold")
        print()

        if stats.top_artists:
            artist_data = []
            total_tracks = stats.total_recordings

            for artist, count in stats.top_artists[:self.config.max_top_items]:
                percentage = (count / total_tracks) * 100
                artist_data.append([artist, f"{count:,}", f"{percentage:.1f}%"])

            self.console.table(artist_data, headers=["Artist", "Tracks", "Percentage"])

            # Artist diversity
            unique_artists = stats.total_artists
            if unique_artists > 0:
                tracks_per_artist = total_tracks / unique_artists
                print(f"Artist Diversity: {unique_artists:,} unique artists")
                print(f"Average Tracks per Artist: {tracks_per_artist:.1f}")
        else:
            self.console.print("No artist data available", style="yellow")
        print()

    def _print_temporal_section(self, stats: LibraryStatistics):
        """Print temporal/decade analysis."""
        self.console.print("⏰ TEMPORAL ANALYSIS", style="bold")
        print()

        if stats.decade_distribution:
            decade_data = []
            total = sum(stats.decade_distribution.values())

            # Sort decades chronologically
            for decade in sorted(stats.decade_distribution.keys()):
                count = stats.decade_distribution[decade]
                percentage = (count / total) * 100
                decade_data.append([decade, f"{count:,}", f"{percentage:.1f}%"])

            self.console.table(decade_data, headers=["Decade", "Tracks", "Percentage"])

            # Era analysis
            if self.config.include_charts:
                print()
                max_count = max(stats.decade_distribution.values())
                for decade in sorted(stats.decade_distribution.keys()):
                    count = stats.decade_distribution[decade]
                    bar_length = int((count / max_count) * 30)
                    bar = "█" * bar_length
                    print(f"{decade:12} {bar} {count:,}")
        else:
            self.console.print("No temporal data available", style="yellow")
        print()

    def _print_file_details_section(self, stats: LibraryStatistics):
        """Print detailed file statistics."""
        self.console.print("📁 FILE DETAILS", style="bold")
        print()

        # Calculate file statistics
        avg_size_mb = (stats.total_size_gb * 1024) / stats.total_recordings if stats.total_recordings > 0 else 0
        avg_duration_min = (stats.total_duration_hours * 60) / stats.total_recordings if stats.total_recordings > 0 else 0

        file_data = [
            ["Average File Size", f"{avg_size_mb:.1f} MB"],
            ["Average Duration", f"{avg_duration_min:.1f} minutes"],
            ["Total Size", f"{stats.total_size_gb:.1f} GB"],
            ["Total Duration", f"{stats.total_duration_hours:.1f} hours"],
            ["Tracks per GB", f"{stats.total_recordings / stats.total_size_gb:.0f}" if stats.total_size_gb > 0 else "N/A"],
            ["Hours per GB", f"{stats.total_duration_hours / stats.total_size_gb:.1f}" if stats.total_size_gb > 0 else "N/A"]
        ]

        self.console.table(file_data, headers=["Metric", "Value"])
        print()

    async def show_artist_details(self, artist_name: str):
        """Show detailed statistics for a specific artist."""
        if not self.recordings:
            self.console.print("Dashboard not initialized. Call initialize() first.", style="red")
            return

        # Filter recordings by artist
        artist_recordings = [
            r for r in self.recordings
            if artist_name.lower() in r['metadata'].get('artist', '').lower()
        ]

        if not artist_recordings:
            self.console.print(f"Artist '{artist_name}' not found in library", style="yellow")
            return

        self.console.rule(f"🎤 ARTIST DETAILS: {artist_name.upper()}")
        print()

        # Calculate artist statistics
        albums_set = set()
        total_duration = 0
        years = []
        genres_set = set()

        for recording in artist_recordings:
            metadata = recording['metadata']

            album = metadata.get('album', '')
            if album:
                albums_set.add(album)

            duration = metadata.get('duration', 0)
            if isinstance(duration, (int, float)):
                total_duration += duration

            year = metadata.get('year')
            if year and isinstance(year, int):
                years.append(year)

            genre = metadata.get('genre', '')
            if genre:
                for g in genre.split(','):
                    genres_set.add(g.strip())

        # Artist overview
        self.console.print("📊 ARTIST OVERVIEW", style="bold")
        print()

        overview_data = [
            ["Artist", artist_name],
            ["Total Tracks", f"{len(artist_recordings):,}"],
            ["Total Albums", f"{len(albums_set):,}"],
            ["Total Duration", f"{total_duration / 3600:.1f} hours"],
            ["Average Year", f"{int(sum(years) / len(years)):,}" if years else "N/A"],
            ["Primary Genres", ", ".join(list(genres_set)[:3])]
        ]

        self.console.table(overview_data, headers=["Metric", "Value"])
        print()

    async def export_statistics(self, output_path: Path):
        """Export complete statistics to file."""
        if not self.recordings:
            self.console.print("Dashboard not initialized. Call initialize() first.", style="red")
            return

        self.console.print(f"📤 Exporting statistics to {output_path}...", style="cyan")

        # Calculate statistics
        library_stats = self._calculate_statistics()

        # Prepare export data
        export_data = {
            "generated_at": datetime.now().isoformat(),
            "library_overview": {
                "total_recordings": library_stats.total_recordings,
                "total_artists": library_stats.total_artists,
                "total_releases": library_stats.total_releases,
                "total_size_gb": library_stats.total_size_gb,
                "total_duration_hours": library_stats.total_duration_hours,
                "average_bitrate": library_stats.average_bitrate,
                "recently_added": library_stats.recently_added,
                "duplicates_count": library_stats.duplicates_count
            },
            "format_distribution": library_stats.format_distribution,
            "genre_distribution": library_stats.genre_distribution,
            "quality_distribution": library_stats.quality_distribution,
            "decade_distribution": library_stats.decade_distribution,
            "top_artists": library_stats.top_artists,
            "top_genres": library_stats.top_genres,
            "detailed_genre_distribution": library_stats.genre_distribution
        }

        # Write to file
        if self.config.export_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        elif self.config.export_format == "csv":
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for key, value in export_data.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            writer.writerow([f"{key}_{subkey}", subvalue])
                    else:
                        writer.writerow([key, value])
        else:  # txt
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("MUSIC LIBRARY STATISTICS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {export_data['generated_at']}\n\n")

                for key, value in export_data.items():
                    if key != "generated_at" and isinstance(value, dict):
                        f.write(f"{key.replace('_', ' ').title()}\n")
                        f.write("-" * len(key) + "\n")
                        for subkey, subvalue in value.items():
                            f.write(f"  {subkey}: {subvalue}\n")
                        f.write("\n")

        self.console.print(f"✅ Statistics exported successfully!", style="green")

    async def interactive_dashboard(self, library_path: Path):
        """Run interactive dashboard mode."""
        await self.initialize(library_path)

        while True:
            print("\n" + "=" * 50)
            self.console.print("📊 MUSIC LIBRARY DASHBOARD", style="bold")
            print("=" * 50)
            print("1. Library Overview")
            print("2. Genre Analysis")
            print("3. Artist Details")
            print("4. Quality Analysis")
            print("5. Format Distribution")
            print("6. Export Statistics")
            print("7. Refresh Data")
            print("0. Exit")
            print("=" * 50)

            choice = input("\nSelect an option (0-7): ").strip()

            if choice == "0":
                self.console.print("Goodbye! 👋", style="green")
                break
            elif choice == "1":
                await self.show_library_overview()
            elif choice == "2":
                # Detailed genre analysis
                await self.show_library_overview()  # Includes genre section
            elif choice == "3":
                artist_name = input("Enter artist name: ").strip()
                if artist_name:
                    await self.show_artist_details(artist_name)
            elif choice == "4":
                # Detailed quality analysis
                await self.show_library_overview()  # Includes quality section
            elif choice == "5":
                # Detailed format analysis
                await self.show_library_overview()  # Includes format section
            elif choice == "6":
                filename = input("Enter export filename (default: library_stats.json): ").strip()
                if not filename:
                    filename = "library_stats.json"

                format_choice = input("Export format (json/csv/txt) [default: json]: ").strip().lower()
                if format_choice in ['csv', 'txt']:
                    self.config.export_format = format_choice

                output_path = Path(filename)
                await self.export_statistics(output_path)
            elif choice == "7":
                self.console.print("Refreshing library data...", style="cyan")
                await self._load_library_data(library_path)
                self.console.print("Library data refreshed!", style="green")
            else:
                self.console.print("Invalid option. Please try again.", style="red")

            if choice != "0":
                input("\nPress Enter to continue...")


# Simple progress bar class for loading
class SimpleProgress:
    """Simple progress bar implementation without external dependencies."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()

    def update(self, advance: int = 1):
        """Update progress bar."""
        self.current += advance
        self._draw()

    def _draw(self):
        """Draw the progress bar."""
        if self.total == 0:
            return

        percent = min(100, (self.current * 100) // self.total)
        bar_length = 50
        filled = (percent * bar_length) // 100
        bar = '█' * filled + '░' * (bar_length - filled)

        elapsed = datetime.now() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f"{int(eta.total_seconds())}s"
        else:
            eta_str = "?:??"

        print(f"\r{self.description}: [{bar}] {percent}% ({self.current}/{self.total}) ETA: {eta_str}", end='')

        if self.current >= self.total:
            print()  # New line when complete