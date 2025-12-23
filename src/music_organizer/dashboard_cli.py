"""Command line interface for the statistics dashboard."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from .dashboard import StatisticsDashboard, DashboardConfig


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Music Library Statistics Dashboard - Comprehensive insights into your music collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive dashboard mode
  music-organize-dashboard /path/to/music

  # Quick overview
  music-organize-dashboard /path/to/music --overview

  # Export statistics
  music-organize-dashboard /path/to/music --export library_stats.json

  # Artist-specific analysis
  music-organize-dashboard /path/to/music --artist "The Beatles"

  # Genre distribution
  music-organize-dashboard /path/to/music --genre-analysis

  # Quality analysis only
  music-organize-dashboard /path/to/music --quality-only

  # Export as CSV
  music-organize-dashboard /path/to/music --export stats.csv --format csv

  # Full report with ASCII charts
  music-organize-dashboard /path/to/music --overview --charts
        """
    )

    parser.add_argument(
        "library_path",
        type=Path,
        help="Path to the music library directory to analyze"
    )

    # Display options
    parser.add_argument(
        "--overview", "-o",
        action="store_true",
        help="Show complete library overview dashboard"
    )

    parser.add_argument(
        "--artist", "-a",
        type=str,
        help="Show detailed statistics for specific artist"
    )

    parser.add_argument(
        "--genre-analysis", "-g",
        action="store_true",
        help="Show detailed genre distribution and analysis"
    )

    parser.add_argument(
        "--quality-only", "-q",
        action="store_true",
        help="Show only quality/bitrate analysis"
    )

    parser.add_argument(
        "--format-only", "-f",
        action="store_true",
        help="Show only audio format distribution"
    )

    parser.add_argument(
        "--charts", "-c",
        action="store_true",
        help="Include ASCII charts in the output"
    )

    # Export options
    parser.add_argument(
        "--export", "-e",
        type=str,
        help="Export statistics to file"
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv", "txt"],
        default="json",
        help="Export format (default: json)"
    )

    # Configuration options
    parser.add_argument(
        "--max-items", "-m",
        type=int,
        default=10,
        help="Maximum number of items to show in top lists (default: 10)"
    )

    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Hide detailed file analysis"
    )

    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip quality/bitrate analysis"
    )

    # Interactive mode
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode with menu navigation"
    )

    return parser


async def run_dashboard(args: argparse.Namespace):
    """Run the statistics dashboard with given arguments."""
    # Validate library path
    if not args.library_path.exists():
        print(f"Error: Library path '{args.library_path}' does not exist", file=sys.stderr)
        return 1

    if not args.library_path.is_dir():
        print(f"Error: '{args.library_path}' is not a directory", file=sys.stderr)
        return 1

    # Create dashboard configuration
    config = DashboardConfig(
        include_charts=args.charts,
        max_top_items=args.max_items,
        show_file_details=not args.no_details,
        show_quality_analysis=not args.no_quality,
        export_format=args.format
    )

    # Initialize dashboard
    dashboard = StatisticsDashboard(config)

    try:
        # Interactive mode
        if args.interactive or not any([args.overview, args.artist, args.genre_analysis,
                                       args.quality_only, args.format_only, args.export]):
            await dashboard.interactive_dashboard(args.library_path)
            return 0

        # Initialize dashboard for non-interactive mode
        await dashboard.initialize(args.library_path)

        # Run specific analyses
        if args.overview:
            await dashboard.show_library_overview()

        if args.artist:
            await dashboard.show_artist_details(args.artist)

        if args.genre_analysis:
            # Genre analysis is part of overview, but we can focus on it
            from .console_utils import SimpleConsole
            console = SimpleConsole()
            console.rule("🎸 DETAILED GENRE ANALYSIS")

            stats = await dashboard.query_bus.dispatch(
                dashboard.GetLibraryStatisticsQuery(include_detailed_breakdown=True)
            )
            dashboard._print_genre_section(stats)

        if args.quality_only:
            from .console_utils import SimpleConsole
            console = SimpleConsole()
            console.rule("🎯 QUALITY ANALYSIS")

            stats = await dashboard.query_bus.dispatch(
                dashboard.GetLibraryStatisticsQuery(include_detailed_breakdown=True)
            )
            dashboard._print_quality_section(stats)

        if args.format_only:
            from .console_utils import SimpleConsole
            console = SimpleConsole()
            console.rule("🎵 FORMAT DISTRIBUTION")

            stats = await dashboard.query_bus.dispatch(
                dashboard.GetLibraryStatisticsQuery(include_detailed_breakdown=True)
            )
            dashboard._print_format_section(stats)

        # Export if requested
        if args.export:
            output_path = Path(args.export)
            await dashboard.export_statistics(output_path)

        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    """Main entry point for the dashboard CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Run the async dashboard
    exit_code = asyncio.run(run_dashboard(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()