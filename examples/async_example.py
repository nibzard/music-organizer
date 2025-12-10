#!/usr/bin/env python3
"""Example demonstrating async music organization."""

import asyncio
from pathlib import Path
from music_organizer.core.async_organizer import AsyncMusicOrganizer
from music_organizer.models.config import Config


async def main():
    """Demonstrate async organization."""
    # Setup paths
    source_dir = Path("/path/to/your/music")
    target_dir = Path("/path/to/organized/music")

    # Create config
    config = Config(
        source_directory=source_dir,
        target_directory=target_dir
    )

    # Create async organizer with 8 worker threads
    async with AsyncMusicOrganizer(
        config,
        dry_run=True,  # Set to False to actually move files
        max_workers=8
    ) as organizer:

        print("Scanning music library...")
        file_count = 0

        # Count files with async generator
        async for _ in organizer.scan_directory(source_dir):
            file_count += 1

        print(f"Found {file_count} audio files")

        # Process files in streaming mode (memory efficient)
        print("\nProcessing files...")
        processed = 0

        async for file_path, success, error in organizer.organize_files_streaming(
            organizer.scan_directory(source_dir),
            batch_size=50  # Process 50 files at a time
        ):
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} files...")

        # Get summary
        summary = await organizer.get_operation_summary()
        print(f"\nOperation complete!")
        print(f"Files moved: {summary.get('total_files', 0)}")
        print(f"Directories created: {summary.get('directories_created', 0)}")


if __name__ == "__main__":
    asyncio.run(main())