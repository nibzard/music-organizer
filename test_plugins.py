#!/usr/bin/env python3
"""Test script to demonstrate the plugin system."""

import asyncio
from pathlib import Path
from src.music_organizer.plugins import PluginManager
from src.music_organizer.models.audio_file import AudioFile


async def test_plugin_system():
    """Test the plugin system with example plugins."""
    print("=== Testing Music Organizer Plugin System ===\n")

    # Create plugin manager
    plugin_manager = PluginManager()

    # Discover plugins
    print("Discovering plugins...")
    discovered = plugin_manager.discover_plugins()
    print(f"Found plugins: {discovered}\n")

    # List all available plugins
    print("Available plugins:")
    plugins_info = plugin_manager.list_plugins()
    for name, info in plugins_info.items():
        print(f"  - {name}: {info.description} (v{info.version})")
    print()

    # Load the example classifier plugin
    print("Loading example classifier plugin...")
    classifier = plugin_manager.load_plugin(
        "example_classifier",
        config={
            "enabled": True,
            "tags": ["decade", "energy"],
            "high_energy_genres": ["rock", "metal", "electronic"],
            "low_energy_genres": ["classical", "ambient"]
        }
    )

    if classifier:
        print(f"Loaded plugin: {classifier.info.name}\n")

        # Create a test audio file
        test_file = AudioFile(
            path=Path("/test/rock_song.mp3"),
            file_type="MP3",
            title="Rock Anthem",
            artists=["The Rockers"],
            genre="Rock",
            year=2020
        )

        # Test classification
        print("Testing classification...")
        result = await classifier.classify(test_file)
        print(f"Classification result: {result}\n")

        # Show plugin configuration schema
        schema = plugin_manager.get_plugin_schema("example_classifier")
        if schema:
            print("Plugin configuration schema:")
            for name, option in schema._options.items():
                default = option.default if option.default is not None else "None"
                print(f"  - {name}: {option.type.__name__} (default: {default})")
                if option.description:
                    print(f"    Description: {option.description}")
            print()

    # Load the M3U exporter plugin
    print("Loading M3U exporter plugin...")
    exporter = plugin_manager.load_plugin(
        "m3u_exporter",
        config={
            "enabled": True,
            "output_dir": "/tmp/playlists",
            "overwrite": True
        }
    )

    if exporter:
        print(f"Loaded plugin: {exporter.info.name}\n")

        # Test export
        print("Testing playlist export...")
        test_files = [
            test_file,
            AudioFile(
                path=Path("/test/jazz_track.flac"),
                file_type="FLAC",
                title="Smooth Jazz",
                artists=["Jazz Master"],
                year=1995
            )
        ]

        output_path = Path("/tmp/test_playlist.m3u")
        success = await exporter.export(test_files, output_path)
        if success:
            print(f"Successfully exported playlist to {output_path}")
        else:
            print("Failed to export playlist")

    # Save plugin configuration
    print("\nSaving plugin configuration...")
    config_path = Path("/tmp/plugin_config.json")
    plugin_manager.save_config(config_path)
    print(f"Configuration saved to {config_path}")

    # Cleanup
    print("\nCleaning up plugins...")
    plugin_manager.unload_plugin("example_classifier")
    plugin_manager.unload_plugin("m3u_exporter")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(test_plugin_system())