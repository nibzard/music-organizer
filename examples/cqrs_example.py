#!/usr/bin/env python3
"""
Example demonstrating the CQRS pattern in the Music Organizer.

This example shows:
1. Command side: Adding and updating recordings
2. Query side: Retrieving recordings and statistics
3. Event-driven updates between commands and queries
4. Caching of query results
"""

import asyncio
from pathlib import Path
from datetime import datetime

# Import CQRS components
from music_organizer.application.commands import CommandBus
from music_organizer.application.queries import QueryBus, QueryCache
from music_organizer.application.events import EventBus, EventStore

# Import catalog commands
from music_organizer.application.commands.catalog import (
    AddRecordingCommand,
    AddRecordingCommandHandler,
    UpdateMetadataCommand,
    UpdateMetadataCommandHandler
)

# Import catalog queries
from music_organizer.application.queries.catalog import (
    GetRecordingByIdQuery,
    GetRecordingsByArtistQuery,
    SearchRecordingsQuery,
    GetLibraryStatisticsQuery,
    GetGenreDistributionQuery
)

# Import repositories
from music_organizer.domain.catalog.repositories import InMemoryRecordingRepository

# Import read model cache updater
from music_organizer.application.read_models import ReadModelProjector


class MusicLibraryCQRS:
    """Example CQRS-based music library service."""

    def __init__(self):
        # Create infrastructure
        self.recording_repo = InMemoryRecordingRepository()
        self.event_bus = EventBus()
        self.event_store = EventStore()
        self.query_cache = QueryCache()

        # Create buses
        self.command_bus = CommandBus()
        self.query_bus = QueryBus()
        self.query_bus.set_cache(self.query_cache)

        # Register command handlers
        self._register_command_handlers()

        # Register query handlers
        self._register_query_handlers()

        # Register event handlers
        self._register_event_handlers()

        # Create read model projector
        self.read_model_projector = ReadModelProjector()

    def _register_command_handlers(self):
        """Register all command handlers."""
        add_handler = AddRecordingCommandHandler(self.recording_repo, self.event_bus)
        update_handler = UpdateMetadataCommandHandler(self.recording_repo, self.event_bus)

        self.command_bus.register(AddRecordingCommand, add_handler)
        self.command_bus.register(UpdateMetadataCommand, update_handler)

    def _register_query_handlers(self):
        """Register all query handlers."""
        from music_organizer.application.queries.catalog.recording_queries import (
            GetRecordingByIdHandler,
            GetRecordingsByArtistHandler,
            SearchRecordingsHandler
        )
        from music_organizer.application.queries.catalog.statistics_queries import (
            GetLibraryStatisticsHandler,
            GetGenreDistributionHandler
        )

        self.query_bus.register(GetRecordingByIdQuery, GetRecordingByIdHandler(self.recording_repo))
        self.query_bus.register(GetRecordingsByArtistQuery, GetRecordingsByArtistHandler(self.recording_repo))
        self.query_bus.register(SearchRecordingsQuery, SearchRecordingsHandler(self.recording_repo))
        self.query_bus.register(GetLibraryStatisticsQuery, GetLibraryStatisticsHandler(self.recording_repo, None))
        self.query_bus.register(GetGenreDistributionQuery, GetGenreDistributionHandler(self.recording_repo))

    def _register_event_handlers(self):
        """Register event handlers."""
        # Store all events
        self.event_bus.subscribe("*", EventStoreHandler(self.event_store))

        # Update read models
        self.event_bus.subscribe("RecordingAdded", self.read_model_projector)
        self.event_bus.subscribe("MetadataUpdated", self.read_model_projector)
        self.event_bus.subscribe("RecordingRemoved", self.read_model_projector)

    async def add_recording(self, file_path: Path, metadata: dict) -> str:
        """Add a new recording to the library."""
        command = AddRecordingCommand(
            file_path=file_path,
            metadata=metadata
        )

        result = await self.command_bus.dispatch(command)
        if result.success:
            print(f"✅ Added recording: {file_path}")
            return result.result_data["recording_id"]
        else:
            print(f"❌ Failed to add recording: {result.errors}")
            raise Exception(result.errors[0])

    async def update_metadata(self, recording_id: str, updates: dict) -> None:
        """Update metadata for a recording."""
        command = UpdateMetadataCommand(
            recording_id=recording_id,
            metadata_updates=updates
        )

        result = await self.command_bus.dispatch(command)
        if result.success:
            print(f"✅ Updated metadata for: {recording_id}")
        else:
            print(f"❌ Failed to update metadata: {result.errors}")

    async def get_recording(self, recording_id: str):
        """Get a recording by ID."""
        query = GetRecordingByIdQuery(recording_id=recording_id)
        result = await self.query_bus.dispatch(query)

        if result.success:
            cache_status = "📦 (from cache)" if result.from_cache else "🔄 (computed)"
            print(f"✅ Retrieved recording {cache_status}")
            return result.data
        else:
            print(f"❌ Failed to get recording: {result.errors}")
            return None

    async def search_recordings(self, search_term: str):
        """Search recordings by text."""
        query = SearchRecordingsQuery(
            search_term=search_term,
            search_fields=["title", "artist", "album"]
        )
        result = await self.query_bus.dispatch(query)

        if result.success:
            cache_status = "📦 (from cache)" if result.from_cache else "🔄 (computed)"
            print(f"✅ Found {result.data.total_count} recordings matching '{search_term}' {cache_status}")
            return result.data
        else:
            print(f"❌ Failed to search: {result.errors}")
            return None

    async def get_library_stats(self):
        """Get library statistics."""
        query = GetLibraryStatisticsQuery()
        result = await self.query_bus.dispatch(query)

        if result.success:
            cache_status = "📦 (from cache)" if result.from_cache else "🔄 (computed)"
            print(f"✅ Library statistics retrieved {cache_status}")
            return result.data
        else:
            print(f"❌ Failed to get stats: {result.errors}")
            return None

    async def get_events(self, event_type: str = None):
        """Get events from the event store."""
        if event_type:
            events = await self.event_store.get_events_by_type(event_type)
        else:
            events = await self.event_store.get_all_events(limit=10)
        return events


class EventStoreHandler:
    """Simple event handler that stores events in the event store."""

    def __init__(self, event_store: EventStore):
        self.event_store = event_store

    async def handle(self, event):
        await self.event_store.save_event(event)

    def can_handle(self, event_type: str) -> bool:
        return True  # Handle all events


async def main():
    """Demonstrate CQRS pattern with music library operations."""
    print("🎵 Music Organizer CQRS Example\n")

    # Initialize the CQRS-based library
    library = MusicLibraryCQRS()

    # Sample music data
    sample_recordings = [
        {
            "file_path": Path("/music/The Beatles/Abbey Road/01 Come Together.flac"),
            "metadata": {
                "title": "Come Together",
                "artists": ["The Beatles"],
                "album": "Abbey Road",
                "year": 1969,
                "genre": "Rock",
                "track_number": 1
            }
        },
        {
            "file_path": Path("/music/Pink Floyd/The Dark Side of the Moon/01 Speak to Me.flac"),
            "metadata": {
                "title": "Speak to Me",
                "artists": ["Pink Floyd"],
                "album": "The Dark Side of the Moon",
                "year": 1973,
                "genre": "Progressive Rock",
                "track_number": 1
            }
        },
        {
            "file_path": Path("/music/Led Zeppelin/Led Zeppelin IV/01 Black Dog.flac"),
            "metadata": {
                "title": "Black Dog",
                "artists": ["Led Zeppelin"],
                "album": "Led Zeppelin IV",
                "year": 1971,
                "genre": "Hard Rock",
                "track_number": 1
            }
        }
    ]

    # 1. COMMAND SIDE: Add recordings
    print("\n📝 COMMAND SIDE: Adding recordings...")
    recording_ids = []
    for recording_data in sample_recordings:
        recording_id = await library.add_recording(
            recording_data["file_path"],
            recording_data["metadata"]
        )
        recording_ids.append(recording_id)

    # 2. QUERY SIDE: Retrieve recordings
    print("\n🔍 QUERY SIDE: Retrieving recordings...")
    for i, recording_id in enumerate(recording_ids):
        recording = await library.get_recording(recording_id)
        if recording:
            print(f"   - {recording.metadata.title} by {recording.metadata.artists[0]}")

    # 3. Demonstrate caching
    print("\n📦 CACHING: Querying same recording again...")
    await library.get_recording(recording_ids[0])  # Should hit cache

    # 4. COMMAND SIDE: Update metadata
    print("\n✏️  COMMAND SIDE: Updating metadata...")
    await library.update_metadata(
        recording_ids[0],
        {"genre": "Classic Rock", "comment": "Added comment via CQRS"}
    )

    # 5. QUERY SIDE: Search
    print("\n🔎 QUERY SIDE: Searching recordings...")
    search_results = await library.search_recordings("Rock")
    if search_results:
        for recording in search_results.recordings[:3]:
            print(f"   - {recording['title']} (score: {recording.get('match_score', 0)})")

    # 6. QUERY SIDE: Statistics
    print("\n📊 QUERY SIDE: Library statistics...")
    stats = await library.get_library_stats()
    if stats:
        print(f"   Total recordings: {stats.total_recordings}")
        print(f"   Total artists: {stats.total_artists}")
        print(f"   Genre distribution: {stats.genre_distribution}")
        print(f"   Decade distribution: {stats.decade_distribution}")

    # 7. EVENT STORE: Review events
    print("\n📋 EVENT STORE: Recent events...")
    events = await library.get_events()
    for event in events[-3:]:
        print(f"   - {event.event_type} at {event.occurred_on}")

    # 8. Demonstrate query invalidation
    print("\n🗑️  CACHE INVALIDATION: Adding new recording invalidates stats cache...")
    await library.add_recording(
        Path("/music/Queen/A Night at the Opera/01 Bohemian Rhapsody.flac"),
        {
            "title": "Bohemian Rhapsody",
            "artists": ["Queen"],
            "album": "A Night at the Opera",
            "year": 1975,
            "genre": "Rock"
        }
    )

    # Stats should be recomputed
    stats = await library.get_library_stats()
    if stats:
        print(f"   Updated total recordings: {stats.total_recordings}")

    print("\n✨ CQRS demonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())