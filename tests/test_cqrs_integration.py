"""Integration tests for CQRS implementation."""

import pytest
from pathlib import Path
from datetime import datetime

from music_organizer.application.commands import CommandBus, CommandResult
from music_organizer.application.queries import QueryBus, QueryResult, QueryCache
from music_organizer.application.events import EventBus
from music_organizer.application.commands.catalog import (
    AddRecordingCommand,
    AddRecordingCommandHandler,
    UpdateMetadataCommand,
    UpdateMetadataCommandHandler
)
from music_organizer.application.queries.catalog import (
    GetRecordingByIdQuery,
    GetRecordingsByArtistQuery,
    GetLibraryStatisticsQuery
)
from music_organizer.domain.catalog.repositories import InMemoryRecordingRepository
from music_organizer.domain.value_objects import AudioPath, Metadata


@pytest.fixture
def cqrs_components():
    """Set up CQRS components for testing."""
    # Create repositories
    recording_repo = InMemoryRecordingRepository()

    # Create event bus
    event_bus = EventBus()

    # Create caches
    query_cache = QueryCache()

    # Create buses
    command_bus = CommandBus()
    query_bus = QueryBus()
    query_bus.set_cache(query_cache)

    # Register command handlers
    add_handler = AddRecordingCommandHandler(recording_repo, event_bus)
    update_handler = UpdateMetadataCommandHandler(recording_repo, event_bus)

    command_bus.register(AddRecordingCommand, add_handler)
    command_bus.register(UpdateMetadataCommand, update_handler)

    # Register query handlers
    from music_organizer.application.queries.catalog.recording_queries import (
        GetRecordingByIdHandler,
        GetRecordingsByArtistHandler
    )
    from music_organizer.application.queries.catalog.statistics_queries import (
        GetLibraryStatisticsHandler
    )

    query_bus.register(GetRecordingByIdQuery, GetRecordingByIdHandler(recording_repo))
    query_bus.register(GetRecordingsByArtistQuery, GetRecordingsByArtistHandler(recording_repo))
    query_bus.register(GetLibraryStatisticsQuery, GetLibraryStatisticsHandler(recording_repo, None))

    return {
        "command_bus": command_bus,
        "query_bus": query_bus,
        "event_bus": event_bus,
        "recording_repo": recording_repo,
        "query_cache": query_cache
    }


@pytest.mark.asyncio
async def test_cqrs_add_and_query_recording(cqrs_components):
    """Test adding a recording via command and querying it back."""
    command_bus = cqrs_components["command_bus"]
    query_bus = cqrs_components["query_bus"]

    # Create command to add a recording
    test_file = Path("/test/artist/album/track.flac")
    metadata = {
        "title": "Test Song",
        "artists": ["Test Artist"],
        "album": "Test Album",
        "year": 2023,
        "genre": "Rock",
        "track_number": 1
    }

    add_command = AddRecordingCommand(
        file_path=test_file,
        metadata=metadata
    )

    # Execute command
    command_result = await command_bus.dispatch(add_command)

    # Verify command succeeded
    assert command_result.success is True
    assert "Successfully added recording" in command_result.message
    assert "recording_id" in command_result.result_data

    recording_id = command_result.result_data["recording_id"]

    # Query the recording back
    get_query = GetRecordingByIdQuery(recording_id=recording_id)
    query_result = await query_bus.dispatch(get_query)

    # Verify query succeeded
    assert query_result.success is True
    assert query_result.data is not None
    assert query_result.data.metadata.title == "Test Song"
    assert str(query_result.data.metadata.artists[0]) == "Test Artist"


@pytest.mark.asyncio
async def test_cqrs_update_metadata(cqrs_components):
    """Test updating recording metadata."""
    command_bus = cqrs_components["command_bus"]
    query_bus = cqrs_components["query_bus"]

    # First add a recording
    add_command = AddRecordingCommand(
        file_path=Path("/test/artist/album/track.flac"),
        metadata={
            "title": "Original Title",
            "artists": ["Original Artist"],
            "year": 2020
        }
    )

    add_result = await command_bus.dispatch(add_command)
    recording_id = add_result.result_data["recording_id"]

    # Update metadata
    update_command = UpdateMetadataCommand(
        recording_id=recording_id,
        metadata_updates={
            "title": "Updated Title",
            "year": 2023,
            "genre": "Pop"
        }
    )

    update_result = await command_bus.dispatch(update_command)

    # Verify update succeeded
    assert update_result.success is True
    assert "title" in update_result.result_data["updated_fields"]
    assert "year" in update_result.result_data["updated_fields"]
    assert "genre" in update_result.result_data["updated_fields"]

    # Query to verify changes
    get_query = GetRecordingByIdQuery(recording_id=recording_id)
    query_result = await query_bus.dispatch(get_query)

    assert query_result.success is True
    assert query_result.data.metadata.title == "Updated Title"
    assert query_result.data.metadata.year == 2023
    assert query_result.data.metadata.genre == "Pop"
    # Artist should remain unchanged due to merge strategy
    assert str(query_result.data.metadata.artists[0]) == "Original Artist"


@pytest.mark.asyncio
async def test_cqrs_query_caching(cqrs_components):
    """Test that query results are cached properly."""
    command_bus = cqrs_components["command_bus"]
    query_bus = cqrs_components["query_bus"]
    query_cache = cqrs_components["query_cache"]

    # Add a recording
    add_command = AddRecordingCommand(
        file_path=Path("/test/artist/album/track.flac"),
        metadata={
            "title": "Cache Test Song",
            "artists": ["Cache Test Artist"],
            "album": "Cache Test Album"
        }
    )

    await command_bus.dispatch(add_command)

    # Query by artist - first time (should miss cache)
    query1 = GetRecordingsByArtistQuery(
        artist_name="Cache Test Artist",
        cache_key="test_artist_query"
    )
    result1 = await query_bus.dispatch(query1)

    assert result1.success is True
    assert result1.from_cache is False
    assert len(result1.data) == 1

    # Query by artist - second time (should hit cache)
    query2 = GetRecordingsByArtistQuery(
        artist_name="Cache Test Artist",
        cache_key="test_artist_query"
    )
    result2 = await query_bus.dispatch(query2)

    assert result2.success is True
    assert result2.from_cache is True
    assert result2.cached_at is not None
    assert len(result2.data) == 1


@pytest.mark.asyncio
async def test_cqrs_library_statistics(cqrs_components):
    """Test library statistics query."""
    command_bus = cqrs_components["command_bus"]
    query_bus = cqrs_components["query_bus"]

    # Add multiple recordings
    recordings_data = [
        {
            "file_path": Path("/test/rock/album1/track1.flac"),
            "metadata": {"title": "Rock Song 1", "artists": ["Rock Artist"], "genre": "Rock", "year": 2020}
        },
        {
            "file_path": Path("/test/pop/album1/track1.mp3"),
            "metadata": {"title": "Pop Song 1", "artists": ["Pop Artist"], "genre": "Pop", "year": 2023}
        },
        {
            "file_path": Path("/test/rock/album2/track2.flac"),
            "metadata": {"title": "Rock Song 2", "artists": ["Rock Artist"], "genre": "Rock", "year": 2021}
        }
    ]

    for data in recordings_data:
        command = AddRecordingCommand(
            file_path=data["file_path"],
            metadata=data["metadata"]
        )
        await command_bus.dispatch(command)

    # Query library statistics
    stats_query = GetLibraryStatisticsQuery()
    stats_result = await query_bus.dispatch(stats_query)

    assert stats_result.success is True
    stats = stats_result.data

    assert stats.total_recordings == 3
    assert stats.total_artists == 2
    assert stats.genre_distribution["Rock"] == 2
    assert stats.genre_distribution["Pop"] == 1
    assert "2020s" in stats.decade_distribution


@pytest.mark.asyncio
async def test_cqrs_command_error_handling(cqrs_components):
    """Test that command errors are properly handled."""
    command_bus = cqrs_components["command_bus"]

    # Try to add a recording with invalid metadata
    add_command = AddRecordingCommand(
        file_path=Path("/test/artist/album/track.flac"),
        metadata={
            "title": "Test Song",
            "year": -1  # Invalid year
        }
    )

    result = await command_bus.dispatch(add_command)

    # Should fail due to validation error
    assert result.success is False
    assert len(result.errors) > 0
    assert result.execution_time_ms is not None


@pytest.mark.asyncio
async def test_cqrs_event_publishing(cqrs_components):
    """Test that events are published when commands are executed."""
    command_bus = cqrs_components["command_bus"]
    event_bus = cqrs_components["event_bus"]

    # Track published events
    published_events = []

    async def event_handler(event):
        published_events.append(event)

    # Subscribe to recording events
    event_bus.subscribe("RecordingAdded", type("TestHandler", (), {"handle": event_handler})())

    # Add a recording
    add_command = AddRecordingCommand(
        file_path=Path("/test/artist/album/track.flac"),
        metadata={
            "title": "Event Test Song",
            "artists": ["Event Test Artist"]
        }
    )

    await command_bus.dispatch(add_command)

    # Verify event was published
    assert len(published_events) == 1
    assert published_events[0].event_type == "RecordingAdded"
    assert "Event Test Song" in published_events[0].event_data["metadata"]["title"]