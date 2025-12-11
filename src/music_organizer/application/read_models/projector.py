"""Read model projector for updating denormalized views."""

from typing import Dict, Any, List
from ..events.base import DomainEvent, EventHandler


class ReadModelProjector(EventHandler):
    """Projects domain events to update read models."""

    def __init__(self):
        self.library_stats = {
            "total_recordings": 0,
            "total_artists": set(),
            "genre_distribution": {},
            "recent_recordings": []
        }

    async def handle(self, event: DomainEvent) -> None:
        """Handle domain events to update read models."""
        if event.event_type == "RecordingAdded":
            await self._handle_recording_added(event)
        elif event.event_type == "MetadataUpdated":
            await self._handle_metadata_updated(event)
        elif event.event_type == "RecordingRemoved":
            await self._handle_recording_removed(event)

    async def _handle_recording_added(self, event: DomainEvent) -> None:
        """Update read models when a recording is added."""
        metadata = event.event_data["metadata"]

        # Update counts
        self.library_stats["total_recordings"] += 1

        # Track artists
        for artist in metadata.get("artists", []):
            self.library_stats["total_artists"].add(artist)

        # Update genre distribution
        genre = metadata.get("genre")
        if genre:
            self.library_stats["genre_distribution"][genre] = \
                self.library_stats["genre_distribution"].get(genre, 0) + 1

    async def _handle_metadata_updated(self, event: DomainEvent) -> None:
        """Update read models when metadata is updated."""
        # Invalidate relevant caches
        pass

    async def _handle_recording_removed(self, event: DomainEvent) -> None:
        """Update read models when a recording is removed."""
        self.library_stats["total_recordings"] -= 1

    def can_handle(self, event_type: str) -> bool:
        """Check if this projector can handle the given event type."""
        return event_type in ["RecordingAdded", "MetadataUpdated", "RecordingRemoved"]