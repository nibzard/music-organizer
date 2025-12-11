"""Base classes for domain events in CQRS pattern."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar
from uuid import uuid4

from typing import Callable

E = TypeVar("E", bound="DomainEvent")


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """Base domain event class."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    aggregate_id: str = ""
    aggregate_type: str = ""
    event_type: str = ""
    occurred_on: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    event_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Set event_type based on class name if not provided
        if not self.event_type:
            object.__setattr__(self, 'event_type', self.__class__.__name__)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "event_type": self.event_type,
            "occurred_on": self.occurred_on.isoformat(),
            "version": self.version,
            "event_data": self.event_data,
            "metadata": self.metadata
        }


class EventHandler(ABC):
    """Abstract base class for event handlers."""

    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle the domain event."""
        pass

    @abstractmethod
    def can_handle(self, event_type: str) -> bool:
        """Check if this handler can handle the given event type."""
        pass


class EventBus:
    """Mediates domain events to registered handlers."""

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._middleware: List[Callable] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def register_middleware(self, middleware: Callable) -> None:
        """Register middleware for event processing pipeline."""
        self._middleware.append(middleware)

    async def publish(self, event: DomainEvent) -> None:
        """Publish an event to all registered handlers."""
        event_type = event.event_type
        handlers = self._handlers.get(event_type, [])

        # Apply middleware and execute handlers
        for handler in handlers:
            try:
                # Apply middleware pipeline
                current_handler = handler.handle
                for middleware in reversed(self._middleware):
                    current_handler = middleware(current_handler)

                await current_handler(event)
            except Exception as e:
                # Log error but continue with other handlers
                # In a real implementation, you'd have proper logging
                print(f"Error handling event {event_type}: {e}")

    async def publish_batch(self, events: List[DomainEvent]) -> None:
        """Publish multiple events in sequence."""
        for event in events:
            await self.publish(event)

    def get_subscribed_events(self) -> Dict[str, int]:
        """Get count of handlers for each event type."""
        return {event_type: len(handlers) for event_type, handlers in self._handlers.items()}


class EventStore:
    """Simple in-memory event store for domain events."""

    def __init__(self):
        self._events: List[DomainEvent] = []
        self._events_by_aggregate: Dict[str, List[DomainEvent]] = {}

    async def save_event(self, event: DomainEvent) -> None:
        """Save a domain event."""
        self._events.append(event)

        # Index by aggregate
        if event.aggregate_id:
            if event.aggregate_id not in self._events_by_aggregate:
                self._events_by_aggregate[event.aggregate_id] = []
            self._events_by_aggregate[event.aggregate_id].append(event)

    async def save_events(self, events: List[DomainEvent]) -> None:
        """Save multiple domain events."""
        for event in events:
            await self.save_event(event)

    async def get_events_for_aggregate(
        self,
        aggregate_id: str,
        from_version: Optional[int] = None
    ) -> List[DomainEvent]:
        """Get all events for an aggregate, optionally from a specific version."""
        events = self._events_by_aggregate.get(aggregate_id, [])

        if from_version is not None:
            events = [e for e in events if e.version >= from_version]

        return events

    async def get_events_by_type(
        self,
        event_type: str,
        from_timestamp: Optional[datetime] = None
    ) -> List[DomainEvent]:
        """Get all events of a specific type, optionally from a timestamp."""
        events = [e for e in self._events if e.event_type == event_type]

        if from_timestamp is not None:
            events = [e for e in events if e.occurred_on >= from_timestamp]

        return events

    async def get_all_events(self, limit: Optional[int] = None) -> List[DomainEvent]:
        """Get all events, optionally limited."""
        if limit:
            return self._events[-limit:]
        return self._events.copy()

    async def count_events(self, event_type: Optional[str] = None) -> int:
        """Count events, optionally filtered by type."""
        if event_type:
            return len([e for e in self._events if e.event_type == event_type])
        return len(self._events)