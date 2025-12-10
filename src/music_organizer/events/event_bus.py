"""
Event Bus - Event-driven communication system.

This module provides a lightweight event bus for domain events,
enabling loose coupling between components.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from enum import Enum
import weakref


class EventPriority(Enum):
    """Priority levels for events."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


T = TypeVar('T', bound='DomainEvent')


@dataclass
class DomainEvent:
    """Base class for all domain events."""
    event_id: str = field(default_factory=lambda: f"evt_{datetime.now().timestamp()}")
    timestamp: datetime = field(default_factory=datetime.now)
    aggregate_id: Optional[str] = None
    aggregate_type: Optional[str] = None
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing."""
        # Set aggregate_type from class name if not provided
        if not self.aggregate_type and self.aggregate_id:
            self.aggregate_type = self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.__class__.__name__,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "version": self.version,
            "metadata": self.metadata,
            "data": self._get_event_data(),
        }

    def _get_event_data(self) -> Dict[str, Any]:
        """Get event-specific data for serialization."""
        return {}


class EventHandler(ABC):
    """Base class for event handlers."""

    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle the event."""
        pass

    @property
    @abstractmethod
    def event_types(self) -> List[Type[DomainEvent]]:
        """Return the list of event types this handler can handle."""
        pass


class EventBus:
    """
    Central event bus for publishing and subscribing to domain events.

    Provides both synchronous and asynchronous event handling.
    """

    def __init__(self):
        self._handlers: Dict[Type[DomainEvent], List[weakref.ref]] = {}
        self._middleware: List[Callable] = []
        self._event_store: List[DomainEvent] = []
        self._max_events_in_memory = 1000

    def subscribe(
        self,
        event_type: Type[T],
        handler: Callable[[T], Any]
    ) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: The event class to subscribe to
            handler: The handler function/method
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        # Use weak reference to avoid memory leaks
        if hasattr(handler, '__self__'):
            # Method handler - use weak reference to instance
            ref = weakref.WeakMethod(handler)
        else:
            # Function handler - use weak reference to function
            ref = weakref.ref(handler)

        self._handlers[event_type].append(ref)

    def unsubscribe(
        self,
        event_type: Type[DomainEvent],
        handler: Callable
    ) -> None:
        """
        Unsubscribe from events.
        """
        if event_type in self._handlers:
            # Find and remove the handler
            self._handlers[event_type] = [
                ref for ref in self._handlers[event_type]
                if ref() is not None and ref() != handler
            ]

    async def publish(
        self,
        event: DomainEvent,
        priority: EventPriority = EventPriority.NORMAL
    ) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: The event to publish
            priority: Event priority for processing order
        """
        # Store event
        self._event_store.append(event)
        if len(self._event_store) > self._max_events_in_memory:
            self._event_store.pop(0)

        # Apply middleware
        for middleware in self._middleware:
            event = await middleware(event)

        # Get handlers for this event type and its parent classes
        handlers = []
        event_type = type(event)

        # Find handlers for this exact type
        if event_type in self._handlers:
            handlers.extend(ref() for ref in self._handlers[event_type] if ref() is not None)

        # Find handlers for parent types
        for parent_type in event_type.__mro__[1:]:
            if issubclass(parent_type, DomainEvent) and parent_type in self._handlers:
                handlers.extend(ref() for ref in self._handlers[parent_type] if ref() is not None)

        # Handle the event
        if handlers:
            if priority == EventPriority.CRITICAL:
                # Critical events are handled synchronously
                await self._handle_sync(event, handlers)
            else:
                # Other events are handled asynchronously
                await self._handle_async(event, handlers)

    async def publish_batch(
        self,
        events: List[DomainEvent],
        priority: EventPriority = EventPriority.NORMAL
    ) -> None:
        """
        Publish multiple events.

        Args:
            events: List of events to publish
            priority: Event priority
        """
        if priority == EventPriority.CRITICAL:
            # Process critical events in order
            for event in events:
                await self.publish(event, priority)
        else:
            # Process other events concurrently
            tasks = [self.publish(event, priority) for event in events]
            await asyncio.gather(*tasks, return_exceptions=True)

    def add_middleware(self, middleware: Callable[[DomainEvent], DomainEvent]) -> None:
        """
        Add middleware to process events before handling.

        Middleware can modify, filter, or log events.
        """
        self._middleware.append(middleware)

    def get_events(
        self,
        aggregate_id: Optional[str] = None,
        aggregate_type: Optional[str] = None,
        since: Optional[datetime] = None,
        event_type: Optional[Type[DomainEvent]] = None
    ) -> List[DomainEvent]:
        """
        Get events from the store with optional filtering.
        """
        filtered_events = self._event_store

        if aggregate_id:
            filtered_events = [e for e in filtered_events if e.aggregate_id == aggregate_id]

        if aggregate_type:
            filtered_events = [e for e in filtered_events if e.aggregate_type == aggregate_type]

        if since:
            filtered_events = [e for e in filtered_events if e.timestamp >= since]

        if event_type:
            filtered_events = [e for e in filtered_events if isinstance(e, event_type)]

        return filtered_events

    async def _handle_sync(self, event: DomainEvent, handlers: List[Callable]) -> None:
        """Handle event synchronously (blocking)."""
        for handler in handlers:
            try:
                result = handler(event)
                # If handler is async, await it
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"Error in event handler {handler}: {e}")

    async def _handle_async(self, event: DomainEvent, handlers: List[Callable]) -> None:
        """Handle event asynchronously (non-blocking)."""
        tasks = []
        for handler in handlers:
            # Create task for each handler
            task = asyncio.create_task(self._safe_handle(handler, event))
            tasks.append(task)

        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_handle(self, handler: Callable, event: DomainEvent) -> None:
        """Safely handle an event, catching exceptions."""
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            print(f"Error in async event handler {handler}: {e}")

    def clear(self) -> None:
        """Clear all handlers and events."""
        self._handlers.clear()
        self._middleware.clear()
        self._event_store.clear()


# Global event bus instance
event_bus = EventBus()


# Decorators for easy subscription
def event_handler(event_type: Type[T]):
    """Decorator for marking methods as event handlers."""
    def decorator(handler):
        # Auto-subscribe the handler
        event_bus.subscribe(event_type, handler)
        return handler
    return decorator


def domain_event(aggregate_type: Optional[str] = None):
    """Decorator for marking domain events."""
    def decorator(cls):
        # Add class to registry if needed
        return cls
    return decorator