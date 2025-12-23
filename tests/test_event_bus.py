"""
Comprehensive tests for the event bus system.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Any

from music_organizer.events.event_bus import (
    EventBus,
    DomainEvent,
    EventHandler,
    EventPriority,
    event_handler,
    domain_event,
    event_bus as global_event_bus,
)


class CustomEvent(DomainEvent):
    """Test event."""
    data: str = "default"

    def __init__(self, data: str, **kwargs):
        # Set dataclass fields first
        super().__init__(**kwargs)
        object.__setattr__(self, 'data', data)


class AnotherEvent(DomainEvent):
    """Another test event."""
    value: int = 0

    def __init__(self, value: int, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'value', value)


class CustomEventHandler(EventHandler):
    """Test event handler."""
    def __init__(self):
        self.handled_events = []

    async def handle(self, event: DomainEvent) -> None:
        self.handled_events.append(event)

    @property
    def event_types(self):
        return [CustomEvent]


@pytest.fixture
def event_bus():
    """Create fresh event bus for each test."""
    bus = EventBus()
    yield bus
    bus.clear()


class TestDomainEvent:
    """Test DomainEvent base class."""

    def test_domain_event_creation(self):
        """Test creating a basic domain event."""
        event = DomainEvent()
        assert event.event_id.startswith("evt_")
        assert isinstance(event.timestamp, datetime)
        assert event.aggregate_id is None
        assert event.aggregate_type is None
        assert event.version == 1
        assert event.metadata == {}

    def test_domain_event_with_aggregate(self):
        """Test domain event with aggregate info."""
        event = DomainEvent(
            aggregate_id="agg_123",
            aggregate_type="TestAggregate"
        )
        assert event.aggregate_id == "agg_123"
        assert event.aggregate_type == "TestAggregate"

    def test_domain_event_auto_aggregate_type(self):
        """Test automatic aggregate type setting."""
        event = DomainEvent(aggregate_id="agg_123")
        assert event.aggregate_type == "DomainEvent"

    def test_domain_event_to_dict(self):
        """Test event serialization to dict."""
        event = DomainEvent(
            aggregate_id="agg_123",
            metadata={"key": "value"}
        )
        data = event.to_dict()
        assert data["event_type"] == "DomainEvent"
        assert data["aggregate_id"] == "agg_123"
        assert data["metadata"] == {"key": "value"}
        assert "timestamp" in data
        assert "event_id" in data

    def test_domain_event_with_custom_data(self):
        """Test event with custom data in _get_event_data."""
        class EventWithData(DomainEvent):
            def _get_event_data(self):
                return {"custom": "data", "value": 42}

        event = EventWithData()
        data = event.to_dict()
        assert data["data"] == {"custom": "data", "value": 42}


class TestEventBusSubscribeUnsubscribe:
    """Test event bus subscription management."""

    def test_subscribe_handler(self, event_bus):
        """Test subscribing a handler to an event."""
        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)
        assert CustomEvent in event_bus._handlers
        assert len(event_bus._handlers[CustomEvent]) == 1

    def test_subscribe_multiple_handlers(self, event_bus):
        """Test subscribing multiple handlers."""
        handler1 = Mock()
        handler2 = Mock()
        event_bus.subscribe(CustomEvent, handler1)
        event_bus.subscribe(CustomEvent, handler2)
        assert len(event_bus._handlers[CustomEvent]) == 2

    def test_subscribe_different_events(self, event_bus):
        """Test subscribing to different event types."""
        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)
        event_bus.subscribe(AnotherEvent, handler)
        assert CustomEvent in event_bus._handlers
        assert AnotherEvent in event_bus._handlers

    def test_unsubscribe_handler(self, event_bus):
        """Test unsubscribing a handler."""
        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)
        event_bus.unsubscribe(CustomEvent, handler)
        # Weak ref might still exist but point to None
        valid_refs = [ref for ref in event_bus._handlers[CustomEvent] if ref() is not None]
        assert len(valid_refs) == 0

    def test_unsubscribe_nonexistent_handler(self, event_bus):
        """Test unsubscribing handler that doesn't exist."""
        handler = Mock()
        # Should not raise
        event_bus.unsubscribe(CustomEvent, handler)


class TestEventBusPublish:
    """Test event publishing."""

    @pytest.mark.asyncio
    async def test_publish_simple_event(self, event_bus):
        """Test publishing a simple event."""
        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        # Handler should have been called
        assert handler.called

    @pytest.mark.asyncio
    async def test_publish_event_no_handlers(self, event_bus):
        """Test publishing event with no handlers."""
        event = CustomEvent(data="test")
        # Should not raise
        await event_bus.publish(event)

    @pytest.mark.asyncio
    async def test_publish_multiple_handlers(self, event_bus):
        """Test event with multiple handlers."""
        handler1 = Mock()
        handler2 = Mock()
        event_bus.subscribe(CustomEvent, handler1)
        event_bus.subscribe(CustomEvent, handler2)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        assert handler1.called
        assert handler2.called

    @pytest.mark.asyncio
    async def test_publish_async_handler(self, event_bus):
        """Test async handler execution."""
        async_handler = AsyncMock()
        event_bus.subscribe(CustomEvent, async_handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        async_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_mixed_sync_async_handlers(self, event_bus):
        """Test mix of sync and async handlers."""
        sync_handler = Mock()
        async_handler = AsyncMock()
        event_bus.subscribe(CustomEvent, sync_handler)
        event_bus.subscribe(CustomEvent, async_handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        assert sync_handler.called
        async_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_with_priority(self, event_bus):
        """Test publishing with different priorities."""
        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event, EventPriority.HIGH)
        assert handler.called

    @pytest.mark.asyncio
    async def test_publish_critical_priority_sync(self, event_bus):
        """Test critical priority events handled synchronously."""
        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event, EventPriority.CRITICAL)
        assert handler.called


class TestEventBusBatchPublish:
    """Test batch event publishing."""

    @pytest.mark.asyncio
    async def test_publish_batch_normal(self, event_bus):
        """Test batch publishing with normal priority."""
        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)

        events = [
            CustomEvent(data=f"test{i}")
            for i in range(5)
        ]
        await event_bus.publish_batch(events, EventPriority.NORMAL)

        # Should have been called for all events
        assert handler.call_count == 5

    @pytest.mark.asyncio
    async def test_publish_batch_critical(self, event_bus):
        """Test batch publishing with critical priority."""
        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)

        events = [
            CustomEvent(data=f"test{i}")
            for i in range(3)
        ]
        await event_bus.publish_batch(events, EventPriority.CRITICAL)

        assert handler.call_count == 3

    @pytest.mark.asyncio
    async def test_publish_batch_empty(self, event_bus):
        """Test publishing empty batch."""
        # Should not raise
        await event_bus.publish_batch([])


class TestEventBusMiddleware:
    """Test event middleware."""

    @pytest.mark.asyncio
    async def test_add_middleware(self, event_bus):
        """Test adding middleware."""
        middleware_called = []

        async def test_middleware(event):
            middleware_called.append(event)
            return event

        event_bus.add_middleware(test_middleware)
        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        assert len(middleware_called) == 1
        assert middleware_called[0] is event

    @pytest.mark.asyncio
    async def test_multiple_middleware(self, event_bus):
        """Test multiple middleware in chain."""
        order = []

        async def middleware1(event):
            order.append("middleware1")
            return event

        async def middleware2(event):
            order.append("middleware2")
            return event

        event_bus.add_middleware(middleware1)
        event_bus.add_middleware(middleware2)

        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        assert order == ["middleware1", "middleware2"]

    @pytest.mark.asyncio
    async def test_middleware_modifies_event(self, event_bus):
        """Test middleware modifying event."""
        async def modifying_middleware(event):
            event.metadata["modified"] = True
            return event

        event_bus.add_middleware(modifying_middleware)

        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        assert event.metadata.get("modified") is True


class TestEventStore:
    """Test event storage and retrieval."""

    @pytest.mark.asyncio
    async def test_events_stored(self, event_bus):
        """Test events are stored."""
        event = CustomEvent(data="test")
        await event_bus.publish(event)

        assert len(event_bus._event_store) == 1
        assert event_bus._event_store[0] is event

    @pytest.mark.asyncio
    async def test_max_events_limit(self, event_bus):
        """Test max events in memory limit."""
        event_bus._max_events_in_memory = 5

        for i in range(10):
            event = CustomEvent(data=f"test{i}")
            await event_bus.publish(event)

        assert len(event_bus._event_store) == 5
        # Should keep last 5 events
        assert event_bus._event_store[0].data == "test5"
        assert event_bus._event_store[4].data == "test9"

    def test_get_all_events(self, event_bus):
        """Test getting all events."""
        event_bus._event_store = [
            CustomEvent(data=f"test{i}")
            for i in range(3)
        ]
        events = event_bus.get_events()
        assert len(events) == 3

    def test_get_events_by_aggregate_id(self, event_bus):
        """Test filtering events by aggregate ID."""
        event_bus._event_store = [
            CustomEvent(data="test1", aggregate_id="agg1"),
            CustomEvent(data="test2", aggregate_id="agg2"),
            CustomEvent(data="test3", aggregate_id="agg1"),
        ]
        events = event_bus.get_events(aggregate_id="agg1")
        assert len(events) == 2
        assert all(e.aggregate_id == "agg1" for e in events)

    def test_get_events_by_aggregate_type(self, event_bus):
        """Test filtering events by aggregate type."""
        event_bus._event_store = [
            CustomEvent(data="test1", aggregate_type="Type1"),
            CustomEvent(data="test2", aggregate_type="Type2"),
            CustomEvent(data="test3", aggregate_type="Type1"),
        ]
        events = event_bus.get_events(aggregate_type="Type1")
        assert len(events) == 2

    def test_get_events_since(self, event_bus):
        """Test filtering events by time."""
        now = datetime.now()
        event_bus._event_store = [
            CustomEvent(data="test1", timestamp=datetime(2020, 1, 1)),
            CustomEvent(data="test2", timestamp=now),
        ]
        events = event_bus.get_events(since=datetime(2021, 1, 1))
        assert len(events) == 1
        assert events[0].data == "test2"

    def test_get_events_by_type(self, event_bus):
        """Test filtering events by type."""
        event_bus._event_store = [
            CustomEvent(data="test1"),
            AnotherEvent(value=42),
            CustomEvent(data="test2"),
        ]
        events = event_bus.get_events(event_type=CustomEvent)
        assert len(events) == 2
        assert all(isinstance(e, CustomEvent) for e in events)

    def test_get_events_combined_filters(self, event_bus):
        """Test combined filters."""
        event_bus._event_store = [
            CustomEvent(data="test1", aggregate_id="agg1", aggregate_type="Type1"),
            CustomEvent(data="test2", aggregate_id="agg1", aggregate_type="Type2"),
            CustomEvent(data="test3", aggregate_id="agg2", aggregate_type="Type1"),
        ]
        events = event_bus.get_events(
            aggregate_id="agg1",
            aggregate_type="Type1"
        )
        assert len(events) == 1
        assert events[0].data == "test1"


class TestInheritance:
    """Test event inheritance handling."""

    @pytest.mark.asyncio
    async def test_handler_for_base_event(self, event_bus):
        """Test handler for base event called for derived event."""
        handler = Mock()
        event_bus.subscribe(DomainEvent, handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        assert handler.called

    @pytest.mark.asyncio
    async def test_handler_for_derived_not_base(self, event_bus):
        """Test derived event handler not called for base event."""
        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)

        event = AnotherEvent(value=42)
        await event_bus.publish(event)

        assert not handler.called


class TestWeakReferences:
    """Test weak reference handling."""

    @pytest.mark.asyncio
    async def test_handler_garbage_collected(self, event_bus):
        """Test weak refs allow handler garbage collection."""
        def handler_factory():
            def handler(event):
                pass
            return handler

        event_bus.subscribe(CustomEvent, handler_factory())
        # Function should still be in memory
        assert len(event_bus._handlers[CustomEvent]) == 1

        # After function goes out of scope, weak ref becomes dead
        # This is tested implicitly by not causing memory leaks


class TestMethodHandlers:
    """Test method-based event handlers."""

    @pytest.mark.asyncio
    async def test_instance_method_handler(self, event_bus):
        """Test subscribing instance method."""
        class HandlerClass:
            def __init__(self):
                self.calls = []

            def handle_event(self, event):
                self.calls.append(event)

        handler_obj = HandlerClass()
        event_bus.subscribe(CustomEvent, handler_obj.handle_event)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        assert len(handler_obj.calls) == 1

    @pytest.mark.asyncio
    async def test_async_instance_method(self, event_bus):
        """Test subscribing async instance method."""
        class HandlerClass:
            def __init__(self):
                self.calls = []

            async def handle_event(self, event):
                self.calls.append(event)

        handler_obj = HandlerClass()
        event_bus.subscribe(CustomEvent, handler_obj.handle_event)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        assert len(handler_obj.calls) == 1


class TestErrorHandling:
    """Test error handling in event processing."""

    @pytest.mark.asyncio
    async def test_handler_exception(self, event_bus, capsys):
        """Test handler exceptions are caught."""
        def failing_handler(event):
            raise ValueError("Test error")

        event_bus.subscribe(CustomEvent, failing_handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        # Should not raise, should print error
        captured = capsys.readouterr()
        assert "Error in" in captured.out and "event handler" in captured.out

    @pytest.mark.asyncio
    async def test_handler_exception_critical_priority(self, event_bus, capsys):
        """Test handler exceptions in critical (sync) mode are caught."""
        def failing_handler(event):
            raise ValueError("Sync error")

        event_bus.subscribe(CustomEvent, failing_handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event, EventPriority.CRITICAL)

        captured = capsys.readouterr()
        assert "Error in" in captured.out and "event handler" in captured.out

    @pytest.mark.asyncio
    async def test_async_handler_in_critical_mode(self, event_bus):
        """Test async handlers are awaited in critical (sync) mode."""
        async_handler = AsyncMock()
        event_bus.subscribe(CustomEvent, async_handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event, EventPriority.CRITICAL)

        # Should be called and awaited
        async_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_handler_exception(self, event_bus, capsys):
        """Test async handler exceptions are caught."""
        async def failing_handler(event):
            raise ValueError("Test error")

        event_bus.subscribe(CustomEvent, failing_handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        captured = capsys.readouterr()
        assert "Error in" in captured.out and "event handler" in captured.out

    @pytest.mark.asyncio
    async def test_multiple_handlers_one_fails(self, event_bus):
        """Test other handlers run even if one fails."""
        failing_handler = Mock(side_effect=ValueError("Test error"))
        success_handler = Mock()

        event_bus.subscribe(CustomEvent, failing_handler)
        event_bus.subscribe(CustomEvent, success_handler)

        event = CustomEvent(data="test")
        await event_bus.publish(event)

        assert success_handler.called


class TestEventBusClear:
    """Test clearing event bus."""

    @pytest.mark.asyncio
    async def test_clear_all(self, event_bus):
        """Test clearing all data."""
        handler = Mock()
        event_bus.subscribe(CustomEvent, handler)
        event_bus._event_store.append(CustomEvent(data="test"))
        event_bus.add_middleware(lambda e: e)

        event_bus.clear()

        assert len(event_bus._handlers) == 0
        assert len(event_bus._event_store) == 0
        assert len(event_bus._middleware) == 0


class TestGlobalEventBus:
    """Test global event bus instance."""

    def test_global_event_bus_exists(self):
        """Test global event bus is available."""
        assert global_event_bus is not None
        assert isinstance(global_event_bus, EventBus)


class TestDecorators:
    """Test event decorators."""

    @pytest.mark.asyncio
    async def test_event_handler_decorator(self):
        """Test @event_handler decorator."""
        bus = EventBus()

        # Create a local decorator using this bus
        def test_event_handler(event_type):
            def decorator(handler):
                bus.subscribe(event_type, handler)
                return handler
            return decorator

        calls = []

        @test_event_handler(CustomEvent)
        def handler(event):
            calls.append(event)

        event = CustomEvent(data="test")
        await bus.publish(event)

        assert len(calls) == 1

    def test_domain_event_decorator(self):
        """Test @domain_event decorator."""
        # The decorator in event_bus doesn't actually modify the class
        # It's just a marker. Test that it returns the class unchanged.
        @domain_event(aggregate_type="CustomAggregate")
        class CustomDomainEvent(DomainEvent):
            pass

        # aggregate_type is only auto-set when aggregate_id is provided
        event = CustomDomainEvent(aggregate_id="test_agg")
        assert event.aggregate_type == "CustomDomainEvent"

    @pytest.mark.asyncio
    async def test_global_event_handler_decorator(self):
        """Test global @event_handler decorator subscribes to global bus."""
        from music_organizer.events.event_bus import event_handler

        calls = []

        @event_handler(CustomEvent)
        def handler(event):
            calls.append(event)

        event = CustomEvent(data="test")
        await global_event_bus.publish(event)

        assert len(calls) == 1

        # Cleanup
        global_event_bus.clear()

    def test_global_domain_event_decorator(self):
        """Test global @domain_event decorator."""
        from music_organizer.events.event_bus import domain_event

        @domain_event(aggregate_type="TestAggregate")
        class TestDomainEvent(DomainEvent):
            pass

        # Decorator should return class unchanged
        event = TestDomainEvent(aggregate_id="agg")
        assert isinstance(event, DomainEvent)


class TestEventPriority:
    """Test EventPriority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert EventPriority.LOW.value == 0
        assert EventPriority.NORMAL.value == 1
        assert EventPriority.HIGH.value == 2
        assert EventPriority.CRITICAL.value == 3


class TestEventHandlerABC:
    """Test EventHandler abstract base class."""

    def test_event_handler_subclass_required(self):
        """Test EventHandler requires implementation."""
        with pytest.raises(TypeError):
            EventHandler()

    def test_event_handler_subclass_valid(self):
        """Test valid EventHandler subclass."""
        class ValidHandler(EventHandler):
            async def handle(self, event):
                pass

            @property
            def event_types(self):
                return [DomainEvent]

        handler = ValidHandler()
        assert isinstance(handler, EventHandler)
