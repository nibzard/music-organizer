"""Hook system for plugin integration."""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..models.audio_file import AudioFile


class HookEvent(Enum):
    """Events that plugins can hook into."""
    PRE_SCAN = "pre_scan"
    POST_SCAN = "post_scan"
    PRE_METADATA_EXTRACT = "pre_metadata_extract"
    POST_METADATA_EXTRACT = "post_metadata_extract"
    PRE_CLASSIFY = "pre_classify"
    POST_CLASSIFY = "post_classify"
    PRE_MOVE = "pre_move"
    POST_MOVE = "post_move"
    ERROR_OCCURRED = "error_occurred"
    PROGRESS_UPDATE = "progress_update"


@dataclass
class HookContext:
    """Context passed to hook handlers."""
    event: HookEvent
    data: Dict[str, Any]
    error: Optional[Exception] = None


class PluginHooks:
    """Manages plugin hooks and event handling."""

    def __init__(self) -> None:
        """Initialize hook system."""
        self._hooks: Dict[HookEvent, List[Callable]] = {
            event: [] for event in HookEvent
        }

    def register(self, event: HookEvent, handler: Callable) -> None:
        """Register a handler for a specific event.

        Args:
            event: The event to handle
            handler: Function to call when event occurs
        """
        self._hooks[event].append(handler)

    def unregister(self, event: HookEvent, handler: Callable) -> None:
        """Unregister a handler for a specific event.

        Args:
            event: The event to stop handling
            handler: Function to remove
        """
        if handler in self._hooks[event]:
            self._hooks[event].remove(handler)

    async def emit(self, event: HookEvent, **data) -> HookContext:
        """Emit an event to all registered handlers.

        Args:
            event: The event to emit
            **data: Additional data to pass to handlers

        Returns:
            HookContext with event information
        """
        context = HookContext(event=event, data=data)

        for handler in self._hooks[event]:
            try:
                if callable(handler):
                    # Support both sync and async handlers
                    import inspect
                    if inspect.iscoroutinefunction(handler):
                        await handler(context)
                    else:
                        handler(context)
            except Exception as e:
                # Emit error event but don't stop processing
                error_context = HookContext(
                    event=HookEvent.ERROR_OCCURRED,
                    data={"original_event": event, "handler": handler.__name__},
                    error=e
                )
                await self.emit(HookEvent.ERROR_OCCURRED, **error_context.data)

        return context

    def clear(self, event: Optional[HookEvent] = None) -> None:
        """Clear all handlers or handlers for a specific event.

        Args:
            event: Specific event to clear, or None for all events
        """
        if event:
            self._hooks[event].clear()
        else:
            for handlers in self._hooks.values():
                handlers.clear()