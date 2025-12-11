"""Base classes for CQRS command pattern."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import uuid4

from ...events import DomainEvent

# Type variables for generic command handling
C = TypeVar("C", bound="Command")
R = TypeVar("R", bound="CommandResult")


@dataclass(frozen=True, slots=True)
class Command:
    """Base command class with metadata."""

    command_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert command to dictionary for serialization."""
        return {
            "command_id": self.command_id,
            "command_type": self.__class__.__name__,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
            **{
                field.name: getattr(self, field.name)
                for field in self.__dataclass_fields__.values()
                if field.name not in {"command_id", "timestamp", "correlation_id", "metadata"}
            }
        }


class CommandHandler(ABC, Generic[C, R]):
    """Abstract base class for command handlers."""

    @abstractmethod
    async def handle(self, command: C) -> R:
        """Handle the command and return a result."""
        pass

    @abstractmethod
    def can_handle(self, command_type: type) -> bool:
        """Check if this handler can handle the given command type."""
        pass


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Base result class for command execution."""

    success: bool
    command_id: str
    message: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    events: List[DomainEvent] = field(default_factory=list)
    result_data: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None


class CommandBus:
    """Mediates commands to appropriate handlers."""

    def __init__(self):
        self._handlers: Dict[type, CommandHandler] = {}
        self._middleware: List[Callable] = []

    def register(self, command_type: type, handler: CommandHandler) -> None:
        """Register a handler for a command type."""
        self._handlers[command_type] = handler

    def register_middleware(self, middleware: Callable) -> None:
        """Register middleware for command processing pipeline."""
        self._middleware.append(middleware)

    async def dispatch(self, command: Command) -> CommandResult:
        """Dispatch a command to its registered handler."""
        command_type = type(command)

        if command_type not in self._handlers:
            return CommandResult(
                success=False,
                command_id=command.command_id,
                errors=[f"No handler registered for command type: {command_type.__name__}"]
            )

        handler = self._handlers[command_type]
        start_time = datetime.utcnow()

        try:
            # Apply middleware pipeline
            current_handler = handler.handle
            for middleware in reversed(self._middleware):
                current_handler = middleware(current_handler)

            # Execute command
            result = await current_handler(command)

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time

            return result

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return CommandResult(
                success=False,
                command_id=command.command_id,
                errors=[str(e)],
                execution_time_ms=execution_time
            )

    def get_registered_commands(self) -> List[type]:
        """Get list of registered command types."""
        return list(self._handlers.keys())