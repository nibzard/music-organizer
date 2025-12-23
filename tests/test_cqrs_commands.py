"""Comprehensive tests for CQRS command modules."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from music_organizer.application.commands import (
    Command,
    CommandHandler,
    CommandBus,
    CommandResult,
)
from music_organizer.events import DomainEvent, RecordingAdded


def async_test(func):
    """Decorator to run async tests."""
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


class TestCommand:
    """Test Command base class."""

    def test_command_creation_with_defaults(self):
        """Test command creation with default values."""
        cmd = Command()

        assert cmd.command_id is not None
        assert isinstance(cmd.command_id, str)
        assert cmd.timestamp is not None
        assert isinstance(cmd.timestamp, datetime)
        assert cmd.correlation_id is None
        assert cmd.metadata == {}

    def test_command_creation_with_values(self):
        """Test command creation with provided values."""
        correlation_id = "test-correlation-123"
        metadata = {"user": "test", "source": "api"}

        cmd = Command(
            correlation_id=correlation_id,
            metadata=metadata
        )

        assert cmd.correlation_id == correlation_id
        assert cmd.metadata == metadata

    def test_command_to_dict(self):
        """Test command serialization to dictionary."""
        correlation_id = "test-correlation-123"
        metadata = {"key": "value"}

        cmd = Command(
            correlation_id=correlation_id,
            metadata=metadata
        )

        result = cmd.to_dict()

        assert result["command_id"] == cmd.command_id
        assert result["command_type"] == "Command"
        assert "timestamp" in result
        assert result["correlation_id"] == correlation_id
        assert result["metadata"] == metadata

    def test_command_is_frozen(self):
        """Test that command instances are immutable."""
        from dataclasses import FrozenInstanceError
        cmd = Command()

        with pytest.raises(FrozenInstanceError):
            cmd.command_id = "new-id"

    def test_command_with_custom_fields(self):
        """Test command subclass with custom fields."""
        @dataclass(frozen=True, slots=True, kw_only=True)
        class CustomCommand(Command):
            name: str
            value: int

        cmd = CustomCommand(name="test", value=42)

        result = cmd.to_dict()
        assert result["name"] == "test"
        assert result["value"] == 42


class TestCommandResult:
    """Test CommandResult class."""

    def test_result_creation_success(self):
        """Test successful result creation."""
        result = CommandResult(
            success=True,
            command_id="cmd-123",
            message="Operation completed"
        )

        assert result.success is True
        assert result.command_id == "cmd-123"
        assert result.message == "Operation completed"
        assert result.errors == []
        assert result.events == []
        assert result.result_data == {}
        assert result.execution_time_ms is None

    def test_result_creation_failure(self):
        """Test failed result creation."""
        result = CommandResult(
            success=False,
            command_id="cmd-456",
            errors=["Error 1", "Error 2"],
            execution_time_ms=150.5
        )

        assert result.success is False
        assert result.command_id == "cmd-456"
        assert result.errors == ["Error 1", "Error 2"]
        assert result.execution_time_ms == 150.5

    def test_result_with_events(self):
        """Test result with domain events."""
        event = RecordingAdded(
            recording_id="rec-1",
            file_path="/path/to/file.mp3",
            title="Test Song",
            artist="Test Artist"
        )

        result = CommandResult(
            success=True,
            command_id="cmd-789",
            events=[event]
        )

        assert len(result.events) == 1
        assert result.events[0] == event

    def test_result_with_data(self):
        """Test result with additional data."""
        result = CommandResult(
            success=True,
            command_id="cmd-101",
            result_data={"created_id": "123", "count": 5}
        )

        assert result.result_data == {"created_id": "123", "count": 5}


class TestCommandHandler:
    """Test CommandHandler abstract base class."""

    def test_handler_requires_handle_implementation(self):
        """Test that handle method must be implemented."""
        with pytest.raises(TypeError):
            class IncompleteHandler(CommandHandler):
                pass

            IncompleteHandler()

    def test_handler_requires_can_handle_implementation(self):
        """Test that can_handle method must be implemented."""
        with pytest.raises(TypeError):
            class IncompleteHandler(CommandHandler):
                async def handle(self, command):
                    return CommandResult(success=True, command_id="123")

            IncompleteHandler()

    def test_concrete_handler_implementation(self):
        """Test concrete handler implementation."""
        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestCommand(Command):
            test_field: str

        class TestHandler(CommandHandler):
            async def handle(self, command: TestCommand) -> CommandResult:
                return CommandResult(
                    success=True,
                    command_id=command.command_id,
                    message=f"Handled: {command.test_field}"
                )

            def can_handle(self, command_type: type) -> bool:
                return command_type == TestCommand

        handler = TestHandler()
        cmd = TestCommand(test_field="test-value")

        assert handler.can_handle(TestCommand) is True
        assert handler.can_handle(Command) is False


class TestCommandBus:
    """Test CommandBus class."""

    def test_bus_initialization(self):
        """Test bus initialization."""
        bus = CommandBus()

        assert bus.get_registered_commands() == []
        assert bus._handlers == {}
        assert bus._middleware == []

    @async_test
    async def test_register_handler(self):
        """Test registering a command handler."""
        bus = CommandBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestCommand(Command):
            value: str

        class TestHandler(CommandHandler):
            async def handle(self, command: TestCommand) -> CommandResult:
                return CommandResult(success=True, command_id=command.command_id)

            def can_handle(self, command_type: type) -> bool:
                return command_type == TestCommand

        handler = TestHandler()
        bus.register(TestCommand, handler)

        assert TestCommand in bus.get_registered_commands()
        assert bus._handlers[TestCommand] == handler

    @async_test
    async def test_dispatch_success(self):
        """Test successful command dispatch."""
        bus = CommandBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestCommand(Command):
            value: str

        class TestHandler(CommandHandler):
            async def handle(self, command: TestCommand) -> CommandResult:
                return CommandResult(
                    success=True,
                    command_id=command.command_id,
                    message=f"Processed: {command.value}"
                )

            def can_handle(self, command_type: type) -> bool:
                return command_type == TestCommand

        handler = TestHandler()
        bus.register(TestCommand, handler)

        cmd = TestCommand(value="test-data")
        result = await bus.dispatch(cmd)

        assert result.success is True
        assert result.command_id == cmd.command_id
        assert result.message == "Processed: test-data"
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0

    @async_test
    async def test_dispatch_no_handler(self):
        """Test dispatch when no handler is registered."""
        bus = CommandBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class UnhandledCommand(Command):
            value: str

        cmd = UnhandledCommand(value="test")
        result = await bus.dispatch(cmd)

        assert result.success is False
        assert result.command_id == cmd.command_id
        assert len(result.errors) == 1
        assert "No handler registered" in result.errors[0]

    @async_test
    async def test_dispatch_handler_exception(self):
        """Test dispatch when handler raises exception."""
        bus = CommandBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class FailingCommand(Command):
            value: str

        class FailingHandler(CommandHandler):
            async def handle(self, command: FailingCommand) -> CommandResult:
                raise ValueError("Handler error")

            def can_handle(self, command_type: type) -> bool:
                return command_type == FailingCommand

        handler = FailingHandler()
        bus.register(FailingCommand, handler)

        cmd = FailingCommand(value="test")
        result = await bus.dispatch(cmd)

        assert result.success is False
        assert result.command_id == cmd.command_id
        assert len(result.errors) == 1
        assert "Handler error" in result.errors[0]
        assert result.execution_time_ms is not None

    @async_test
    async def test_middleware_execution(self):
        """Test middleware pipeline execution."""
        bus = CommandBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestCommand(Command):
            value: str

        class TestHandler(CommandHandler):
            async def handle(self, command: TestCommand) -> CommandResult:
                return CommandResult(
                    success=True,
                    command_id=command.command_id,
                    result_data={"value": command.value}
                )

            def can_handle(self, command_type: type) -> bool:
                return command_type == TestCommand

        middleware_calls = []

        def logging_middleware(next_handler):
            async def wrapper(command):
                middleware_calls.append("logging")
                return await next_handler(command)
            return wrapper

        def validation_middleware(next_handler):
            async def wrapper(command):
                middleware_calls.append("validation")
                return await next_handler(command)
            return wrapper

        handler = TestHandler()
        bus.register(TestCommand, handler)
        bus.register_middleware(logging_middleware)
        bus.register_middleware(validation_middleware)

        cmd = TestCommand(value="test-data")
        result = await bus.dispatch(cmd)

        assert result.success is True
        assert len(middleware_calls) == 2
        assert middleware_calls == ["logging", "validation"]

    @async_test
    async def test_middleware_exception_handling(self):
        """Test exception handling in middleware."""
        bus = CommandBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestCommand(Command):
            value: str

        class TestHandler(CommandHandler):
            async def handle(self, command: TestCommand) -> CommandResult:
                return CommandResult(success=True, command_id=command.command_id)

            def can_handle(self, command_type: type) -> bool:
                return command_type == TestCommand

        def failing_middleware(next_handler):
            async def wrapper(command):
                raise RuntimeError("Middleware error")
            return wrapper

        handler = TestHandler()
        bus.register(TestCommand, handler)
        bus.register_middleware(failing_middleware)

        cmd = TestCommand(value="test")
        result = await bus.dispatch(cmd)

        assert result.success is False
        assert "Middleware error" in result.errors[0]

    @async_test
    async def test_execution_time_measurement(self):
        """Test execution time is measured accurately."""
        bus = CommandBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestCommand(Command):
            delay_ms: int

        class SlowHandler(CommandHandler):
            async def handle(self, command: TestCommand) -> CommandResult:
                import asyncio
                await asyncio.sleep(command.delay_ms / 1000)
                return CommandResult(success=True, command_id=command.command_id)

            def can_handle(self, command_type: type) -> bool:
                return command_type == TestCommand

        handler = SlowHandler()
        bus.register(TestCommand, handler)

        cmd = TestCommand(delay_ms=100)
        result = await bus.dispatch(cmd)

        assert result.success is True
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 90  # Allow some tolerance

    @async_test
    async def test_get_registered_commands(self):
        """Test getting list of registered commands."""
        bus = CommandBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class Command1(Command):
            pass

        @dataclass(frozen=True, slots=True, kw_only=True)
        class Command2(Command):
            pass

        class Handler1(CommandHandler):
            async def handle(self, command):
                return CommandResult(success=True, command_id=command.command_id)
            def can_handle(self, command_type):
                return command_type == Command1

        class Handler2(CommandHandler):
            async def handle(self, command):
                return CommandResult(success=True, command_id=command.command_id)
            def can_handle(self, command_type):
                return command_type == Command2

        bus.register(Command1, Handler1())
        bus.register(Command2, Handler2())

        registered = bus.get_registered_commands()
        assert Command1 in registered
        assert Command2 in registered
        assert len(registered) == 2


class TestCommandIntegration:
    """Integration tests for command pattern."""

    @async_test
    async def test_full_command_lifecycle(self):
        """Test complete command lifecycle with events."""
        bus = CommandBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class CreateRecordingCommand(Command):
            file_path: str
            title: str
            artist: str

        class CreateRecordingHandler(CommandHandler):
            async def handle(self, command: CreateRecordingCommand) -> CommandResult:
                event = RecordingAdded(
                    recording_id="rec-" + command.command_id[:8],
                    file_path=command.file_path,
                    title=command.title,
                    artist=command.artist
                )

                return CommandResult(
                    success=True,
                    command_id=command.command_id,
                    message="Recording created",
                    events=[event],
                    result_data={"recording_id": event.recording_id}
                )

            def can_handle(self, command_type: type) -> bool:
                return command_type == CreateRecordingCommand

        handler = CreateRecordingHandler()
        bus.register(CreateRecordingCommand, handler)

        cmd = CreateRecordingCommand(
            file_path="/music/song.mp3",
            title="Test Song",
            artist="Test Artist"
        )

        result = await bus.dispatch(cmd)

        assert result.success is True
        assert "Recording created" in result.message
        assert len(result.events) == 1
        assert isinstance(result.events[0], RecordingAdded)
        assert result.events[0].recording_id in result.result_data["recording_id"]
        assert result.execution_time_ms is not None

    @async_test
    async def test_command_with_correlation_id(self):
        """Test command correlation through handler chain."""
        bus = CommandBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class Step1Command(Command):
            data: str

        @dataclass(frozen=True, slots=True, kw_only=True)
        class Step2Command(Command):
            data: str

        class Step1Handler(CommandHandler):
            async def handle(self, command: Step1Command) -> CommandResult:
                return CommandResult(
                    success=True,
                    command_id=command.command_id,
                    result_data={"step1_output": command.data + "_processed"}
                )

            def can_handle(self, command_type: type) -> bool:
                return command_type == Step1Command

        class Step2Handler(CommandHandler):
            async def handle(self, command: Step2Command) -> CommandResult:
                return CommandResult(
                    success=True,
                    command_id=command.command_id,
                    result_data={"step2_output": command.data + "_final"}
                )

            def can_handle(self, command_type: type) -> bool:
                return command_type == Step2Command

        bus.register(Step1Command, Step1Handler())
        bus.register(Step2Command, Step2Handler())

        correlation_id = "workflow-123"
        cmd1 = Step1Command(data="input1", correlation_id=correlation_id)
        cmd2 = Step2Command(data="input2", correlation_id=correlation_id)

        result1 = await bus.dispatch(cmd1)
        result2 = await bus.dispatch(cmd2)

        assert result1.success is True
        assert result2.success is True
        assert result1.command_id != result2.command_id
        assert cmd1.correlation_id == cmd2.correlation_id == correlation_id


# Import dataclass for custom command tests
from dataclasses import dataclass
