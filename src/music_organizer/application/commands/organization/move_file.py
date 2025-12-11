"""Move file command."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from ...commands.base import Command, CommandHandler, CommandResult
from ...events.base import DomainEvent
from ....domain.organization.services import FileMover


@dataclass(frozen=True, slots=True)
class MoveFileCommand(Command):
    """Command to move a file to a specific location."""

    source_path: Path
    target_path: Path
    conflict_strategy: str = "skip"  # "skip", "rename", "replace", "keep_both"
    create_directories: bool = True
    preserve_metadata: bool = True


class FileMovedEvent(DomainEvent):
    """Event raised when a file is moved."""

    def __init__(
        self,
        source_path: Path,
        target_path: Path,
        conflict_strategy: str,
        preserved_metadata: bool
    ):
        super().__init__(
            aggregate_type="FileOperation",
            event_type="FileMoved",
            event_data={
                "source_path": str(source_path),
                "target_path": str(target_path),
                "conflict_strategy": conflict_strategy,
                "preserved_metadata": preserved_metadata
            }
        )


class MoveFileCommandHandler(CommandHandler[MoveFileCommand, CommandResult]):
    """Handler for moving files."""

    def __init__(self, file_mover: FileMover, event_bus=None):
        self.file_mover = file_mover
        self.event_bus = event_bus

    async def handle(self, command: MoveFileCommand) -> CommandResult:
        """Handle the move file command."""
        try:
            # Validate source file exists
            if not command.source_path.exists():
                return CommandResult(
                    success=False,
                    command_id=command.command_id,
                    message="Source file does not exist",
                    errors=[f"File not found: {command.source_path}"]
                )

            # Create target directory if needed
            if command.create_directories:
                command.target_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the file
            moved_file = await self.file_mover.move_file(
                source_path=command.source_path,
                target_path=command.target_path,
                conflict_strategy=command.conflict_strategy,
                preserve_metadata=command.preserve_metadata
            )

            # Create event
            event = FileMovedEvent(
                source_path=command.source_path,
                target_path=command.target_path,
                conflict_strategy=command.conflict_strategy,
                preserved_metadata=command.preserve_metadata
            )

            result = CommandResult(
                success=True,
                command_id=command.command_id,
                message=f"Successfully moved file to: {command.target_path}",
                result_data={
                    "source_path": str(command.source_path),
                    "target_path": str(command.target_path),
                    "conflict_strategy": command.conflict_strategy,
                    "moved": moved_file.moved,
                    "preserved_metadata": command.preserve_metadata
                },
                events=[event] if self.event_bus else []
            )

            # Publish event if event bus is available
            if self.event_bus:
                await self.event_bus.publish(event)

            return result

        except Exception as e:
            return CommandResult(
                success=False,
                command_id=command.command_id,
                message="Failed to move file",
                errors=[str(e)]
            )

    def can_handle(self, command_type: type) -> bool:
        """Check if this handler can handle the given command type."""
        return command_type == MoveFileCommand