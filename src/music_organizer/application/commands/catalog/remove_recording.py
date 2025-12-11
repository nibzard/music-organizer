"""Remove recording command."""

from dataclasses import dataclass

from ...commands.base import Command, CommandHandler, CommandResult
from ...events.base import DomainEvent
from ....domain.catalog.repositories import RecordingRepository


@dataclass(frozen=True, slots=True)
class RemoveRecordingCommand(Command):
    """Command to remove a recording from the catalog."""

    recording_id: str
    delete_file: bool = False  # Whether to delete the actual file


class RecordingRemovedEvent(DomainEvent):
    """Event raised when a recording is removed from the catalog."""

    def __init__(
        self,
        recording_id: str,
        file_path: str,
        delete_file: bool
    ):
        super().__init__(
            aggregate_id=recording_id,
            aggregate_type="Recording",
            event_type="RecordingRemoved",
            event_data={
                "file_path": file_path,
                "delete_file": delete_file
            }
        )


class RemoveRecordingCommandHandler(CommandHandler[RemoveRecordingCommand, CommandResult]):
    """Handler for removing recordings from the catalog."""

    def __init__(self, recording_repo: RecordingRepository, event_bus=None):
        self.recording_repo = recording_repo
        self.event_bus = event_bus

    async def handle(self, command: RemoveRecordingCommand) -> CommandResult:
        """Handle the remove recording command."""
        try:
            # Get existing recording
            recording = await self.recording_repo.find_by_id(command.recording_id)
            if not recording:
                return CommandResult(
                    success=False,
                    command_id=command.command_id,
                    message="Recording not found",
                    errors=[f"Recording ID not found: {command.recording_id}"]
                )

            # Store file path for event
            file_path = str(recording.path)

            # Delete actual file if requested
            if command.delete_file:
                try:
                    recording.path.path.unlink()
                except FileNotFoundError:
                    # File already deleted, continue with removal
                    pass
                except Exception as e:
                    return CommandResult(
                        success=False,
                        command_id=command.command_id,
                        message="Failed to delete file",
                        errors=[f"Could not delete file {file_path}: {str(e)}"]
                    )

            # Remove from repository
            await self.recording_repo.delete(command.recording_id)

            # Create and publish event
            event = RecordingRemovedEvent(
                recording_id=command.recording_id,
                file_path=file_path,
                delete_file=command.delete_file
            )

            result = CommandResult(
                success=True,
                command_id=command.command_id,
                message=f"Successfully removed recording: {command.recording_id}",
                result_data={
                    "recording_id": command.recording_id,
                    "file_path": file_path,
                    "deleted_file": command.delete_file
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
                message="Failed to remove recording",
                errors=[str(e)]
            )

    def can_handle(self, command_type: type) -> bool:
        """Check if this handler can handle the given command type."""
        return command_type == RemoveRecordingCommand