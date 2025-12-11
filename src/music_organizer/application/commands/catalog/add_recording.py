"""Add recording command."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from ...commands.base import Command, CommandHandler, CommandResult
from ...events.base import DomainEvent
from ....domain.catalog.entities import Recording
from ....domain.catalog.repositories import RecordingRepository
from ....domain.value_objects import AudioPath, Metadata


@dataclass(frozen=True, slots=True)
class AddRecordingCommand(Command):
    """Command to add a new recording to the catalog."""

    file_path: Path
    metadata: Dict[str, Any]
    catalog_id: str = "default"


class RecordingAddedEvent(DomainEvent):
    """Event raised when a recording is added to the catalog."""

    def __init__(
        self,
        recording_id: str,
        file_path: Path,
        catalog_id: str,
        metadata: Dict[str, Any]
    ):
        super().__init__(
            aggregate_id=recording_id,
            aggregate_type="Recording",
            event_type="RecordingAdded",
            event_data={
                "file_path": str(file_path),
                "catalog_id": catalog_id,
                "metadata": metadata
            }
        )


class AddRecordingCommandHandler(CommandHandler[AddRecordingCommand, CommandResult]):
    """Handler for adding recordings to the catalog."""

    def __init__(self, recording_repo: RecordingRepository, event_bus=None):
        self.recording_repo = recording_repo
        self.event_bus = event_bus

    async def handle(self, command: AddRecordingCommand) -> CommandResult:
        """Handle the add recording command."""
        try:
            # Create value objects
            audio_path = AudioPath(command.file_path)
            metadata = Metadata(**command.metadata)

            # Create recording entity
            recording = Recording(path=audio_path, metadata=metadata)

            # Check for duplicates
            existing = await self.recording_repo.find_by_path(audio_path)
            if existing:
                return CommandResult(
                    success=False,
                    command_id=command.command_id,
                    message="Recording already exists in catalog",
                    errors=[f"File already cataloged: {command.file_path}"]
                )

            # Save to repository
            await self.recording_repo.save(recording)

            # Create and publish event
            event = RecordingAddedEvent(
                recording_id=recording.id,
                file_path=command.file_path,
                catalog_id=command.catalog_id,
                metadata=command.metadata
            )

            result = CommandResult(
                success=True,
                command_id=command.command_id,
                message=f"Successfully added recording: {command.file_path}",
                result_data={
                    "recording_id": recording.id,
                    "file_path": str(command.file_path),
                    "catalog_id": command.catalog_id
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
                message="Failed to add recording",
                errors=[str(e)]
            )

    def can_handle(self, command_type: type) -> bool:
        """Check if this handler can handle the given command type."""
        return command_type == AddRecordingCommand