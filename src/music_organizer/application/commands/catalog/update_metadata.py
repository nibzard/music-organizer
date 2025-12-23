"""Update metadata command."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from ...commands.base import Command, CommandHandler, CommandResult
from ...events.base import DomainEvent
from ....domain.catalog.repositories import RecordingRepository
from ....domain.value_objects import Metadata


@dataclass(frozen=True, slots=True, kw_only=True)
class UpdateMetadataCommand(Command):
    """Command to update recording metadata."""

    recording_id: str
    metadata_updates: Dict[str, Any]
    merge_strategy: str = "merge"  # "merge" or "replace"


class MetadataUpdatedEvent(DomainEvent):
    """Event raised when recording metadata is updated."""

    def __init__(
        self,
        recording_id: str,
        previous_metadata: Dict[str, Any],
        new_metadata: Dict[str, Any],
        updates: Dict[str, Any]
    ):
        super().__init__(
            aggregate_id=recording_id,
            aggregate_type="Recording",
            event_type="MetadataUpdated",
            event_data={
                "previous_metadata": previous_metadata,
                "new_metadata": new_metadata,
                "updates": updates
            }
        )


class UpdateMetadataCommandHandler(CommandHandler[UpdateMetadataCommand, CommandResult]):
    """Handler for updating recording metadata."""

    def __init__(self, recording_repo: RecordingRepository, event_bus=None):
        self.recording_repo = recording_repo
        self.event_bus = event_bus

    async def handle(self, command: UpdateMetadataCommand) -> CommandResult:
        """Handle the update metadata command."""
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

            # Store previous metadata for event
            previous_metadata = recording.metadata.to_dict()

            # Apply metadata updates
            if command.merge_strategy == "replace":
                # Create new metadata with updates
                new_metadata_dict = command.metadata_updates
            else:
                # Merge with existing metadata
                current_metadata_dict = recording.metadata.to_dict()
                new_metadata_dict = {**current_metadata_dict, **command.metadata_updates}

            # Validate and create new metadata
            new_metadata = Metadata(**new_metadata_dict)

            # Update recording metadata
            recording.update_metadata(new_metadata)

            # Save changes
            await self.recording_repo.update(recording)

            # Create and publish event
            event = MetadataUpdatedEvent(
                recording_id=command.recording_id,
                previous_metadata=previous_metadata,
                new_metadata=new_metadata.to_dict(),
                updates=command.metadata_updates
            )

            result = CommandResult(
                success=True,
                command_id=command.command_id,
                message=f"Successfully updated metadata for recording: {command.recording_id}",
                result_data={
                    "recording_id": command.recording_id,
                    "updated_fields": list(command.metadata_updates.keys())
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
                message="Failed to update metadata",
                errors=[str(e)]
            )

    def can_handle(self, command_type: type) -> bool:
        """Check if this handler can handle the given command type."""
        return command_type == UpdateMetadataCommand