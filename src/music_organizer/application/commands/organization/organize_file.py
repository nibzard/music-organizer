"""Organize file command."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from ...commands.base import Command, CommandHandler, CommandResult
from ...events.base import DomainEvent
from ....domain.organization.entities import OrganizationSession, MovedFile
from ....domain.organization.services import OrganizationService


@dataclass(frozen=True, slots=True)
class OrganizeFileCommand(Command):
    """Command to organize a single file according to rules."""

    source_path: Path
    target_directory: Path
    organization_pattern: Optional[str] = None
    conflict_strategy: str = "skip"  # "skip", "rename", "replace", "keep_both"
    dry_run: bool = False
    session_id: Optional[str] = None


class FileOrganizedEvent(DomainEvent):
    """Event raised when a file is organized."""

    def __init__(
        self,
        session_id: Optional[str],
        source_path: Path,
        target_path: Path,
        conflict_strategy: str,
        dry_run: bool
    ):
        super().__init__(
            aggregate_id=session_id or "single_file",
            aggregate_type="OrganizationSession",
            event_type="FileOrganized",
            event_data={
                "source_path": str(source_path),
                "target_path": str(target_path),
                "conflict_strategy": conflict_strategy,
                "dry_run": dry_run
            }
        )


class OrganizeFileCommandHandler(CommandHandler[OrganizeFileCommand, CommandResult]):
    """Handler for organizing individual files."""

    def __init__(self, organization_service: OrganizationService, event_bus=None):
        self.organization_service = organization_service
        self.event_bus = event_bus

    async def handle(self, command: OrganizeFileCommand) -> CommandResult:
        """Handle the organize file command."""
        try:
            # Validate source file exists
            if not command.source_path.exists():
                return CommandResult(
                    success=False,
                    command_id=command.command_id,
                    message="Source file does not exist",
                    errors=[f"File not found: {command.source_path}"]
                )

            # Create organization session if session_id provided
            session = None
            if command.session_id:
                session = OrganizationSession(
                    session_id=command.session_id,
                    source_directory=command.source_path.parent,
                    target_directory=command.target_directory
                )

            # Perform file organization
            moved_file = await self.organization_service.organize_file(
                source_path=command.source_path,
                target_directory=command.target_directory,
                pattern=command.organization_pattern,
                conflict_strategy=command.conflict_strategy,
                dry_run=command.dry_run,
                session=session
            )

            # Create event
            event = FileOrganizedEvent(
                session_id=command.session_id,
                source_path=command.source_path,
                target_path=moved_file.target_path,
                conflict_strategy=command.conflict_strategy,
                dry_run=command.dry_run
            )

            result = CommandResult(
                success=True,
                command_id=command.command_id,
                message=f"Successfully organized file: {command.source_path}",
                result_data={
                    "source_path": str(command.source_path),
                    "target_path": str(moved_file.target_path),
                    "conflict_strategy": command.conflict_strategy,
                    "dry_run": command.dry_run,
                    "moved": moved_file.moved
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
                message="Failed to organize file",
                errors=[str(e)]
            )

    def can_handle(self, command_type: type) -> bool:
        """Check if this handler can handle the given command type."""
        return command_type == OrganizeFileCommand