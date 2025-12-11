"""Create directory structure command."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

from ...commands.base import Command, CommandHandler, CommandResult
from ...events.base import DomainEvent
from ....domain.organization.entities import FolderStructure


@dataclass(frozen=True, slots=True)
class CreateDirectoryStructureCommand(Command):
    """Command to create a directory structure based on a template."""

    root_directory: Path
    structure_template: Dict[str, Any]
    dry_run: bool = False


class DirectoryStructureCreatedEvent(DomainEvent):
    """Event raised when a directory structure is created."""

    def __init__(
        self,
        root_directory: Path,
        created_directories: List[str],
        dry_run: bool
    ):
        super().__init__(
            aggregate_type="DirectoryStructure",
            event_type="DirectoryStructureCreated",
            event_data={
                "root_directory": str(root_directory),
                "created_directories": created_directories,
                "dry_run": dry_run
            }
        )


class CreateDirectoryStructureCommandHandler(CommandHandler[CreateDirectoryStructureCommand, CommandResult]):
    """Handler for creating directory structures."""

    def __init__(self, event_bus=None):
        self.event_bus = event_bus

    async def handle(self, command: CreateDirectoryStructureCommand) -> CommandResult:
        """Handle the create directory structure command."""
        try:
            created_directories = []

            def create_structure_recursive(current_path: Path, structure: Dict[str, Any]):
                """Recursively create directory structure."""
                for name, content in structure.items():
                    dir_path = current_path / name

                    if isinstance(content, dict):
                        # It's a subdirectory
                        if not command.dry_run:
                            dir_path.mkdir(parents=True, exist_ok=True)
                        created_directories.append(str(dir_path))
                        create_structure_recursive(dir_path, content)
                    elif isinstance(content, list):
                        # It's a list of subdirectories or files
                        for item in content:
                            if isinstance(item, dict):
                                # Nested structure
                                create_structure_recursive(dir_path, item)
                            else:
                                # Simple subdirectory
                                sub_path = dir_path / item
                                if not command.dry_run:
                                    sub_path.mkdir(parents=True, exist_ok=True)
                                created_directories.append(str(sub_path))
                    else:
                        # Simple subdirectory
                        if not command.dry_run:
                            dir_path.mkdir(parents=True, exist_ok=True)
                        created_directories.append(str(dir_path))

            # Start creating the structure
            if not command.dry_run:
                command.root_directory.mkdir(parents=True, exist_ok=True)
            created_directories.append(str(command.root_directory))

            create_structure_recursive(command.root_directory, command.structure_template)

            # Create event
            event = DirectoryStructureCreatedEvent(
                root_directory=command.root_directory,
                created_directories=created_directories,
                dry_run=command.dry_run
            )

            result = CommandResult(
                success=True,
                command_id=command.command_id,
                message=f"Successfully created directory structure: {command.root_directory}",
                result_data={
                    "root_directory": str(command.root_directory),
                    "created_directories": created_directories,
                    "total_directories": len(created_directories),
                    "dry_run": command.dry_run
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
                message="Failed to create directory structure",
                errors=[str(e)]
            )

    def can_handle(self, command_type: type) -> bool:
        """Check if this handler can handle the given command type."""
        return command_type == CreateDirectoryStructureCommand