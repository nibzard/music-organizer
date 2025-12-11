"""Catalog commands."""

from .add_recording import AddRecordingCommand, AddRecordingCommandHandler
from .update_metadata import UpdateMetadataCommand, UpdateMetadataCommandHandler
from .remove_recording import RemoveRecordingCommand, RemoveRecordingCommandHandler

__all__ = [
    "AddRecordingCommand",
    "AddRecordingCommandHandler",
    "UpdateMetadataCommand",
    "UpdateMetadataCommandHandler",
    "RemoveRecordingCommand",
    "RemoveRecordingCommandHandler",
]