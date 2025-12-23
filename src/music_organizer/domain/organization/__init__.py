"""
Organization Context - Managing physical file organization.

This bounded context is responsible for:
- File organization rules and patterns
- Path generation and transformation
- File move operations
- Conflict resolution
"""

from .entities import OrganizationRule, FolderStructure, MovedFile, ConflictResolution
from .value_objects import TargetPath, OrganizationPattern, ConflictStrategy
from .services import OrganizationService, PathGenerationService, RecordingLoaderService

__all__ = [
    # Entities
    "OrganizationRule",
    "FolderStructure",
    "MovedFile",
    "ConflictResolution",
    # Value Objects
    "TargetPath",
    "OrganizationPattern",
    "ConflictStrategy",
    # Services
    "OrganizationService",
    "PathGenerationService",
    "RecordingLoaderService",
]