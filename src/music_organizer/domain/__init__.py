"""
Domain Layer - Music Organizer

This module contains the domain layer with bounded contexts following Domain-Driven Design principles.

Bounded Contexts:
- Catalog: Managing the music catalog and metadata
- Organization: Managing physical file organization
- Classification: Content classification and duplicate detection
"""

# Legacy exports for backward compatibility
from .entities import (
    Recording as LegacyRecording,
    Release as LegacyRelease,
    Collection as LegacyCollection,
    AudioLibrary as LegacyAudioLibrary,
    DuplicateResolutionMode,
)
from .value_objects import (
    AudioPath,
    ArtistName,
    TrackNumber,
    Metadata,
    ContentPattern,
    FileFormat,
)

# Bounded Contexts
from .catalog import (
    Recording as CatalogRecording,
    Release as CatalogRelease,
    Artist,
    Catalog,
    RecordingRepository,
    ReleaseRepository,
    ArtistRepository,
    CatalogService,
    MetadataService,
)

from .organization import (
    OrganizationRule,
    FolderStructure,
    MovedFile,
    ConflictResolution,
    TargetPath,
    OrganizationPattern,
    ConflictStrategy,
    OrganizationService,
    PathGenerationService,
)

from .classification import (
    Classifier,
    DuplicateGroup,
    ContentType,
    ClassificationRule,
    SimilarityScore,
    ContentTypeEnum,
    ClassificationPattern,
    SimilarityThreshold,
    ClassificationService,
    DuplicateService,
    ContentAnalysisService,
)

# Result pattern for error handling
from .result import (
    Result,
    Success,
    Failure,
    success,
    failure,
    as_result,
    as_result_async,
    collect,
    partition,
    try_catch,
    ResultBuilder,
    DomainError,
    ValidationError,
    NotFoundError,
    DuplicateError,
    OrganizationError,
    MetadataError,
)

# Unified exports
__all__ = [
    # Legacy entities (deprecated, use bounded context versions)
    "LegacyRecording",
    "LegacyRelease",
    "LegacyCollection",
    "LegacyAudioLibrary",
    "DuplicateResolutionMode",
    # Common value objects
    "AudioPath",
    "ArtistName",
    "TrackNumber",
    "Metadata",
    "ContentPattern",
    "FileFormat",
    # Catalog context
    "CatalogRecording",
    "CatalogRelease",
    "Artist",
    "Catalog",
    "RecordingRepository",
    "ReleaseRepository",
    "ArtistRepository",
    "CatalogService",
    "MetadataService",
    # Organization context
    "OrganizationRule",
    "FolderStructure",
    "MovedFile",
    "ConflictResolution",
    "TargetPath",
    "OrganizationPattern",
    "ConflictStrategy",
    "OrganizationService",
    "PathGenerationService",
    # Classification context
    "Classifier",
    "DuplicateGroup",
    "ContentType",
    "ClassificationRule",
    "SimilarityScore",
    "ContentTypeEnum",
    "ClassificationPattern",
    "SimilarityThreshold",
    "ClassificationService",
    "DuplicateService",
    "ContentAnalysisService",
    # Result pattern
    "Result",
    "Success",
    "Failure",
    "success",
    "failure",
    "as_result",
    "as_result_async",
    "collect",
    "partition",
    "try_catch",
    "ResultBuilder",
    "DomainError",
    "ValidationError",
    "NotFoundError",
    "DuplicateError",
    "OrganizationError",
    "MetadataError",
]

# Context markers for type checking
CATALOG_CONTEXT = "catalog"
ORGANIZATION_CONTEXT = "organization"
CLASSIFICATION_CONTEXT = "classification"