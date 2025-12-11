"""Music Library Organizer

A tool for organizing music libraries using metadata-aware categorization.
"""

__version__ = "0.1.0"

# Export rollback and operation history components
from .core.operation_history import (
    OperationHistoryTracker,
    OperationRollbackService,
    OperationType,
    OperationStatus,
    OperationRecord,
    OperationSession,
    operation_session,
    create_operation_record
)

from .core.enhanced_file_mover import (
    EnhancedAsyncFileMover,
    file_operation_session
)

from .core.enhanced_async_organizer import (
    EnhancedAsyncMusicOrganizer
)

# Export duplicate resolution components
from .core.interactive_duplicate_resolver import (
    InteractiveDuplicateResolver,
    DuplicateAction,
    ResolutionStrategy,
    DuplicateQualityScorer,
    quick_duplicate_resolution
)

from .core.duplicate_resolver_organizer import (
    DuplicateResolverOrganizer
)

from .ui.duplicate_resolver_ui import (
    DuplicateResolverUI
)

__all__ = [
    # Core components
    "OperationHistoryTracker",
    "OperationRollbackService",
    "EnhancedAsyncFileMover",
    "EnhancedAsyncMusicOrganizer",

    # Duplicate resolution components
    "InteractiveDuplicateResolver",
    "DuplicateResolverOrganizer",
    "DuplicateQualityScorer",
    "DuplicateResolverUI",

    # Types and enums
    "OperationType",
    "OperationStatus",
    "OperationRecord",
    "OperationSession",
    "DuplicateAction",
    "ResolutionStrategy",

    # Utilities
    "operation_session",
    "create_operation_record",
    "file_operation_session",
    "quick_duplicate_resolution"
]