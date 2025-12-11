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

__all__ = [
    # Core components
    "OperationHistoryTracker",
    "OperationRollbackService",
    "EnhancedAsyncFileMover",
    "EnhancedAsyncMusicOrganizer",

    # Types and enums
    "OperationType",
    "OperationStatus",
    "OperationRecord",
    "OperationSession",

    # Utilities
    "operation_session",
    "create_operation_record",
    "file_operation_session"
]