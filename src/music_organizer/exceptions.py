"""Custom exceptions for music organizer."""


class MusicOrganizerError(Exception):
    """Base exception for music organizer errors."""
    pass


class MetadataError(MusicOrganizerError):
    """Raised when there's an error extracting or processing metadata."""
    pass


class FileOperationError(MusicOrganizerError):
    """Raised when file operations fail."""
    pass


class ClassificationError(MusicOrganizerError):
    """Raised when content classification fails."""
    pass


class ConfigurationError(MusicOrganizerError):
    """Raised when there's an error in configuration."""
    pass