"""
Security utilities for file operations.

Provides path validation, sanitization, and security checks for file operations.
"""

import os
from pathlib import Path
from typing import Optional, Set


class PathValidationError(ValueError):
    """Raised when path validation fails."""
    pass


class SecurityUtils:
    """Security utilities for file operations."""

    # Allowed audio file extensions
    ALLOWED_EXTENSIONS: Set[str] = {
        '.flac', '.mp3', '.wav', '.m4a', '.aac',
        '.ogg', '.opus', '.wma', '.mp4', '.aiff', '.aif'
    }

    @staticmethod
    def is_valid_path(path: Path, base_path: Optional[Path] = None) -> bool:
        """
        Ensure path is within allowed base directory and doesn't escape.

        Args:
            path: Path to validate
            base_path: Optional base directory to check against

        Returns:
            True if path is valid and safe

        Raises:
            PathValidationError: If path contains suspicious patterns or escapes base
        """
        if not path:
            raise PathValidationError("Path cannot be empty")

        # Convert to string to check for suspicious patterns
        path_str = str(path)

        # Check for path traversal attempts
        if '../' in path_str or '..\\' in path_str:
            raise PathValidationError("Path contains parent directory references")

        # Check for null bytes (potential exploit)
        if '\x00' in path_str:
            raise PathValidationError("Path contains null bytes")

        # If base_path provided, ensure resolved path stays within it
        if base_path is not None:
            try:
                base_resolved = base_path.resolve()
                path_resolved = path.resolve()
                # Use is_relative_to (Python 3.9+) or check common path
                try:
                    if not path_resolved.is_relative_to(base_resolved):
                        raise PathValidationError(
                            f"Path escapes base directory: {path}"
                        )
                except AttributeError:
                    # Python < 3.9 fallback
                    base_resolved_str = str(base_resolved)
                    path_resolved_str = str(path_resolved)
                    if not path_resolved_str.startswith(base_resolved_str):
                        raise PathValidationError(
                            f"Path escapes base directory: {path}"
                        )
            except OSError as e:
                raise PathValidationError(f"Cannot resolve path: {e}")

        return True

    @staticmethod
    def sanitize_path(path: Path, base_path: Optional[Path] = None) -> Path:
        """
        Sanitize a path by resolving it and validating it's safe.

        Args:
            path: Path to sanitize
            base_path: Optional base directory to check against

        Returns:
            Sanitized absolute Path

        Raises:
            PathValidationError: If path is invalid or unsafe
        """
        SecurityUtils.is_valid_path(path, base_path)
        return path.resolve()

    @staticmethod
    def is_allowed_file_type(file_path: Path) -> bool:
        """
        Check if file extension is in allowed types.

        Args:
            file_path: Path to check

        Returns:
            True if file type is allowed
        """
        return file_path.suffix.lower() in SecurityUtils.ALLOWED_EXTENSIONS

    @staticmethod
    def check_symlink_safety(path: Path) -> bool:
        """
        Check if path is a symlink (potential security risk).

        Args:
            path: Path to check

        Returns:
            True if path is not a symlink

        Raises:
            PathValidationError: If path is a symlink
        """
        try:
            if path.is_symlink():
                raise PathValidationError(
                    f"Symlinks are not allowed for security reasons: {path}"
                )
            return True
        except OSError as e:
            raise PathValidationError(f"Cannot check symlink status: {e}")

    @staticmethod
    def check_file_permissions(
        file_path: Path,
        read_required: bool = False,
        write_required: bool = False
    ) -> bool:
        """
        Check file permissions before access.

        Args:
            file_path: Path to check
            read_required: Whether read permission is needed
            write_required: Whether write permission is needed

        Returns:
            True if permissions are sufficient

        Raises:
            PathValidationError: If permissions are insufficient
        """
        if read_required:
            if not os.access(file_path, os.R_OK):
                raise PathValidationError(
                    f"Read permission denied: {file_path.name}"
                )

        if write_required:
            if not os.access(file_path, os.W_OK):
                raise PathValidationError(
                    f"Write permission denied: {file_path.name}"
                )

        return True

    @staticmethod
    def sanitize_error_message(path: Path, error: Exception) -> str:
        """
        Sanitize error messages to avoid exposing sensitive paths.

        Args:
            path: Original path (will not be included in output)
            error: Original exception

        Returns:
            Sanitized error message
        """
        # Only include filename, not full path
        try:
            size = path.stat().st_size
            return f"File operation failed (size: {size} bytes): {type(error).__name__}"
        except (OSError, AttributeError):
            return f"File operation failed: {type(error).__name__}"


def validate_path(path: Path, base_path: Optional[Path] = None) -> Path:
    """
    Convenience function to validate and sanitize a path.

    Args:
        path: Path to validate
        base_path: Optional base directory

    Returns:
        Validated, sanitized Path

    Raises:
        PathValidationError: If validation fails
    """
    return SecurityUtils.sanitize_path(path, base_path)
