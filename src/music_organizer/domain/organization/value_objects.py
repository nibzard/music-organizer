"""Organization Context Value Objects.

This module defines value objects for the Organization bounded context.
Value objects are immutable objects that are defined by their attributes rather than identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..catalog.value_objects import ArtistName, TrackNumber, Metadata


class ConflictStrategy(Enum):
    """Strategies for handling naming conflicts."""
    SKIP = "skip"
    RENAME = "rename"
    REPLACE = "replace"
    KEEP_BOTH = "keep_both"
    APPEND_SUFFIX = "append_suffix"


class OrganizationLevel(Enum):
    """Organization levels for folder structure."""
    FLAT = "flat"  # All files in one directory
    ARTIST = "artist"  # Artist/Album/Track
    GENRE = "genre"  # Genre/Artist/Album/Track
    YEAR = "year"  # Year/Artist/Album/Track
    DECADE = "decade"  # Decade/Artist/Album/Track
    CUSTOM = "custom"  # User-defined pattern


@dataclass(frozen=True, slots=True)
class OrganizationPattern:
    """
    Value object representing a file organization pattern.
    """

    path_pattern: str  # e.g., "{genre}/{artist}/{album} ({year})/"
    filename_pattern: str  # e.g., "{track_number} {title}{file_extension}"
    level: OrganizationLevel = OrganizationLevel.ARTIST

    # Pattern validation rules
    required_variables: List[str] = field(default_factory=lambda: ["artist", "title"])
    optional_variables: List[str] = field(default_factory=lambda: ["album", "year", "track_number", "genre"])

    # Character cleanup settings
    replace_spaces: bool = True
    space_replacement: str = "_"
    max_filename_length: int = 255

    def __post_init__(self) -> None:
        """Validate the pattern."""
        # Check if required variables are present
        for var in self.required_variables:
            if var not in self.path_pattern and var not in self.filename_pattern:
                raise ValueError(f"Required variable '{var}' not found in pattern")

    def generate_path(
        self,
        metadata: Metadata,
        source_path: Optional[Path] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Generate a target path based on the pattern."""
        context = self._build_context(metadata, source_path, additional_context or {})

        # Replace variables in path pattern
        path_str = self._replace_variables(self.path_pattern, context)

        # Replace variables in filename pattern
        filename = self._replace_variables(self.filename_pattern, context)

        # Clean up the path
        path_str = self._clean_path_component(path_str)
        filename = self._clean_filename(filename)

        # Combine and return
        full_path = Path(path_str) / filename
        return full_path

    def _build_context(
        self,
        metadata: Metadata,
        source_path: Optional[Path],
        additional_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build context dictionary for variable replacement."""
        context = {
            "artist": str(metadata.artists[0]) if metadata.artists else "Unknown Artist",
            "albumartist": str(metadata.albumartist) if metadata.albumartist else "",
            "album": metadata.album or "",
            "title": metadata.title or "",
            "year": str(metadata.year) if metadata.year else "",
            "decade": f"{(metadata.year // 10) * 10}s" if metadata.year else "",
            "genre": metadata.genre or "",
            "track_number": metadata.track_number.formatted() if metadata.track_number else "",
            "disc_number": str(metadata.disc_number) if metadata.disc_number else "",
            "total_tracks": str(metadata.total_tracks) if metadata.total_tracks else "",
            "total_discs": str(metadata.total_discs) if metadata.total_discs else "",
            "composer": metadata.composer or "",
        }

        # Add file extension from source path
        if source_path:
            context["file_extension"] = source_path.suffix
            context["filename_without_ext"] = source_path.stem

        # Add additional context
        context.update(additional_context)

        return context

    def _replace_variables(self, pattern: str, context: Dict[str, Any]) -> str:
        """Replace variables in a pattern with values from context."""
        # Handle conditional blocks {if:var}...{endif}
        pattern = self._process_conditionals(pattern, context)

        # Replace simple variables
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            pattern = pattern.replace(placeholder, str(value))

        return pattern

    def _process_conditionals(self, pattern: str, context: Dict[str, Any]) -> str:
        """Process conditional blocks in the pattern."""
        # Match {if:var}...{endif} blocks
        conditional_pattern = re.compile(r'\{if:(\w+)\}(.*?)\{endif\}', re.DOTALL)

        def replace_conditional(match: re.Match) -> str:
            var_name = match.group(1)
            content = match.group(2)

            # Check if variable exists and has a value
            value = context.get(var_name)
            if value and str(value).strip():
                return content
            return ""

        return conditional_pattern.sub(replace_conditional, pattern)

    def _clean_path_component(self, component: str) -> str:
        """Clean up a path component."""
        # Remove invalid filesystem characters
        invalid_chars = '<>:"|?*'
        for char in invalid_chars:
            component = component.replace(char, '')

        # Replace forward slashes with underscores to prevent directory creation
        component = component.replace('/', '_')

        # Replace spaces if requested
        if self.replace_spaces:
            component = component.replace(' ', self.space_replacement)

        # Collapse multiple separators
        component = re.sub(r'[_-]+', '_', component)

        # Remove leading/trailing separators
        component = component.strip('_-')

        return component

    def _clean_filename(self, filename: str) -> str:
        """Clean up a filename."""
        # Apply path cleaning first
        filename = self._clean_path_component(filename)

        # Ensure it doesn't exceed maximum length
        if len(filename) > self.max_filename_length:
            # Preserve extension if present
            stem = Path(filename).stem
            ext = Path(filename).suffix

            max_stem_length = self.max_filename_length - len(ext)
            if max_stem_length > 0:
                filename = stem[:max_stem_length] + ext
            else:
                # If extension is too long, truncate everything
                filename = filename[:self.max_filename_length]

        return filename


@dataclass(frozen=True, slots=True)
class TargetPath:
    """
    Value object representing a target file path.
    """

    path: Path
    pattern_used: Optional[OrganizationPattern] = None
    conflict_strategy: ConflictStrategy = ConflictStrategy.SKIP
    original_path: Optional[Path] = None

    @property
    def exists(self) -> bool:
        """Check if the target path exists."""
        return self.path.exists()

    @property
    def parent(self) -> Path:
        """Get the parent directory."""
        return self.path.parent

    @property
    def stem(self) -> str:
        """Get the filename without extension."""
        return self.path.stem

    @property
    def suffix(self) -> str:
        """Get the file extension."""
        return self.path.suffix

    def with_conflict_suffix(self, suffix: str) -> "TargetPath":
        """Create a new TargetPath with a suffix added for conflict resolution."""
        new_name = f"{self.stem}_{suffix}{self.suffix}"
        new_path = self.parent / new_name
        return TargetPath(
            path=new_path,
            pattern_used=self.pattern_used,
            conflict_strategy=ConflictStrategy.APPEND_SUFFIX,
            original_path=self.original_path
        )

    def with_new_name(self, new_name: str) -> "TargetPath":
        """Create a new TargetPath with a different name."""
        new_path = self.parent / new_name
        return TargetPath(
            path=new_path,
            pattern_used=self.pattern_used,
            conflict_strategy=self.conflict_strategy,
            original_path=self.original_path
        )

    def relative_to(self, base: Path) -> "TargetPath":
        """Create a relative version of this TargetPath."""
        relative_path = self.path.relative_to(base)
        return TargetPath(
            path=relative_path,
            pattern_used=self.pattern_used,
            conflict_strategy=self.conflict_strategy,
            original_path=self.original_path.relative_to(base) if self.original_path else None
        )


@dataclass(frozen=True, slots=True)
class PathTemplate:
    """
    Value object for path templates with validation.
    """

    template: str
    description: Optional[str] = None
    example: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def validate(self, required_vars: List[str]) -> List[str]:
        """Validate that the template contains required variables."""
        missing_vars = []
        for var in required_vars:
            if f"{{{var}}}" not in self.template:
                # Check if it's in a conditional
                if not re.search(r'\{if:' + var + r'\}.*?\{endif\}', self.template):
                    missing_vars.append(var)
        return missing_vars

    def get_variables(self) -> List[str]:
        """Extract all variables from the template."""
        # Find simple variables
        simple_vars = re.findall(r'\{(\w+)\}', self.template)

        # Find conditional variables
        conditional_vars = re.findall(r'\{if:(\w+)\}', self.template)

        # Combine and deduplicate
        all_vars = list(set(simple_vars + conditional_vars))

        # Remove control keywords
        control_words = {"if", "endif"}
        return [var for var in all_vars if var not in control_words]