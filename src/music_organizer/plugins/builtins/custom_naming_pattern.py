"""Custom naming pattern plugin for user-defined organization rules."""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..base import PathPlugin, PluginInfo
from ...models.audio_file import AudioFile
from ...domain.classification.value_objects import ContentType


def create_plugin(config: Optional[Dict[str, Any]] = None) -> PathPlugin:
    """Factory function to create the plugin."""
    return CustomNamingPatternPlugin(config)


class PatternTemplate:
    """Template system for path and filename patterns."""

    # Mapping of template variables to AudioFile attributes/methods
    VARIABLE_MAP = {
        'artist': lambda af: af.primary_artist or af.artists[0] if af.artists else 'Unknown Artist',
        'artists': lambda af: ', '.join(af.artists) if af.artists else 'Unknown Artist',
        'primary_artist': lambda af: af.primary_artist or 'Unknown Artist',
        'album': lambda af: af.album or 'Unknown Album',
        'year': lambda af: str(af.year) if af.year else '',
        'track_number': lambda af: f"{af.track_number:02d}" if af.track_number is not None else '',
        'title': lambda af: af.title or af.path.stem,
        'genre': lambda af: af.genre or 'Unknown',
        'format': lambda af: af.file_type or af.path.suffix.lstrip('.').upper(),
        'content_type': lambda af: af.content_type.value if af.content_type else 'unknown',
        'date': lambda af: af.date or '',
        'location': lambda af: af.location or '',
    }

    # Additional computed variables
    COMPUTED_VARIABLES = {
        'albumartist': lambda af: af.metadata.get('albumartist', '') or af.primary_artist or '',
        'disc_number': lambda af: f"{int(af.metadata.get('disc_number', 0)):02d}" if af.metadata.get('disc_number') else '',
        'total_discs': lambda af: str(af.metadata.get('total_discs', '')) if af.metadata.get('total_discs') else '',
        'total_tracks': lambda af: str(af.metadata.get('total_tracks', '')) if af.metadata.get('total_tracks') else '',
        'first_letter': lambda af: (af.primary_artist or af.artists[0] if af.artists else 'Unknown')[0].upper(),
        'decade': lambda af: f"{(af.year // 10) * 10}s" if af.year else '',
        'file_extension': lambda af: af.path.suffix,
        # Legacy variables for backward compatibility (now read from metadata)
        'duration': lambda af: str(af.metadata.get('duration', '')) if af.metadata.get('duration') else '',
        'bitrate': lambda af: str(af.metadata.get('bitrate', '')) if af.metadata.get('bitrate') else '',
    }

    def __init__(self):
        """Initialize the template system."""
        self.all_variables = {**self.VARIABLE_MAP, **self.COMPUTED_VARIABLES}

    def render(self, template: str, audio_file: AudioFile) -> str:
        """Render a template string with variables from an AudioFile.

        Args:
            template: Template string with variables in {variable} format
            audio_file: AudioFile object to extract values from

        Returns:
            Rendered string with variables replaced
        """
        # Mark template path separators with a placeholder so we don't clean them
        # Use a string that won't appear in music metadata
        TEMPLATE_PATH_PLACEHOLDER = '___PATH_SEPARATOR___'
        template = template.replace('/', TEMPLATE_PATH_PLACEHOLDER)
        template = template.replace('\\', TEMPLATE_PATH_PLACEHOLDER)

        # First pass: replace all variables
        result = template

        for var_name, getter in self.all_variables.items():
            pattern = re.compile(rf'\{{{re.escape(var_name)}\}}')
            value = getter(audio_file)
            # Clean the value for filesystem compatibility (removes slashes from values)
            clean_value = self._clean_value(str(value))
            result = pattern.sub(clean_value, result)

        # Handle conditional sections {if:variable}...{endif}
        result = self._handle_conditionals(result, audio_file)

        # Remove any remaining {unknown} variables (replace with empty string)
        # But exclude conditional tags and other special patterns
        result = re.sub(r'\{(?!if:|else|endif)([^}]+)\}', '', result)

        # Clean up any remaining filesystem-incompatible characters in the result
        result = self._clean_result(result)

        # Restore template path separators
        result = result.replace(TEMPLATE_PATH_PLACEHOLDER, '/')

        # Remove multiple consecutive separators
        result = re.sub(r'/+', '/', result)

        # Remove trailing slash, but only if removing it doesn't leave an empty path
        # Keep trailing slash if it would leave us with just a directory name without content
        # (e.g., "The Beatles/" is valid, but "The Beatles/Abbey Road/" -> "The Beatles/Abbey Road")
        if result.endswith('/'):
            result_without_trailing = result.rstrip('/')
            # Only remove trailing slash if there's at least one other slash in the path
            # (meaning we have a multi-component path, not just a single directory)
            if '/' in result_without_trailing:
                result = result_without_trailing

        return result

    def _clean_value(self, value: str) -> str:
        """Clean a value (individual field) to be filesystem-safe.
        This cleans problematic characters within values including slashes.

        Args:
            value: String to clean

        Returns:
            Filesystem-safe string
        """
        # Replace problematic characters (including slashes in values)
        replacements = {
            '/': ' - ',
            '\\': ' - ',
            ':': ' - ',
            '*': '',
            '?': '',
            '"': "'",
            '<': '',
            '>': '',
            '|': '-',
            '\n': ' ',
            '\r': ' ',
            '\t': ' ',
        }

        for old, new in replacements.items():
            value = value.replace(old, new)

        # Remove control characters
        value = ''.join(char for char in value if ord(char) >= 32)

        # Collapse multiple spaces
        value = re.sub(r'\s+', ' ', value).strip()

        return value

    def _clean_result(self, value: str) -> str:
        """Clean the final result string.
        This only cleans characters that shouldn't be in paths at all,
        not path separators (those are handled with placeholders).

        Args:
            value: String to clean

        Returns:
            Filesystem-safe string
        """
        # Replace problematic characters (not slashes - those are placeholders)
        replacements = {
            ':': ' - ',
            '*': '',
            '?': '',
            '"': "'",
            '<': '',
            '>': '',
            '|': '-',
            '\n': ' ',
            '\r': ' ',
            '\t': ' ',
        }

        for old, new in replacements.items():
            value = value.replace(old, new)

        # Remove control characters
        value = ''.join(char for char in value if ord(char) >= 32)

        # Collapse multiple spaces
        value = re.sub(r'\s+', ' ', value).strip()

        return value

    def _clean_for_filesystem(self, value: str) -> str:
        """Clean a string to be filesystem-safe (legacy method for backward compatibility).

        Args:
            value: String to clean

        Returns:
            Filesystem-safe string
        """
        return self._clean_value(value)

    def _handle_conditionals(self, template: str, audio_file: AudioFile) -> str:
        """Handle conditional sections in templates.

        Supports formats:
        - {if:variable}content{endif}
        - {if:variable}content{else}alternative{endif}
        The content is only included if the variable has a non-empty value.

        Args:
            template: Template string with conditionals
            audio_file: AudioFile to get variable values from

        Returns:
            Template with conditionals processed
        """
        # Pattern to match conditional blocks (with optional else)
        pattern = re.compile(r'\{if:(\w+)\}(.*?)\{(else)\}(.*?)\{endif\}|\{if:(\w+)\}(.*?)\{endif\}', re.DOTALL)

        def replace_conditional(match: re.Match) -> str:
            # Handle if-else-endif pattern (groups 1-4)
            if match.group(1):  # var_name from if-else-endif
                var_name = match.group(1)
                if_content = match.group(2)
                else_content = match.group(4)

                # Check if variable exists and has a non-empty value
                if var_name in self.all_variables:
                    value = self.all_variables[var_name](audio_file)
                    if value and str(value).strip():
                        return if_content

                return else_content

            # Handle if-endif pattern without else (groups 5-6)
            if match.group(5):  # var_name from if-endif
                var_name = match.group(5)
                content = match.group(6)

                # Check if variable exists and has a non-empty value
                if var_name in self.all_variables:
                    value = self.all_variables[var_name](audio_file)
                    if value and str(value).strip():
                        return content

                return ''

            return ''

        # Process all conditionals
        result = pattern.sub(replace_conditional, template)

        # Handle nested conditionals
        if '{if:' in result:
            result = self._handle_conditionals(result, audio_file)

        return result

    def validate_template(self, template: str) -> List[str]:
        """Validate a template string and return any errors.

        Args:
            template: Template string to validate

        Returns:
            List of validation error messages
        """
        errors = []

        # Check for malformed conditionals
        if_stack = []
        i = 0
        while i < len(template):
            if template[i:i+4] == '{if:':
                if_stack.append(i)
                i += 4
            elif template[i:i+6] == '{endif}' and if_stack:
                if_stack.pop()
                i += 6
            elif template[i:i+6] == '{endif}' and not if_stack:
                errors.append("Unmatched {endif} found")
                break
            else:
                i += 1

        if if_stack:
            errors.append(f"Unclosed conditional block starting at position {if_stack[-1]}")

        # Check for unknown variables
        var_pattern = re.compile(r'\{(\w+)\}')
        for match in var_pattern.finditer(template):
            var_name = match.group(1)
            if var_name not in ['if', 'endif'] and var_name not in self.all_variables:
                errors.append(f"Unknown variable: {var_name}")

        return errors


class CustomNamingPatternPlugin(PathPlugin):
    """Plugin that allows users to define custom naming patterns for music organization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the plugin with configuration."""
        super().__init__(config)
        self.template_engine = PatternTemplate()

        # Default patterns
        self.default_path_pattern = "{content_type}/{artist}/{album} ({year})"
        self.default_filename_pattern = "{track_number} {title}{file_extension}"

        # Load from config
        self.path_patterns = self.config.get('path_patterns', {})
        self.filename_pattern = self.config.get('filename_pattern', self.default_filename_pattern)
        self.fallback_to_default = self.config.get('fallback_to_default', True)
        self.create_date_dirs = self.config.get('create_date_dirs', False)
        self.organize_by_genre = self.config.get('organize_by_genre', False)
        self.organize_by_decade = self.config.get('organize_by_decade', False)

    @property
    def info(self) -> PluginInfo:
        """Return plugin information."""
        return PluginInfo(
            name="custom_naming_pattern",
            version="1.0.0",
            description="Plugin for user-defined music organization rules with flexible naming patterns",
            author="Music Organizer Team",
            dependencies=[]
        )

    def initialize(self) -> None:
        """Initialize the plugin."""
        # Validate patterns
        all_patterns = list(self.path_patterns.values()) + [self.filename_pattern]

        for pattern in all_patterns:
            errors = self.template_engine.validate_template(pattern)
            if errors:
                raise ValueError(f"Invalid pattern '{pattern}': {'; '.join(errors)}")

    def cleanup(self) -> None:
        """Cleanup resources when plugin is disabled."""
        pass

    async def generate_target_path(self, audio_file: AudioFile, base_dir: Path) -> Path:
        """Generate target directory path for an audio file."""
        # Select appropriate pattern based on content type and other criteria
        pattern = self._select_path_pattern(audio_file)

        # Generate base path
        rendered_path = self.template_engine.render(pattern, audio_file)

        # Add additional organization levels
        path_parts = [rendered_path]

        # Add date-based organization if enabled
        if self.create_date_dirs and audio_file.year:
            path_parts.append(str(audio_file.year))

        # Add genre organization if enabled
        if self.organize_by_genre and audio_file.genre:
            clean_genre = self.template_engine._clean_for_filesystem(audio_file.genre)
            path_parts.append(clean_genre)

        # Add decade organization if enabled
        if self.organize_by_decade and audio_file.year:
            decade = f"{(audio_file.year // 10) * 10}s"
            path_parts.append(decade)

        # Combine all parts
        final_path = base_dir
        for part in path_parts:
            if part:  # Skip empty parts
                final_path = final_path / part

        return final_path

    async def generate_filename(self, audio_file: AudioFile) -> str:
        """Generate filename for an audio file."""
        rendered = self.template_engine.render(self.filename_pattern, audio_file)

        # Ensure the filename has an extension
        if not Path(rendered).suffix:
            rendered += audio_file.path.suffix

        return rendered

    def _select_path_pattern(self, audio_file: AudioFile) -> str:
        """Select the appropriate path pattern for an audio file."""
        content_type = audio_file.content_type or ContentType.STUDIO

        # Try content-type specific pattern
        type_key = content_type.value.lower()
        if type_key in self.path_patterns:
            return self.path_patterns[type_key]

        # Try genre-specific pattern
        if audio_file.genre:
            genre_key = audio_file.genre.lower().replace(' ', '_')
            if genre_key in self.path_patterns:
                return self.path_patterns[genre_key]

        # Try artist-specific pattern
        if audio_file.primary_artist:
            artist_key = audio_file.primary_artist.lower().replace(' ', '_')
            if artist_key in self.path_patterns:
                return self.path_patterns[artist_key]

        # Use default pattern
        return self.default_path_pattern

    def get_supported_patterns(self) -> Dict[str, str]:
        """Return a dictionary of supported pattern examples."""
        return {
            "Default Album": "{content_type}/{artist}/{album} ({year})",
            "Flat Structure": "{artist} - {album} ({year})",
            "Genre-based": "{genre}/{artist}/{album}",
            "Year-based": "{year}/{artist}/{album}",
            "Decade-based": "{decade}/{artist}/{album}",
            "Artist-centric": "{first_letter}/{artist}/{album}",
            "With Disc Number": "{artist}/{album} (Disc {disc_number})",
            "Compilation": "Compilations/{album} ({year})",
            "Live Albums": "Live/{artist}/{date} - {location}",
            "Soundtracks": "Soundtracks/{album} ({year})",
            "Custom": "{if:albumartist}{albumartist}{else}{artist}{endif}/{album} ({year})",
        }

    def get_pattern_examples(self) -> Dict[str, str]:
        """Return examples of how patterns render with sample data."""
        examples = {
            "Simple": {
                "pattern": "{artist}/{album}",
                "result": "The Beatles/Abbey Road"
            },
            "With Year": {
                "pattern": "{artist}/{album} ({year})",
                "result": "Pink Floyd/The Dark Side of the Moon (1973)"
            },
            "With Track": {
                "pattern": "{artist}/{album}/{track_number} {title}",
                "result": "Queen/A Night at the Opera/05 Bohemian Rhapsody"
            },
            "Conditional": {
                "pattern": "{artist}/{album}{if:year} ({year}){endif}",
                "result": "Led Zeppelin/Led Zeppelin IV (1971)"
            },
            "Genre": {
                "pattern": "{genre}/{artist}/{album}",
                "result": "Rock/The Rolling Stones/Exile on Main St."
            },
            "Decade": {
                "pattern": "{decade}/{artist}/{album}",
                "result": "1980s/Michael Jackson/Thriller"
            }
        }

        return examples