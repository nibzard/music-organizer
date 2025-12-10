"""Plugin configuration system with validation."""

from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

from ..exceptions import ConfigurationError


@dataclass
class ConfigOption:
    """Definition of a configuration option."""
    name: str
    type: type
    default: Any = None
    required: bool = False
    description: str = ""
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    validator: Optional[Callable[[Any], bool]] = None

    def validate(self, value: Any) -> Any:
        """Validate and convert a value to the correct type.

        Args:
            value: Value to validate

        Returns:
            Validated value (possibly converted)

        Raises:
            ConfigurationError: If validation fails
        """
        # Check required
        if value is None:
            if self.required:
                raise ConfigurationError(f"Required option '{self.name}' is missing")
            return self.default

        # Type conversion
        try:
            if self.type == bool:
                if isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    value = bool(value)
            elif self.type == int:
                value = int(value)
            elif self.type == float:
                value = float(value)
            elif self.type == str:
                value = str(value)
            elif self.type == list:
                if isinstance(value, str):
                    # Split comma-separated values
                    value = [v.strip() for v in value.split(',')]
                elif not isinstance(value, list):
                    value = [value]
            elif self.type == dict:
                if isinstance(value, str):
                    value = json.loads(value)
                elif not isinstance(value, dict):
                    raise ConfigurationError(f"Option '{self.name}' must be a dictionary")
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Invalid type for option '{self.name}': {e}")

        # Check choices
        if self.choices is not None and value not in self.choices:
            raise ConfigurationError(
                f"Option '{self.name}' must be one of: {self.choices}, got: {value}"
            )

        # Check numeric ranges
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                raise ConfigurationError(
                    f"Option '{self.name}' must be >= {self.min_value}, got: {value}"
                )
            if self.max_value is not None and value > self.max_value:
                raise ConfigurationError(
                    f"Option '{self.name}' must be <= {self.max_value}, got: {value}"
                )

        # Custom validator
        if self.validator is not None and not self.validator(value):
            raise ConfigurationError(f"Custom validation failed for option '{self.name}'")

        return value


class PluginConfigSchema:
    """Configuration schema for a plugin."""

    def __init__(self, options: List[ConfigOption]) -> None:
        """Initialize schema with options.

        Args:
            options: List of configuration options
        """
        self._options: Dict[str, ConfigOption] = {opt.name: opt for opt in options}

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration dictionary.

        Args:
            config: Configuration to validate

        Returns:
            Validated configuration with defaults applied

        Raises:
            ConfigurationError: If validation fails
        """
        validated = {}

        # Validate each option
        for name, option in self._options.items():
            value = config.get(name)
            validated[name] = option.validate(value)

        # Check for unknown options
        for name in config:
            if name not in self._options:
                # Warn but don't fail for unknown options
                import warnings
                warnings.warn(f"Unknown configuration option: {name}")

        return validated

    def get_option(self, name: str) -> Optional[ConfigOption]:
        """Get an option definition by name.

        Args:
            name: Option name

        Returns:
            ConfigOption or None if not found
        """
        return self._options.get(name)

    def add_option(self, option: ConfigOption) -> None:
        """Add a new configuration option.

        Args:
            option: Option to add
        """
        self._options[option.name] = option

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation.

        Returns:
            Dictionary with schema information
        """
        result = {}
        for name, option in self._options.items():
            result[name] = {
                'type': option.type.__name__,
                'default': option.default,
                'required': option.required,
                'description': option.description,
                'choices': option.choices,
                'min_value': option.min_value,
                'max_value': option.max_value,
            }
        return result


# Built-in schema definitions
def create_metadata_plugin_schema() -> PluginConfigSchema:
    """Create default schema for metadata plugins."""
    return PluginConfigSchema([
        ConfigOption(
            name="enabled",
            type=bool,
            default=True,
            description="Enable this metadata plugin"
        ),
        ConfigOption(
            name="timeout",
            type=float,
            default=10.0,
            min_value=0.1,
            max_value=60.0,
            description="Timeout for network requests (seconds)"
        ),
        ConfigOption(
            name="cache_ttl",
            type=int,
            default=86400,
            min_value=0,
            description="Cache time-to-live for metadata (seconds, 0 = no cache)"
        ),
        ConfigOption(
            name="overwrite_existing",
            type=bool,
            default=False,
            description="Overwrite existing metadata"
        ),
    ])


def create_classification_plugin_schema() -> PluginConfigSchema:
    """Create default schema for classification plugins."""
    return PluginConfigSchema([
        ConfigOption(
            name="enabled",
            type=bool,
            default=True,
            description="Enable this classification plugin"
        ),
        ConfigOption(
            name="confidence_threshold",
            type=float,
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            description="Minimum confidence for classifications"
        ),
        ConfigOption(
            name="tags",
            type=list,
            default=[],
            description="List of tags to generate (empty = all supported)"
        ),
    ])


def create_output_plugin_schema() -> PluginConfigSchema:
    """Create default schema for output plugins."""
    return PluginConfigSchema([
        ConfigOption(
            name="enabled",
            type=bool,
            default=True,
            description="Enable this output plugin"
        ),
        ConfigOption(
            name="output_dir",
            type=str,
            description="Output directory for exported files"
        ),
        ConfigOption(
            name="overwrite",
            type=bool,
            default=False,
            description="Overwrite existing output files"
        ),
        ConfigOption(
            name="format_options",
            type=dict,
            default={},
            description="Format-specific options"
        ),
    ])


class ConfigValidator:
    """Validates plugin configurations."""

    def __init__(self) -> None:
        """Initialize validator."""
        self._schemas: Dict[str, PluginConfigSchema] = {}

    def register_schema(self, plugin_name: str, schema: PluginConfigSchema) -> None:
        """Register a configuration schema for a plugin.

        Args:
            plugin_name: Name of the plugin
            schema: Configuration schema
        """
        self._schemas[plugin_name] = schema

    def validate_config(self, plugin_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration for a specific plugin.

        Args:
            plugin_name: Name of the plugin
            config: Configuration to validate

        Returns:
            Validated configuration

        Raises:
            ConfigurationError: If validation fails or schema not found
        """
        if plugin_name not in self._schemas:
            # No schema registered, return as-is
            return config

        schema = self._schemas[plugin_name]
        return schema.validate(config)

    def get_schema(self, plugin_name: str) -> Optional[PluginConfigSchema]:
        """Get the schema for a plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            PluginConfigSchema or None if not found
        """
        return self._schemas.get(plugin_name)

    def load_schemas_from_plugin(self, plugin: Any) -> None:
        """Load configuration schema from a plugin if it provides one.

        Args:
            plugin: Plugin instance
        """
        if hasattr(plugin, 'get_config_schema') and callable(getattr(plugin, 'get_config_schema')):
            try:
                schema = plugin.get_config_schema()
                if isinstance(schema, PluginConfigSchema):
                    self.register_schema(plugin.info.name, schema)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load schema from plugin {plugin.info.name}: {e}")