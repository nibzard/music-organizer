"""
Plugin validation utilities.

This module provides comprehensive validation for plugin implementations,
including interface compliance, configuration validation, and performance checks.
"""

import inspect
import asyncio
import time
import traceback
from typing import Dict, Any, List, Optional, Type, Callable
from pathlib import Path

from ..base import (
    Plugin, MetadataPlugin, ClassificationPlugin,
    OutputPlugin, PathPlugin, PluginInfo
)
from ..config import ConfigValidator
from ...models.audio_file import AudioFile


class PluginValidationError(Exception):
    """Exception raised for plugin validation failures."""
    pass


class ValidationWarning:
    """Represents a validation warning that doesn't prevent plugin usage."""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message} (Suggestion: {self.suggestion})"
        return self.message


class ValidationResult:
    """Result of plugin validation."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[ValidationWarning] = []
        self.success: bool = True

    def add_error(self, message: str):
        """Add an error to the validation result."""
        self.errors.append(message)
        self.success = False

    def add_warning(self, message: str, suggestion: Optional[str] = None):
        """Add a warning to the validation result."""
        self.warnings.append(ValidationWarning(message, suggestion))

    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.success

    def get_summary(self) -> str:
        """Get a summary of validation results."""
        if self.success:
            if self.warnings:
                return f"✅ Valid with {len(self.warnings)} warning(s)"
            return "✅ Valid"
        else:
            return f"❌ Invalid with {len(self.errors)} error(s) and {len(self.warnings)} warning(s)"


class PluginValidator:
    """
    Comprehensive plugin validator.

    Validates plugin implementation against interface requirements,
    configuration schemas, and best practices.
    """

    def __init__(self, strict: bool = False):
        """
        Initialize plugin validator.

        Args:
            strict: Whether to enforce strict validation rules
        """
        self.strict = strict
        self.config_validator = ConfigValidator()

    def validate_plugin(self, plugin_class: Type[Plugin]) -> ValidationResult:
        """
        Validate a plugin class.

        Args:
            plugin_class: Plugin class to validate

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult()

        try:
            # Basic class validation
            self._validate_class_structure(plugin_class, result)

            # Interface compliance validation
            self._validate_interface_compliance(plugin_class, result)

            # Method signature validation
            self._validate_method_signatures(plugin_class, result)

            # Configuration validation
            self._validate_configuration(plugin_class, result)

            # Best practices validation
            self._validate_best_practices(plugin_class, result)

        except Exception as e:
            result.add_error(f"Validation failed with exception: {e}")

        return result

    def validate_plugin_instance(self, plugin: Plugin) -> ValidationResult:
        """
        Validate a plugin instance.

        Args:
            plugin: Plugin instance to validate

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult()

        try:
            # Test plugin info
            self._validate_plugin_info(plugin, result)

            # Test plugin lifecycle
            self._test_plugin_lifecycle(plugin, result)

            # Test plugin functionality
            self._test_plugin_functionality(plugin, result)

        except Exception as e:
            result.add_error(f"Instance validation failed: {e}")

        return result

    def _validate_class_structure(self, plugin_class: Type[Plugin], result: ValidationResult):
        """Validate basic class structure."""
        # Check if class is concrete (not abstract)
        if inspect.isabstract(plugin_class):
            result.add_error("Plugin class must not be abstract")

        # Check inheritance
        base_classes = [cls.__name__ for cls in plugin_class.__mro__]
        if 'Plugin' not in base_classes:
            result.add_error("Plugin must inherit from Plugin base class")

        # Check required attributes
        required_class_methods = ['create_plugin']
        for method in required_class_methods:
            if not hasattr(plugin_class, method):
                result.add_error(f"Plugin must have {method} method")

    def _validate_interface_compliance(self, plugin_class: Type[Plugin], result: ValidationResult):
        """Validate plugin interface compliance."""
        # Check for proper plugin type
        plugin_type = None
        if issubclass(plugin_class, MetadataPlugin):
            plugin_type = 'metadata'
        elif issubclass(plugin_class, ClassificationPlugin):
            plugin_type = 'classification'
        elif issubclass(plugin_class, OutputPlugin):
            plugin_type = 'output'
        elif issubclass(plugin_class, PathPlugin):
            plugin_type = 'path'
        else:
            result.add_error("Plugin must inherit from a specific plugin type")
            return

        # Validate type-specific abstract methods
        abstract_methods = inspect.getmembers(plugin_class, predicate=inspect.ismethod)
        abstract_property_names = [name for name, _ in abstract_methods if name.startswith('_')]

        if plugin_type == 'metadata':
            required_methods = ['enhance_metadata']
            self._check_required_methods(plugin_class, required_methods, result)
        elif plugin_type == 'classification':
            required_methods = ['classify', 'get_supported_tags']
            self._check_required_methods(plugin_class, required_methods, result)
        elif plugin_type == 'output':
            required_methods = ['export', 'get_supported_formats', 'get_file_extension']
            self._check_required_methods(plugin_class, required_methods, result)
        elif plugin_type == 'path':
            required_methods = ['generate_target_path', 'generate_filename']
            self._check_required_methods(plugin_class, required_methods, result)

    def _validate_method_signatures(self, plugin_class: Type[Plugin], result: ValidationResult):
        """Validate method signatures."""
        # Check create_plugin signature
        if hasattr(plugin_class, 'create_plugin'):
            sig = inspect.signature(plugin_class.create_plugin)
            params = list(sig.parameters.keys())
            if 'config' not in params:
                result.add_warning(
                    "create_plugin should accept a 'config' parameter",
                    "Add 'config: Optional[Dict[str, Any]] = None' parameter"
                )

        # Check info property
        if hasattr(plugin_class, 'info'):
            if not isinstance(getattr(plugin_class, 'info', None), property):
                result.add_error("info must be a property")

    def _validate_configuration(self, plugin_class: Type[Plugin], result: ValidationResult):
        """Validate plugin configuration."""
        # Check for configuration schema method
        if hasattr(plugin_class, 'get_config_schema'):
            try:
                schema = plugin_class.get_config_schema()
                if not isinstance(schema, dict):
                    result.add_error("Configuration schema must be a dictionary")
                else:
                    # Validate JSON Schema structure
                    self._validate_json_schema(schema, result)
            except Exception as e:
                result.add_error(f"Configuration schema validation failed: {e}")
        else:
            result.add_warning(
                "Plugin does not provide configuration schema",
                "Implement get_config_schema() method for better configuration validation"
            )

        # Check for default configuration method
        if not hasattr(plugin_class, 'get_default_config'):
            result.add_warning(
                "Plugin does not provide default configuration",
                "Implement get_default_config() method for better user experience"
            )

    def _validate_best_practices(self, plugin_class: Type[Plugin], result: ValidationResult):
        """Validate plugin best practices."""
        source = inspect.getsource(plugin_class)

        # Check for docstrings
        if not inspect.getdoc(plugin_class):
            result.add_warning("Plugin class should have a docstring")

        # Check for proper error handling
        if 'raise' in source and 'except' not in source:
            result.add_warning(
                "Plugin raises exceptions but may not handle them properly",
                "Add try-except blocks for error handling"
            )

        # Check for logging
        if 'import logging' not in source and 'logger' not in source:
            result.add_warning(
                "Plugin does not appear to use logging",
                "Add logging for better debugging and monitoring"
            )

        # Check for async methods
        async_methods = [name for name, method in inspect.getmembers(plugin_class, predicate=inspect.iscoroutinefunction)]
        if not async_methods:
            result.add_warning(
                "Plugin does not implement any async methods",
                "Consider using async for I/O operations"
            )

    def _validate_plugin_info(self, plugin: Plugin, result: ValidationResult):
        """Validate plugin info."""
        try:
            info = plugin.info

            # Check required fields
            if not info.name:
                result.add_error("Plugin name is required")
            if not info.version:
                result.add_error("Plugin version is required")
            if not info.description:
                result.add_error("Plugin description is required")
            if not info.author:
                result.add_error("Plugin author is required")

            # Check version format
            if info.version and not self._is_valid_version(info.version):
                result.add_warning(
                    f"Plugin version '{info.version}' may not follow semantic versioning",
                    "Use semantic versioning (e.g., '1.0.0', '1.2.3-beta.1')"
                )

        except Exception as e:
            result.add_error(f"Failed to get plugin info: {e}")

    def _test_plugin_lifecycle(self, plugin: Plugin, result: ValidationResult):
        """Test plugin lifecycle methods."""
        try:
            # Test initialization
            plugin.initialize()

            # Test enable/disable
            plugin.enable()
            if not plugin.enabled:
                result.add_error("Plugin should be enabled after calling enable()")

            plugin.disable()
            if plugin.enabled:
                result.add_error("Plugin should be disabled after calling disable()")

            # Test cleanup
            plugin.cleanup()

        except Exception as e:
            result.add_error(f"Plugin lifecycle test failed: {e}")

    def _test_plugin_functionality(self, plugin: Plugin, result: ValidationResult):
        """Test plugin-specific functionality."""
        from .mocks import create_mock_audio_file

        test_file = create_mock_audio_file()

        if isinstance(plugin, MetadataPlugin):
            self._test_metadata_plugin(plugin, test_file, result)
        elif isinstance(plugin, ClassificationPlugin):
            self._test_classification_plugin(plugin, test_file, result)
        elif isinstance(plugin, OutputPlugin):
            self._test_output_plugin(plugin, test_file, result)
        elif isinstance(plugin, PathPlugin):
            self._test_path_plugin(plugin, test_file, result)

    def _test_metadata_plugin(self, plugin: MetadataPlugin, test_file: AudioFile, result: ValidationResult):
        """Test metadata plugin functionality."""
        try:
            # Test enhance_metadata
            enhanced = asyncio.run(plugin.enhance_metadata(test_file))

            # Check if it looks like an AudioFile (duck typing)
            has_metadata = hasattr(enhanced, 'metadata')
            has_path = hasattr(enhanced, 'file_path') or hasattr(enhanced, 'path')
            if not (has_metadata and has_path):
                result.add_error("enhance_metadata must return an AudioFile-like object")

            if enhanced == test_file and plugin.enabled:
                result.add_warning("enhance_metadata did not modify the audio file")

        except Exception as e:
            result.add_error(f"Metadata plugin test failed: {e}")

    def _test_classification_plugin(self, plugin: ClassificationPlugin, test_file: AudioFile, result: ValidationResult):
        """Test classification plugin functionality."""
        try:
            # Test get_supported_tags
            tags = plugin.get_supported_tags()
            if not isinstance(tags, list) or not tags:
                result.add_error("get_supported_tags must return a non-empty list")

            # Test classify
            result_dict = asyncio.run(plugin.classify(test_file))

            if not isinstance(result_dict, dict):
                result.add_error("classify must return a dictionary")

        except Exception as e:
            result.add_error(f"Classification plugin test failed: {e}")

    def _test_output_plugin(self, plugin: OutputPlugin, test_file: AudioFile, result: ValidationResult):
        """Test output plugin functionality."""
        try:
            from pathlib import Path
            import tempfile

            # Test get_supported_formats
            formats = plugin.get_supported_formats()
            if not isinstance(formats, list) or not formats:
                result.add_error("get_supported_formats must return a non-empty list")

            # Test get_file_extension
            extension = plugin.get_file_extension()
            if not isinstance(extension, str) or not extension:
                result.add_error("get_file_extension must return a non-empty string")

            # Test export
            with tempfile.NamedTemporaryFile() as tmp:
                export_path = Path(tmp.name).with_suffix(f".{extension}")
                success = asyncio.run(plugin.export([test_file], export_path))

                if not isinstance(success, bool):
                    result.add_error("export must return a boolean")

        except Exception as e:
            result.add_error(f"Output plugin test failed: {e}")

    def _test_path_plugin(self, plugin: PathPlugin, test_file: AudioFile, result: ValidationResult):
        """Test path plugin functionality."""
        try:
            from pathlib import Path

            base_dir = Path("/test/base")

            # Test generate_target_path
            target_path = asyncio.run(plugin.generate_target_path(test_file, base_dir))
            if not isinstance(target_path, Path):
                result.add_error("generate_target_path must return a Path")

            # Test generate_filename
            filename = asyncio.run(plugin.generate_filename(test_file))
            if not isinstance(filename, str) or not filename:
                result.add_error("generate_filename must return a non-empty string")

            # Test get_supported_variables
            variables = plugin.get_supported_variables()
            if not isinstance(variables, list) or not variables:
                result.add_error("get_supported_variables must return a non-empty list")

        except Exception as e:
            result.add_error(f"Path plugin test failed: {e}")

    def _check_required_methods(self, plugin_class: Type[Plugin], required_methods: List[str], result: ValidationResult):
        """Check if required methods are implemented."""
        for method in required_methods:
            if not hasattr(plugin_class, method):
                result.add_error(f"Plugin must implement {method} method")
            elif getattr(plugin_class, method) is None:
                result.add_error(f"{method} method cannot be None")

    def _validate_json_schema(self, schema: Dict[str, Any], result: ValidationResult):
        """Validate JSON Schema structure."""
        if 'type' not in schema:
            result.add_error("Configuration schema must specify a type")

        if 'properties' in schema:
            properties = schema['properties']
            if not isinstance(properties, dict):
                result.add_error("Properties must be a dictionary")

    def _is_valid_version(self, version: str) -> bool:
        """Check if version follows semantic versioning."""
        import re
        # Simple semantic versioning check
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+(\.\d+)?)?$'
        return bool(re.match(pattern, version))


def validate_plugin_interface(plugin_class: Type[Plugin], strict: bool = False) -> ValidationResult:
    """
    Validate plugin interface compliance.

    Args:
        plugin_class: Plugin class to validate
        strict: Whether to enforce strict validation

    Returns:
        ValidationResult with validation details
    """
    validator = PluginValidator(strict=strict)
    return validator.validate_plugin(plugin_class)


def validate_plugin_config(
    plugin_class: Type[Plugin],
    config: Dict[str, Any],
    strict: bool = False
) -> ValidationResult:
    """
    Validate plugin configuration.

    Args:
        plugin_class: Plugin class to validate config for
        config: Configuration dictionary
        strict: Whether to enforce strict validation

    Returns:
        ValidationResult with validation details
    """
    result = ValidationResult()

    try:
        # Get configuration schema
        if hasattr(plugin_class, 'get_config_schema'):
            schema = plugin_class.get_config_schema()

            # Use ConfigValidator to validate
            validator = ConfigValidator()
            validation_result = validator.validate(config, schema)

            if not validation_result.valid:
                result.errors.extend([str(e) for e in validation_result.errors])
                result.success = False

        # Check against default configuration
        if hasattr(plugin_class, 'get_default_config'):
            default_config = plugin_class.get_default_config()

            # Check for missing required fields
            for key, value in default_config.items():
                if key not in config and strict:
                    result.add_error(f"Missing required configuration field: {key}")
                elif key not in config:
                    result.add_warning(
                        f"Missing configuration field: {key}",
                        f"Consider adding '{key}': {repr(value)} to your configuration"
                    )

    except Exception as e:
        result.add_error(f"Configuration validation failed: {e}")

    return result


def validate_plugin_performance(
    plugin_class: Type[Plugin],
    config: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0
) -> ValidationResult:
    """
    Validate plugin performance.

    Args:
        plugin_class: Plugin class to validate
        config: Optional configuration for plugin
        timeout: Maximum allowed time for operations

    Returns:
        ValidationResult with performance validation details
    """
    result = ValidationResult()

    try:
        # Create plugin instance
        if hasattr(plugin_class, 'create_plugin'):
            plugin = plugin_class.create_plugin(config)
        else:
            plugin = plugin_class(config)

        from .mocks import create_mock_audio_file
        test_file = create_mock_audio_file()

        # Test initialization performance
        start_time = time.time()
        plugin.initialize()
        init_time = time.time() - start_time

        if init_time > timeout:
            result.add_warning(
                f"Plugin initialization took {init_time:.2f}s (threshold: {timeout}s)",
                "Consider optimizing initialization code"
            )

        # Test method performance
        if isinstance(plugin, MetadataPlugin):
            result = _test_method_performance(
                plugin, 'enhance_metadata', test_file, timeout, result
            )
        elif isinstance(plugin, ClassificationPlugin):
            result = _test_method_performance(
                plugin, 'classify', test_file, timeout, result
            )

        # Cleanup
        plugin.cleanup()

    except Exception as e:
        result.add_error(f"Performance validation failed: {e}")

    return result


def _test_method_performance(
    plugin: Plugin,
    method_name: str,
    test_arg: Any,
    timeout: float,
    result: ValidationResult
) -> ValidationResult:
    """Test performance of a specific method."""
    try:
        method = getattr(plugin, method_name)

        start_time = time.time()
        if asyncio.iscoroutinefunction(method):
            asyncio.run(method(test_arg))
        else:
            method(test_arg)
        execution_time = time.time() - start_time

        if execution_time > timeout:
            result.add_warning(
                f"{method_name} took {execution_time:.2f}s (threshold: {timeout}s)",
                f"Consider optimizing {method_name} method"
            )

    except Exception as e:
        result.add_error(f"Performance test for {method_name} failed: {e}")

    return result