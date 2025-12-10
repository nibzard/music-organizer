"""Plugin manager for discovering and loading plugins."""

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import json
import logging

from .base import Plugin, PluginInfo, PluginFactory, MetadataPlugin, ClassificationPlugin, OutputPlugin
from .hooks import PluginHooks, HookEvent
from .config import (
    ConfigValidator,
    PluginConfigSchema,
    create_metadata_plugin_schema,
    create_classification_plugin_schema,
    create_output_plugin_schema
)

logger = logging.getLogger(__name__)


class PluginManager:
    """Manages plugin discovery, loading, and lifecycle."""

    def __init__(self, plugin_dirs: Optional[List[Path]] = None) -> None:
        """Initialize plugin manager.

        Args:
            plugin_dirs: List of directories to search for plugins
        """
        self.plugin_dirs = plugin_dirs or []
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}
        self.hooks = PluginHooks()
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        self.config_validator = ConfigValidator()

        # Add default plugin directory
        current_dir = Path(__file__).parent
        default_plugins_dir = current_dir / "builtins"
        if default_plugins_dir.exists():
            self.plugin_dirs.append(default_plugins_dir)

    def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugin directories.

        Returns:
            List of plugin names found
        """
        discovered = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue

            # Look for .py files
            for py_file in plugin_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                plugin_name = py_file.stem
                discovered.append(plugin_name)
                self._load_plugin_from_file(py_file, plugin_name)

            # Look for plugin packages
            for subdir in plugin_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("_"):
                    init_file = subdir / "__init__.py"
                    if init_file.exists():
                        plugin_name = subdir.name
                        discovered.append(plugin_name)
                        self._load_plugin_from_directory(subdir, plugin_name)

        return discovered

    def _load_plugin_from_file(self, file_path: Path, plugin_name: str) -> None:
        """Load a plugin from a Python file.

        Args:
            file_path: Path to the plugin file
            plugin_name: Name to register the plugin under
        """
        try:
            spec = importlib.util.spec_from_file_location(plugin_name, file_path)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load spec for plugin {plugin_name}")
                return

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Plugin) and obj != Plugin:
                    self._plugin_classes[plugin_name] = obj
                    logger.debug(f"Loaded plugin class: {plugin_name}.{name}")

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name} from {file_path}: {e}")

    def _load_plugin_from_directory(self, dir_path: Path, plugin_name: str) -> None:
        """Load a plugin from a directory package.

        Args:
            dir_path: Path to the plugin directory
            plugin_name: Name to register the plugin under
        """
        try:
            # Try to import as a package
            spec = importlib.util.spec_from_file_location(
                plugin_name,
                dir_path / "__init__.py"
            )
            if spec is None or spec.loader is None:
                return

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for a main plugin class or factory function
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Plugin) and obj != Plugin:
                    self._plugin_classes[plugin_name] = obj
                elif callable(obj) and hasattr(obj, '__annotations__'):
                    # Check if it's a factory function
                    return_annotation = obj.__annotations__.get('return')
                    if return_annotation and (
                        return_annotation is Plugin or
                        (hasattr(return_annotation, '__origin__') and
                         return_annotation.__origin__ in (Plugin, Type[Plugin]))
                    ):
                        self._plugin_classes[plugin_name] = obj

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name} from {dir_path}: {e}")

    def load_plugin(
        self,
        plugin_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Plugin]:
        """Load and initialize a plugin.

        Args:
            plugin_name: Name of the plugin to load
            config: Optional configuration for the plugin

        Returns:
            Loaded plugin instance or None if failed
        """
        if plugin_name in self._plugins:
            return self._plugins[plugin_name]

        if plugin_name not in self._plugin_classes:
            logger.error(f"Plugin {plugin_name} not found")
            return None

        try:
            plugin_class = self._plugin_classes[plugin_name]

            # Get or create config
            if config is None:
                config = self.get_plugin_config(plugin_name)

            # Validate config if schema is available
            validated_config = self.config_validator.validate_config(plugin_name, config or {})

            # Handle factory functions
            if callable(plugin_class) and not inspect.isclass(plugin_class):
                plugin = plugin_class(validated_config)
            else:
                plugin = plugin_class(validated_config)

            # Validate plugin
            if not isinstance(plugin, Plugin):
                logger.error(f"Plugin {plugin_name} is not a valid Plugin instance")
                return None

            # Register default schema based on plugin type if not provided
            if plugin_name not in self.config_validator._schemas:
                if isinstance(plugin, MetadataPlugin):
                    self.config_validator.register_schema(plugin_name, create_metadata_plugin_schema())
                elif isinstance(plugin, ClassificationPlugin):
                    self.config_validator.register_schema(plugin_name, create_classification_plugin_schema())
                elif isinstance(plugin, OutputPlugin):
                    self.config_validator.register_schema(plugin_name, create_output_plugin_schema())

            # Load schema from plugin if it provides one
            self.config_validator.load_schemas_from_plugin(plugin)

            # Initialize plugin
            plugin.initialize()
            self._plugins[plugin_name] = plugin

            # Store validated config for later
            self._config_cache[plugin_name] = validated_config

            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return plugin

        except Exception as e:
            logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
            return None

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin.

        Args:
            plugin_name: Name of the plugin to unload

        Returns:
            True if plugin was unloaded
        """
        if plugin_name not in self._plugins:
            return False

        try:
            plugin = self._plugins[plugin_name]
            plugin.cleanup()
            del self._plugins[plugin_name]
            if plugin_name in self._config_cache:
                del self._config_cache[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get a loaded plugin by name.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin instance or None if not loaded
        """
        return self._plugins.get(plugin_name)

    def list_plugins(self) -> Dict[str, PluginInfo]:
        """List all available plugins with their info.

        Returns:
            Dictionary of plugin name -> PluginInfo
        """
        plugins_info = {}

        # Include loaded plugins
        for name, plugin in self._plugins.items():
            plugins_info[name] = plugin.info

        # Include discovered but unloaded plugins
        for name, plugin_class in self._plugin_classes.items():
            if name not in self._plugins:
                try:
                    # Create temporary instance to get info
                    temp_instance = plugin_class()
                    plugins_info[name] = temp_instance.info
                    if hasattr(temp_instance, 'cleanup'):
                        temp_instance.cleanup()
                except Exception as e:
                    logger.warning(f"Could not get info for plugin {name}: {e}")

        return plugins_info

    def get_plugins_by_type(self, plugin_type: Type[Plugin]) -> List[Plugin]:
        """Get all loaded plugins of a specific type.

        Args:
            plugin_type: The plugin type to filter by

        Returns:
            List of plugins of the specified type
        """
        return [
            plugin for plugin in self._plugins.values()
            if isinstance(plugin, plugin_type) and plugin.enabled
        ]

    async def execute_hook(self, event: HookEvent, **data) -> None:
        """Execute a hook event.

        Args:
            event: The hook event to execute
            **data: Data to pass to hook handlers
        """
        await self.hooks.emit(event, **data)

    def load_config_file(self, config_path: Path) -> None:
        """Load plugin configuration from a JSON file.

        Args:
            config_path: Path to the configuration file
        """
        if not config_path.exists():
            logger.warning(f"Plugin config file not found: {config_path}")
            return

        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            if 'plugins' in config_data:
                for plugin_name, plugin_config in config_data['plugins'].items():
                    self._config_cache[plugin_name] = plugin_config

        except Exception as e:
            logger.error(f"Failed to load plugin config from {config_path}: {e}")

    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin configuration dictionary
        """
        return self._config_cache.get(plugin_name, {})

    def validate_all_configs(self) -> Dict[str, List[str]]:
        """Validate all cached plugin configurations.

        Returns:
            Dictionary of plugin name -> list of validation errors
        """
        errors = {}

        for plugin_name, config in self._config_cache.items():
            try:
                self.config_validator.validate_config(plugin_name, config)
            except Exception as e:
                errors[plugin_name] = [str(e)]

        return errors

    def get_plugin_schema(self, plugin_name: str) -> Optional[PluginConfigSchema]:
        """Get the configuration schema for a plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Configuration schema or None if not found
        """
        return self.config_validator.get_schema(plugin_name)

    def save_config(self, config_path: Path) -> None:
        """Save all plugin configurations to a file.

        Args:
            config_path: Path to save configuration file
        """
        config_data = {
            "plugins": self._config_cache,
            "schemas": {
                name: schema.to_dict()
                for name, schema in self.config_validator._schemas.items()
            }
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)

        logger.info(f"Saved plugin configuration to {config_path}")