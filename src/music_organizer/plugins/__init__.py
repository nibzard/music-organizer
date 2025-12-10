"""Plugin system for the music organizer."""

from .base import Plugin, MetadataPlugin, ClassificationPlugin, OutputPlugin
from .manager import PluginManager
from .hooks import PluginHooks

__all__ = [
    'Plugin',
    'MetadataPlugin',
    'ClassificationPlugin',
    'OutputPlugin',
    'PluginManager',
    'PluginHooks',
]