"""
Plugin testing utilities and framework.

This module provides utilities for testing music-organizer plugins,
including mock objects, test fixtures, validation utilities, and performance testing.
"""

from .mocks import (
    MockAudioFile,
    MockPluginManager,
    MockExternalAPI,
    MockMetadataPlugin,
    MockClassificationPlugin,
    MockOutputPlugin,
    MockPathPlugin,
    create_mock_audio_file,
    create_mock_plugin
)

from .fixtures import (
    AudioTestFixture,
    MetadataTestFixture,
    ClassificationTestFixture,
    get_test_audio_files,
    get_test_metadata
)

from .validators import (
    PluginValidator,
    validate_plugin_interface,
    validate_plugin_config,
    validate_plugin_performance
)

from .performance import (
    PerformanceProfiler,
    benchmark_plugin,
    profile_memory_usage
)

__all__ = [
    # Mock objects
    'MockAudioFile',
    'MockPluginManager',
    'MockExternalAPI',
    'MockMetadataPlugin',
    'MockClassificationPlugin',
    'MockOutputPlugin',
    'MockPathPlugin',
    'create_mock_audio_file',
    'create_mock_plugin',

    # Test fixtures
    'AudioTestFixture',
    'MetadataTestFixture',
    'ClassificationTestFixture',
    'get_test_audio_files',
    'get_test_metadata',

    # Validators
    'PluginValidator',
    'validate_plugin_interface',
    'validate_plugin_config',
    'validate_plugin_performance',

    # Performance testing
    'PerformanceProfiler',
    'benchmark_plugin',
    'profile_memory_usage'
]