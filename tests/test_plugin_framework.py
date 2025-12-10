"""
Comprehensive tests for plugin development utilities.

This test suite validates the plugin testing framework, including
mocks, fixtures, validators, and performance testing utilities.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from music_organizer.plugins.testing import (
    MockAudioFile, MockMetadataPlugin, MockClassificationPlugin,
    MockOutputPlugin, MockPathPlugin, MockPluginManager,
    create_mock_audio_file, create_mock_plugin
)
from music_organizer.plugins.testing.fixtures import (
    AudioTestFixture, MetadataTestFixture, ClassificationTestFixture,
    get_test_audio_files, get_test_metadata
)
from music_organizer.plugins.testing.validators import (
    PluginValidator, validate_plugin_interface,
    validate_plugin_config, validate_plugin_performance
)
from music_organizer.plugins.testing.performance import (
    PerformanceProfiler, benchmark_plugin, profile_memory_usage
)
from music_organizer.plugins.base import (
    MetadataPlugin, ClassificationPlugin, OutputPlugin, PathPlugin
)
from music_organizer.models.audio_file import AudioFile


class TestMockAudioFile:
    """Test cases for MockAudioFile."""

    def test_creation(self):
        """Test mock audio file creation."""
        file = MockAudioFile()
        assert file.path == Path("/test/mock_file.mp3")
        assert file.duration == 180.5
        assert file.bitrate == 320
        assert file.format == "mp3"

    def test_custom_creation(self):
        """Test mock audio file creation with custom parameters."""
        metadata = {"title": "Test Song", "artist": "Test Artist"}
        file = MockAudioFile(
            path="/custom/path.flac",
            metadata=metadata,
            duration=240.0,
            bitrate=1000,
            format="flac"
        )
        assert file.path == Path("/custom/path.flac")
        assert file.metadata == metadata
        assert file.duration == 240.0
        assert file.bitrate == 1000
        assert file.format == "flac"

    def test_copy(self):
        """Test copying mock audio file."""
        original = MockAudioFile(metadata={"title": "Original"})
        copy = original.copy()
        assert copy.metadata == original.metadata
        assert copy.path == original.path
        assert copy is not original

    def test_get_target_path(self):
        """Test getting target path for organization."""
        file = MockAudioFile(
            metadata={"artist": "Test Artist", "album": "Test Album"}
        )
        base_dir = Path("/music")
        target = file.get_target_path(base_dir)
        expected = base_dir / "Test Artist" / "Test Album" / file.path.name
        assert target == expected

    def test_equality(self):
        """Test equality comparison."""
        file1 = MockAudioFile(
            path="/test/song.mp3",
            metadata={"title": "Song"},
            duration=180.0
        )
        file2 = MockAudioFile(
            path="/test/song.mp3",
            metadata={"title": "Song"},
            duration=180.1  # Small difference
        )
        file3 = MockAudioFile(
            path="/test/different.mp3",
            metadata={"title": "Song"},
            duration=180.0
        )

        assert file1 == file2  # Within tolerance
        assert file1 != file3


class TestMockPlugins:
    """Test cases for mock plugins."""

    def test_mock_metadata_plugin(self):
        """Test mock metadata plugin."""
        plugin = MockMetadataPlugin(
            enhance_result={"genre": "Rock"},
            should_fail=False
        )

        assert plugin.info.name == "mock-metadata-plugin"
        assert plugin.enabled

        # Test enhancement
        audio_file = create_mock_audio_file()
        result = asyncio.run(plugin.enhance_metadata(audio_file))
        assert result.metadata.get("genre") == "Rock"
        assert result.metadata.get("enhanced_by") == plugin.name

    def test_mock_metadata_plugin_failure(self):
        """Test mock metadata plugin with failure."""
        plugin = MockMetadataPlugin(should_fail=True)
        audio_file = create_mock_audio_file()

        with pytest.raises(Exception, match="Mock failure"):
            asyncio.run(plugin.enhance_metadata(audio_file))

    def test_mock_classification_plugin(self):
        """Test mock classification plugin."""
        plugin = MockClassificationPlugin(
            classification_result={"genre": "Jazz", "confidence": 0.9},
            supported_tags=["genre", "confidence"]
        )

        assert "genre" in plugin.get_supported_tags()
        assert "confidence" in plugin.get_supported_tags()

        audio_file = create_mock_audio_file()
        result = asyncio.run(plugin.classify(audio_file))
        assert result["genre"] == "Jazz"
        assert result["confidence"] == 0.9

    def test_mock_output_plugin(self):
        """Test mock output plugin."""
        plugin = MockOutputPlugin(
            supported_formats=["m3u", "json"],
            file_extension="m3u"
        )

        assert "m3u" in plugin.get_supported_formats()
        assert "json" in plugin.get_supported_formats()
        assert plugin.get_file_extension() == "m3u"

        audio_files = [create_mock_audio_file()]
        success = asyncio.run(plugin.export(audio_files, Path("/test/playlist.m3u")))
        assert success
        assert len(plugin.exported_files) == 1

    def test_mock_path_plugin(self):
        """Test mock path plugin."""
        plugin = MockPathPlugin(
            path_pattern="{artist}/{year}",
            filename_pattern="{track_number:02d} - {title}"
        )

        audio_file = create_mock_audio_file(
            metadata={"artist": "Test Artist", "year": "2023", "track_number": "5", "title": "Test Song"}
        )

        base_dir = Path("/music")
        target_path = asyncio.run(plugin.generate_target_path(audio_file, base_dir))
        filename = asyncio.run(plugin.generate_filename(audio_file))

        assert target_path == base_dir / "Test Artist" / "2023"
        assert filename == "05 - Test Song.mp3"


class TestPluginManager:
    """Test cases for MockPluginManager."""

    def test_plugin_loading(self):
        """Test loading plugins."""
        manager = MockPluginManager()
        plugin = MockMetadataPlugin()

        manager.load_plugin(plugin)
        assert plugin.info.name in manager.plugins
        assert plugin.info.name in manager.load_history

    def test_plugin_retrieval(self):
        """Test retrieving loaded plugins."""
        manager = MockPluginManager()
        plugin = MockMetadataPlugin(name="test-plugin")

        manager.load_plugin(plugin)
        retrieved = manager.get_plugin("test-plugin")
        assert retrieved is plugin

    def test_metadata_processing(self):
        """Test metadata processing through manager."""
        manager = MockPluginManager()
        plugin = MockMetadataPlugin(
            enhance_result={"genre": "Test Genre"}
        )

        manager.load_plugin(plugin)
        audio_file = create_mock_audio_file()

        processed = asyncio.run(manager.process_metadata(audio_file))
        assert processed.metadata.get("genre") == "Test Genre"
        assert "metadata:test-plugin" in manager.execution_history

    def test_classification_processing(self):
        """Test classification through manager."""
        manager = MockPluginManager()
        plugin = MockClassificationPlugin(
            classification_result={"mood": "Happy"}
        )

        manager.load_plugin(plugin)
        audio_file = create_mock_audio_file()

        results = asyncio.run(manager.classify_file(audio_file))
        assert results.get("mood") == "Happy"
        assert "classify:test-plugin" in manager.execution_history


class TestFixtures:
    """Test cases for test fixtures."""

    def test_audio_test_fixture(self):
        """Test audio test fixture."""
        fixture = AudioTestFixture()

        assert fixture.complete_file.metadata.get("title") == "Complete Song"
        assert fixture.minimal_file.metadata.get("title") == "Minimal Song"
        assert fixture.compilation_file.metadata.get("compilation") is True

        all_files = fixture.get_all_files()
        assert len(all_files) >= 7  # Basic files + genre files

        fixture.cleanup()

    def test_metadata_test_fixture(self):
        """Test metadata test fixture."""
        fixture = MetadataTestFixture()

        # Test missing year case
        case = fixture.get_test_case("missing_year")
        assert "year" not in case["input"].metadata
        assert "year" in case["expected_enhancement"]

        # Test all test cases
        all_cases = fixture.get_all_test_cases()
        assert len(all_cases) >= 5  # Basic test cases

    def test_classification_test_fixture(self):
        """Test classification test fixture."""
        fixture = ClassificationTestFixture()

        rock_song = fixture.get_test_file("rock_song")
        assert rock_song.metadata.get("genre") == "Rock"

        expected = fixture.get_expected_classification("rock_song")
        assert expected.get("genre") == "Rock"

        all_files = fixture.get_all_files()
        assert len(all_files) >= 5

    def test_get_test_audio_files(self):
        """Test getting test audio files."""
        files = get_test_audio_files(5)
        assert len(files) == 5

        for i, file in enumerate(files):
            assert file.metadata.get("title") == f"Test Song {i + 1}"
            assert file.metadata.get("artist") is not None

    def test_get_test_metadata(self):
        """Test getting test metadata."""
        metadata = get_test_metadata()

        assert "basic" in metadata
        assert "complete" in metadata
        assert "minimal" in metadata
        assert "problematic" in metadata
        assert "international" in metadata

        # Check complete metadata has many fields
        complete = metadata["complete"]
        assert len(complete) >= 20  # Should have many metadata fields


class TestValidators:
    """Test cases for plugin validators."""

    def test_plugin_validator_metadata_plugin(self):
        """Test validation of metadata plugin."""
        class TestMetadataPlugin(MetadataPlugin):
            @property
            def info(self):
                from ..base import PluginInfo
                return PluginInfo(
                    name="test",
                    version="1.0.0",
                    description="Test",
                    author="Test"
                )

            def initialize(self):
                pass

            def cleanup(self):
                pass

            async def enhance_metadata(self, audio_file):
                return audio_file

            @staticmethod
            def create_plugin(config=None):
                return TestMetadataPlugin(config)

        validator = PluginValidator()
        result = validator.validate_plugin(TestMetadataPlugin)

        assert result.is_valid()

    def test_plugin_validator_missing_methods(self):
        """Test validation of plugin with missing methods."""
        class IncompleteMetadataPlugin(MetadataPlugin):
            @property
            def info(self):
                from ..base import PluginInfo
                return PluginInfo(
                    name="incomplete",
                    version="1.0.0",
                    description="Incomplete",
                    author="Test"
                )

            def initialize(self):
                pass

            def cleanup(self):
                pass
            # Missing enhance_metadata method

        validator = PluginValidator()
        result = validator.validate_plugin(IncompleteMetadataPlugin)

        assert not result.is_valid()
        assert any("enhance_metadata" in error for error in result.errors)

    def test_validate_plugin_interface(self):
        """Test plugin interface validation function."""
        # This would test the actual validation function
        # For now, we'll test the function exists and works
        result = validate_plugin_interface(MockMetadataPlugin)
        assert result is not None

    def test_validate_plugin_config(self):
        """Test plugin configuration validation."""
        config = {"timeout": 30, "cache_enabled": True}
        result = validate_plugin_config(MockMetadataPlugin, config)
        assert result is not None


class TestPerformance:
    """Test cases for performance testing utilities."""

    def test_performance_profiler(self):
        """Test performance profiler."""
        profiler = PerformanceProfiler()

        with profiler.profile():
            # Simulate some work
            import time
            time.sleep(0.01)

        metrics = profiler.get_current_metrics()
        assert metrics.execution_time > 0.01
        assert metrics.success

    def test_method_measurement(self):
        """Test measuring method performance."""
        plugin = MockMetadataPlugin()
        audio_file = create_mock_audio_file()

        profiler = PerformanceProfiler()
        metrics = profiler.measure_method(plugin, "enhance_metadata", audio_file)

        assert metrics.execution_time >= 0
        assert metrics.success
        assert metrics.error_message is None

    def test_benchmark_plugin(self):
        """Test plugin benchmarking."""
        plugin = MockMetadataPlugin()
        test_files = get_test_audio_files(3)

        results = benchmark_plugin(plugin, test_files, iterations=2)

        assert "enhance_metadata" in results
        assert results["enhance_metadata"].avg_time >= 0
        assert results["enhance_metadata"].iterations == 6  # 3 files * 2 iterations

    def test_memory_profiling(self):
        """Test memory profiling."""
        from music_organizer.plugins.testing.performance import MemoryProfiler

        profiler = MemoryProfiler()

        with profiler.track_memory():
            # Allocate some memory
            data = [i for i in range(1000)]

        memory_trend = profiler.get_memory_trend()
        assert len(memory_trend) > 0


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def test_create_mock_audio_file(self):
        """Test creating mock audio file with factory."""
        file = create_mock_audio_file()
        assert isinstance(file, MockAudioFile)
        assert file.metadata.get("title") == "Mock Song"

    def test_create_mock_audio_file_custom(self):
        """Test creating custom mock audio file."""
        file = create_mock_audio_file(
            path="/custom/test.flac",
            metadata={"title": "Custom Title"},
            duration=200.0
        )
        assert file.path == Path("/custom/test.flac")
        assert file.metadata.get("title") == "Custom Title"
        assert file.duration == 200.0

    def test_create_mock_plugin(self):
        """Test creating mock plugins with factory."""
        # Test metadata plugin
        metadata_plugin = create_mock_plugin("metadata")
        assert isinstance(metadata_plugin, MetadataPlugin)

        # Test classification plugin
        classification_plugin = create_mock_plugin("classification")
        assert isinstance(classification_plugin, ClassificationPlugin)

        # Test output plugin
        output_plugin = create_mock_plugin("output")
        assert isinstance(output_plugin, OutputPlugin)

        # Test path plugin
        path_plugin = create_mock_plugin("path")
        assert isinstance(path_plugin, PathPlugin)

        # Test generic plugin
        generic_plugin = create_mock_plugin("generic")
        assert hasattr(generic_plugin, "info")


class TestIntegration:
    """Integration tests for the plugin testing framework."""

    def test_complete_testing_workflow(self):
        """Test a complete plugin testing workflow."""
        # 1. Create test data
        test_files = get_test_audio_files(5)

        # 2. Create plugin to test
        plugin = MockMetadataPlugin(
            enhance_result={"test_field": "test_value"}
        )

        # 3. Validate plugin
        validator = PluginValidator()
        validation_result = validator.validate_plugin_instance(plugin)
        assert validation_result.is_valid()

        # 4. Benchmark performance
        benchmark_results = benchmark_plugin(plugin, test_files, iterations=2)
        assert "enhance_metadata" in benchmark_results

        # 5. Profile memory usage
        memory_profile = profile_memory_usage(plugin, "enhance_metadata", test_files)
        assert "method" in memory_profile
        assert "files_processed" in memory_profile

    def test_plugin_with_real_metadata(self):
        """Test plugin with realistic metadata scenarios."""
        fixture = AudioTestFixture()
        plugin = MockMetadataPlugin()

        # Test various file types
        test_cases = [
            fixture.complete_file,
            fixture.minimal_file,
            fixture.compilation_file,
            fixture.special_chars_file
        ]

        for test_file in test_cases:
            enhanced = asyncio.run(plugin.enhance_metadata(test_file))
            assert enhanced.metadata.get("enhanced_by") == plugin.name

        fixture.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])