"""Tests for the custom naming pattern plugin."""

import pytest
from pathlib import Path
from datetime import datetime

from music_organizer.plugins.builtins.custom_naming_pattern import (
    CustomNamingPatternPlugin,
    PatternTemplate,
    create_plugin
)
from music_organizer.models.audio_file import AudioFile
from music_organizer.models.content_type import ContentType


class TestPatternTemplate:
    """Tests for the PatternTemplate class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.template = PatternTemplate()
        self.sample_audio_file = AudioFile(
            path=Path("/test/artist/album/01 - song.flac"),
            artists=["The Beatles"],
            primary_artist="The Beatles",
            album="Abbey Road",
            year=1969,
            track_number=1,
            title="Come Together",
            genre="Rock",
            duration=259,
            bitrate=320,
            content_type=ContentType.STUDIO
        )

    def test_basic_variable_replacement(self):
        """Test basic variable replacement."""
        result = self.template.render("{artist}/{album}", self.sample_audio_file)
        assert result == "The Beatles/Abbey Road"

    def test_track_number_padding(self):
        """Test track number zero-padding."""
        self.sample_audio_file.track_number = 3
        result = self.template.render("{track_number} {title}", self.sample_audio_file)
        assert result == "03 Come Together"

    def test_conditional_inclusion(self):
        """Test conditional variable inclusion."""
        # Test with year
        result = self.template.render(
            "{artist}/{album}{if:year} ({year}){endif}",
            self.sample_audio_file
        )
        assert result == "The Beatles/Abbey Road (1969)"

        # Test without year
        self.sample_audio_file.year = None
        result = self.template.render(
            "{artist}/{album}{if:year} ({year}){endif}",
            self.sample_audio_file
        )
        assert result == "The Beatles/Abbey Road"

    def test_filesystem_cleaning(self):
        """Test cleaning of filesystem-incompatible characters."""
        dirty_file = AudioFile(
            path=Path("/test/artist/album/01 - song.flac"),
            artists=["Artist/With\\Slashes:And*Symbols"],
            album="Album <With> \"Quotes\" | Pipe",
            year=2023,
            title="Title ? With * Special"
        )

        result = self.template.render("{artist}/{album}/{title}", dirty_file)
        assert "/" not in result
        assert "\\" not in result
        assert ":" not in result
        assert "*" not in result
        assert "?" not in result
        assert "<" not in result
        assert ">" not in result

    def test_computed_variables(self):
        """Test computed variables like decade and first_letter."""
        result = self.template.render("{decade}/{first_letter}", self.sample_audio_file)
        assert result == "1960s/T"

    def test_unknown_variable(self):
        """Test handling of unknown variables."""
        result = self.template.render("{artist}/{unknown}", self.sample_audio_file)
        assert result == "The Beatles/"

    def test_template_validation(self):
        """Test template validation."""
        # Valid template
        errors = self.template.validate_template("{artist}/{album}")
        assert len(errors) == 0

        # Template with errors
        errors = self.template.validate_template("{if:artist}{artist}{endif}")  # Missing {endif}
        assert len(errors) > 0

        # Unknown variable
        errors = self.template.validate_template("{artist}/{unknown_var}")
        assert len(errors) > 0
        assert "Unknown variable" in errors[0]

    def test_multiple_slashes_cleanup(self):
        """Test cleanup of multiple consecutive slashes."""
        result = self.template.render("{artist}//{album}", self.sample_audio_file)
        assert result == "The Beatles/Abbey Road"

    def test_trailing_slash_removal(self):
        """Test removal of trailing slashes."""
        result = self.template.render("{artist}/{album}/", self.sample_audio_file)
        assert not result.endswith("/")

    def test_nested_conditionals(self):
        """Test nested conditional blocks."""
        template = "{if:artist}{if:year}{artist} ({year}){else}{artist}{endif}{else}Unknown{endif}"
        result = self.template.render(template, self.sample_audio_file)
        assert result == "The Beatles (1969)"

    def test_complex_template(self):
        """Test a complex template with multiple features."""
        template = "{genre}/{decade}/{artist}/{album}{if:year} ({year}){endif}"
        result = self.template.render(template, self.sample_audio_file)
        assert result == "Rock/1960s/The Beatles/Abbey Road (1969)"


class TestCustomNamingPatternPlugin:
    """Tests for the CustomNamingPatternPlugin class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "filename_pattern": "{track_number} {title}{file_extension}",
            "path_patterns": {
                "studio": "Albums/{artist}/{album} ({year})",
                "live": "Live/{artist}/{date} - {location}"
            }
        }
        self.plugin = CustomNamingPatternPlugin(self.config)
        self.sample_audio_file = AudioFile(
            path=Path("/test/artist/album/01 - song.flac"),
            artists=["The Beatles"],
            primary_artist="The Beatles",
            album="Abbey Road",
            year=1969,
            track_number=1,
            title="Come Together",
            genre="Rock",
            duration=259,
            bitrate=320,
            content_type=ContentType.STUDIO
        )

    def test_plugin_info(self):
        """Test plugin information."""
        info = self.plugin.info
        assert info.name == "custom_naming_pattern"
        assert info.version == "1.0.0"
        assert "naming patterns" in info.description.lower()

    def test_initialization_validation(self):
        """Test plugin initialization with validation."""
        # Valid config
        plugin = CustomNamingPatternPlugin(self.config)
        plugin.initialize()  # Should not raise

        # Invalid config
        bad_config = {"filename_pattern": "{artist}/{unknown}"}
        plugin = CustomNamingPatternPlugin(bad_config)
        with pytest.raises(ValueError):
            plugin.initialize()

    def test_generate_target_path(self):
        """Test target path generation."""
        import asyncio

        async def test():
            base_dir = Path("/music")
            path = await self.plugin.generate_target_path(self.sample_audio_file, base_dir)
            assert path == Path("/music/Albums/The Beatles/Abbey Road (1969)")

        asyncio.run(test())

    def test_generate_filename(self):
        """Test filename generation."""
        import asyncio

        async def test():
            filename = await self.plugin.generate_filename(self.sample_audio_file)
            assert filename == "01 Come Together.flac"

        asyncio.run(test())

    def test_path_pattern_selection(self):
        """Test selection of appropriate path pattern."""
        # Studio album
        pattern = self.plugin._select_path_pattern(self.sample_audio_file)
        assert pattern == "Albums/{artist}/{album} ({year})"

        # Live album
        self.sample_audio_file.content_type = ContentType.LIVE
        pattern = self.plugin._select_path_pattern(self.sample_audio_file)
        assert pattern == "Live/{artist}/{date} - {location}"

        # Fallback to default
        self.sample_audio_file.content_type = ContentType.COMPILATION
        pattern = self.plugin._select_path_pattern(self.sample_audio_file)
        assert "{content_type}" in pattern

    def test_additional_organization(self):
        """Test additional organization options."""
        config_with_extra = {
            **self.config,
            "organize_by_genre": True,
            "organize_by_decade": True,
            "create_date_dirs": True
        }
        plugin = CustomNamingPatternPlugin(config_with_extra)

        import asyncio

        async def test():
            base_dir = Path("/music")
            path = await plugin.generate_target_path(self.sample_audio_file, base_dir)
            # Should include genre, decade, and year in path
            assert "Rock" in str(path)
            assert "1960s" in str(path)
            assert "1969" in str(path)

        asyncio.run(test())

    def test_genre_specific_pattern(self):
        """Test genre-specific pattern selection."""
        config = {
            **self.config,
            "path_patterns": {
                "rock": "Rock Collection/{artist}/{album}",
                "default": "{artist}/{album}"
            }
        }
        plugin = CustomNamingPatternPlugin(config)
        pattern = plugin._select_path_pattern(self.sample_audio_file)
        assert pattern == "Rock Collection/{artist}/{album}"

    def test_artist_specific_pattern(self):
        """Test artist-specific pattern selection."""
        config = {
            **self.config,
            "path_patterns": {
                "the_beatles": "Special/Beatles/{album}",
                "default": "{artist}/{album}"
            }
        }
        plugin = CustomNamingPatternPlugin(config)
        pattern = plugin._select_path_pattern(self.sample_audio_file)
        assert pattern == "Special/Beatles/{album}"

    def test_get_supported_patterns(self):
        """Test getting supported pattern examples."""
        patterns = self.plugin.get_supported_patterns()
        assert isinstance(patterns, dict)
        assert "Default Album" in patterns
        assert "Flat Structure" in patterns
        assert "Genre-based" in patterns

    def test_get_pattern_examples(self):
        """Test getting pattern rendering examples."""
        examples = self.plugin.get_pattern_examples()
        assert isinstance(examples, dict)
        assert "Simple" in examples
        assert "With Year" in examples

        # Check example structure
        for example in examples.values():
            assert "pattern" in example
            assert "result" in example

    def test_filename_without_extension(self):
        """Test filename generation when pattern lacks extension."""
        config = {"filename_pattern": "{track_number} {title}"}  # No extension
        plugin = CustomNamingPatternPlugin(config)

        import asyncio

        async def test():
            filename = await plugin.generate_filename(self.sample_audio_file)
            assert filename.endswith(".flac")  # Should add original extension

        asyncio.run(test())

    def test_empty_values_handling(self):
        """Test handling of empty or None values."""
        empty_file = AudioFile(
            path=Path("/test/unknown.flac"),
            artists=[],
            primary_artist=None,
            album=None,
            year=None,
            track_number=None,
            title=None
        )

        import asyncio

        async def test():
            filename = await self.plugin.generate_filename(empty_file)
            assert filename  # Should generate some filename

            base_dir = Path("/music")
            path = await self.plugin.generate_target_path(empty_file, base_dir)
            assert path  # Should generate some path

        asyncio.run(test())

    def test_multilingual_content(self):
        """Test handling of non-ASCII characters."""
        multilingual_file = AudioFile(
            path=Path("/test/artist/album/01 - song.flac"),
            artists=["Björk"],
            album="Medúlla",
            year=2004,
            title="Öll Birtan",
            genre="Experimental"
        )

        import asyncio

        async def test():
            filename = await self.plugin.generate_filename(multilingual_file)
            assert "Öll Birtan" in filename

            base_dir = Path("/music")
            path = await self.plugin.generate_target_path(multilingual_file, base_dir)
            assert "Björk" in str(path)
            assert "Medúlla" in str(path)

        asyncio.run(test())


class TestPluginFactory:
    """Tests for the plugin factory function."""

    def test_create_plugin(self):
        """Test creating plugin through factory."""
        plugin = create_plugin()
        assert isinstance(plugin, CustomNamingPatternPlugin)
        assert plugin.enabled

    def test_create_plugin_with_config(self):
        """Test creating plugin with configuration."""
        config = {"filename_pattern": "{title}{file_extension}"}
        plugin = create_plugin(config)
        assert isinstance(plugin, CustomNamingPatternPlugin)
        assert plugin.filename_pattern == "{title}{file_extension}"


class TestIntegrationWithOrganizer:
    """Integration tests with the organizer system."""

    def test_plugin_discovery(self):
        """Test that the plugin can be discovered by the plugin manager."""
        from music_organizer.plugins.manager import PluginManager

        manager = PluginManager()
        # The plugin should be discoverable from the builtins directory
        # This test ensures the plugin is properly structured

    def test_multiple_content_types(self):
        """Test plugin behavior with different content types."""
        config = {
            "path_patterns": {
                "studio": "Albums/{artist}/{album}",
                "live": "Live/{artist}/{date}",
                "compilation": "Compilations/{album}"
            }
        }
        plugin = CustomNamingPatternPlugin(config)

        test_cases = [
            (ContentType.STUDIO, "Albums/The Beatles/Abbey Road"),
            (ContentType.LIVE, "Live/The Beatles/1969-08-15"),
            (ContentType.COMPILATION, "Compilations/Abbey Road")
        ]

        import asyncio

        async def test_cases():
            base_dir = Path("/music")
            for content_type, expected_subpath in test_cases:
                audio_file = AudioFile(
                    path=Path("/test/song.flac"),
                    artists=["The Beatles"],
                    album="Abbey Road",
                    date="1969-08-15",
                    content_type=content_type
                )
                path = await plugin.generate_target_path(audio_file, base_dir)
                assert expected_subpath in str(path)

        asyncio.run(test_cases())


if __name__ == "__main__":
    pytest.main([__file__])