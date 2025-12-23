"""Tests for the example classifier plugin."""

import pytest
from pathlib import Path
from unittest.mock import patch

from music_organizer.plugins.builtins.example_classifier import ExampleClassifierPlugin
from music_organizer.models.audio_file import AudioFile


class TestExampleClassifierPlugin:
    """Tests for the ExampleClassifierPlugin class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'tags': ['decade', 'energy', 'language'],
            'high_energy_genres': ['rock', 'metal', 'punk', 'electronic', 'dance'],
            'low_energy_genres': ['ambient', 'classical', 'jazz', 'folk']
        }
        self.plugin = ExampleClassifierPlugin(self.config)

    def test_plugin_info(self):
        """Test plugin information."""
        info = self.plugin.info
        assert info.name == "example_classifier"
        assert info.version == "1.0.0"
        assert "classifies music" in info.description.lower()
        assert info.author == "Music Organizer Team"
        assert info.dependencies == []

    def test_initialization(self):
        """Test plugin initialization."""
        with patch('builtins.print') as mock_print:
            self.plugin.initialize()
            mock_print.assert_called_once_with("Example classifier plugin initialized")

    def test_cleanup(self):
        """Test plugin cleanup."""
        with patch('builtins.print') as mock_print:
            self.plugin.cleanup()
            mock_print.assert_called_once_with("Example classifier plugin cleaned up")

    def test_classify_decade(self):
        """Test decade classification."""
        import asyncio

        async def test():
            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                year=1969
            )
            result = await self.plugin.classify(audio_file)
            assert 'decade' in result
            assert result['decade'] == "1960s"

            # Test different decades
            test_years = [
                (1975, "1970s"),
                (1989, "1980s"),
                (2000, "2000s"),
                (2023, "2020s")
            ]

            for year, expected_decade in test_years:
                audio_file.year = year
                result = await self.plugin.classify(audio_file)
                assert result['decade'] == expected_decade

        asyncio.run(test())

    def test_classify_energy_high(self):
        """Test energy classification for high energy genres."""
        import asyncio

        async def test():
            high_genres = ['rock', 'metal', 'punk', 'electronic', 'dance']

            for genre in high_genres:
                audio_file = AudioFile(
                    path=Path("/test/song.mp3"),
                    file_type="mp3",
                    genre=genre
                )
                result = await self.plugin.classify(audio_file)
                assert 'energy' in result
                assert result['energy'] == 'high'

        asyncio.run(test())

    def test_classify_energy_low(self):
        """Test energy classification for low energy genres."""
        import asyncio

        async def test():
            low_genres = ['ambient', 'classical', 'jazz', 'folk']

            for genre in low_genres:
                audio_file = AudioFile(
                    path=Path("/test/song.mp3"),
                    file_type="mp3",
                    genre=genre
                )
                result = await self.plugin.classify(audio_file)
                assert 'energy' in result
                assert result['energy'] == 'low'

        asyncio.run(test())

    def test_classify_energy_medium(self):
        """Test energy classification for medium energy genres."""
        import asyncio

        async def test():
            medium_genres = ['pop', 'r&b', 'country', 'blues']

            for genre in medium_genres:
                audio_file = AudioFile(
                    path=Path("/test/song.mp3"),
                    file_type="mp3",
                    genre=genre
                )
                result = await self.plugin.classify(audio_file)
                assert 'energy' in result
                assert result['energy'] == 'medium'

        asyncio.run(test())

    def test_classify_energy_case_insensitive(self):
        """Test energy classification is case insensitive."""
        import asyncio

        async def test():
            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                genre="ROCK"  # uppercase
            )
            result = await self.plugin.classify(audio_file)
            assert result['energy'] == 'high'

            audio_file.genre = "ElEcTrOnIc"  # mixed case
            result = await self.plugin.classify(audio_file)
            assert result['energy'] == 'high'

        asyncio.run(test())

    def test_classify_energy_partial_match(self):
        """Test energy classification with partial genre matches."""
        import asyncio

        async def test():
            # Genre that contains high energy keyword
            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                genre="progressive rock"
            )
            result = await self.plugin.classify(audio_file)
            assert result['energy'] == 'high'

            # Genre that contains low energy keyword
            audio_file.genre = "smooth jazz"
            result = await self.plugin.classify(audio_file)
            assert result['energy'] == 'low'

        asyncio.run(test())

    def test_classify_language_english(self):
        """Test language classification for English."""
        import asyncio

        async def test():
            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                title="Hello World"
            )
            result = await self.plugin.classify(audio_file)
            assert 'language' in result
            assert result['language'] == 'english'

        asyncio.run(test())

    def test_classify_language_non_english(self):
        """Test language classification for non-English characters."""
        import asyncio

        async def test():
            # The actual implementation checks for specific accented characters
            # French - uses è
            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                title="C'était déjà toi"
            )
            result = await self.plugin.classify(audio_file)
            assert result['language'] == 'non_english'

            # Spanish - uses ñ
            audio_file.title = "Mañana de sol"
            result = await self.plugin.classify(audio_file)
            assert result['language'] == 'non_english'

            # Portuguese - uses ã and ç
            audio_file.title = "canção"
            result = await self.plugin.classify(audio_file)
            assert result['language'] == 'non_english'

            # Italian - uses à
            audio_file.title = "città"
            result = await self.plugin.classify(audio_file)
            assert result['language'] == 'non_english'

        asyncio.run(test())

    def test_classify_disabled_plugin(self):
        """Test classification when plugin is disabled."""
        import asyncio

        async def test():
            self.plugin.disable()
            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                year=1969,
                genre="rock",
                title="Test Song"
            )
            result = await self.plugin.classify(audio_file)
            assert result == {}

        asyncio.run(test())

    def test_classify_partial_tags(self):
        """Test classification with only some tags enabled."""
        import asyncio

        async def test():
            config = {'tags': ['decade']}  # Only decade
            plugin = ExampleClassifierPlugin(config)

            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                year=1969,
                genre="rock",
                title="Test Song"
            )
            result = await plugin.classify(audio_file)
            assert 'decade' in result
            assert 'energy' not in result
            assert 'language' not in result

        asyncio.run(test())

    def test_classify_missing_metadata(self):
        """Test classification with missing metadata."""
        import asyncio

        async def test():
            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3"
                # No year, genre, or title
            )
            result = await self.plugin.classify(audio_file)
            # Should return empty dict for missing data
            assert result == {}

        asyncio.run(test())

    def test_classify_all_tags(self):
        """Test classification with all tags enabled."""
        import asyncio

        async def test():
            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                year=2000,
                genre="electronic",
                title="Electronic Beats"
            )
            result = await self.plugin.classify(audio_file)
            assert 'decade' in result
            assert result['decade'] == "2000s"
            assert 'energy' in result
            assert result['energy'] == "high"
            assert 'language' in result
            assert result['language'] == "english"

        asyncio.run(test())

    def test_get_supported_tags(self):
        """Test getting supported classification tags."""
        tags = self.plugin.get_supported_tags()
        assert isinstance(tags, list)
        assert 'decade' in tags
        assert 'energy' in tags
        assert 'language' in tags

    def test_get_config_schema(self):
        """Test getting configuration schema."""
        schema = self.plugin.get_config_schema()
        assert schema is not None

        # Check that all expected options are present
        option = schema.get_option('enabled')
        assert option is not None
        assert option.type == bool

        option = schema.get_option('confidence_threshold')
        assert option is not None
        assert option.type == float

        option = schema.get_option('tags')
        assert option is not None
        assert option.type == list

        option = schema.get_option('high_energy_genres')
        assert option is not None
        assert option.type == list

        option = schema.get_option('low_energy_genres')
        assert option is not None
        assert option.type == list

    def test_custom_energy_genres(self):
        """Test classification with custom energy genres."""
        import asyncio

        async def test():
            config = {
                'tags': ['energy'],
                'high_energy_genres': ['pop', 'disco'],
                'low_energy_genres': ['ambient', 'new age']
            }
            plugin = ExampleClassifierPlugin(config)

            # Test custom high energy
            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                genre="pop"
            )
            result = await plugin.classify(audio_file)
            assert result['energy'] == 'high'

            # Test custom low energy
            audio_file.genre = "new age"
            result = await plugin.classify(audio_file)
            assert result['energy'] == 'low'

            # Test medium for others
            audio_file.genre = "rock"
            result = await plugin.classify(audio_file)
            assert result['energy'] == 'medium'

        asyncio.run(test())

    def test_enable_disable(self):
        """Test enabling and disabling plugin."""
        assert self.plugin.enabled is True

        self.plugin.disable()
        assert self.plugin.enabled is False

        self.plugin.enable()
        assert self.plugin.enabled is True

    def test_config_property(self):
        """Test plugin config property."""
        assert self.plugin.config == self.config

        # Test with default config
        plugin = ExampleClassifierPlugin()
        assert plugin.config == {}

    def test_config_access_in_classify(self):
        """Test that config is properly accessed during classification."""
        import asyncio

        async def test():
            config = {
                'tags': ['energy'],
                'high_energy_genres': ['custom_high']
            }
            plugin = ExampleClassifierPlugin(config)

            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                genre="custom_high"
            )
            result = await plugin.classify(audio_file)
            assert result['energy'] == 'high'

        asyncio.run(test())

    def test_year_none_for_decade(self):
        """Test decade classification when year is None."""
        import asyncio

        async def test():
            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                year=None
            )
            result = await self.plugin.classify(audio_file)
            assert 'decade' not in result

        asyncio.run(test())

    def test_genre_none_for_energy(self):
        """Test energy classification when genre is None."""
        import asyncio

        async def test():
            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                genre=None
            )
            result = await self.plugin.classify(audio_file)
            assert 'energy' not in result

        asyncio.run(test())

    def test_title_none_for_language(self):
        """Test language classification when title is None."""
        import asyncio

        async def test():
            audio_file = AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3",
                title=None
            )
            result = await self.plugin.classify(audio_file)
            assert 'language' not in result

        asyncio.run(test())


class TestPluginFactory:
    """Tests for plugin instantiation."""

    def test_create_without_config(self):
        """Test creating plugin without configuration."""
        plugin = ExampleClassifierPlugin()
        assert plugin is not None
        assert plugin.config == {}

    def test_create_with_empty_config(self):
        """Test creating plugin with empty configuration."""
        plugin = ExampleClassifierPlugin({})
        assert plugin is not None
        assert plugin.config == {}
