"""Example classification plugin."""

from typing import Dict, Any, List
from ..base import ClassificationPlugin, PluginInfo
from ..config import PluginConfigSchema, ConfigOption
from ...models.audio_file import AudioFile


class ExampleClassifierPlugin(ClassificationPlugin):
    """Example plugin that classifies music by decade and energy."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="example_classifier",
            version="1.0.0",
            description="Classifies music by decade and energy level",
            author="Music Organizer Team",
            dependencies=[],
        )

    def initialize(self) -> None:
        """Initialize the plugin."""
        print("Example classifier plugin initialized")

    def cleanup(self) -> None:
        """Cleanup resources."""
        print("Example classifier plugin cleaned up")

    async def classify(self, audio_file: AudioFile) -> Dict[str, Any]:
        """Classify an audio file."""
        if not self.enabled:
            return {}

        classifications = {}

        # Get configured tags to generate
        tags = self.config.get('tags', ['decade', 'energy', 'language'])

        # Classify by decade
        if 'decade' in tags and audio_file.year:
            decade = (audio_file.year // 10) * 10
            classifications['decade'] = f"{decade}s"

        # Classify by energy (simple heuristic based on genre)
        if 'energy' in tags and audio_file.genre:
            high_energy_genres = self.config.get('high_energy_genres', ['rock', 'metal', 'punk', 'electronic', 'dance'])
            low_energy_genres = self.config.get('low_energy_genres', ['ambient', 'classical', 'jazz', 'folk'])

            genre_lower = audio_file.genre.lower()
            if any(eg in genre_lower for eg in high_energy_genres):
                classifications['energy'] = 'high'
            elif any(eg in genre_lower for eg in low_energy_genres):
                classifications['energy'] = 'low'
            else:
                classifications['energy'] = 'medium'

        # Classify by language (simple heuristic)
        if 'language' in tags and audio_file.title:
            title_lower = audio_file.title.lower()
            if any(c in title_lower for c in 'éàèùâêîôûçñ'):
                classifications['language'] = 'non_english'
            else:
                classifications['language'] = 'english'

        return classifications

    def get_supported_tags(self) -> List[str]:
        """Return supported classification tags."""
        return ['decade', 'energy', 'language']

    def get_config_schema(self) -> PluginConfigSchema:
        """Return configuration schema for this plugin."""
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
                default=['decade', 'energy', 'language'],
                description="List of tags to generate",
                choices=['decade', 'energy', 'language']
            ),
            ConfigOption(
                name="high_energy_genres",
                type=list,
                default=['rock', 'metal', 'punk', 'electronic', 'dance'],
                description="Genres considered high energy"
            ),
            ConfigOption(
                name="low_energy_genres",
                type=list,
                default=['ambient', 'classical', 'jazz', 'folk'],
                description="Genres considered low energy"
            ),
        ])