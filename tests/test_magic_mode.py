"""Tests for Magic Mode - zero configuration organization with AI suggestions."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from music_organizer.core.magic_mode import (
    MagicAnalyzer,
    MagicStrategyRecommender,
    MagicModeOrchestrator,
    LibraryAnalysis,
    OrganizationStrategy,
    MagicSuggestion,
    analyze_music_library,
    create_magic_organization_config
)
from music_organizer.core.magic_organizer import MagicMusicOrganizer
from music_organizer.models.audio_file import AudioFile, ContentType, FileFormat
from music_organizer.models.config import Config
from music_organizer.exceptions import MagicModeError
from music_organizer.domain.value_objects import ArtistName, Metadata


@pytest.fixture
def sample_audio_files() -> List[AudioFile]:
    """Create sample audio files for testing."""
    files = []

    # Rock albums
    for i in range(5):
        metadata = Metadata(
            title=f"Rock Song {i+1}",
            artists=[ArtistName(f"Rock Artist {i//2 + 1}")],
            album=f"Rock Album {i//3 + 1}",
            year=2000 + i,
            genre="rock",
            track_number=i+1
        )

        file = AudioFile(
            path=Path(f"/music/rock/artist{i//2 + 1}/album{i//3 + 1}/track{i+1:02d}.flac"),
            format=FileFormat.FLAC,
            metadata=metadata,
            content_type=ContentType.STUDIO,
            size_mb=50.0
        )
        files.append(file)

    # Jazz albums
    for i in range(3):
        metadata = Metadata(
            title=f"Jazz Track {i+1}",
            artists=[ArtistName("Jazz Master")],
            album="Jazz Collection",
            year=1985,
            genre="jazz",
            track_number=i+1
        )

        file = AudioFile(
            path=Path(f"/music/jazz/master/collection/track{i+1:02d}.mp3"),
            format=FileFormat.MP3,
            metadata=metadata,
            content_type=ContentType.STUDIO,
            size_mb=8.0
        )
        files.append(file)

    # Live recordings
    for i in range(2):
        metadata = Metadata(
            title=f"Live Song {i+1}",
            artists=[ArtistName("Live Band")],
            album="Live at Venue",
            year=2020,
            genre="rock",
            track_number=i+1
        )

        file = AudioFile(
            path=Path(f"/music/live/band/venue/track{i+1:02d}.wav"),
            format=FileFormat.WAV,
            metadata=metadata,
            content_type=ContentType.LIVE,
            size_mb=80.0
        )
        files.append(file)

    # Compilation
    metadata = Metadata(
        title="Various Hit",
        artists=[ArtistName("Various Artists")],
        album="Greatest Hits 2020",
        year=2020,
        genre="pop",
        track_number=1,
        is_compilation=True
    )

    file = AudioFile(
        path=Path("/music/compilations/greatest_hits/track01.flac"),
        format=FileFormat.FLAC,
        metadata=metadata,
        content_type=ContentType.COMPILATION,
        size_mb=45.0
    )
    files.append(file)

    return files


@pytest.fixture
def magic_analyzer():
    """Create MagicAnalyzer instance."""
    return MagicAnalyzer()


@pytest.fixture
def magic_recommender():
    """Create MagicStrategyRecommender instance."""
    return MagicStrategyRecommender()


@pytest.fixture
def magic_orchestrator():
    """Create MagicModeOrchestrator instance."""
    return MagicModeOrchestrator()


class TestMagicAnalyzer:
    """Test MagicAnalyzer functionality."""

    @pytest.mark.asyncio
    async def test_analyze_library_basic(self, magic_analyzer, sample_audio_files):
        """Test basic library analysis."""
        analysis = await magic_analyzer.analyze_library(sample_audio_files)

        assert isinstance(analysis, LibraryAnalysis)
        assert analysis.total_files == len(sample_audio_files)
        assert analysis.total_size_mb > 0
        assert analysis.artist_count > 0
        assert analysis.album_count > 0
        assert analysis.metadata_completeness > 0

        # Check format distribution
        assert "flac" in analysis.format_distribution
        assert "mp3" in analysis.format_distribution
        assert "wav" in analysis.format_distribution

        # Check content type distribution
        assert "studio" in analysis.content_type_distribution
        assert "live" in analysis.content_type_distribution
        assert "compilation" in analysis.content_type_distribution

        # Check genre distribution
        assert "rock" in analysis.genre_distribution
        assert "jazz" in analysis.genre_distribution

        # Check decade distribution
        assert "2000s" in analysis.decade_distribution
        assert "1980s" in analysis.decade_distribution
        assert "2020s" in analysis.decade_distribution

    @pytest.mark.asyncio
    async def test_analyze_empty_library(self, magic_analyzer):
        """Test analysis of empty library."""
        with pytest.raises(MagicModeError, match="No audio files provided"):
            await magic_analyzer.analyze_library([])

    def test_calculate_metadata_completeness(self, magic_analyzer):
        """Test metadata completeness calculation."""
        # Complete metadata
        complete_metadata = Metadata(
            title="Song",
            artists=[ArtistName("Artist")],
            album="Album",
            year=2020,
            genre="Rock",
            track_number=1,
            albumartist="Album Artist",
            composer="Composer",
            disc_number=1,
            is_compilation=False
        )
        score = magic_analyzer._calculate_metadata_completeness(complete_metadata)
        assert score == 1.0

        # Minimal metadata
        minimal_metadata = Metadata(
            title="Song",
            artists=[ArtistName("Artist")],
            album=None,
            year=None,
            genre=None,
            track_number=None
        )
        score = magic_analyzer._calculate_metadata_completeness(minimal_metadata)
        assert 0 < score < 1

    def test_normalize_genre(self, magic_analyzer):
        """Test genre normalization."""
        assert magic_analyzer._normalize_genre("alternative rock") == "rock"
        assert magic_analyzer._normalize_genre("techno house") == "electronic"
        assert magic_analyzer._normalize_genre("bebop jazz") == "jazz"
        assert magic_analyzer._normalize_genre("unknown genre") == "unknowngenre"

    def test_get_decade(self, magic_analyzer):
        """Test decade calculation."""
        assert magic_analyzer._get_decade(2020) == "2020s"
        assert magic_analyzer._get_decade(1995) == "1990s"
        assert magic_analyzer._get_decade(1989) == "1980s"
        assert magic_analyzer._get_decade(0) == "unknown"

    @pytest.mark.asyncio
    async def test_calculate_chaos_score(self, magic_analyzer, sample_audio_files):
        """Test chaos score calculation."""
        analysis = await magic_analyzer.analyze_library(sample_audio_files)
        chaos_score = magic_analyzer._calculate_chaos_score(sample_audio_files, analysis)

        assert 0 <= chaos_score <= 1
        # Our sample should have moderate chaos (good metadata but mixed formats)
        assert 0.2 <= chaos_score <= 0.8

    def test_estimate_duplicate_likelihood(self, magic_analyzer):
        """Test duplicate likelihood estimation."""
        # No duplicates
        files = [
            AudioFile(
                path=Path(f"/music/song{i}.flac"),
                format=FileFormat.FLAC,
                metadata=Metadata(title=f"Song {i}", artists=[ArtistName(f"Artist {i}")]),
                content_type=ContentType.STUDIO,
                size_mb=50.0
            )
            for i in range(10)
        ]
        likelihood = magic_analyzer._estimate_duplicate_likelihood(files)
        assert likelihood == 0.0

        # With duplicates
        duplicate_files = files + files[:2]  # Add 2 duplicates
        likelihood = magic_analyzer._estimate_duplicate_likelihood(duplicate_files)
        assert likelihood > 0


class TestMagicStrategyRecommender:
    """Test MagicStrategyRecommender functionality."""

    @pytest.mark.asyncio
    async def test_recommend_strategy(self, magic_recommender, sample_audio_files):
        """Test strategy recommendation."""
        # First analyze the library
        analyzer = MagicAnalyzer()
        analysis = await analyzer.analyze_library(sample_audio_files)

        # Get recommendations
        recommendations = await magic_recommender.recommend_strategy(analysis)

        assert len(recommendations) > 0
        assert all(isinstance(r, OrganizationStrategy) for r in recommendations)

        # Check that strategies are sorted by confidence
        confidences = [r.confidence for r in recommendations]
        assert confidences == sorted(confidences, reverse=True)

        # Check top strategy
        top_strategy = recommendations[0]
        assert 0 <= top_strategy.confidence <= 1
        assert top_strategy.name
        assert top_strategy.path_pattern
        assert top_strategy.filename_pattern
        assert top_strategy.reasoning
        assert top_strategy.pros
        assert top_strategy.cons

    def test_score_strategy(self, magic_recommender, sample_audio_files):
        """Test strategy scoring."""
        analyzer = MagicAnalyzer()
        analysis = asyncio.run(analyzer.analyze_library(sample_audio_files))

        # Test different strategies
        strategies = ["artist_album", "genre_artist_album", "decade_artist_album"]
        scores = []

        for strategy_id in strategies:
            strategy_info = magic_recommender.strategies[strategy_id]
            score = magic_recommender._score_strategy(strategy_id, strategy_info, analysis)
            scores.append(score)

        # All scores should be valid
        assert all(0 <= score <= 1 for score in scores)

    def test_generate_reasoning(self, magic_recommender):
        """Test reasoning generation."""
        # Create sample analysis
        analysis = LibraryAnalysis(
            total_files=1000,
            metadata_completeness=0.9,
            organization_chaos_score=0.3
        )

        strategy_info = {
            "name": "Test Strategy",
            "reasons": ["basic reason"],
            "complexity": "simple"
        }

        reasoning = magic_recommender._generate_reasoning("test_strategy", strategy_info, analysis)

        assert isinstance(reasoning, list)
        assert len(reasoning) > 0
        assert "basic reason" in reasoning


class TestMagicModeOrchestrator:
    """Test MagicModeOrchestrator functionality."""

    @pytest.mark.asyncio
    async def test_analyze_and_suggest(self, magic_orchestrator, sample_audio_files):
        """Test analyze and suggest functionality."""
        suggestion = await magic_orchestrator.analyze_and_suggest(sample_audio_files)

        assert isinstance(suggestion, MagicSuggestion)
        assert isinstance(suggestion.strategy, OrganizationStrategy)
        assert isinstance(suggestion.analysis, LibraryAnalysis)
        assert isinstance(suggestion.quick_wins, list)
        assert isinstance(suggestion.potential_issues, list)
        assert isinstance(suggestion.custom_rules, list)
        assert isinstance(suggestion.preprocessing_steps, list)

        # Check strategy quality
        assert suggestion.strategy.confidence > 0
        assert suggestion.strategy.name
        assert suggestion.strategy.path_pattern

    @pytest.mark.asyncio
    async def test_generate_magic_config(self, magic_orchestrator, sample_audio_files, tmp_path):
        """Test magic configuration generation."""
        suggestion = await magic_orchestrator.analyze_and_suggest(sample_audio_files)

        # Test without output path
        config = await magic_orchestrator.generate_magic_config(suggestion)
        assert isinstance(config, dict)
        assert "magic_mode" in config
        assert "organization" in config
        assert "preprocessing" in config
        assert "plugins" in config

        # Test with output path
        output_path = tmp_path / "magic_config.json"
        config = await magic_orchestrator.generate_magic_config(suggestion, output_path)

        assert output_path.exists()
        assert isinstance(config, dict)

        # Verify file contents
        import json
        with open(output_path) as f:
            saved_config = json.load(f)
        assert saved_config == config

    @pytest.mark.asyncio
    async def test_analyze_and_suggest_empty_files(self, magic_orchestrator):
        """Test analyze and suggest with no files."""
        with pytest.raises(MagicModeError, match="No audio files to analyze"):
            await magic_orchestrator.analyze_and_suggest([])

    def test_generate_quick_wins(self, magic_orchestrator):
        """Test quick wins generation."""
        analysis = LibraryAnalysis(
            metadata_completeness=0.5,
            organization_chaos_score=0.8,
            duplicate_likelihood=0.2
        )
        strategy = OrganizationStrategy(
            name="Test Strategy",
            description="Test",
            path_pattern="{artist}/{album}",
            filename_pattern="{title}",
            confidence=0.7,
            reasoning=[],
            estimated_time_minutes=10,
            complexity="simple",
            pros=[],
            cons=[]
        )

        suggestion = MagicSuggestion(
            strategy=strategy,
            analysis=analysis,
            quick_wins=[],
            potential_issues=[],
            custom_rules=[],
            preprocessing_steps=[]
        )

        quick_wins = magic_orchestrator._generate_quick_wins(analysis, strategy)
        assert isinstance(quick_wins, list)
        # Should suggest fixes for identified issues
        assert any("metadata" in win.lower() for win in quick_wins)
        assert any("duplicate" in win.lower() for win in quick_wins)

    def test_generate_custom_rules(self, magic_orchestrator):
        """Test custom rules generation."""
        analysis = LibraryAnalysis(
            total_files=1000,
            content_type_distribution={
                "compilation": 150,  # 15% compilations
                "live": 50,  # 5% live
                "studio": 800
            }
        )

        custom_rules = magic_orchestrator._generate_custom_rules(analysis)
        assert isinstance(custom_rules, list)

        # Should have compilation rule
        compilation_rules = [r for r in custom_rules if "compilation" in r["name"].lower()]
        assert len(compilation_rules) > 0

        # Should have live recording rule
        live_rules = [r for r in custom_rules if "live" in r["name"].lower()]
        assert len(live_rules) > 0


class TestMagicModeIntegration:
    """Test Magic Mode integration and convenience functions."""

    @pytest.mark.asyncio
    async def test_analyze_music_library_function(self, sample_audio_files):
        """Test convenience function for library analysis."""
        suggestion = await analyze_music_library(sample_audio_files)

        assert isinstance(suggestion, MagicSuggestion)
        assert suggestion.strategy
        assert suggestion.analysis

    @pytest.mark.asyncio
    async def test_create_magic_organization_config_function(self, sample_audio_files, tmp_path):
        """Test convenience function for config creation."""
        output_path = tmp_path / "config.json"
        config = await create_magic_organization_config(sample_audio_files, output_path)

        assert isinstance(config, dict)
        assert output_path.exists()

        # Verify structure
        assert "magic_mode" in config
        assert "organization" in config
        assert "preprocessing" in config
        assert "plugins" in config


class TestMagicMusicOrganizer:
    """Test MagicMusicOrganizer integration."""

    @pytest.fixture
    def magic_organizer(self):
        """Create MagicMusicOrganizer instance."""
        config = Config(
            source_directory=Path("/source"),
            target_directory=Path("/target")
        )
        return MagicMusicOrganizer(config=config)

    @pytest.mark.asyncio
    async def test_analyze_library_for_magic_mode(self, magic_organizer, sample_audio_files):
        """Test library analysis in MagicMusicOrganizer."""
        with patch.object(magic_organizer, 'scan_directory') as mock_scan, \
             patch.object(magic_organizer, 'extract_metadata') as mock_extract:

            # Mock scan to return our sample file paths
            mock_scan.return_value = (f.path for f in sample_audio_files)
            mock_extract.return_value = sample_audio_files[0]  # Return first file's metadata

            suggestion = await magic_organizer.analyze_library_for_magic_mode(
                Path("/music"),
                sample_size=5
            )

            assert isinstance(suggestion, MagicSuggestion)
            assert magic_organizer._current_suggestion == suggestion
            assert magic_organizer._library_analysis == suggestion.analysis

    def test_generate_magic_target_path(self, magic_organizer):
        """Test magic target path generation."""
        source_path = Path("/music/test/song.flac")

        # Create test metadata
        metadata = Metadata(
            title="Test Song",
            artists=[ArtistName("Test Artist")],
            album="Test Album",
            year=2020,
            genre="Rock",
            track_number=5
        )

        path_pattern = "{artist}/{album} ({year})"
        filename_pattern = "{track_number:02d} {title}"

        target_path = magic_organizer._generate_magic_target_path(
            source_path,
            metadata,
            Path("/target"),
            path_pattern,
            filename_pattern
        )

        expected = Path("/target/Test Artist/Test Album (2020)/05 Test Song.flac")
        assert target_path == expected

    def test_clean_path_for_filesystem(self, magic_organizer):
        """Test path cleaning for filesystem safety."""
        # Test problematic characters
        problematic_path = Path("/music/artist/album/song:with*bad?characters.flac")
        cleaned_path = magic_organizer._clean_path_for_filesystem(problematic_path)

        # Should replace problematic characters with underscores
        assert "*" not in str(cleaned_path)
        assert "?" not in str(cleaned_path)
        assert ":" not in str(cleaned_path)

        # Test reserved names (Windows)
        reserved_path = Path("/music/CON.flac")
        cleaned_path = magic_organizer._clean_path_for_filesystem(reserved_path)
        assert cleaned_path.name.startswith("_")

    def test_calculate_decade(self, magic_organizer):
        """Test decade calculation."""
        assert magic_organizer._calculate_decade(2020) == "2020s"
        assert magic_organizer._calculate_decade(1995) == "1990s"
        assert magic_organizer._calculate_decade(0) == "unknown"
        assert magic_organizer._calculate_decade("invalid") == "unknown"


@pytest.mark.asyncio
async def test_magic_mode_error_handling():
    """Test Magic Mode error handling."""
    orchestrator = MagicModeOrchestrator()

    # Test with empty files list
    with pytest.raises(MagicModeError):
        await orchestrator.analyze_and_suggest([])

    # Test invalid analysis data
    with pytest.raises(MagicModeError):
        await analyze_music_library([])


@pytest.mark.asyncio
async def test_magic_mode_performance(sample_audio_files):
    """Test Magic Mode performance with larger datasets."""
    import time

    # Create a larger dataset
    large_files = []
    for i in range(100):  # 100 files
        metadata = Metadata(
            title=f"Song {i}",
            artists=[ArtistName(f"Artist {i % 20}")],  # 20 different artists
            album=f"Album {i // 10}",  # 10 different albums
            year=2000 + (i % 23),  # Span 23 years
            genre=["rock", "pop", "jazz", "electronic"][i % 4],
            track_number=(i % 12) + 1
        )

        file = AudioFile(
            path=Path(f"/music/artist{i % 20}/album{i // 10}/track{i:03d}.flac"),
            format=FileFormat.FLAC,
            metadata=metadata,
            content_type=ContentType.STUDIO,
            size_mb=50.0
        )
        large_files.append(file)

    # Test analysis performance
    start_time = time.time()
    suggestion = await analyze_music_library(large_files)
    end_time = time.time()

    # Should complete within reasonable time (adjust threshold as needed)
    analysis_time = end_time - start_time
    assert analysis_time < 5.0  # 5 seconds for 100 files

    # Should still produce valid results
    assert suggestion.strategy.confidence > 0
    assert suggestion.analysis.total_files == 100
    assert len(suggestion.analysis.genre_distribution) > 1


if __name__ == "__main__":
    pytest.main([__file__])