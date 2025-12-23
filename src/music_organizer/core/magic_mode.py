"""Magic Mode - Zero Configuration Organization with AI Suggestions.

This module provides intelligent, zero-configuration music organization that
automatically analyzes the user's music library and suggests the best organization
strategy based on patterns, metadata quality, and user preferences.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from datetime import datetime
import re

from ..models.audio_file import AudioFile, ContentType
from ..models.config import Config
from ..domain.value_objects import ArtistName, Metadata
from ..domain.catalog.services import CatalogService
from ..domain.organization.services import OrganizationService, PathGenerationService
from ..domain.classification.services import ClassificationService
from ..exceptions import MagicModeError


@dataclass
class LibraryAnalysis:
    """Analysis results for a music library."""
    total_files: int = 0
    total_size_mb: float = 0.0
    format_distribution: Dict[str, int] = field(default_factory=dict)
    content_type_distribution: Dict[str, int] = field(default_factory=dict)
    genre_distribution: Dict[str, int] = field(default_factory=dict)
    decade_distribution: Dict[str, int] = field(default_factory=dict)
    artist_count: int = 0
    album_count: int = 0
    average_tracks_per_album: float = 0.0
    metadata_completeness: float = 0.0
    folder_structure_patterns: List[str] = field(default_factory=list)
    organization_chaos_score: float = 0.0  # 0 = perfectly organized, 1 = completely chaotic
    duplicate_likelihood: float = 0.0
    quality_variance: Dict[str, float] = field(default_factory=dict)  # bitrate, sample rate variance


@dataclass
class OrganizationStrategy:
    """Recommended organization strategy."""
    name: str
    description: str
    path_pattern: str
    filename_pattern: str
    confidence: float
    reasoning: List[str]
    estimated_time_minutes: int
    complexity: str  # "simple", "moderate", "complex"
    pros: List[str]
    cons: List[str]


@dataclass
class MagicSuggestion:
    """AI-powered suggestion for organization."""
    strategy: OrganizationStrategy
    analysis: LibraryAnalysis
    quick_wins: List[str]
    potential_issues: List[str]
    custom_rules: List[Dict[str, Any]]
    preprocessing_steps: List[str]


class MagicAnalyzer:
    """Analyzes music libraries to determine optimal organization strategies."""

    def __init__(self):
        self.genre_patterns = self._load_genre_patterns()
        self.decade_patterns = self._load_decade_patterns()

    def _load_genre_patterns(self) -> Dict[str, List[str]]:
        """Load genre classification patterns."""
        return {
            "rock": ["rock", "alternative", "indie", "punk", "metal", "hardcore"],
            "electronic": ["electronic", "techno", "house", "trance", "dubstep", "ambient", "edm"],
            "classical": ["classical", "orchestra", "symphony", "opera", "baroque", "romantic"],
            "jazz": ["jazz", "blues", "swing", "bebop", "fusion", "smooth"],
            "hip-hop": ["hip", "hop", "rap", "trap", "r&b", "soul"],
            "pop": ["pop", "top", "chart", "mainstream"],
            "world": ["folk", "traditional", "ethnic", "world", "reggae"],
            "soundtrack": ["soundtrack", "film", "movie", "game", "anime"],
        }

    def _load_decade_patterns(self) -> Dict[str, Tuple[int, int]]:
        """Load decade year ranges."""
        return {
            "1950s": (1950, 1959),
            "1960s": (1960, 1969),
            "1970s": (1970, 1979),
            "1980s": (1980, 1989),
            "1990s": (1990, 1999),
            "2000s": (2000, 2009),
            "2010s": (2010, 2019),
            "2020s": (2020, 2029),
        }

    async def analyze_library(self, audio_files: List[AudioFile]) -> LibraryAnalysis:
        """Comprehensive analysis of a music library."""
        if not audio_files:
            raise MagicModeError("No audio files provided for analysis")

        analysis = LibraryAnalysis()
        analysis.total_files = len(audio_files)

        # Analyze formats and sizes
        format_sizes = defaultdict(float)
        for file in audio_files:
            fmt = file.file_type.lower()
            analysis.format_distribution[fmt] = analysis.format_distribution.get(fmt, 0) + 1
            try:
                size_mb = file.path.stat().st_size / (1024 * 1024)
            except (FileNotFoundError, OSError):
                size_mb = 0.0
            analysis.total_size_mb += size_mb
            format_sizes[fmt] += size_mb

            # Quality variance
            bitrate = file.metadata.get('bitrate') if isinstance(file.metadata, dict) else None
            if bitrate:
                if 'bitrate' not in analysis.quality_variance:
                    analysis.quality_variance['bitrate'] = []
                analysis.quality_variance['bitrate'].append(bitrate)

        # Calculate average size per format
        for fmt, total_size in format_sizes.items():
            count = analysis.format_distribution[fmt]
            if count > 0:
                analysis.quality_variance[f'avg_size_{fmt}'] = total_size / count

        # Analyze content types
        for file in audio_files:
            content_type = file.content_type.value if file.content_type else "unknown"
            analysis.content_type_distribution[content_type] = (
                analysis.content_type_distribution.get(content_type, 0) + 1
            )

        # Extract unique artists and albums
        artists = set()
        albums = set()
        tracks_per_album = defaultdict(int)
        metadata_scores = []

        for file in audio_files:
            # Artist extraction
            if file.artists:
                for artist in file.artists:
                    artists.add(artist)

            # Album extraction
            if file.album:
                album_key = (file.album, tuple(file.artists) if file.artists else ())
                albums.add(album_key)
                tracks_per_album[album_key] += 1

            # Metadata completeness
            score = self._calculate_metadata_completeness_for_audiofile(file)
            metadata_scores.append(score)

            # Genre analysis
            if file.genre:
                normalized_genre = self._normalize_genre(file.genre)
                analysis.genre_distribution[normalized_genre] = (
                    analysis.genre_distribution.get(normalized_genre, 0) + 1
                )

            # Decade analysis
            if file.year:
                decade = self._get_decade(file.year)
                analysis.decade_distribution[decade] = (
                    analysis.decade_distribution.get(decade, 0) + 1
                )

        analysis.artist_count = len(artists)
        analysis.album_count = len(albums)
        analysis.average_tracks_per_album = (
            sum(tracks_per_album.values()) / len(tracks_per_album) if tracks_per_album else 0
        )
        analysis.metadata_completeness = sum(metadata_scores) / len(metadata_scores) if metadata_scores else 0

        # Analyze folder structure
        analysis.folder_structure_patterns = self._analyze_folder_structure(audio_files)

        # Calculate organization chaos score
        analysis.organization_chaos_score = self._calculate_chaos_score(audio_files, analysis)

        # Estimate duplicate likelihood
        analysis.duplicate_likelihood = self._estimate_duplicate_likelihood(audio_files)

        return analysis

    def _calculate_metadata_completeness(self, metadata: Metadata) -> float:
        """Calculate metadata completeness score (0-1)."""
        fields = ['title', 'artists', 'album', 'year', 'genre', 'track_number']
        present = sum(1 for field in fields if getattr(metadata, field, None))
        # Bonus fields
        bonus_fields = ['albumartist', 'composer', 'disc_number']
        present_bonus = sum(1 for field in bonus_fields if getattr(metadata, field, None))

        return (present + (present_bonus * 0.5)) / (len(fields) + (len(bonus_fields) * 0.5))

    def _calculate_metadata_completeness_for_audiofile(self, file: AudioFile) -> float:
        """Calculate metadata completeness score for AudioFile (0-1)."""
        fields = ['title', 'artists', 'album', 'year', 'genre', 'track_number']
        present = sum(1 for field in fields if getattr(file, field, None))
        # Bonus fields
        bonus_fields = ['primary_artist', 'date', 'location']
        present_bonus = sum(1 for field in bonus_fields if getattr(file, field, None))

        return (present + (present_bonus * 0.5)) / (len(fields) + (len(bonus_fields) * 0.5))

    def _normalize_genre(self, genre: str) -> str:
        """Normalize genre to broader category."""
        genre_lower = genre.lower()

        for broad_genre, patterns in self.genre_patterns.items():
            for pattern in patterns:
                if pattern in genre_lower:
                    return broad_genre

        # If no match, return cleaned original
        return re.sub(r'[^a-z]', '', genre_lower.lower())

    def _get_decade(self, year: int) -> str:
        """Get decade from year."""
        for decade, (start, end) in self.decade_patterns.items():
            if start <= year <= end:
                return decade
        return "unknown"

    def _analyze_folder_structure(self, audio_files: List[AudioFile]) -> List[str]:
        """Analyze existing folder structure patterns."""
        patterns = Counter()

        for file in audio_files[:100]:  # Sample first 100 files for performance
            path_parts = file.path.relative_to(file.path.anchor).parts

            if len(path_parts) >= 2:
                # Create pattern from path structure
                pattern = " -> ".join(path_parts[:-1])  # Exclude filename
                patterns[pattern] += 1

        # Return most common patterns
        return [pattern for pattern, count in patterns.most_common(10)]

    def _calculate_chaos_score(self, audio_files: List[AudioFile], analysis: LibraryAnalysis) -> float:
        """Calculate how chaotic/disorganized the library is."""
        score = 0.0

        # Metadata incompleteness contributes to chaos
        score += (1 - analysis.metadata_completeness) * 0.3

        # Many different formats in same folders
        format_entropy = sum(
            (count / analysis.total_files) *
            (1 if count / analysis.total_files < 0.7 else 0)
            for count in analysis.format_distribution.values()
        )
        score += format_entropy * 0.2

        # Content type mixing
        content_entropy = len(analysis.content_type_distribution) / max(len(ContentType), 1)
        score += min(content_entropy * 0.1, 0.2)

        # Folder structure consistency
        if analysis.folder_structure_patterns:
            dominant_pattern_ratio = (
                len([p for p in analysis.folder_structure_patterns[:5] if " -> " in p]) /
                min(5, len(analysis.folder_structure_patterns))
            )
            score += (1 - dominant_pattern_ratio) * 0.4

        return min(score, 1.0)

    def _estimate_duplicate_likelihood(self, audio_files: List[AudioFile]) -> float:
        """Estimate likelihood of duplicates in library."""
        if len(audio_files) < 10:
            return 0.0

        # Check for likely duplicates based on metadata
        metadata_signatures = Counter()

        for file in audio_files:
            if file.title and file.artists:
                # Create normalized signature
                title = re.sub(r'[^a-z0-9]', '', file.title.lower())
                artist = re.sub(r'[^a-z0-9]', '', file.artists[0].lower())
                signature = f"{artist}_{title}"
                metadata_signatures[signature] += 1

        # Calculate duplicate ratio
        duplicate_signatures = sum(1 for count in metadata_signatures.values() if count > 1)
        total_signatures = len(metadata_signatures)

        return duplicate_signatures / total_signatures if total_signatures > 0 else 0.0


class MagicStrategyRecommender:
    """Recommends organization strategies based on library analysis."""

    def __init__(self):
        self.strategies = self._initialize_strategies()

    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize predefined organization strategies."""
        return {
            "artist_album": {
                "name": "Artist/Album Structure",
                "path_pattern": "{artist}/{album} ({year})",
                "filename_pattern": "{track_number:02d} {title}",
                "complexity": "simple",
                "best_for": ["well_tagged", "traditional", "large_libraries"],
                "reasons": ["traditional structure", "easy navigation", "compatible with most players"]
            },
            "genre_artist_album": {
                "name": "Genre/Artist/Album Structure",
                "path_pattern": "{genre}/{artist}/{album} ({year})",
                "filename_pattern": "{track_number:02d} {title}",
                "complexity": "moderate",
                "best_for": ["diverse_genres", "large_libraries", "eclectic_taste"],
                "reasons": ["genre separation", "good for diverse collections", "playlist friendly"]
            },
            "decade_artist_album": {
                "name": "Decade/Artist/Album Structure",
                "path_pattern": "{decade}/{artist}/{album} ({year})",
                "filename_pattern": "{track_number:02d} {title}",
                "complexity": "moderate",
                "best_for": ["historical_collections", "chronological_listening", "music_historians"],
                "reasons": ["chronological organization", "shows music evolution", "era-based discovery"]
            },
            "smart_content_type": {
                "name": "Smart Content-Based Structure",
                "path_pattern": "{content_type}/{artist}/{album} ({year})",
                "filename_pattern": "{track_number:02d} {title}",
                "complexity": "complex",
                "best_for": ["live_recordings", "compilations", "mixed_content"],
                "reasons": ["separates live/studio", "compilations organized", "content-aware"]
            },
            "flat_smart": {
                "name": "Smart Flat Structure",
                "path_pattern": "{artist_first_letter}/{artist} - {album} - {title}",
                "filename_pattern": "",
                "complexity": "simple",
                "best_for": ["small_libraries", "portable_devices", "simple_browsing"],
                "reasons": ["minimal folders", "easy searching", "portable friendly"]
            },
            "collection_based": {
                "name": "Collection-Based Structure",
                "path_pattern": "{main_genre}/{content_type}/{artist}/{album} ({year})",
                "filename_pattern": "{track_number:02d} {title}",
                "complexity": "complex",
                "best_for": ["huge_libraries", "serious_collectors", "dj_collections"],
                "reasons": ["very organized", "excellent discovery", "professional structure"]
            }
        }

    async def recommend_strategy(self, analysis: LibraryAnalysis) -> List[OrganizationStrategy]:
        """Recommend best organization strategies based on analysis."""
        recommendations = []

        # Score each strategy
        strategy_scores = []
        for strategy_id, strategy_info in self.strategies.items():
            score = self._score_strategy(strategy_id, strategy_info, analysis)
            strategy_scores.append((score, strategy_id, strategy_info))

        # Sort by score and create recommendations
        strategy_scores.sort(reverse=True)

        for i, (score, strategy_id, strategy_info) in enumerate(strategy_scores[:5]):
            strategy = OrganizationStrategy(
                name=strategy_info["name"],
                description=self._generate_description(strategy_id, strategy_info, analysis),
                path_pattern=strategy_info["path_pattern"],
                filename_pattern=strategy_info["filename_pattern"],
                confidence=score,
                reasoning=self._generate_reasoning(strategy_id, strategy_info, analysis),
                estimated_time_minutes=self._estimate_time(analysis, strategy_info["complexity"]),
                complexity=strategy_info["complexity"],
                pros=self._generate_pros(strategy_id, analysis),
                cons=self._generate_cons(strategy_id, analysis)
            )
            recommendations.append(strategy)

        return recommendations

    def _score_strategy(self, strategy_id: str, strategy_info: Dict[str, Any], analysis: LibraryAnalysis) -> float:
        """Score how well a strategy fits the library."""
        score = 0.5  # Base score

        # Metadata completeness bonus
        if analysis.metadata_completeness > 0.8:
            if strategy_id in ["artist_album", "genre_artist_album", "smart_content_type"]:
                score += 0.3
        elif analysis.metadata_completeness < 0.5:
            if strategy_id in ["flat_smart"]:
                score += 0.2

        # Library size considerations
        if analysis.total_files > 10000:
            if strategy_id in ["collection_based", "genre_artist_album"]:
                score += 0.2
            elif strategy_id in ["flat_smart"]:
                score -= 0.1
        elif analysis.total_files < 1000:
            if strategy_id in ["flat_smart", "artist_album"]:
                score += 0.2
            elif strategy_id in ["collection_based"]:
                score -= 0.1

        # Genre diversity
        genre_diversity = len(analysis.genre_distribution)
        if genre_diversity > 8:
            if "genre" in strategy_id:
                score += 0.25
        elif genre_diversity < 3:
            if "genre" in strategy_id:
                score -= 0.1

        # Content type diversity
        content_diversity = len(analysis.content_type_distribution)
        if content_diversity > 3:
            if strategy_id == "smart_content_type":
                score += 0.3
        elif content_diversity == 1:
            if strategy_id == "smart_content_type":
                score -= 0.1

        # Decade spread
        if len(analysis.decade_distribution) > 4:
            if "decade" in strategy_id:
                score += 0.2

        # Organization chaos score
        if analysis.organization_chaos_score > 0.7:
            # Prefer more structured approaches for chaotic libraries
            if strategy_info["complexity"] in ["moderate", "complex"]:
                score += 0.2
        elif analysis.organization_chaos_score < 0.3:
            # Library is already organized, prefer simpler structures
            if strategy_info["complexity"] == "simple":
                score += 0.1

        return min(score, 1.0)

    def _generate_description(self, strategy_id: str, strategy_info: Dict[str, Any], analysis: LibraryAnalysis) -> str:
        """Generate a description for the strategy."""
        base_descriptions = {
            "artist_album": "Organizes music by artist and album, maintaining a traditional and widely-compatible structure.",
            "genre_artist_album": "First groups by genre, then by artist and album within each genre category.",
            "decade_artist_album": "Organizes music chronologically by decade, perfect for historical music collections.",
            "smart_content_type": "Intelligently separates studio albums, live recordings, and compilations.",
            "flat_smart": "Minimizes folder nesting while maintaining logical file naming.",
            "collection_based": "Professional-grade organization with multiple levels of categorization."
        }

        return base_descriptions.get(strategy_id, "Custom organization strategy.")

    def _generate_reasoning(self, strategy_id: str, strategy_info: Dict[str, Any], analysis: LibraryAnalysis) -> List[str]:
        """Generate reasoning for why this strategy is recommended."""
        reasoning = []

        # Add generic reasons
        reasoning.extend(strategy_info.get("reasons", []))

        # Add analysis-specific reasons
        if analysis.metadata_completeness > 0.9:
            reasoning.append("excellent metadata quality enables rich organization")
        elif analysis.metadata_completeness < 0.5:
            reasoning.append("minimal metadata requirements")

        if analysis.total_files > 5000:
            reasoning.append("scales well with large libraries")
        elif analysis.total_files < 500:
            reasoning.append("ideal for smaller collections")

        if len(analysis.genre_distribution) > 10:
            reasoning.append("handles diverse genres effectively")

        if len(analysis.content_type_distribution) > 3:
            reasoning.append("properly separates different content types")

        return reasoning[:6]  # Limit to 6 reasons

    def _estimate_time(self, analysis: LibraryAnalysis, complexity: str) -> int:
        """Estimate organization time in minutes."""
        base_time = max(1, analysis.total_files / 100)  # Base time per 100 files

        complexity_multiplier = {
            "simple": 1.0,
            "moderate": 1.5,
            "complex": 2.0
        }.get(complexity, 1.0)

        # Account for chaos score
        chaos_multiplier = 1 + (analysis.organization_chaos_score * 0.5)

        return int(base_time * complexity_multiplier * chaos_multiplier)

    def _generate_pros(self, strategy_id: str, analysis: LibraryAnalysis) -> List[str]:
        """Generate pros for the strategy."""
        pros = []

        if strategy_id == "artist_album":
            pros.extend(["widely compatible", "easy to understand", "minimal folders"])
        elif strategy_id == "genre_artist_album":
            pros.extend(["great for discovery", "playlist friendly", "genre browsing"])
        elif strategy_id == "decade_artist_album":
            pros.extend(["chronological", "shows evolution", "era-based exploration"])
        elif strategy_id == "smart_content_type":
            pros.extend(["content-aware", "live/studio separation", "compilation handling"])
        elif strategy_id == "flat_smart":
            pros.extend(["minimal nesting", "portable friendly", "simple searching"])
        elif strategy_id == "collection_based":
            pros.extend(["highly organized", "professional", "excellent for large libraries"])

        return pros

    def _generate_cons(self, strategy_id: str, analysis: LibraryAnalysis) -> List[str]:
        """Generate cons for the strategy."""
        cons = []

        if strategy_id == "artist_album":
            cons.extend(["no genre separation", "compilations mixed in"])
        elif strategy_id == "genre_artist_album":
            cons.extend(["more folders", "genre classification needed"])
        elif strategy_id == "decade_artist_album":
            cons.extend(["complex", "requires accurate years"])
        elif strategy_id == "smart_content_type":
            cons.extend(["most complex", "requires good metadata"])
        elif strategy_id == "flat_smart":
            cons.extend(["can become crowded", "less structure"])
        elif strategy_id == "collection_based":
            cons.extend(["very complex", "overkill for small libraries"])

        return cons


class MagicModeOrchestrator:
    """Main orchestrator for Magic Mode organization."""

    def __init__(self):
        self.analyzer = MagicAnalyzer()
        self.recommender = MagicStrategyRecommender()

    async def analyze_and_suggest(self, audio_files: List[AudioFile]) -> MagicSuggestion:
        """Analyze library and provide organization suggestions."""
        if not audio_files:
            raise MagicModeError("No audio files to analyze")

        # Analyze the library
        analysis = await self.analyzer.analyze_library(audio_files)

        # Get recommendations
        recommendations = await self.recommender.recommend_strategy(analysis)

        if not recommendations:
            raise MagicModeError("Could not generate organization recommendations")

        best_strategy = recommendations[0]

        # Generate quick wins
        quick_wins = self._generate_quick_wins(analysis, best_strategy)

        # Identify potential issues
        potential_issues = self._identify_potential_issues(analysis, best_strategy)

        # Generate custom rules
        custom_rules = self._generate_custom_rules(analysis)

        # Suggest preprocessing steps
        preprocessing_steps = self._generate_preprocessing_steps(analysis)

        return MagicSuggestion(
            strategy=best_strategy,
            analysis=analysis,
            quick_wins=quick_wins,
            potential_issues=potential_issues,
            custom_rules=custom_rules,
            preprocessing_steps=preprocessing_steps
        )

    def _generate_quick_wins(self, analysis: LibraryAnalysis, strategy: OrganizationStrategy) -> List[str]:
        """Generate quick wins that will improve organization."""
        quick_wins = []

        if analysis.metadata_completeness < 0.8:
            quick_wins.append("Fix missing metadata using MusicBrainz enhancement")

        if analysis.duplicate_likelihood > 0.1:
            quick_wins.append("Remove duplicates to clean up the library")

        if analysis.organization_chaos_score > 0.5:
            quick_wins.append("Consolidate scattered files into logical folders")

        if len(analysis.format_distribution) > 3:
            quick_wins.append("Convert or organize different audio formats consistently")

        if "genre" in strategy.path_pattern and len(analysis.genre_distribution) < 3:
            quick_wins.append("Add genre tags for better categorization")

        return quick_wins

    def _identify_potential_issues(self, analysis: LibraryAnalysis, strategy: OrganizationStrategy) -> List[str]:
        """Identify potential issues with the recommended strategy."""
        issues = []

        if analysis.metadata_completeness < 0.6:
            issues.append("Low metadata quality may result in poor organization")

        if analysis.total_files > 50000 and strategy.complexity == "complex":
            issues.append("Complex organization may be slow with very large libraries")

        if len(analysis.genre_distribution) > 20 and "genre" in strategy.path_pattern:
            issues.append("Too many genres may create deep folder structures")

        if analysis.duplicate_likelihood > 0.2:
            issues.append("High likelihood of duplicates may waste space")

        if analysis.organization_chaos_score > 0.8:
            issues.append("Very disorganized library may need manual cleanup first")

        return issues

    def _generate_custom_rules(self, analysis: LibraryAnalysis) -> List[Dict[str, Any]]:
        """Generate custom organization rules based on analysis."""
        rules = []

        # Rule for compilations
        if analysis.content_type_distribution.get("compilation", 0) > 5:
            rules.append({
                "name": "Compilation Handling",
                "condition": "is_compilation == true",
                "action": "organize_under: Compilations/{albumartist}/{album} ({year})",
                "priority": 80
            })

        # Rule for live recordings
        if analysis.content_type_distribution.get("live", 0) > 5:
            rules.append({
                "name": "Live Recordings",
                "condition": "content_type == 'live'",
                "action": "organize_under: Live/{artist}/{album} ({year})",
                "priority": 85
            })

        # Rule for various artists
        if analysis.content_type_distribution.get("compilation", 0) > analysis.total_files * 0.1:
            rules.append({
                "name": "Various Artists",
                "condition": "has_multiple_artists == true",
                "action": "organize_under: Compilations/{albumartist}/{album} ({year})",
                "priority": 90
            })

        # Rule for high-quality files
        high_quality_formats = ["flac", "wav", "aiff"]
        high_quality_count = sum(
            analysis.format_distribution.get(fmt, 0)
            for fmt in high_quality_formats
        )
        if high_quality_count > analysis.total_files * 0.3:
            rules.append({
                "name": "High Quality Separation",
                "condition": "format in ['flac', 'wav', 'aiff']",
                "action": "organize_under: High Quality/{artist}/{album} ({year})",
                "priority": 70
            })

        return rules

    def _generate_preprocessing_steps(self, analysis: LibraryAnalysis) -> List[str]:
        """Generate recommended preprocessing steps."""
        steps = []

        # Duplicate detection
        if analysis.duplicate_likelihood > 0.1:
            steps.append("Run duplicate detection and remove/rename duplicates")

        # Metadata enhancement
        if analysis.metadata_completeness < 0.7:
            steps.append("Run MusicBrainz metadata enhancement")

        # File format normalization
        dominant_format = max(analysis.format_distribution.items(), key=lambda x: x[1])[0]
        if len(analysis.format_distribution) > 2:
            steps.append(f"Consider converting non-{dominant_format} files for consistency")

        # Folder cleanup
        if analysis.organization_chaos_score > 0.6:
            steps.append("Manual cleanup of folder structure inconsistencies")

        # Filename normalization
        steps.append("Normalize filenames to consistent pattern")

        return steps

    async def generate_magic_config(self, suggestion: MagicSuggestion, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate a complete configuration based on the suggestion."""
        config = {
            "magic_mode": {
                "enabled": True,
                "strategy": suggestion.strategy.name,
                "confidence": suggestion.strategy.confidence,
                "generated_at": datetime.now().isoformat(),
                "analysis": {
                    "total_files": suggestion.analysis.total_files,
                    "metadata_completeness": suggestion.analysis.metadata_completeness,
                    "organization_chaos_score": suggestion.analysis.organization_chaos_score
                }
            },
            "organization": {
                "path_pattern": suggestion.strategy.path_pattern,
                "filename_pattern": suggestion.strategy.filename_pattern,
                "custom_rules": suggestion.custom_rules
            },
            "preprocessing": {
                "steps": suggestion.preprocessing_steps,
                "quick_wins": suggestion.quick_wins,
                "potential_issues": suggestion.potential_issues
            },
            "plugins": {
                "enabled": ["musicbrainz_enhancer", "duplicate_detector"],
                "config": {
                    "musicbrainz_enhancer": {
                        "enabled": True,
                        "enhance_fields": ["year", "genre", "albumartist"]
                    },
                    "duplicate_detector": {
                        "enabled": True,
                        "strategies": ["metadata", "file_hash"],
                        "min_confidence": 0.7
                    }
                }
            }
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        return config


# Convenience function for quick Magic Mode usage
async def analyze_music_library(audio_files: List[AudioFile]) -> MagicSuggestion:
    """Quick function to analyze a music library and get suggestions."""
    orchestrator = MagicModeOrchestrator()
    return await orchestrator.analyze_and_suggest(audio_files)


async def create_magic_organization_config(
    audio_files: List[AudioFile],
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Create a complete Magic Mode organization configuration."""
    orchestrator = MagicModeOrchestrator()
    suggestion = await orchestrator.analyze_and_suggest(audio_files)
    return await orchestrator.generate_magic_config(suggestion, output_path)