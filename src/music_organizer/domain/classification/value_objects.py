"""Classification Context Value Objects.

This module defines value objects for the Classification bounded context.
Value objects are immutable objects that are defined by their attributes rather than identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class ContentTypeEnum(Enum):
    """Types of music content."""
    STUDIO_ALBUM = "studio_album"
    LIVE_RECORDING = "live_recording"
    COMPILATION = "compilation"
    SOUNDTRACK = "soundtrack"
    DJ_MIX = "dj_mix"
    REMIX = "remix"
    COVER = "cover"
    DEMO = "demo"
    SINGLE = "single"
    EP = "ep"
    INTERVIEW = "interview"
    SPOKEN_WORD = "spoken_word"
    AUDIOBOOK = "audiobook"
    PODCAST = "podcast"
    UNKNOWN = "unknown"


class EnergyLevel(Enum):
    """Energy level classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class MatchType(Enum):
    """Types of pattern matching."""
    EXACT = "exact"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    FUZZY = "fuzzy"


@dataclass(frozen=True, slots=True)
class ClassificationPattern:
    """
    Value object for pattern matching in classification.
    """

    pattern: str
    match_type: MatchType = MatchType.CONTAINS
    case_sensitive: bool = False
    weight: float = 1.0
    field: str = "all"  # Field to match against: title, artist, album, genre, all

    def matches(self, text: str) -> bool:
        """Check if the pattern matches the given text."""
        if not text or not self.pattern:
            return False

        # Prepare text for matching
        text_to_match = text if self.case_sensitive else text.lower()
        pattern_to_match = self.pattern if self.case_sensitive else self.pattern.lower()

        if self.match_type == MatchType.EXACT:
            return text_to_match == pattern_to_match
        elif self.match_type == MatchType.CONTAINS:
            return pattern_to_match in text_to_match
        elif self.match_type == MatchType.STARTS_WITH:
            return text_to_match.startswith(pattern_to_match)
        elif self.match_type == MatchType.ENDS_WITH:
            return text_to_match.endswith(pattern_to_match)
        elif self.match_type == MatchType.REGEX:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                return bool(re.search(pattern_to_match, text, flags))
            except re.error:
                return False
        elif self.match_type == MatchType.FUZZY:
            return self._fuzzy_match(text_to_match, pattern_to_match)

        return False

    def _fuzzy_match(self, text: str, pattern: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching using Levenshtein distance."""
        if not text or not pattern:
            return False

        # If pattern is much longer than text, it's unlikely to match
        if len(pattern) > len(text) * 1.5:
            return False

        # Calculate similarity
        distance = self._levenshtein_distance(text, pattern)
        similarity = 1 - distance / max(len(text), len(pattern))

        return similarity >= threshold

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


@dataclass(frozen=True, slots=True)
class SimilarityThreshold:
    """
    Value object defining similarity thresholds for classification.
    """

    title_similarity: float = 0.8
    artist_similarity: float = 0.9
    album_similarity: float = 0.7
    overall_threshold: float = 0.85

    def is_similar(self, scores: Dict[str, float]) -> bool:
        """Check if similarity scores meet the threshold criteria."""
        # Check overall threshold
        overall_score = (
            scores.get("title", 0) * 0.4 +
            scores.get("artist", 0) * 0.35 +
            scores.get("album", 0) * 0.15 +
            scores.get("duration", 0) * 0.1
        )

        # Must meet overall threshold
        if overall_score < self.overall_threshold:
            return False

        # Check individual thresholds
        if scores.get("title", 0) < self.title_similarity:
            return False

        if scores.get("artist", 0) < self.artist_similarity:
            return False

        # Album is less critical
        if scores.get("album", 0) < self.album_similarity * 0.5:  # Allow more lenience
            pass

        return True


@dataclass(frozen=True, slots=True)
class ContentTypeSignature:
    """
    Value object representing the signature of a content type.
    """

    content_type: ContentTypeEnum
    patterns: List[ClassificationPattern] = field(default_factory=list)
    keywords: Set[str] = field(default_factory=set)
    excluded_keywords: Set[str] = field(default_factory=set)
    required_metadata: List[str] = field(default_factory=list)
    forbidden_metadata: List[str] = field(default_factory=list)

    def matches(self, metadata: Any, title: str, artists: List[str], album: str, genre: str) -> float:
        """Calculate how well this signature matches the given metadata."""
        score = 0.0
        total_checks = 0

        # Check patterns in title
        for pattern in self.patterns:
            if pattern.field in ("title", "all"):
                total_checks += pattern.weight
                if pattern.matches(title):
                    score += pattern.weight

        # Check patterns in artist
        for artist in artists:
            for pattern in self.patterns:
                if pattern.field in ("artist", "all"):
                    total_checks += pattern.weight
                    if pattern.matches(artist):
                        score += pattern.weight

        # Check patterns in album
        if album:
            for pattern in self.patterns:
                if pattern.field in ("album", "all"):
                    total_checks += pattern.weight
                    if pattern.matches(album):
                        score += pattern.weight

        # Check keywords
        all_text = " ".join([title] + artists + [album, genre]).lower()
        for keyword in self.keywords:
            total_checks += 1
            if keyword.lower() in all_text:
                score += 1

        # Check excluded keywords (negative scoring)
        for keyword in self.excluded_keywords:
            if keyword.lower() in all_text:
                score -= 2  # Penalty for excluded keywords

        # Normalize score
        return max(0, score / total_checks if total_checks > 0 else 0)


@dataclass(frozen=True, slots=True)
class GenreClassification:
    """
    Value object for genre classification results.
    """

    primary_genre: Optional[str] = None
    secondary_genres: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    subgenres: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

    def add_genre(self, genre: str, confidence: float = 1.0) -> None:
        """Add a genre with confidence score."""
        if confidence > 0.7 and not self.primary_genre:
            self.primary_genre = genre
        elif confidence > 0.5:
            if genre not in self.secondary_genres:
                self.secondary_genres.append(genre)
        elif confidence > 0.3:
            if genre not in self.subgenres:
                self.subgenres.append(genre)

    def has_genre(self, genre: str) -> bool:
        """Check if a genre is present in any category."""
        return (
            (self.primary_genre == genre) or
            (genre in self.secondary_genres) or
            (genre in self.subgenres) or
            (genre in self.tags)
        )

    def get_all_genres(self) -> List[str]:
        """Get all genres in order of confidence."""
        genres = []
        if self.primary_genre:
            genres.append(self.primary_genre)
        genres.extend(self.secondary_genres)
        genres.extend(self.subgenres)
        genres.extend(sorted(self.tags))
        return genres


@dataclass(frozen=True, slots=True)
class FingerprintMatch:
    """
    Value object representing a fingerprint match result.
    """

    track_id: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    match_type: str = "acoustic"  # acoustic, metadata, hybrid

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high confidence match."""
        return self.confidence >= 0.9

    @property
    def is_reliable(self) -> bool:
        """Check if this match is reliable."""
        return self.confidence >= 0.75


@dataclass(frozen=True, slots=True)
class AudioFeatures:
    """
    Value object representing extracted audio features.
    """

    tempo: Optional[float] = None
    key: Optional[str] = None
    mode: Optional[str] = None  # major/minor
    energy: Optional[float] = None
    danceability: Optional[float] = None
    valence: Optional[float] = None  # musical positiveness
    acousticness: Optional[float] = None
    instrumentalness: Optional[float] = None
    speechiness: Optional[float] = None
    loudness: Optional[float] = None

    @property
    def energy_level(self) -> EnergyLevel:
        """Get energy level classification."""
        if self.energy is None:
            return EnergyLevel.MEDIUM

        if self.energy < 0.2:
            return EnergyLevel.VERY_LOW
        elif self.energy < 0.4:
            return EnergyLevel.LOW
        elif self.energy < 0.6:
            return EnergyLevel.MEDIUM
        elif self.energy < 0.8:
            return EnergyLevel.HIGH
        else:
            return EnergyLevel.VERY_HIGH

    @property
    def is_acoustic(self) -> bool:
        """Check if track is likely acoustic."""
        return (self.acousticness or 0) > 0.5

    @property
    def is_instrumental(self) -> bool:
        """Check if track is instrumental."""
        return (self.instrumentalness or 0) > 0.5

    @property
    def is_speech(self) -> bool:
        """Check if track contains speech."""
        return (self.speechness or 0) > 0.5


@dataclass(frozen=True, slots=True)
class ClassificationContext:
    """
    Value object providing context for classification decisions.
    """

    file_path: Optional[str] = None
    directory_structure: Optional[str] = None
    file_naming_pattern: Optional[str] = None
    source_confidence: str = "medium"  # low, medium, high
    user_tags: Set[str] = field(default_factory=set)
    user_corrections: Dict[str, str] = field(default_factory=dict)
    similar_tracks: List[str] = field(default_factory=list)

    def has_user_input(self) -> bool:
        """Check if user has provided input for classification."""
        return bool(self.user_tags or self.user_corrections)

    def get_user_genre(self) -> Optional[str]:
        """Get user-specified genre if available."""
        return self.user_corrections.get("genre")

    def get_confidence_modifier(self) -> float:
        """Get confidence modifier based on source."""
        modifiers = {
            "low": 0.7,
            "medium": 1.0,
            "high": 1.2
        }
        return modifiers.get(self.source_confidence, 1.0)


# Type alias for backward compatibility
ContentType = ContentTypeEnum