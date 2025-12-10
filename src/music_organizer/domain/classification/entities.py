"""Classification Context Entities.

This module defines the core entities for the Classification bounded context.
The Classification context is responsible for content classification and duplicate detection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Iterator
from collections import defaultdict

from .value_objects import (
    ContentTypeEnum,
    ClassificationPattern,
    SimilarityThreshold,
    ContentTypeSignature,
    GenreClassification,
    FingerprintMatch,
    AudioFeatures,
    EnergyLevel,
    MatchType,
)


@dataclass
class SimilarityScore:
    """
    Represents similarity scores between two recordings.
    """

    recording_a_id: str
    recording_b_id: str
    title_similarity: float = 0.0
    artist_similarity: float = 0.0
    album_similarity: float = 0.0
    duration_similarity: float = 0.0
    acoustic_similarity: float = 0.0
    overall_similarity: float = 0.0

    @property
    def is_duplicate(self, threshold: float = 0.85) -> bool:
        """Check if the similarity indicates a duplicate."""
        return self.overall_similarity >= threshold

    def calculate_overall(self) -> float:
        """Calculate overall similarity score."""
        # Weighted average
        weights = {
            "title": 0.4,
            "artist": 0.35,
            "album": 0.15,
            "duration": 0.1
        }

        self.overall_similarity = (
            self.title_similarity * weights["title"] +
            self.artist_similarity * weights["artist"] +
            self.album_similarity * weights["album"] +
            self.duration_similarity * weights["duration"]
        )

        return self.overall_similarity


@dataclass
class ClassificationRule:
    """
    Represents a rule for classifying music content.
    """

    # Rule identity
    id: str
    name: str
    description: Optional[str] = None
    enabled: bool = True
    priority: int = 0

    # Rule conditions
    content_type: Optional[ContentTypeEnum] = None
    patterns: List[ClassificationPattern] = field(default_factory=list)
    genre_patterns: List[str] = field(default_factory=list)
    year_range: Optional[tuple[int, int]] = None
    duration_range: Optional[tuple[float, float]] = None  # in seconds
    required_metadata: List[str] = field(default_factory=list)

    # Rule action
    classification: Dict[str, Any] = field(default_factory=dict)
    confidence_boost: float = 0.0

    # Rule metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: Optional[datetime] = None
    usage_count: int = 0

    def matches(self, metadata: Any, features: Optional[AudioFeatures] = None) -> bool:
        """Check if this rule matches the given metadata."""
        if not self.enabled:
            return False

        # Check content type
        if self.content_type and hasattr(metadata, 'content_type'):
            if metadata.content_type != self.content_type.value:
                return False

        # Check patterns in title, artist, album
        title = getattr(metadata, 'title', '') or ''
        artists = [str(a) for a in getattr(metadata, 'artists', [])] or []
        album = getattr(metadata, 'album', '') or ''

        for pattern in self.patterns:
            matched = False
            if pattern.field in ('title', 'all') and pattern.matches(title):
                matched = True
            elif pattern.field in ('artist', 'all'):
                for artist in artists:
                    if pattern.matches(artist):
                        matched = True
                        break
            elif pattern.field in ('album', 'all') and pattern.matches(album):
                matched = True

            if not matched:
                return False

        # Check genre patterns
        if self.genre_patterns:
            genre = getattr(metadata, 'genre', '') or ''
            genre_lower = genre.lower()
            if not any(pattern.lower() in genre_lower for pattern in self.genre_patterns):
                return False

        # Check year range
        if self.year_range:
            year = getattr(metadata, 'year', None)
            if year and not (self.year_range[0] <= year <= self.year_range[1]):
                return False

        # Check duration range
        if self.duration_range and features:
            duration = getattr(metadata, 'duration_seconds', 0)
            if not (self.duration_range[0] <= duration <= self.duration_range[1]):
                return False

        # Check required metadata
        for field in self.required_metadata:
            if not hasattr(metadata, field) or not getattr(metadata, field):
                return False

        return True

    def apply(self, metadata: Any) -> Dict[str, Any]:
        """Apply the rule to get classification."""
        result = self.classification.copy()
        result['confidence'] = result.get('confidence', 0.5) + self.confidence_boost
        result['rule_id'] = self.id
        result['rule_name'] = self.name
        return result

    def increment_usage(self) -> None:
        """Increment usage count."""
        self.usage_count += 1
        self.last_modified = datetime.now()


@dataclass
class ContentType:
    """
    Represents a type of music content with classification rules.
    """

    # Type identity
    type_enum: ContentTypeEnum
    name: str
    description: Optional[str] = None

    # Classification criteria
    signature: Optional[ContentTypeSignature] = None
    rules: List[ClassificationRule] = field(default_factory=list)

    # Common characteristics
    typical_duration_range: Optional[tuple[float, float]] = None
    typical_keywords: Set[str] = field(default_factory=set)
    forbidden_keywords: Set[str] = field(default_factory=set)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: Optional[datetime] = None

    def classify(self, metadata: Any, features: Optional[AudioFeatures] = None) -> float:
        """Calculate confidence score for this content type."""
        score = 0.0

        # Check signature
        if self.signature:
            title = getattr(metadata, 'title', '') or ''
            artists = [str(a) for a in getattr(metadata, 'artists', [])] or []
            album = getattr(metadata, 'album', '') or ''
            genre = getattr(metadata, 'genre', '') or ''
            score += self.signature.matches(metadata, title, artists, album, genre)

        # Check rules
        for rule in self.rules:
            if rule.matches(metadata, features):
                classification = rule.apply(metadata)
                score = max(score, classification.get('confidence', 0))

        # Check duration
        if self.typical_duration_range and features:
            duration = getattr(metadata, 'duration_seconds', 0)
            if self.typical_duration_range[0] <= duration <= self.typical_duration_range[1]:
                score += 0.1

        return min(1.0, score)

    def add_rule(self, rule: ClassificationRule) -> None:
        """Add a classification rule."""
        self.rules.append(rule)
        # Sort by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        self.last_modified = datetime.now()


@dataclass
class DuplicateGroup:
    """
    Represents a group of duplicate recordings.
    """

    # Group identity
    id: str
    recordings: List[str] = field(default_factory=list)  # Recording IDs
    similarity_matrix: Dict[str, Dict[str, SimilarityScore]] = field(default_factory=dict)
    threshold: float = 0.85

    # Group metadata
    best_recording_id: Optional[str] = None
    resolution_strategy: Optional[str] = None
    resolved_at: Optional[datetime] = None

    @property
    def size(self) -> int:
        """Get number of recordings in the group."""
        return len(self.recordings)

    @property
    def is_resolved(self) -> bool:
        """Check if the duplicate group has been resolved."""
        return self.resolved_at is not None

    def add_recording(self, recording_id: str) -> None:
        """Add a recording to the duplicate group."""
        if recording_id not in self.recordings:
            self.recordings.append(recording_id)

    def remove_recording(self, recording_id: str) -> None:
        """Remove a recording from the duplicate group."""
        if recording_id in self.recordings:
            self.recordings.remove(recording_id)
            # Clean up similarity matrix
            self.similarity_matrix.pop(recording_id, None)
            for scores in self.similarity_matrix.values():
                scores.pop(recording_id, None)

    def get_similarity(self, recording_a: str, recording_b: str) -> Optional[SimilarityScore]:
        """Get similarity score between two recordings."""
        return self.similarity_matrix.get(recording_a, {}).get(recording_b)

    def set_similarity(self, score: SimilarityScore) -> None:
        """Set similarity score between two recordings."""
        if score.recording_a_id not in self.similarity_matrix:
            self.similarity_matrix[score.recording_a_id] = {}
        self.similarity_matrix[score.recording_a_id][score.recording_b_id] = score

    def find_best_recording(self) -> Optional[str]:
        """Find the best recording based on quality criteria."""
        # This is a simplified implementation
        # In practice, you'd consider factors like bitrate, file format, metadata completeness
        if not self.recordings:
            return None

        # For now, just return the first one
        # A real implementation would score each recording
        return self.recordings[0]

    def get_all_pairs(self) -> Iterator[tuple[str, str, SimilarityScore]]:
        """Get all pairs of recordings and their similarities."""
        for i, rec_a in enumerate(self.recordings):
            for rec_b in self.recordings[i+1:]:
                score = self.get_similarity(rec_a, rec_b)
                if score:
                    yield rec_a, rec_b, score


@dataclass
class Classifier:
    """
    Entity responsible for classifying music content.
    """

    # Classifier identity
    name: str
    version: str = "1.0"
    description: Optional[str] = None

    # Classification configuration
    content_types: Dict[ContentTypeEnum, ContentType] = field(default_factory=dict)
    classification_rules: List[ClassificationRule] = field(default_factory=list)
    similarity_threshold: SimilarityThreshold = field(default_factory=SimilarityThreshold)

    # Classifier state
    is_trained: bool = False
    training_data_count: int = 0
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_trained: Optional[datetime] = None

    def classify_content_type(
        self,
        metadata: Any,
        features: Optional[AudioFeatures] = None
    ) -> tuple[ContentTypeEnum, float]:
        """Classify the content type of a recording."""
        best_type = ContentTypeEnum.UNKNOWN
        best_score = 0.0

        for content_type_enum, content_type in self.content_types.items():
            score = content_type.classify(metadata, features)
            if score > best_score:
                best_score = score
                best_type = content_type_enum

        return best_type, best_score

    def classify_genre(
        self,
        metadata: Any,
        features: Optional[AudioFeatures] = None
    ) -> GenreClassification:
        """Classify the genre(s) of a recording."""
        classification = GenreClassification()

        # Extract from metadata first
        metadata_genre = getattr(metadata, 'genre', None)
        if metadata_genre:
            classification.add_genre(metadata_genre, 0.8)

        # Apply classification rules
        for rule in self.classification_rules:
            if rule.matches(metadata, features):
                result = rule.apply(metadata)
                if 'genre' in result:
                    confidence = result.get('confidence', 0.5)
                    classification.add_genre(result['genre'], confidence)

        # Infer from audio features if available
        if features:
            # Simple heuristic rules based on audio features
            if features.energy > 0.7:
                classification.add_genre("electronic", 0.3)
                classification.add_genre("rock", 0.3)
            elif features.energy < 0.3:
                classification.add_genre("ambient", 0.3)
                classification.add_genre("classical", 0.3)

            if features.acousticness > 0.7:
                classification.add_genre("acoustic", 0.4)
                classification.add_genre("folk", 0.3)

            if features.danceability > 0.7:
                classification.add_genre("dance", 0.4)
                classification.add_genre("electronic", 0.3)

        # Set confidence score
        all_genres = classification.get_all_genres()
        if all_genres:
            classification.confidence_score = min(1.0, len(all_genres) * 0.2)

        return classification

    def detect_duplicates(
        self,
        recordings: List[Any],  # List of Recording objects
        threshold: Optional[float] = None
    ) -> List[DuplicateGroup]:
        """Detect duplicate recordings among a list."""
        if threshold is None:
            threshold = self.similarity_threshold.overall_threshold

        duplicate_groups: List[DuplicateGroup] = []
        processed = set()

        for i, recording_a in enumerate(recordings):
            if id(recording_a) in processed:
                continue

            # Start a new potential duplicate group
            current_group = DuplicateGroup(
                id=f"group_{len(duplicate_groups)}",
                threshold=threshold
            )

            # Find all similar recordings
            for recording_b in recordings[i+1:]:
                if id(recording_b) in processed:
                    continue

                # Calculate similarity
                similarity = self._calculate_similarity(recording_a, recording_b)

                if similarity.overall_similarity >= threshold:
                    # Add to group if not already there
                    if not current_group.recordings:
                        current_group.add_recording(id(recording_a))
                        processed.add(id(recording_a))

                    current_group.add_recording(id(recording_b))
                    processed.add(id(recording_b))

                    # Store similarity score
                    current_group.set_similarity(similarity)

            # If we found duplicates, add the group
            if current_group.size > 1:
                duplicate_groups.append(current_group)

        return duplicate_groups

    def _calculate_similarity(self, recording_a: Any, recording_b: Any) -> SimilarityScore:
        """Calculate similarity between two recordings."""
        score = SimilarityScore(
            recording_a_id=id(recording_a),
            recording_b_id=id(recording_b)
        )

        # Title similarity
        title_a = getattr(recording_a, 'title', '') or ''
        title_b = getattr(recording_b, 'title', '') or ''
        score.title_similarity = self._string_similarity(title_a, title_b)

        # Artist similarity
        artists_a = [str(a) for a in getattr(recording_a, 'artists', [])] or []
        artists_b = [str(a) for a in getattr(recording_b, 'artists', [])] or []
        score.artist_similarity = self._artist_similarity(artists_a, artists_b)

        # Album similarity
        album_a = getattr(recording_a, 'album', '') or ''
        album_b = getattr(recording_b, 'album', '') or ''
        score.album_similarity = self._string_similarity(album_a, album_b)

        # Duration similarity
        duration_a = getattr(recording_a, 'duration_seconds', 0) or 0
        duration_b = getattr(recording_b, 'duration_seconds', 0) or 0
        if duration_a and duration_b:
            diff = abs(duration_a - duration_b)
            score.duration_similarity = max(0, 1 - diff / 30)  # 30 second tolerance

        # Acoustic fingerprint similarity if available
        fingerprint_a = getattr(recording_a.metadata, 'acoustic_fingerprint', None)
        fingerprint_b = getattr(recording_b.metadata, 'acoustic_fingerprint', None)
        if fingerprint_a and fingerprint_b:
            score.acoustic_similarity = 1.0 if fingerprint_a == fingerprint_b else 0.0

        # Calculate overall
        score.calculate_overall()

        return score

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity."""
        if not s1 or not s2:
            return 0.0
        if s1.lower() == s2.lower():
            return 1.0

        # Simple word-based similarity
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _artist_similarity(self, artists_a: List[str], artists_b: List[str]) -> float:
        """Calculate artist list similarity."""
        if not artists_a or not artists_b:
            return 0.0

        # Normalize artist names
        normalized_a = [a.lower().strip().replace("the ", "") for a in artists_a]
        normalized_b = [b.lower().strip().replace("the ", "") for b in artists_b]

        # Check for exact matches
        matches = 0
        for a in normalized_a:
            for b in normalized_b:
                if a == b:
                    matches += 1
                    break

        if matches > 0:
            return matches / max(len(normalized_a), len(normalized_b))

        return 0.0

    def train(self, training_data: List[tuple[Any, Dict[str, Any]]]) -> None:
        """Train the classifier with labeled data."""
        # This is a placeholder for ML-based training
        # For now, we'll just update the training count
        self.training_data_count += len(training_data)
        self.is_trained = True
        self.last_trained = datetime.now()

    def add_content_type(self, content_type: ContentType) -> None:
        """Add a content type to the classifier."""
        self.content_types[content_type.type_enum] = content_type

    def add_classification_rule(self, rule: ClassificationRule) -> None:
        """Add a classification rule."""
        self.classification_rules.append(rule)
        # Sort by priority
        self.classification_rules.sort(key=lambda r: r.priority, reverse=True)