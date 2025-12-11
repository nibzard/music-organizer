"""Classification Context Domain Services.

This module defines domain services for the Classification bounded context.
Domain services contain business logic that doesn't naturally fit in entities or value objects.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

from ..result import Result, success, failure, collect
from .entities import Classifier, DuplicateGroup, ClassificationRule, ContentType
from .value_objects import (
    ContentTypeEnum,
    ContentTypeSignature,
    GenreClassification,
    AudioFeatures,
    EnergyLevel,
    ClassificationContext,
)


class ClassificationService:
    """Service for content classification operations."""

    def __init__(self, classifier: Classifier, max_workers: int = 4):
        self.classifier = classifier
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def classify_recording(
        self,
        recording: Any,  # catalog.Recording
        context: Optional[ClassificationContext] = None
    ) -> Result[Dict[str, Any], Exception]:
        """Classify a recording with all available methods."""
        try:
            classification_result = {
                "recording_id": id(recording),
                "content_type": ContentTypeEnum.UNKNOWN,
                "content_type_confidence": 0.0,
                "genres": [],
                "energy_level": EnergyLevel.MEDIUM,
                "features": None,
                "classification_source": "automatic",
                "confidence": 0.0
            }

            # Extract metadata
            metadata = recording.metadata if hasattr(recording, 'metadata') else recording

            # Extract audio features if available
            features = await self._extract_audio_features(recording)
            classification_result["features"] = features

            # Classify content type
            content_type, content_confidence = self.classifier.classify_content_type(
                metadata, features
            )
            classification_result["content_type"] = content_type
            classification_result["content_type_confidence"] = content_confidence

            # Classify genre
            genre_classification = self.classifier.classify_genre(metadata, features)
            classification_result["genres"] = genre_classification.get_all_genres()
            classification_result["genre_confidence"] = genre_classification.confidence_score

            # Get energy level from features
            if features:
                classification_result["energy_level"] = features.energy_level

            # Apply context-based adjustments
            if context:
                classification_result = self._apply_context_adjustments(
                    classification_result, context
                )

            # Calculate overall confidence
            classification_result["confidence"] = (
                content_confidence * 0.4 +
                genre_classification.confidence_score * 0.4 +
                (0.3 if features else 0.0)
            )

            return success(classification_result)
        except Exception as e:
            return failure(e)

    async def batch_classify(
        self,
        recordings: List[Any],
        context: Optional[ClassificationContext] = None
    ) -> Result[List[Dict[str, Any]], List[Exception]]:
        """Classify multiple recordings in parallel."""
        tasks = [
            self.classify_recording(recording, context)
            for recording in recordings
        ]
        results = await asyncio.gather(*tasks)
        return collect(results)

    async def learn_from_corrections(
        self,
        recording_id: str,
        correct_classification: Dict[str, Any],
        incorrect_classification: Dict[str, Any]
    ) -> None:
        """Learn from user corrections to improve classification."""
        # Extract what went wrong
        if correct_classification.get("content_type") != incorrect_classification.get("content_type"):
            # Content type was wrong
            correct_type = ContentTypeEnum(correct_classification["content_type"])
            await self._create_content_type_rule(
                recording_id,
                correct_type,
                correct_classification
            )

        if correct_classification.get("genres") != incorrect_classification.get("genres"):
            # Genres were wrong
            await self._create_genre_rule(
                recording_id,
                set(correct_classification.get("genres", [])),
                set(incorrect_classification.get("genres", []))
            )

    async def _extract_audio_features(self, recording: Any) -> Optional[AudioFeatures]:
        """Extract audio features from a recording."""
        # This is a placeholder implementation
        # In practice, you'd use audio analysis libraries like librosa
        # For now, we'll return None or basic features from metadata

        metadata = recording.metadata if hasattr(recording, 'metadata') else recording

        # Try to infer some basic features
        duration = getattr(metadata, 'duration_seconds', None)
        if duration:
            # Very basic energy inference based on duration and genre
            genre = getattr(metadata, 'genre', '').lower()
            energy = 0.5

            if any(g in genre for g in ['rock', 'metal', 'electronic', 'dance']):
                energy = 0.8
            elif any(g in genre for g in ['ambient', 'classical', 'folk']):
                energy = 0.3

            return AudioFeatures(
                energy=energy,
                # Other features would be extracted from actual audio analysis
            )

        return None

    def _apply_context_adjustments(
        self,
        classification: Dict[str, Any],
        context: ClassificationContext
    ) -> Dict[str, Any]:
        """Apply context-based adjustments to classification."""
        # Use user-specified genre if available
        user_genre = context.get_user_genre()
        if user_genre:
            classification["genres"] = [user_genre]
            classification["genre_confidence"] = 1.0
            classification["classification_source"] = "user_corrected"

        # Add user tags
        if context.user_tags:
            for tag in context.user_tags:
                if tag not in classification["genres"]:
                    classification["genres"].append(tag)

        # Apply confidence modifier
        modifier = context.get_confidence_modifier()
        classification["confidence"] *= modifier
        classification["confidence"] = min(1.0, classification["confidence"])

        return classification

    async def _create_content_type_rule(
        self,
        recording_id: str,
        correct_type: ContentTypeEnum,
        classification: Dict[str, Any]
    ) -> None:
        """Create a rule based on user correction for content type."""
        # This would analyze the recording's characteristics
        # and create a new classification rule
        pass

    async def _create_genre_rule(
        self,
        recording_id: str,
        correct_genres: Set[str],
        incorrect_genres: Set[str]
    ) -> None:
        """Create rules based on user genre corrections."""
        # This would analyze the recording's characteristics
        # and create rules to avoid similar mistakes
        pass


class DuplicateService:
    """Service for duplicate detection and management."""

    def __init__(self, classifier: Classifier, max_workers: int = 4):
        self.classifier = classifier
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def find_duplicates(
        self,
        recordings: List[Any],
        threshold: float = 0.85,
        strategy: str = "comprehensive"
    ) -> List[DuplicateGroup]:
        """Find duplicate recordings using specified strategy."""
        if strategy == "comprehensive":
            return await self._comprehensive_duplicate_search(recordings, threshold)
        elif strategy == "fast":
            return await self._fast_duplicate_search(recordings, threshold)
        elif strategy == "fingerprint_based":
            return await self._fingerprint_duplicate_search(recordings, threshold)
        else:
            raise ValueError(f"Unknown duplicate detection strategy: {strategy}")

    async def resolve_duplicate_group(
        self,
        group: DuplicateGroup,
        strategy: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resolve a duplicate group using the specified strategy."""
        resolution_result = {
            "group_id": group.id,
            "strategy": strategy,
            "actions": [],
            "errors": []
        }

        if strategy == "keep_best":
            best_id = self._select_best_recording(group, preferences)
            for recording_id in group.recordings:
                if recording_id != best_id:
                    resolution_result["actions"].append({
                        "action": "delete",
                        "recording_id": recording_id,
                        "reason": "duplicate - lower quality"
                    })

        elif strategy == "keep_all":
            # No action needed, just mark as resolved
            pass

        elif strategy == "move_duplicates":
            # Move duplicates to a separate folder
            best_id = self._select_best_recording(group, preferences)
            for recording_id in group.recordings:
                if recording_id != best_id:
                    resolution_result["actions"].append({
                        "action": "move",
                        "recording_id": recording_id,
                        "destination": "duplicates/",
                        "reason": "duplicate backup"
                    })

        else:
            resolution_result["errors"].append(f"Unknown resolution strategy: {strategy}")

        # Mark group as resolved
        if not resolution_result["errors"]:
            group.resolution_strategy = strategy
            # In practice, you'd record the timestamp

        return resolution_result

    async def _comprehensive_duplicate_search(
        self,
        recordings: List[Any],
        threshold: float
    ) -> List[DuplicateGroup]:
        """Perform comprehensive duplicate search with multiple criteria."""
        # Use the classifier's duplicate detection
        return self.classifier.detect_duplicates(recordings, threshold)

    async def _fast_duplicate_search(
        self,
        recordings: List[Any],
        threshold: float
    ) -> List[DuplicateGroup]:
        """Perform fast duplicate search using file hashes."""
        # Group by file size first (quick filter)
        size_groups: Dict[int, List[Any]] = {}
        for recording in recordings:
            size = getattr(recording.path, 'size_mb', 0) * 1024 * 1024  # Convert to bytes
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(recording)

        # Only check groups with multiple files of same size
        potential_duplicates = []
        for size, group in size_groups.items():
            if len(group) > 1:
                potential_duplicates.extend(group)

        # Now run full similarity check on potential duplicates
        if potential_duplicates:
            return self.classifier.detect_duplicates(potential_duplicates, threshold)

        return []

    async def _fingerprint_duplicate_search(
        self,
        recordings: List[Any],
        threshold: float
    ) -> List[DuplicateGroup]:
        """Perform fingerprint-based duplicate search."""
        # Group by acoustic fingerprint
        fingerprint_groups: Dict[str, List[Any]] = {}
        for recording in recordings:
            fingerprint = getattr(recording.metadata, 'acoustic_fingerprint', None)
            if fingerprint:
                if fingerprint not in fingerprint_groups:
                    fingerprint_groups[fingerprint] = []
                fingerprint_groups[fingerprint].append(recording)

        # Create groups from fingerprint matches
        duplicate_groups = []
        for fingerprint, group in fingerprint_groups.items():
            if len(group) > 1:
                duplicate_group = DuplicateGroup(
                    id=f"fingerprint_{len(duplicate_groups)}",
                    recordings=[id(r) for r in group],
                    threshold=threshold
                )
                duplicate_groups.append(duplicate_group)

        return duplicate_groups

    def _select_best_recording(
        self,
        group: DuplicateGroup,
        preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """Select the best recording from a duplicate group."""
        # In a real implementation, you'd have access to the actual Recording objects
        # and could compare factors like bitrate, file format, metadata completeness
        # For now, we'll use a simplified approach

        # If we have preferences, use them
        if preferences:
            preferred_format = preferences.get("preferred_format")
            if preferred_format:
                # Find recording with preferred format
                # This is simplified - in practice you'd iterate through recordings
                pass

        # Default to the first recording (placeholder)
        return group.recordings[0] if group.recordings else None


class ContentAnalysisService:
    """Service for deep content analysis and feature extraction."""

    def __init__(self, max_workers: int = 2):  # Usually CPU-intensive
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def analyze_audio_content(self, file_path: Path) -> AudioFeatures:
        """Analyze audio file to extract features."""
        def _analyze():
            # This is where you'd integrate with audio analysis libraries
            # like librosa, madmom, or other audio processing tools
            # For now, return placeholder values

            return AudioFeatures(
                tempo=120.0,  # Would be detected from audio
                key="C",      # Would be detected from audio
                mode="major", # Would be detected from audio
                energy=0.7,   # Would be calculated from audio
                danceability=0.6,
                valence=0.5,
                acousticness=0.3,
                instrumentalness=0.1,
                speechiness=0.05,
                loudness=-8.0
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _analyze)

    async def generate_fingerprint(self, file_path: Path) -> str:
        """Generate acoustic fingerprint for audio file."""
        def _generate_fingerprint():
            # This would integrate with audio fingerprinting libraries
            # like chromaprint, acoustid, or custom implementations
            # For now, generate a hash based on file properties

            stat = file_path.stat()
            # Create a simple hash from file size and modification time
            content = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.sha256(content.encode()).hexdigest()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _generate_fingerprint)

    async def batch_analyze(self, file_paths: List[Path]) -> List[AudioFeatures]:
        """Analyze multiple audio files in parallel."""
        tasks = [self.analyze_audio_content(path) for path in file_paths]
        return await asyncio.gather(*tasks)