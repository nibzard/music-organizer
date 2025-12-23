"""Mood classifier using audio features.

This module implements a mood classifier that uses audio features
(valence, energy, danceability, acousticness) to predict musical mood.

Moods are mapped to positions in a 2D valence-energy space:
- High valence + High energy = Happy, Energetic, Uplifting
- High valence + Low energy = Relaxed, Calm
- Low valence + High energy = Angry, Dark
- Low valence + Low energy = Sad, Melancholic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseModel, ModelNotAvailableError, RuleBasedFallback
from music_organizer.domain.classification.value_objects import Mood, AudioFeatures


# Mood definitions with valence/energy thresholds
MOOD_DEFINITIONS: Dict[Mood, Dict[str, Any]] = {
    Mood.HAPPY: {
        "valence_range": (0.6, 1.0),
        "energy_range": (0.5, 1.0),
        "danceability_range": (0.5, 1.0),
        "description": "Positive and energetic",
    },
    Mood.SAD: {
        "valence_range": (0.0, 0.4),
        "energy_range": (0.0, 0.5),
        "danceability_range": (0.0, 0.5),
        "description": "Negative and low energy",
    },
    Mood.ENERGETIC: {
        "valence_range": (0.4, 1.0),
        "energy_range": (0.7, 1.0),
        "danceability_range": (0.6, 1.0),
        "description": "High energy regardless of valence",
    },
    Mood.CALM: {
        "valence_range": (0.4, 0.8),
        "energy_range": (0.0, 0.4),
        "danceability_range": (0.0, 0.4),
        "description": "Positive and low energy",
    },
    Mood.ANGRY: {
        "valence_range": (0.0, 0.4),
        "energy_range": (0.6, 1.0),
        "danceability_range": (0.3, 0.8),
        "description": "Negative and high energy",
    },
    Mood.RELAXED: {
        "valence_range": (0.5, 1.0),
        "energy_range": (0.0, 0.4),
        "acousticness_range": (0.3, 1.0),
        "description": "Positive, low energy, often acoustic",
    },
    Mood.MELANCHOLIC: {
        "valence_range": (0.2, 0.5),
        "energy_range": (0.0, 0.5),
        "acousticness_range": (0.2, 1.0),
        "description": "Mildly negative, low to medium energy",
    },
    Mood.UPLIFTING: {
        "valence_range": (0.7, 1.0),
        "energy_range": (0.5, 1.0),
        "danceability_range": (0.4, 1.0),
        "description": "Very positive, inspiring",
    },
    Mood.DARK: {
        "valence_range": (0.0, 0.4),
        "energy_range": (0.3, 0.8),
        "acousticness_range": (0.0, 0.6),
        "description": "Negative, often electronic/heavy",
    },
    Mood.BRIGHT: {
        "valence_range": (0.6, 1.0),
        "energy_range": (0.3, 0.8),
        "description": "Positive and clear",
    },
}


@dataclass
class MoodClassificationResult:
    """Result of mood classification."""

    moods: List[Mood] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    primary_mood: Optional[Mood] = None
    method: str = "unknown"  # ml, rule_based, features
    valence: Optional[float] = None
    energy: Optional[float] = None

    @property
    def confidence(self) -> float:
        """Get overall confidence score (max of all scores)."""
        return max(self.confidence_scores.values()) if self.confidence_scores else 0.0

    def add_mood(self, mood: Mood, confidence: float) -> None:
        """Add a mood with confidence score."""
        mood_str = mood.value if isinstance(mood, Mood) else mood

        if mood_str not in [m.value for m in self.moods]:
            self.moods.append(mood)

        self.confidence_scores[mood_str] = max(
            self.confidence_scores.get(mood_str, 0.0),
            confidence
        )

        # Update primary mood if this is the highest confidence
        if self.primary_mood is None or confidence > self.confidence_scores.get(self.primary_mood.value, 0.0):
            self.primary_mood = mood

    def get_top_moods(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get top N moods by confidence."""
        sorted_moods = sorted(
            self.confidence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_moods[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "moods": [m.value for m in self.moods],
            "confidence_scores": self.confidence_scores,
            "primary_mood": self.primary_mood.value if self.primary_mood else None,
            "confidence": self.confidence,
            "method": self.method,
            "valence": self.valence,
            "energy": self.energy,
        }


def _classify_mood_from_features(
    valence: Optional[float],
    energy: Optional[float],
    danceability: Optional[float],
    acousticness: Optional[float],
) -> MoodClassificationResult:
    """Classify mood from raw audio features using rule-based mapping.

    This is a fallback method when ML model is not available.
    It maps the valence-energy space to mood categories.

    Args:
        valence: Musical positiveness (0.0-1.0)
        energy: Energy level (0.0-1.0)
        danceability: Danceability score (0.0-1.0)
        acousticness: Acousticness score (0.0-1.0)

    Returns:
        MoodClassificationResult with predicted moods.
    """
    result = MoodClassificationResult(method="features")
    result.valence = valence
    result.energy = energy

    # If features are missing, return unknown
    if valence is None or energy is None:
        result.add_mood(Mood.UNKNOWN, 1.0)
        return result

    # Score each mood based on feature matching
    mood_scores: Dict[Mood, float] = {}

    for mood, definition in MOOD_DEFINITIONS.items():
        score = 0.0
        weight_sum = 0.0

        # Check valence range
        if "valence_range" in definition:
            v_min, v_max = definition["valence_range"]
            if v_min <= valence <= v_max:
                # Score based on how centered in the range
                v_center = (v_min + v_max) / 2
                v_width = v_max - v_min
                distance = abs(valence - v_center) / (v_width / 2 + 0.01)
                score += 2.0 * (1.0 - min(distance, 1.0))
                weight_sum += 2.0

        # Check energy range
        if "energy_range" in definition:
            e_min, e_max = definition["energy_range"]
            if e_min <= energy <= e_max:
                e_center = (e_min + e_max) / 2
                e_width = e_max - e_min
                distance = abs(energy - e_center) / (e_width / 2 + 0.01)
                score += 2.0 * (1.0 - min(distance, 1.0))
                weight_sum += 2.0

        # Check danceability range
        if danceability is not None and "danceability_range" in definition:
            d_min, d_max = definition["danceability_range"]
            if d_min <= danceability <= d_max:
                d_center = (d_min + d_max) / 2
                d_width = d_max - d_min
                distance = abs(danceability - d_center) / (d_width / 2 + 0.01)
                score += 1.0 * (1.0 - min(distance, 1.0))
                weight_sum += 1.0

        # Check acousticness range
        if acousticness is not None and "acousticness_range" in definition:
            a_min, a_max = definition["acousticness_range"]
            if a_min <= acousticness <= a_max:
                a_center = (a_min + a_max) / 2
                a_width = a_max - a_min
                distance = abs(acousticness - a_center) / (a_width / 2 + 0.01)
                score += 1.0 * (1.0 - min(distance, 1.0))
                weight_sum += 1.0

        # Normalize score
        if weight_sum > 0:
            mood_scores[mood] = score / weight_sum

    # Sort moods by score and add top ones
    sorted_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)

    if not sorted_moods or sorted_moods[0][1] < 0.3:
        # No clear mood match
        result.add_mood(Mood.UNKNOWN, 1.0)
    else:
        # Add top matching moods with diminishing scores
        for mood, score in sorted_moods[:3]:
            confidence = max(score, 0.1)  # Minimum confidence
            result.add_mood(mood, confidence)

    return result


class MoodClassifier(BaseModel):
    """ML-based mood classifier with rule-based fallback.

    The classifier predicts musical mood based on audio features:
    - Valence (musical positiveness)
    - Energy (loudness/dynamics)
    - Danceability (rhythmic patterns)
    - Acousticness (electronic vs organic)

    Usage:
        classifier = MoodClassifier()
        result = classifier.predict_from_features(
            valence=0.8,
            energy=0.7,
            danceability=0.6
        )
        print(result.primary_mood)  # Mood.HAPPY

        # Or classify from AudioFeatures value object
        result = classifier.predict(audio_features)
    """

    model_name = "mood_classifier"
    model_version = "1.0.0"
    model_type = "mood_classification"

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the mood classifier.

        Args:
            model_path: Path to trained model file. If None, uses default.
        """
        super().__init__(model_path)
        self._model: Any = None

    def _get_default_model_path(self) -> Path:
        """Get the default path for this model file."""
        return self.MODELS_DIR / "mood_classifier.pkl"

    def predict(
        self,
        audio_features: Optional[AudioFeatures] = None,
        valence: Optional[float] = None,
        energy: Optional[float] = None,
        danceability: Optional[float] = None,
        acousticness: Optional[float] = None,
    ) -> MoodClassificationResult:
        """Predict mood from audio features.

        Args:
            audio_features: AudioFeatures value object with all features.
            valence: Musical positiveness (0.0-1.0). Overrides audio_features if provided.
            energy: Energy level (0.0-1.0). Overrides audio_features if provided.
            danceability: Danceability (0.0-1.0). Overrides audio_features if provided.
            acousticness: Acousticness (0.0-1.0). Overrides audio_features if provided.

        Returns:
            MoodClassificationResult with predicted moods and confidence scores.
        """
        # Extract features from AudioFeatures if provided
        if audio_features:
            v = valence or audio_features.valence
            e = energy or audio_features.energy
            d = danceability or audio_features.danceability
            a = acousticness or audio_features.acousticness
        else:
            v = valence
            e = energy
            d = danceability
            a = acousticness

        # Try ML model if available
        if self._is_loaded and self._model is not None:
            try:
                return self._predict_with_model(v, e, d, a)
            except Exception:
                # Fall through to rule-based
                pass

        # Use rule-based classification
        return _classify_mood_from_features(v, e, d, a)

    def _predict_with_model(
        self,
        valence: Optional[float],
        energy: Optional[float],
        danceability: Optional[float],
        acousticness: Optional[float],
    ) -> MoodClassificationResult:
        """Predict using trained ML model.

        This method is called when a trained model is available.
        For now, it falls back to rule-based classification.

        TODO: Implement actual ML model prediction.
        """
        # TODO: Implement ML model prediction
        # For now, fall back to rule-based
        return _classify_mood_from_features(valence, energy, danceability, acousticness)

    def predict_from_features(
        self,
        audio_features: AudioFeatures,
    ) -> MoodClassificationResult:
        """Predict mood from AudioFeatures value object.

        Convenience method for classification using AudioFeatures.

        Args:
            audio_features: AudioFeatures value object.

        Returns:
            MoodClassificationResult with predicted moods.
        """
        return self.predict(
            audio_features=audio_features,
        )

    def predict_from_text(
        self,
        text: str,
    ) -> MoodClassificationResult:
        """Predict mood from text metadata (title, album, etc.).

        Uses keyword-based heuristics to infer mood when audio features
        are not available.

        Args:
            text: Combined text from title, album, artist, etc.

        Returns:
            MoodClassificationResult with predicted moods.
        """
        result = MoodClassificationResult(method="text_keywords")
        text_lower = text.lower()

        # Mood keyword patterns
        mood_keywords: Dict[Mood, List[str]] = {
            Mood.HAPPY: ["happy", "joy", "celebration", "upbeat", "fun", "party"],
            Mood.SAD: ["sad", "cry", "tears", "heartbreak", "lonely", "melancholy"],
            Mood.ENERGETIC: ["energy", "power", "blast", "intense", "fierce"],
            Mood.CALM: ["calm", "peace", "serenity", "gentle", "quiet"],
            Mood.ANGRY: ["angry", "rage", "fury", "hate", "aggressive"],
            Mood.RELAXED: ["relax", "chill", "lounge", "mellow", "easy"],
            Mood.MELANCHOLIC: ["melanchol", "nostalgia", "wistful", "longing"],
            Mood.UPLIFTING: ["uplift", "inspire", "hope", "rise", "ascend"],
            Mood.DARK: ["dark", "gloomy", "bleak", "shadow", "nightmare"],
            Mood.BRIGHT: ["bright", "light", "sunny", "radiant", "shine"],
        }

        # Score moods based on keyword matches
        mood_scores: Dict[str, float] = {}
        for mood, keywords in mood_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                mood_scores[mood.value] = min(matches * 0.5, 1.0)

        if mood_scores:
            # Sort by score and add top moods
            for mood_str, score in sorted(mood_scores.items(), key=lambda x: x[1], reverse=True):
                mood = Mood(mood_str)
                result.add_mood(mood, score)
        else:
            result.add_mood(Mood.UNKNOWN, 1.0)

        return result


def classify_mood(
    valence: Optional[float] = None,
    energy: Optional[float] = None,
    danceability: Optional[float] = None,
    acousticness: Optional[float] = None,
    audio_features: Optional[AudioFeatures] = None,
    text: Optional[str] = None,
) -> List[str]:
    """Convenience function to classify mood.

    Args:
        valence: Musical positiveness (0.0-1.0).
        energy: Energy level (0.0-1.0).
        danceability: Danceability (0.0-1.0).
        acousticness: Acousticness (0.0-1.0).
        audio_features: AudioFeatures value object.
        text: Text metadata for keyword-based classification.

    Returns:
        List of mood strings (e.g., ["happy", "energetic"]).
    """
    classifier = MoodClassifier()

    if text and not audio_features and valence is None:
        result = classifier.predict_from_text(text)
    else:
        result = classifier.predict(
            audio_features=audio_features,
            valence=valence,
            energy=energy,
            danceability=danceability,
            acousticness=acousticness,
        )

    return [mood.value for mood in result.moods]


def get_mood_description(mood: Mood) -> str:
    """Get description for a mood.

    Args:
        mood: Mood enum value.

    Returns:
        Description string.
    """
    if mood in MOOD_DEFINITIONS:
        return MOOD_DEFINITIONS[mood]["description"]
    return "Unknown mood"


def get_mood_from_valence_energy(valence: float, energy: float) -> Mood:
    """Get primary mood from valence and energy values.

    This is a simplified mapping that divides the valence-energy
    space into 4 quadrants.

    Args:
        valence: Musical positiveness (0.0-1.0).
        energy: Energy level (0.0-1.0).

    Returns:
        Primary Mood category.
    """
    if valence > 0.5 and energy > 0.5:
        return Mood.HAPPY
    elif valence > 0.5 and energy <= 0.5:
        return Mood.RELAXED
    elif valence <= 0.5 and energy > 0.5:
        return Mood.ANGRY
    else:
        return Mood.SAD
