"""Genre classifier using TF-IDF features.

This module implements a zero-dependency genre classifier that can:
1. Use scikit-learn model if available (high accuracy)
2. Fall back to rule-based classification if ML is not installed
3. Support multi-label genre classification
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import BaseModel, ModelNotAvailableError, RuleBasedFallback


# Top 20 genres for classification
TOP_GENRES = [
    "rock",
    "pop",
    "electronic",
    "hiphop",
    "jazz",
    "classical",
    "country",
    "rnb",
    "soul",
    "reggae",
    "blues",
    "metal",
    "punk",
    "folk",
    "ambient",
    "soundtrack",
    "world",
    "gospel",
    "latin",
    "indie",
]

# Genre synonym mappings for normalization
GENRE_SYNONYMS = {
    "r&b": "rnb",
    "rnb": "rnb",
    "hip-hop": "hiphop",
    "hip hop": "hiphop",
    "r and b": "rnb",
    "rb": "rnb",
    "electronica": "electronic",
    "edm": "electronic",
    "dance": "electronic",
    "classic rock": "rock",
    "heavy metal": "metal",
    "country & western": "country",
    "ost": "soundtrack",
    "soundtrack": "soundtrack",
    "film score": "soundtrack",
    "instrumental": "classical",
    "new age": "ambient",
}


@dataclass
class GenreClassificationResult:
    """Result of genre classification."""
    genres: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    primary_genre: Optional[str] = None
    method: str = "unknown"  # ml, rule_based, metadata

    @property
    def confidence(self) -> float:
        """Get overall confidence score (max of all scores)."""
        return max(self.confidence_scores.values()) if self.confidence_scores else 0.0

    def add_genre(self, genre: str, confidence: float) -> None:
        """Add a genre with confidence score."""
        normalized = _normalize_genre(genre)
        if not normalized:
            return

        if normalized not in self.genres:
            self.genres.append(normalized)

        self.confidence_scores[normalized] = max(
            self.confidence_scores.get(normalized, 0.0),
            confidence
        )

        # Update primary genre if this is the highest confidence
        if self.primary_genre is None or confidence > self.confidence_scores.get(self.primary_genre, 0.0):
            self.primary_genre = normalized

    def get_top_genres(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get top N genres by confidence."""
        sorted_genres = sorted(
            self.confidence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_genres[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "genres": self.genres,
            "confidence_scores": self.confidence_scores,
            "primary_genre": self.primary_genre,
            "confidence": self.confidence,
            "method": self.method,
        }


def _normalize_genre(genre: Optional[str]) -> Optional[str]:
    """Normalize genre name to standard value."""
    if not genre:
        return None

    genre_lower = genre.lower().strip()

    # Return None for empty/whitespace strings
    if not genre_lower:
        return None

    # Check synonyms first
    if genre_lower in GENRE_SYNONYMS:
        return GENRE_SYNONYMS[genre_lower]

    # Direct match with top genres
    if genre_lower in TOP_GENRES:
        return genre_lower

    # Substring matching
    for top_genre in TOP_GENRES:
        if top_genre in genre_lower or genre_lower in top_genre:
            return top_genre

    return None


def _extract_features(
    title: Optional[str] = None,
    album: Optional[str] = None,
    artist: Optional[str] = None,
    year: Optional[int] = None,
    duration: Optional[float] = None,
) -> str:
    """Extract text features for classification.

    Returns:
        Combined text string for TF-IDF processing.
    """
    features = []

    if title:
        # Remove parenthetical content (remix, feat, etc.)
        title_clean = re.sub(r'\([^)]*\)', '', title)
        title_clean = re.sub(r'\[[^\]]*\]', '', title_clean)
        features.append(title_clean)

    if album:
        # Clean album name
        album_clean = re.sub(r'\([^)]*\)', '', album)
        album_clean = re.sub(r'\[[^\]]*\]', '', album_clean)
        features.append(album_clean)

    if artist:
        # Artist name is a strong indicator
        features.append(artist)

    # Year as decade indicator
    if year and year > 1900 and year < 2100:
        decade = (year // 10) * 10
        features.append(f"decade_{decade}")

    # Duration buckets (in minutes)
    if duration:
        minutes = duration / 60
        if minutes < 2:
            features.append("very_short")
        elif minutes < 4:
            features.append("short")
        elif minutes < 6:
            features.append("medium")
        elif minutes < 10:
            features.append("long")
        else:
            features.append("extended")

    return " ".join(features)


class GenreClassifier(BaseModel):
    """ML-based genre classifier with rule-based fallback.

    Usage:
        classifier = GenreClassifier()
        result = classifier.predict(
            title="Stairway to Heaven",
            album="Led Zeppelin IV",
            artist="Led Zeppelin",
            year=1971
        )
        print(result.primary_genre)  # "rock"
    """

    model_name = "genre_classifier"
    model_version = "1.0.0"
    model_type = "tfidf_logreg"

    # TF-IDF settings for sklearn
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)

    def __init__(self, model_path: Optional[Path] = None):
        super().__init__(model_path)
        self._vectorizer: Any = None
        self._use_fallback = False

    def _get_default_model_path(self) -> Path:
        return self.MODELS_DIR / "genre_model.pkl"

    def predict(
        self,
        title: Optional[str] = None,
        album: Optional[str] = None,
        artist: Optional[str] = None,
        year: Optional[int] = None,
        duration: Optional[float] = None,
        existing_genre: Optional[str] = None,
    ) -> GenreClassificationResult:
        """Predict genres for a track.

        Args:
            title: Track title
            album: Album name
            artist: Artist name
            year: Release year
            duration: Track duration in seconds
            existing_genre: Existing genre from metadata (used as hint)

        Returns:
            GenreClassificationResult with predicted genres and confidence
        """
        result = GenreClassificationResult()

        # First, use existing genre if available
        if existing_genre:
            normalized = _normalize_genre(existing_genre)
            if normalized:
                result.add_genre(normalized, 0.9)
                result.method = "metadata"

        # Try ML prediction if model is available
        try:
            if self._is_loaded or self.load():
                ml_genres = self._ml_predict(title, album, artist, year, duration)
                for genre, conf in ml_genres:
                    result.add_genre(genre, conf)
                result.method = "ml" if not result.genres or "ml" not in result.method else result.method
        except ModelNotAvailableError:
            self._use_fallback = True
        except Exception:
            self._use_fallback = True

        # Fall back to rule-based if ML failed or we want more results
        if self._use_fallback or not result.genres:
            rule_genres = self._rule_based_predict(title, album, artist)
            for genre, conf in rule_genres:
                result.add_genre(genre, conf * 0.7)  # Lower confidence for rules
            if result.method == "unknown":
                result.method = "rule_based"

        return result

    def _ml_predict(
        self,
        title: Optional[str],
        album: Optional[str],
        artist: Optional[str],
        year: Optional[int],
        duration: Optional[float],
    ) -> List[Tuple[str, float]]:
        """Run ML prediction using sklearn model."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ModelNotAvailableError(
                "genre classification",
                "Install with: pip install music-organizer[ml]"
            )

        if self._model is None or self._vectorizer is None:
            raise ModelNotAvailableError("genre model not trained")

        # Extract features
        features_text = _extract_features(title, album, artist, year, duration)

        # Vectorize
        X = self._vectorizer.transform([features_text])

        # Predict
        predictions = self._model.predict_proba(X)

        # Get top genres
        if hasattr(self._model, 'classes_'):
            classes = self._model.classes_
        else:
            classes = TOP_GENRES

        results = []
        for i, prob in enumerate(predictions[0]):
            if prob > 0.1:  # Minimum confidence threshold
                results.append((classes[i], float(prob)))

        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:5]  # Top 5

    def _rule_based_predict(
        self,
        title: Optional[str],
        album: Optional[str],
        artist: Optional[str],
    ) -> List[Tuple[str, float]]:
        """Rule-based genre prediction."""
        result = RuleBasedFallback()

        genres = result.infer_from_keywords(title, album, artist)

        # Assign confidence based on number of matches
        if len(genres) == 1:
            return [(genres[0], 0.7)]
        elif len(genres) == 2:
            return [(genres[0], 0.6), (genres[1], 0.5)]
        else:
            return [(g, 0.4) for g in genres[:3]]

    def load(self) -> bool:
        """Load the model and vectorizer."""
        if self._is_loaded:
            return True

        if not self.model_path.exists():
            self._use_fallback = True
            return False

        try:
            import pickle
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self._model = data.get("model")
                self._vectorizer = data.get("vectorizer")
            self._is_loaded = True
            return True
        except Exception as e:
            from .base import ModelLoadError
            raise ModelLoadError(f"Failed to load genre model: {e}")

    def save(self, model: Any, vectorizer: Any) -> None:
        """Save model and vectorizer."""
        import pickle

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": model, "vectorizer": vectorizer}, f)

        self._model = model
        self._vectorizer = vectorizer
        self._is_loaded = True


# Genre patterns for rule-based classification
GENRE_PATTERNS = {
    "electronic": [
        r"\bremix\b",
        r"\bdj\b",
        r"\bclub\b",
        r"\btechno\b",
        r"\bhouse\b",
        r"\btrance\b",
        r"\bdubstep\b",
        r"\bedm\b",
        r"\belectronic\b",
        r"\bsynth\b",
    ],
    "rock": [
        r"\brock\b",
        r"\bmetal\b",
        r"\bpunk\b",
        r"\bgrunge\b",
        r"\balternative\b",
        r"\bindie rock\b",
    ],
    "hiphop": [
        r"\bhip\s*hop\b",
        r"\brap\b",
        r"\btrap\b",
        r"\bmc\b",
    ],
    "jazz": [
        r"\bjazz\b",
        r"\bswing\b",
        r"\bbebop\b",
        r"\bfusion\b",
    ],
    "classical": [
        r"\bclassical\b",
        r"\bsymphony\b",
        r"\borchestra\b",
        r"\bconcerto\b",
        r"\bsonata\b",
        r"\bnocturne\b",
    ],
    "country": [
        r"\bcountry\b",
        r"\bfolk\b",
        r"\bbluegrass\b",
    ],
    "soundtrack": [
        r"\bsoundtrack\b",
        r"\bost\b",
        r"\bscore\b",
        r"\btheme\b",
    ],
    "ambient": [
        r"\bambient\b",
        r"\bchillout\b",
        r"\bdowntempo\b",
    ],
}


def classify_from_text(text: str) -> List[str]:
    """Quick classification from text using patterns."""
    genres = set()
    text_lower = text.lower()

    for genre, patterns in GENRE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                genres.add(genre)
                break

    return sorted(genres)
