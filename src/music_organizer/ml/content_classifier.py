"""Content type classifier using hybrid ML/rule-based approach.

This module implements a content type classifier that can:
1. Detect REMIX, PODCAST, SPOKEN_WORD, SOUNDTRACK content types
2. Use scikit-learn model if available (high accuracy)
3. Fall back to rule-based classification if ML is not installed
4. Support multi-label content type classification
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import BaseModel, ModelNotAvailableError, RuleBasedFallback


class ContentType(Enum):
    """Content types for music classification."""
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


@dataclass
class ContentTypeClassificationResult:
    """Result of content type classification."""
    content_types: List[ContentType] = field(default_factory=list)
    confidence_scores: Dict[ContentType, float] = field(default_factory=dict)
    primary_type: Optional[ContentType] = None
    method: str = "unknown"  # ml, rule_based, metadata

    @property
    def confidence(self) -> float:
        """Get overall confidence score (max of all scores)."""
        return max(self.confidence_scores.values()) if self.confidence_scores else 0.0

    def add_type(self, content_type: ContentType, confidence: float) -> None:
        """Add a content type with confidence score."""
        if content_type not in self.content_types:
            self.content_types.append(content_type)

        self.confidence_scores[content_type] = max(
            self.confidence_scores.get(content_type, 0.0),
            confidence
        )

        # Update primary type if this is the highest confidence
        if self.primary_type is None or confidence > self.confidence_scores.get(self.primary_type, 0.0):
            self.primary_type = content_type

    def get_top_types(self, n: int = 3) -> List[Tuple[ContentType, float]]:
        """Get top N content types by confidence."""
        sorted_types = sorted(
            self.confidence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_types[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content_types": [t.value for t in self.content_types],
            "confidence_scores": {t.value: c for t, c in self.confidence_scores.items()},
            "primary_type": self.primary_type.value if self.primary_type else None,
            "confidence": self.confidence,
            "method": self.method,
        }


# Content type detection patterns
REMIX_PATTERNS = [
    r"\bremix\b",
    r"\bmix\b",
    r"\bedit\b",
    r"\bversion\b",
    r"\bbootleg\b",
    r"\bmashup\b",
    r"\bflip\b",
    r"\brefix\b",
    r"\brework\b",
    r"\b\d+\s*remix\b",  # e.g., "2012 Remix"
]

PODCAST_PATTERNS = [
    r"\bpodcast\b",
    r"\bepisode\b",
    r"\bep\.?\s*\d+\b",  # e.g., "Ep. 42"
    r"\bpt\.?\s*\d+\b",  # e.g., "Pt. 1"
    r"\bfeat\.?\b",
    r"\binterview\b",
    r"\bseason\s+\d+\b",
]

SPOKEN_WORD_PATTERNS = [
    r"\bspoken\s*word\b",
    r"\baudiobook\b",
    r"\bbook\s*\d+\b",
    r"\bchapter\s*\d+\b",
    r"\bnarrated?\b",
    r"\bspeech\b",
    r"\blecture\b",
    r"\bstandup?\b",
    r"\bcomedy\b",
]

SOUNDTRACK_PATTERNS = [
    r"\bsoundtrack\b",
    r"\bost\b",
    r"\bscore\b",
    r"\boriginal\s*motion\s*picture\b",
    r"\bfrom\s+the\s+movie\b",
    r"\bfrom\s+the\s+film\b",
    r"\bgames?\s+soundtrack\b",
    r"\banime\s+soundtrack\b",
    r"\btv\s+soundtrack\b",
]

DJ_ARTIST_PATTERNS = [
    r"\bdj\s+\w",
    r"\bDJ\s+\w",
]

COMPOSER_PATTERNS = [
    r"\bcomposed\s+by\b",
    r"\bscore\s+by\b",
    r"\borchestrated?\b",
]


class ContentTypeClassifier(BaseModel):
    """ML-based content type classifier with rule-based fallback.

    Usage:
        classifier = ContentTypeClassifier()
        result = classifier.predict(
            title="Song Name (DJ Remix)",
            album="Album Name",
            artist="Artist Name",
            genre="Electronic",
            duration=240
        )
        print(result.primary_type)  # ContentType.REMIX
    """

    model_name = "content_type_classifier"
    model_version = "1.0.0"
    model_type = "hybrid"

    def __init__(self, model_path: Optional[Path] = None):
        super().__init__(model_path)
        self._use_fallback = False

    def _get_default_model_path(self) -> Path:
        return self.MODELS_DIR / "content_type_model.pkl"

    def predict(
        self,
        title: Optional[str] = None,
        album: Optional[str] = None,
        artist: Optional[str] = None,
        genre: Optional[str] = None,
        year: Optional[int] = None,
        duration: Optional[float] = None,
        existing_type: Optional[str] = None,
    ) -> ContentTypeClassificationResult:
        """Predict content types for a track.

        Args:
            title: Track title
            album: Album name
            artist: Artist name
            genre: Genre
            year: Release year
            duration: Track duration in seconds
            existing_type: Existing content type from metadata

        Returns:
            ContentTypeClassificationResult with predicted types and confidence
        """
        result = ContentTypeClassificationResult()

        # First, use existing type if available
        if existing_type:
            normalized = self._normalize_content_type(existing_type)
            if normalized and normalized != ContentType.UNKNOWN:
                result.add_type(normalized, 0.9)
                result.method = "metadata"

        # Try ML prediction if model is available
        try:
            if self._is_loaded or self.load():
                ml_types = self._ml_predict(title, album, artist, genre, year, duration)
                for ct, conf in ml_types:
                    result.add_type(ct, conf)
                if result.method == "unknown":
                    result.method = "ml"
        except ModelNotAvailableError:
            self._use_fallback = True
        except Exception:
            self._use_fallback = True

        # Fall back to rule-based if ML failed or we want more results
        if self._use_fallback or not result.content_types:
            rule_types = self._rule_based_predict(
                title, album, artist, genre, duration
            )
            for ct, conf in rule_types:
                result.add_type(ct, conf * 0.8)  # Slightly lower confidence for rules
            if result.method == "unknown":
                result.method = "rule_based"

        # Default to UNKNOWN if nothing detected
        if not result.content_types:
            result.add_type(ContentType.UNKNOWN, 0.5)

        return result

    def _ml_predict(
        self,
        title: Optional[str],
        album: Optional[str],
        artist: Optional[str],
        genre: Optional[str],
        year: Optional[int],
        duration: Optional[float],
    ) -> List[Tuple[ContentType, float]]:
        """Run ML prediction using sklearn model."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ModelNotAvailableError(
                "content type classification",
                "Install with: pip install music-organizer[ml]"
            )

        if self._model is None:
            raise ModelNotAvailableError("content type model not loaded")

        # Extract features
        features_text = self._extract_features(title, album, artist, genre, year, duration)

        # Check if we have a vectorizer
        if hasattr(self, '_vectorizer') and self._vectorizer:
            X = self._vectorizer.transform([features_text])
            predictions = self._model.predict_proba(X)

            if hasattr(self._model, 'classes_'):
                classes = self._model.classes_
            else:
                classes = [ct for ct in ContentType if ct != ContentType.UNKNOWN]

            results = []
            for i, prob in enumerate(predictions[0]):
                if prob > 0.1:  # Minimum confidence threshold
                    results.append((classes[i], float(prob)))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:5]
        else:
            # No vectorizer available, use fallback
            return []

    def _rule_based_predict(
        self,
        title: Optional[str],
        album: Optional[str],
        artist: Optional[str],
        genre: Optional[str],
        duration: Optional[float],
    ) -> List[Tuple[ContentType, float]]:
        """Rule-based content type prediction."""
        results = []

        # Combine all text for pattern matching
        all_text = " ".join(filter(None, [
            title or "",
            album or "",
            artist or "",
            genre or ""
        ])).lower()

        title_lower = (title or "").lower()
        album_lower = (album or "").lower()
        artist_lower = (artist or "").lower()
        genre_lower = (genre or "").lower()

        # REMIX detection
        remix_conf = self._detect_remix(title_lower, album_lower, artist_lower, genre_lower)
        if remix_conf > 0:
            results.append((ContentType.REMIX, remix_conf))

        # PODCAST detection
        podcast_conf = self._detect_podcast(
            title_lower, album_lower, artist_lower, genre_lower, duration
        )
        if podcast_conf > 0:
            results.append((ContentType.PODCAST, podcast_conf))

        # SPOKEN_WORD detection
        spoken_conf = self._detect_spoken_word(
            title_lower, album_lower, artist_lower, genre_lower
        )
        if spoken_conf > 0:
            results.append((ContentType.SPOKEN_WORD, spoken_conf))

        # SOUNDTRACK detection
        soundtrack_conf = self._detect_soundtrack(
            title_lower, album_lower, artist_lower, genre_lower
        )
        if soundtrack_conf > 0:
            results.append((ContentType.SOUNDTRACK, soundtrack_conf))

        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _detect_remix(
        self, title: str, album: str, artist: str, genre: str
    ) -> float:
        """Detect if content is a remix."""
        confidence = 0.0

        # Check title patterns
        for pattern in REMIX_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                confidence += 0.3

        # Check album patterns
        for pattern in REMIX_PATTERNS:
            if re.search(pattern, album, re.IGNORECASE):
                confidence += 0.2

        # Check artist patterns (DJ in name)
        for pattern in DJ_ARTIST_PATTERNS:
            if re.search(pattern, artist, re.IGNORECASE):
                confidence += 0.2

        # Genre hint
        if "electronic" in genre or "edm" in genre or "dance" in genre:
            confidence += 0.1

        return min(confidence, 1.0)

    def _detect_podcast(
        self, title: str, album: str, artist: str, genre: str, duration: Optional[float]
    ) -> float:
        """Detect if content is a podcast."""
        confidence = 0.0

        # Check title patterns
        for pattern in PODCAST_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                confidence += 0.25

        # Check album patterns
        if "podcast" in album:
            confidence += 0.3

        # Genre hint
        if "podcast" in genre:
            confidence += 0.4

        # Duration hint (podcasts are usually long)
        if duration and duration > 900:  # 15+ minutes
            confidence += 0.2
        elif duration and duration > 1800:  # 30+ minutes
            confidence += 0.3

        return min(confidence, 1.0)

    def _detect_spoken_word(
        self, title: str, album: str, artist: str, genre: str
    ) -> float:
        """Detect if content is spoken word."""
        confidence = 0.0

        # Check title patterns
        for pattern in SPOKEN_WORD_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                confidence += 0.3

        # Check album patterns
        for pattern in SPOKEN_WORD_PATTERNS:
            if re.search(pattern, album, re.IGNORECASE):
                confidence += 0.3

        # Genre hint
        spoken_genres = ["speech", "spoken", "audiobook", "comedy", "lecture"]
        if any(g in genre for g in spoken_genres):
            confidence += 0.4

        # Artist hint (author, narrator)
        if "author" in artist.lower() or "narrator" in artist.lower():
            confidence += 0.2

        return min(confidence, 1.0)

    def _detect_soundtrack(
        self, title: str, album: str, artist: str, genre: str
    ) -> float:
        """Detect if content is a soundtrack."""
        confidence = 0.0

        # Check album patterns (strong indicator)
        for pattern in SOUNDTRACK_PATTERNS:
            if re.search(pattern, album, re.IGNORECASE):
                confidence += 0.35

        # Check title patterns
        for pattern in SOUNDTRACK_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                confidence += 0.2

        # Genre hint
        if "soundtrack" in genre or "score" in genre or "ost" in genre:
            confidence += 0.3

        # Artist hint (composer, conductor)
        for pattern in COMPOSER_PATTERNS:
            if re.search(pattern, artist, re.IGNORECASE):
                confidence += 0.2
        if "composer" in artist.lower() or "conductor" in artist.lower():
            confidence += 0.15

        return min(confidence, 1.0)

    def _normalize_content_type(self, content_type: Optional[str]) -> Optional[ContentType]:
        """Normalize content type string to ContentType enum."""
        if not content_type:
            return None

        ct_lower = content_type.lower().strip()

        # Direct mapping
        mapping = {
            "remix": ContentType.REMIX,
            "podcast": ContentType.PODCAST,
            "spoken_word": ContentType.SPOKEN_WORD,
            "spoken-word": ContentType.SPOKEN_WORD,
            "spoken word": ContentType.SPOKEN_WORD,
            "audiobook": ContentType.AUDIOBOOK,
            "soundtrack": ContentType.SOUNDTRACK,
            "ost": ContentType.SOUNDTRACK,
            "score": ContentType.SOUNDTRACK,
            "dj_mix": ContentType.DJ_MIX,
            "dj-mix": ContentType.DJ_MIX,
            "dj mix": ContentType.DJ_MIX,
            "live": ContentType.LIVE_RECORDING,
            "live_recording": ContentType.LIVE_RECORDING,
            "compilation": ContentType.COMPILATION,
            "studio_album": ContentType.STUDIO_ALBUM,
            "studio": ContentType.STUDIO_ALBUM,
            "ep": ContentType.EP,
            "single": ContentType.SINGLE,
            "demo": ContentType.DEMO,
            "cover": ContentType.COVER,
        }

        for key, value in mapping.items():
            if key in ct_lower:
                return value

        return ContentType.UNKNOWN

    def _extract_features(
        self,
        title: Optional[str] = None,
        album: Optional[str] = None,
        artist: Optional[str] = None,
        genre: Optional[str] = None,
        year: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> str:
        """Extract text features for classification."""
        features = []

        if title:
            title_clean = re.sub(r'\([^)]*\)', '', title)
            title_clean = re.sub(r'\[[^\]]*\]', '', title_clean)
            features.append(title_clean)

        if album:
            album_clean = re.sub(r'\([^)]*\)', '', album)
            album_clean = re.sub(r'\[[^\]]*\]', '', album_clean)
            features.append(album_clean)

        if artist:
            features.append(artist)

        if genre:
            features.append(genre)

        if duration:
            minutes = duration / 60
            if minutes < 3:
                features.append("short")
            elif minutes < 6:
                features.append("medium")
            elif minutes < 15:
                features.append("long")
            else:
                features.append("extended")

        return " ".join(features)

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
            raise ModelLoadError(f"Failed to load content type model: {e}")

    def save(self, model: Any, vectorizer: Any = None) -> None:
        """Save model and vectorizer."""
        import pickle

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": model, "vectorizer": vectorizer}, f)

        self._model = model
        self._vectorizer = vectorizer
        self._is_loaded = True


def classify_from_text(
    title: Optional[str] = None,
    album: Optional[str] = None,
    artist: Optional[str] = None,
    genre: Optional[str] = None,
    duration: Optional[float] = None,
) -> List[ContentType]:
    """Quick content type classification using rules only."""
    classifier = ContentTypeClassifier()
    result = classifier.predict(
        title=title,
        album=album,
        artist=artist,
        genre=genre,
        duration=duration
    )
    return result.content_types
