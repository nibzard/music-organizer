"""Base classes for ML models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar
import pickle


class ModelLoadError(Exception):
    """Raised when a model file cannot be loaded."""
    pass


class ModelNotAvailableError(Exception):
    """Raised when an optional ML model is not available."""
    def __init__(self, feature_name: str, install_hint: Optional[str] = None):
        self.feature_name = feature_name
        self.install_hint = install_hint
        message = f"ML feature '{feature_name}' is not available."
        if install_hint:
            message += f" {install_hint}"
        super().__init__(message)


T = TypeVar('T')


class BaseModel(ABC):
    """Base class for all ML models in the music organizer."""

    # Default model directory
    MODELS_DIR = Path(__file__).parent / "models"

    # Model metadata
    model_name: str
    model_version: str
    model_type: str

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the model.

        Args:
            model_path: Path to the model file. If None, uses default path.
        """
        self.model_path = model_path or self._get_default_model_path()
        self._model: Any = None
        self._is_loaded = False

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Make a prediction using the model.

        Raises:
            ModelNotAvailableError: If the model is not available
        """
        pass

    @abstractmethod
    def _get_default_model_path(self) -> Path:
        """Get the default path for this model file."""
        pass

    def load(self) -> bool:
        """Load the model from disk.

        Returns:
            True if model was loaded successfully, False otherwise.
        """
        if self._is_loaded:
            return True

        if not self.model_path.exists():
            return False

        try:
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)
            self._is_loaded = True
            return True
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {self.model_path}: {e}")

    def save(self, model: Any) -> None:
        """Save a model to disk.

        Args:
            model: The model object to save.
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)
        self._model = model
        self._is_loaded = True

    def is_available(self) -> bool:
        """Check if the model is available (exists and can be loaded)."""
        return self.model_path.exists()

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded in memory."""
        return self._is_loaded

    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "name": self.model_name,
            "version": self.model_version,
            "type": self.model_type,
            "path": str(self.model_path),
            "loaded": self._is_loaded,
        }


class RuleBasedFallback:
    """Fallback rule-based classifier for when ML models are not available.

    This provides basic functionality using heuristics and patterns
    without requiring ML dependencies.
    """

    @staticmethod
    def normalize_genre(genre: Optional[str]) -> Optional[str]:
        """Normalize genre names to standard values."""
        if not genre:
            return None

        genre_lower = genre.lower().strip()

        # Common mappings
        mappings = {
            "r&b": "rnb",
            "rnb": "rnb",
            "r and b": "rnb",
            "hip-hop": "hiphop",
            "hip hop": "hiphop",
            "electronic": "electronic",
            "edm": "electronic",
            "dance": "electronic",
            "rock & roll": "rock",
            "rock'n'roll": "rock",
            "classic rock": "rock",
            "country & western": "country",
            "soundtrack": "soundtrack",
            "ost": "soundtrack",
            "film score": "soundtrack",
        }

        for key, value in mappings.items():
            if key in genre_lower:
                return value

        return genre_lower

    @staticmethod
    def infer_from_keywords(
        title: Optional[str] = None,
        album: Optional[str] = None,
        artist: Optional[str] = None,
    ) -> List[str]:
        """Infer genres from text keywords."""
        genres = set()

        text = " ".join(filter(None, [title, album, artist])).lower()

        # Genre keyword patterns
        patterns = {
            "electronic": ["remix", "dj", "club", "electronic", "edm", "techno", "house", "trance", "dubstep"],
            "rock": ["rock", "metal", "punk", "grunge", "alternative"],
            "hiphop": ["hip", "hop", "rap", "trap"],
            "jazz": ["jazz", "swing", "bebop", "fusion"],
            "classical": ["classical", "symphony", "orchestra", "concerto", "sonata"],
            "country": ["country", "folk", "bluegrass"],
            "reggae": ["reggae", "dancehall"],
            "blues": ["blues"],
            "soul": ["soul", "funk", "motown"],
            "soundtrack": ["soundtrack", "ost", "score", "theme"],
            "ambient": ["ambient", "chillout", "downtempo"],
            "pop": ["pop"],
        }

        for genre, keywords in patterns.items():
            if any(keyword in text for keyword in keywords):
                genres.add(genre)

        return sorted(genres)

    @staticmethod
    def infer_from_artist(artist: str) -> Optional[str]:
        """Infer genre from artist name (database lookup would go here)."""
        if not artist:
            return None

        # This is where you'd integrate with MusicBrainz or similar
        # For now, just return None
        return None
