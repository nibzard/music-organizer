"""ML-based classification for music organizer.

This module provides machine learning models for music classification,
including genre classification, mood detection, and content type classification.

All models are designed to work locally with no external API calls,
ensuring privacy and fast inference.
"""

from .base import BaseModel, ModelLoadError, ModelNotAvailableError
from .genre_classifier import GenreClassifier, GenreClassificationResult

__all__ = [
    "BaseModel",
    "ModelLoadError",
    "ModelNotAvailableError",
    "GenreClassifier",
    "GenreClassificationResult",
]
