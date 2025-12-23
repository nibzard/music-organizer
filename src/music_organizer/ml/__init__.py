"""ML-based classification for music organizer.

This module provides machine learning models for music classification,
including genre classification, mood detection, and content type classification.

All models are designed to work locally with no external API calls,
ensuring privacy and fast inference.
"""

from .base import BaseModel, ModelLoadError, ModelNotAvailableError
from .content_classifier import ContentTypeClassifier, ContentTypeClassificationResult, ContentType
from .genre_classifier import GenreClassifier, GenreClassificationResult
from .audio_features import (
    AudioFeatureExtractor,
    AudioFeatureError,
    FeatureCache,
    BatchAudioFeatureExtractor,
    BatchProgress,
    is_available as audio_features_available,
)
from .mood_classifier import (
    MoodClassifier,
    MoodClassificationResult,
    classify_mood,
    get_mood_description,
    get_mood_from_valence_energy,
)
from .acoustic_similarity import (
    AcousticSimilarityAnalyzer,
    AcousticSimilarityError,
    ChromaCache,
    SimilarityResult,
    SimilarityBatchResult,
    BatchAcousticSimilarityAnalyzer,
    is_available as acoustic_similarity_available,
)

__all__ = [
    "BaseModel",
    "ModelLoadError",
    "ModelNotAvailableError",
    "GenreClassifier",
    "GenreClassificationResult",
    "ContentTypeClassifier",
    "ContentTypeClassificationResult",
    "ContentType",
    "AudioFeatureExtractor",
    "AudioFeatureError",
    "FeatureCache",
    "BatchAudioFeatureExtractor",
    "BatchProgress",
    "audio_features_available",
    "MoodClassifier",
    "MoodClassificationResult",
    "classify_mood",
    "get_mood_description",
    "get_mood_from_valence_energy",
    "AcousticSimilarityAnalyzer",
    "AcousticSimilarityError",
    "ChromaCache",
    "SimilarityResult",
    "SimilarityBatchResult",
    "BatchAcousticSimilarityAnalyzer",
    "acoustic_similarity_available",
]
