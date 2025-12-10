"""
Classification Context - Content classification and duplicate detection.

This bounded context is responsible for:
- Classifying music content
- Detecting duplicates
- Content analysis
- Similarity matching
"""

from .entities import Classifier, DuplicateGroup, ContentType, ClassificationRule, SimilarityScore
from .value_objects import ClassificationPattern, ContentTypeEnum, SimilarityThreshold
from .services import ClassificationService, DuplicateService, ContentAnalysisService

__all__ = [
    # Entities
    "Classifier",
    "DuplicateGroup",
    "ContentType",
    "ClassificationRule",
    "SimilarityScore",
    # Value Objects
    "ClassificationPattern",
    "ContentTypeEnum",
    "SimilarityThreshold",
    # Services
    "ClassificationService",
    "DuplicateService",
    "ContentAnalysisService",
]