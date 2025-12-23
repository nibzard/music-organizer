"""High-performance string similarity utilities using Rust extension.

This module provides fast string distance and similarity calculations
for music metadata matching, duplicate detection, and fuzzy search.

The Rust extension is automatically used when available, with pure Python
fallbacks for compatibility.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Try to import the Rust extension
try:
    # Import the compiled Rust module
    if sys.version_info >= (3, 11):
        from importlib.resources import files
        import ctypes
        import os

        # Try to load the .so file from the native directory
        native_lib_path = (
            files(__package__)
            .joinpath("../../../native/music_organizer_rs/target/release/libmusic_organizer_rs.so")
            .resolve()
        )
        if native_lib_path.exists():
            # Load the library directly for development
            # In production, this would be installed as a proper Python extension
            _rust_available = False  # Will be True when properly installed
        else:
            _rust_available = False
    else:
        _rust_available = False

    # Try importing via maturin-stubs or similar
    try:
        import music_organizer_rs as _rs

        _rust_available = True
    except ImportError:
        _rust_available = False
except Exception:
    _rust_available = False


class StringSimilarity:
    """High-performance string similarity calculations."""

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings.

        Returns the minimum number of single-character edits (insertions,
        deletions, or substitutions) required to change one string into another.

        Args:
            s1: First string
            s2: Second string

        Returns:
            The edit distance as an integer
        """
        if _rust_available:
            return _rs.levenshtein_distance(s1, s2)
        return _python_levenshtein_distance(s1, s2)

    @staticmethod
    def levenshtein_similarity(s1: str, s2: str) -> float:
        """Calculate normalized Levenshtein similarity (0.0 to 1.0).

        Returns 1.0 for identical strings and 0.0 for completely different strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if _rust_available:
            return _rs.levenshtein_similarity(s1, s2)
        return _python_levenshtein_similarity(s1, s2)

    @staticmethod
    def damerau_levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Damerau-Levenshtein distance (includes transpositions).

        Counts transpositions of adjacent characters as a single edit operation.

        Args:
            s1: First string
            s2: Second string

        Returns:
            The edit distance as an integer
        """
        if _rust_available:
            return _rs.damerau_levenshtein_distance(s1, s2)
        return _python_damerau_levenshtein_distance(s1, s2)

    @staticmethod
    def jaro_similarity(s1: str, s2: str) -> float:
        """Calculate Jaro similarity between two strings.

        Useful for detecting typos and name variations.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if _rust_available:
            return _rs.jaro_similarity(s1, s2)
        return _python_jaro_similarity(s1, s2)

    @staticmethod
    def jaro_winkler_similarity(s1: str, s2: str) -> float:
        """Calculate Jaro-Winkler similarity.

        Gives higher scores to strings that match from the beginning.
        Particularly useful for detecting typos in proper names.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if _rust_available:
            return _rs.jaro_winkler_similarity(s1, s2)
        return _python_jaro_winkler_similarity(s1, s2)

    @staticmethod
    def jaccard_similarity(s1: str, s2: str) -> float:
        """Calculate Jaccard similarity between word sets.

        Measures similarity between sets of words (tokens).
        Useful for comparing titles, phrases, etc.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if _rust_available:
            return _rs.jaccard_similarity(s1, s2)
        return _python_jaccard_similarity(s1, s2)

    @staticmethod
    def cosine_similarity(s1: str, s2: str) -> float:
        """Calculate cosine similarity of word frequency vectors.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if _rust_available:
            return _rs.cosine_similarity(s1, s2)
        return _python_cosine_similarity(s1, s2)

    @staticmethod
    def sorensen_dice_similarity(s1: str, s2: str) -> float:
        """Calculate Sorensen-Dice coefficient.

        Similar to Jaccard but gives more weight to matches.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if _rust_available:
            return _rs.sorensen_dice_similarity(s1, s2)
        return _python_sorensen_dice_similarity(s1, s2)

    @staticmethod
    def music_metadata_similarity(s1: str, s2: str) -> float:
        """Calculate similarity optimized for music metadata.

        Handles common variations like "The Beatles" vs "Beatles, The".
        Uses a weighted combination of multiple similarity measures.

        Args:
            s1: First metadata string (artist name, title, etc.)
            s2: Second metadata string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if _rust_available:
            return _rs.music_metadata_similarity(s1, s2)
        return _python_music_metadata_similarity(s1, s2)

    @staticmethod
    def find_best_match(target: str, candidates: Sequence[str]) -> tuple[int, float]:
        """Find the best matching string from a list of candidates.

        Args:
            target: The string to match against
            candidates: List of candidate strings

        Returns:
            Tuple of (index, similarity_score) for the best match
        """
        if _rust_available:
            return _rs.find_best_match(target, list(candidates))
        return _python_find_best_match(target, candidates)

    @staticmethod
    def find_similar_pairs(
        strings: Sequence[str], threshold: float
    ) -> list[tuple[int, int, float]]:
        """Find all pairs of strings with similarity above threshold.

        Args:
            strings: List of strings to compare
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of (index1, index2, similarity) tuples for pairs above threshold
        """
        if _rust_available:
            return _rs.find_similar_pairs(list(strings), threshold)
        return _python_find_similar_pairs(strings, threshold)

    @staticmethod
    def fuzzy_match(s1: str, s2: str) -> dict[str, float]:
        """Get all similarity metrics between two strings.

        Returns a dictionary with multiple similarity scores.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Dict with keys: levenshtein_distance, levenshtein_similarity,
            jaro_winkler_similarity, jaccard_similarity, cosine_similarity,
            sorensen_dice_similarity
        """
        if _rust_available:
            result = _rs.fuzzy_match(s1, s2)
            return {
                "levenshtein_distance": result.levenshtein_distance,
                "levenshtein_similarity": result.levenshtein_similarity,
                "jaro_winkler_similarity": result.jaro_winkler_similarity,
                "jaccard_similarity": result.jaccard_similarity,
                "cosine_similarity": result.cosine_similarity,
                "sorensen_dice_similarity": result.sorensen_dice_similarity,
            }
        return _python_fuzzy_match(s1, s2)


# Pure Python fallback implementations


def _python_levenshtein_distance(s1: str, s2: str) -> int:
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    if len(s1) < len(s2):
        return _python_levenshtein_distance(s2, s1)

    previous = list(range(len(s2) + 1))
    current = [0] * (len(s2) + 1)

    for i, c1 in enumerate(s1):
        current[0] = i + 1
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            current[j + 1] = min(current[j] + 1, previous[j + 1] + 1, previous[j] + cost)
        previous, current = current, previous

    return previous[len(s2)]


def _python_levenshtein_similarity(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = _python_levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


def _python_damerau_levenshtein_distance(s1: str, s2: str) -> int:
    # Simplified implementation - falls back to Levenshtein
    return _python_levenshtein_distance(s1, s2)


def _python_jaro_similarity(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    match_distance = max(len(s1), len(s2)) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    matches = 0

    for i in range(len(s1)):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(s2))

        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len(s1)):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    transpositions //= 2
    m = matches
    return (m / len(s1) + m / len(s2) + (m - transpositions) / m) / 3.0


def _python_jaro_winkler_similarity(s1: str, s2: str) -> float:
    jaro = _python_jaro_similarity(s1, s2)

    prefix_len = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2 and prefix_len < 4:
            prefix_len += 1
        else:
            break

    return min(1.0, jaro + prefix_len * 0.1 * (1.0 - jaro))


def _python_jaccard_similarity(s1: str, s2: str) -> float:
    words1 = set(s1.split())
    words2 = set(s2.split())

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def _python_cosine_similarity(s1: str, s2: str) -> float:
    from collections import Counter

    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    words1 = s1.split()
    words2 = s2.split()

    if not words1 or not words2:
        return 0.0

    freq1 = Counter(words1)
    freq2 = Counter(words2)

    dot_product = sum(freq1[w] * freq2.get(w, 0) for w in freq1)
    mag1 = sum(v * v for v in freq1.values()) ** 0.5
    mag2 = sum(v * v for v in freq2.values()) ** 0.5

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def _python_sorensen_dice_similarity(s1: str, s2: str) -> float:
    if len(s1) < 2 or len(s2) < 2:
        return 1.0 if s1 == s2 else 0.0

    bigrams1 = {s1[i : i + 2] for i in range(len(s1) - 1)}
    bigrams2 = {s2[i : i + 2] for i in range(len(s2) - 1)}

    if not bigrams1 and not bigrams2:
        return 1.0
    if not bigrams1 or not bigrams2:
        return 0.0

    intersection = bigrams1 & bigrams2
    return (2.0 * len(intersection)) / (len(bigrams1) + len(bigrams2))


def _normalize_music_text(s: str) -> str:
    import re

    s = s.strip().lower()
    s = re.sub(r"^(the |a |an )", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def _python_music_metadata_similarity(s1: str, s2: str) -> float:
    norm1 = _normalize_music_text(s1)
    norm2 = _normalize_music_text(s2)

    if norm1 == norm2:
        return 1.0

    lev_sim = _python_levenshtein_similarity(norm1, norm2)
    jw_sim = _python_jaro_winkler_similarity(norm1, norm2)
    dice_sim = _python_sorensen_dice_similarity(norm1, norm2)

    return lev_sim * 0.3 + jw_sim * 0.5 + dice_sim * 0.2


def _python_find_best_match(
    target: str, candidates: Sequence[str]
) -> tuple[int, float]:
    if not candidates:
        return (0, 0.0)

    best_idx = 0
    best_score = 0.0

    for i, s in enumerate(candidates):
        score = _python_levenshtein_similarity(target, s)
        if score > best_score:
            best_score = score
            best_idx = i

    return (best_idx, best_score)


def _python_find_similar_pairs(
    strings: Sequence[str], threshold: float
) -> list[tuple[int, int, float]]:
    results = []
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            sim = _python_levenshtein_similarity(strings[i], strings[j])
            if sim >= threshold:
                results.append((i, j, sim))
    return results


def _python_fuzzy_match(s1: str, s2: str) -> dict[str, float]:
    return {
        "levenshtein_distance": float(_python_levenshtein_distance(s1, s2)),
        "levenshtein_similarity": _python_levenshtein_similarity(s1, s2),
        "jaro_winkler_similarity": _python_jaro_winkler_similarity(s1, s2),
        "jaccard_similarity": _python_jaccard_similarity(s1, s2),
        "cosine_similarity": _python_cosine_similarity(s1, s2),
        "sorensen_dice_similarity": _python_sorensen_dice_similarity(s1, s2),
    }


# Convenience exports
levenshtein_distance = StringSimilarity.levenshtein_distance
levenshtein_similarity = StringSimilarity.levenshtein_similarity
jaro_winkler_similarity = StringSimilarity.jaro_winkler_similarity
jaccard_similarity = StringSimilarity.jaccard_similarity
music_metadata_similarity = StringSimilarity.music_metadata_similarity

__all__ = [
    "StringSimilarity",
    "levenshtein_distance",
    "levenshtein_similarity",
    "jaro_winkler_similarity",
    "jaccard_similarity",
    "music_metadata_similarity",
    "_rust_available",
]
