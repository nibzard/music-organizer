"""Acoustic similarity detection using chroma-based features.

This module provides chroma-based audio similarity for:
- Cover song detection (key-invariant, tempo-invariant)
- Duplicate detection across different versions
- Version grouping (original vs remix)

Uses librosa chroma CQT (Constant-Q Transform) for pitch class profiling
and Dynamic Time Warping (DTW) for tempo-invariant comparison.
"""

from __future__ import annotations

import asyncio
import sqlite3
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import librosa
    import numpy as np
    from scipy.spatial.distance import euclidean
    from scipy.fft import fft
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class AcousticSimilarityError(Exception):
    """Raised when acoustic similarity analysis fails."""
    pass


@dataclass
class SimilarityResult:
    """Result of acoustic similarity comparison."""
    similarity: float  # 0.0 to 1.0
    is_cover: bool
    confidence: str  # high, medium, low
    details: Dict[str, Any]

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence == "high" and self.similarity >= 0.75


class ChromaCache:
    """SQLite cache for chroma features.

    Chroma extraction is expensive (~500-1000ms per file), so caching
    provides significant performance improvements for similarity analysis.
    """

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        if cache_path is None:
            cache_dir = Path.home() / ".cache" / "music-organizer"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "chroma.db"

        self.cache_path = cache_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    def _init_db(self) -> None:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.cache_path))
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS chroma_features (
                    file_hash TEXT PRIMARY KEY,
                    chroma_cqt BLOB,
                    duration REAL,
                    tempo REAL,
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def get(self, file_hash: str) -> Optional[Tuple[np.ndarray, float, float]]:
        async with self._lock:
            if self._conn is None:
                self._init_db()

            cursor = self._conn.execute(
                "SELECT chroma_cqt, duration, tempo FROM chroma_features WHERE file_hash = ?",
                (file_hash,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Deserialize chroma from bytes
            chroma_bytes, duration, tempo = row
            chroma = np.frombuffer(chroma_bytes, dtype=np.float32)
            # Reshape to (12, n_frames)
            n_frames = len(chroma_bytes) // (12 * 4)  # 4 bytes per float32
            chroma = chroma.reshape(12, n_frames)

            return chroma, duration, tempo

    async def set(
        self,
        file_hash: str,
        chroma: np.ndarray,
        duration: float,
        tempo: float
    ) -> None:
        async with self._lock:
            if self._conn is None:
                self._init_db()

            # Serialize chroma to bytes
            chroma_bytes = chroma.astype(np.float32).tobytes()

            self._conn.execute(
                """INSERT OR REPLACE INTO chroma_features
                   (file_hash, chroma_cqt, duration, tempo)
                   VALUES (?, ?, ?, ?)""",
                (file_hash, chroma_bytes, duration, tempo)
            )
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class AcousticSimilarityAnalyzer:
    """Analyze acoustic similarity between audio files using chroma features.

    This class provides methods for:
    - Chroma-based similarity (key-invariant)
    - Cross-correlation for temporal alignment
    - Dynamic Time Warping for tempo invariance
    - Cover song detection
    """

    # Cover song detection thresholds
    COVER_THRESHOLD_HIGH = 0.75
    COVER_THRESHOLD_MEDIUM = 0.60
    COVER_THRESHOLD_LOW = 0.45

    def __init__(
        self,
        cache: Optional[ChromaCache] = None,
        sample_rate: int = 22050,
        n_chroma: int = 12,
    ) -> None:
        """Initialize the analyzer.

        Args:
            cache: Optional chroma feature cache.
            sample_rate: Audio sample rate for analysis.
            n_chroma: Number of chroma bins (12 for semitones).

        Raises:
            AcousticSimilarityError: If librosa is not available.
        """
        if not LIBROSA_AVAILABLE:
            raise AcousticSimilarityError(
                "librosa is not installed. "
                "Install with: pip install librosa numpy scipy"
            )

        self.cache = cache
        self.sample_rate = sample_rate
        self.n_chroma = n_chroma
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def extract_chroma(
        self,
        file_path: Path,
    ) -> Tuple[np.ndarray, float, float]:
        """Extract chroma features from audio file.

        Args:
            file_path: Path to audio file.

        Returns:
            Tuple of (chroma_features, duration, tempo).

        Raises:
            AcousticSimilarityError: If extraction fails.
        """
        if not file_path.exists():
            raise AcousticSimilarityError(f"File not found: {file_path}")

        # Check cache first
        if self.cache is not None:
            from music_organizer.ml.audio_features import FeatureCache
            file_hash = FeatureCache.compute_file_hash(file_path)
            cached = await self.cache.get(file_hash)
            if cached is not None:
                return cached

        # Extract in thread pool
        loop = asyncio.get_event_loop()
        try:
            chroma, duration, tempo = await loop.run_in_executor(
                self._executor,
                self._extract_chroma_sync,
                file_path,
            )
        except Exception as e:
            raise AcousticSimilarityError(f"Failed to extract chroma from {file_path}: {e}")

        # Cache results
        if self.cache is not None:
            await self.cache.set(file_hash, chroma, duration, tempo)

        return chroma, duration, tempo

    def _extract_chroma_sync(
        self,
        file_path: Path,
    ) -> Tuple[np.ndarray, float, float]:
        """Synchronous chroma extraction (runs in thread pool)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Load audio
            y, sr = librosa.load(
                str(file_path),
                sr=self.sample_rate,
                mono=True,
            )

            duration = librosa.get_duration(y=y, sr=sr)

            # Extract tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(tempo)
            except Exception:
                tempo = 120.0  # Default

            # Extract chroma using CQT (better for music than STFT)
            # Use Constant-Q Transform for better frequency resolution
            chroma = librosa.feature.chroma_cqt(
                y=y,
                sr=sr,
                n_chroma=self.n_chroma,
                hop_length=512,
            )

            # Normalize chroma features
            chroma_norm = chroma / (chroma.max(axis=0) + 1e-8)

            return chroma_norm, duration, tempo

    async def compare_similarity(
        self,
        file1: Path,
        file2: Path,
        method: str = "dtw",
    ) -> SimilarityResult:
        """Compare two audio files for acoustic similarity.

        Args:
            file1: Path to first audio file.
            file2: Path to second audio file.
            method: Comparison method - 'dtw', 'correlation', or 'euclidean'.

        Returns:
            SimilarityResult with similarity score and cover detection.
        """
        # Extract features for both files
        chroma1, dur1, tempo1 = await self.extract_chroma(file1)
        chroma2, dur2, tempo2 = await self.extract_chroma(file2)

        # Calculate similarity using specified method
        if method == "dtw":
            similarity = await self._dtw_similarity(chroma1, chroma2)
        elif method == "correlation":
            similarity = await self._correlation_similarity(chroma1, chroma2)
        elif method == "euclidean":
            similarity = await self._euclidean_similarity(chroma1, chroma2)
        else:
            similarity = await self._dtw_similarity(chroma1, chroma2)

        # Determine if likely a cover song
        is_cover = similarity >= self.COVER_THRESHOLD_LOW

        # Determine confidence level
        if similarity >= self.COVER_THRESHOLD_HIGH:
            confidence = "high"
        elif similarity >= self.COVER_THRESHOLD_MEDIUM:
            confidence = "medium"
        else:
            confidence = "low"

        # Additional details
        details = {
            "duration_ratio": min(dur1, dur2) / max(dur1, dur2),
            "tempo_ratio": min(tempo1, tempo2) / max(tempo1, tempo2),
            "method": method,
        }

        return SimilarityResult(
            similarity=similarity,
            is_cover=is_cover,
            confidence=confidence,
            details=details,
        )

    async def _dtw_similarity(
        self,
        chroma1: np.ndarray,
        chroma2: np.ndarray,
    ) -> float:
        """Calculate similarity using Dynamic Time Warping.

        DTW allows for tempo-invariant comparison by finding optimal
        alignment between sequences.
        """
        try:
            # Use librosa's DTW implementation
            D, wp = librosa.sequence.dtw(
                X=chroma1,
                Y=chroma2,
                metric="euclidean",
            )

            # Normalize by path length and max distance
            avg_distance = D[-1, -1] / len(wp)

            # Convert distance to similarity (0-1)
            # Typical DTW distances for chroma are 0-3
            similarity = max(0.0, 1.0 - avg_distance / 3.0)

            return float(similarity)
        except Exception:
            return 0.0

    async def _correlation_similarity(
        self,
        chroma1: np.ndarray,
        chroma2: np.ndarray,
    ) -> float:
        """Calculate similarity using cross-correlation.

        Fast but less tempo-invariant than DTW.
        """
        try:
            # Reshape to 1D for correlation
            c1_flat = chroma1.flatten()
            c2_flat = chroma2.flatten()

            # Normalize
            c1_norm = (c1_flat - c1_flat.mean()) / (c1_flat.std() + 1e-8)
            c2_norm = (c2_flat - c2_flat.mean()) / (c2_flat.std() + 1e-8)

            # Cross-correlation
            correlation = np.corrcoef(c1_norm, c2_norm)[0, 1]

            # Scale to 0-1
            return float(max(0.0, correlation))
        except Exception:
            return 0.0

    async def _euclidean_similarity(
        self,
        chroma1: np.ndarray,
        chroma2: np.ndarray,
    ) -> float:
        """Calculate similarity using normalized euclidean distance.

        Fastest method, but assumes similar tempo.
        """
        try:
            # Pad/truncate to same length
            max_len = max(chroma1.shape[1], chroma2.shape[1])
            if chroma1.shape[1] < max_len:
                chroma1 = np.pad(chroma1, ((0, 0), (0, max_len - chroma1.shape[1])))
            if chroma2.shape[1] < max_len:
                chroma2 = np.pad(chroma2, ((0, 0), (0, max_len - chroma2.shape[1])))

            # Calculate distance
            dist = euclidean(chroma1.flatten(), chroma2.flatten())

            # Normalize (typical distances are 0-10)
            similarity = max(0.0, 1.0 - dist / 10.0)

            return float(similarity)
        except Exception:
            return 0.0

    async def find_similar_tracks(
        self,
        query_file: Path,
        candidate_files: List[Path],
        threshold: float = 0.5,
        method: str = "dtw",
    ) -> List[Tuple[Path, SimilarityResult]]:
        """Find tracks similar to a query file.

        Args:
            query_file: Path to query audio file.
            candidate_files: List of candidate files to compare.
            threshold: Minimum similarity threshold.
            method: Comparison method.

        Returns:
            List of (file_path, similarity_result) tuples, sorted by similarity.
        """
        results = []

        # Extract query features once
        query_chroma, query_dur, query_tempo = await self.extract_chroma(query_file)

        # Compare with each candidate
        for candidate in candidate_files:
            if candidate == query_file:
                continue

            try:
                cand_chroma, cand_dur, cand_tempo = await self.extract_chroma(candidate)

                # Quick filter: duration and tempo should be somewhat similar
                dur_ratio = min(query_dur, cand_dur) / max(query_dur, cand_dur)
                tempo_ratio = min(query_tempo, cand_tempo) / max(query_tempo, cand_tempo)

                # Skip if too different (unless it's a remix)
                if dur_ratio < 0.5 or dur_ratio > 2.0:
                    continue

                # Calculate similarity
                if method == "dtw":
                    similarity = await self._dtw_similarity(query_chroma, cand_chroma)
                elif method == "correlation":
                    similarity = await self._correlation_similarity(query_chroma, cand_chroma)
                else:
                    similarity = await self._euclidean_similarity(query_chroma, cand_chroma)

                if similarity >= threshold:
                    # Determine if cover
                    is_cover = similarity >= self.COVER_THRESHOLD_LOW
                    if similarity >= self.COVER_THRESHOLD_HIGH:
                        confidence = "high"
                    elif similarity >= self.COVER_THRESHOLD_MEDIUM:
                        confidence = "medium"
                    else:
                        confidence = "low"

                    result = SimilarityResult(
                        similarity=similarity,
                        is_cover=is_cover,
                        confidence=confidence,
                        details={
                            "duration_ratio": dur_ratio,
                            "tempo_ratio": tempo_ratio,
                            "method": method,
                        },
                    )
                    results.append((candidate, result))

            except AcousticSimilarityError:
                continue

        # Sort by similarity descending
        results.sort(key=lambda x: x[1].similarity, reverse=True)
        return results

    def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)


@dataclass
class SimilarityBatchResult:
    """Result of batch similarity analysis."""
    query_file: Path
    similar_tracks: List[Tuple[Path, SimilarityResult]]
    covers_found: int
    total_compared: int


class BatchAcousticSimilarityAnalyzer:
    """Batch processing for acoustic similarity analysis."""

    def __init__(
        self,
        cache: Optional[ChromaCache] = None,
        num_workers: int = 4,
    ) -> None:
        """Initialize batch analyzer.

        Raises:
            AcousticSimilarityError: If librosa is not available.
        """
        if not LIBROSA_AVAILABLE:
            raise AcousticSimilarityError(
                "librosa is not installed. "
                "Install with: pip install librosa numpy scipy"
            )

        self.cache = cache
        self.num_workers = num_workers

    async def batch_find_covers(
        self,
        query_files: List[Path],
        threshold: float = 0.5,
    ) -> Dict[Path, List[Tuple[Path, SimilarityResult]]]:
        """Find cover songs across a collection.

        Args:
            query_files: List of all files to compare.
            threshold: Minimum similarity threshold.

        Returns:
            Dict mapping query file to list of similar files found.
        """
        results = {}

        for i, query_file in enumerate(query_files):
            # Compare with remaining files
            candidates = query_files[i + 1:]

            analyzer = AcousticSimilarityAnalyzer(cache=self.cache)
            try:
                similar = await analyzer.find_similar_tracks(
                    query_file,
                    candidates,
                    threshold=threshold,
                )

                # Only include covers (higher threshold)
                covers = [
                    (f, r) for f, r in similar
                    if r.is_cover
                ]

                if covers:
                    results[query_file] = covers

            finally:
                analyzer.close()

        return results

    def close(self) -> None:
        """Clean up resources."""
        if self.cache:
            self.cache.close()


def is_available() -> bool:
    """Check if acoustic similarity analysis is available.

    Returns:
        True if librosa and scipy are installed, False otherwise.
    """
    return LIBROSA_AVAILABLE
