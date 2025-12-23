"""Audio feature extraction using librosa.

This module provides psychoacoustic feature extraction from audio files,
including tempo, key, energy, danceability, valence, and more.
Features are cached in SQLite for performance.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from music_organizer.domain.classification.value_objects import AudioFeatures


class AudioFeatureError(Exception):
    """Raised when audio feature extraction fails."""
    pass


class FeatureCache:
    """SQLite cache for extracted audio features.

    Caches features by file hash to avoid re-extracting from unchanged files.
    Audio feature extraction is expensive (500-2000ms per file), so caching
    provides significant performance improvements.
    """

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        """Initialize the feature cache.

        Args:
            cache_path: Path to SQLite cache file. Defaults to ~/.cache/music-organizer/features.db
        """
        if cache_path is None:
            cache_dir = Path.home() / ".cache" / "music-organizer"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "features.db"

        self.cache_path = cache_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    def _init_db(self) -> None:
        """Initialize database schema."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.cache_path))
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_features (
                    file_hash TEXT PRIMARY KEY,
                    tempo REAL,
                    key TEXT,
                    mode TEXT,
                    energy REAL,
                    danceability REAL,
                    valence REAL,
                    acousticness REAL,
                    instrumentalness REAL,
                    speechiness REAL,
                    loudness REAL,
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def get(self, file_hash: str) -> Optional[AudioFeatures]:
        """Get cached features for a file.

        Args:
            file_hash: SHA-256 hash of the audio file.

        Returns:
            Cached AudioFeatures or None if not found.
        """
        async with self._lock:
            if self._conn is None:
                self._init_db()

            cursor = self._conn.execute(
                "SELECT tempo, key, mode, energy, danceability, valence, "
                "acousticness, instrumentalness, speechiness, loudness "
                "FROM audio_features WHERE file_hash = ?",
                (file_hash,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return AudioFeatures(
                tempo=row[0],
                key=row[1],
                mode=row[2],
                energy=row[3],
                danceability=row[4],
                valence=row[5],
                acousticness=row[6],
                instrumentalness=row[7],
                speechiness=row[8],
                loudness=row[9],
            )

    async def set(self, file_hash: str, features: AudioFeatures) -> None:
        """Cache features for a file.

        Args:
            file_hash: SHA-256 hash of the audio file.
            features: AudioFeatures to cache.
        """
        async with self._lock:
            if self._conn is None:
                self._init_db()

            self._conn.execute(
                """INSERT OR REPLACE INTO audio_features
                   (file_hash, tempo, key, mode, energy, danceability, valence,
                    acousticness, instrumentalness, speechiness, loudness)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (file_hash, features.tempo, features.key, features.mode,
                 features.energy, features.danceability, features.valence,
                 features.acousticness, features.instrumentalness,
                 features.speechiness, features.loudness)
            )
            self._conn.commit()

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of an audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class AudioFeatureExtractor:
    """Extract psychoacoustic features from audio files using librosa.

    This class provides methods to extract various audio features:
    - Tempo (BPM)
    - Musical key and mode (major/minor)
    - Energy (RMS loudness dynamics)
    - Danceability (rhythmic patterns)
    - Valence (musical positiveness/mood)
    - Acousticness (spectral features)
    - Instrumentalness (vocal vs instrumental)
    - Speechiness (MFCC analysis)
    - Loudness (overall dB)
    """

    # Musical key names from chroma indices
    _KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(
        self,
        cache: Optional[FeatureCache] = None,
        sample_rate: int = 22050,
        duration: Optional[float] = None,
    ) -> None:
        """Initialize the feature extractor.

        Args:
            cache: Optional feature cache for performance.
            sample_rate: Audio sample rate for analysis.
            duration: Max duration to analyze in seconds. None for full file.

        Raises:
            AudioFeatureError: If librosa is not available.
        """
        if not LIBROSA_AVAILABLE:
            raise AudioFeatureError(
                "librosa is not installed. "
                "Install with: pip install librosa numpy"
            )

        self.cache = cache
        self.sample_rate = sample_rate
        self.duration = duration
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def extract_features(self, file_path: Path) -> AudioFeatures:
        """Extract all audio features from a file.

        Args:
            file_path: Path to the audio file.

        Returns:
            AudioFeatures containing all extracted features.

        Raises:
            AudioFeatureError: If extraction fails.
        """
        if not file_path.exists():
            raise AudioFeatureError(f"File not found: {file_path}")

        # Check cache first
        if self.cache is not None:
            file_hash = FeatureCache.compute_file_hash(file_path)
            cached = await self.cache.get(file_hash)
            if cached is not None:
                return cached

        # Extract features in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            features = await loop.run_in_executor(
                self._executor,
                self._extract_features_sync,
                file_path,
            )
        except Exception as e:
            raise AudioFeatureError(f"Failed to extract features from {file_path}: {e}")

        # Cache the results
        if self.cache is not None:
            await self.cache.set(file_hash, features)

        return features

    def _extract_features_sync(self, file_path: Path) -> AudioFeatures:
        """Synchronous feature extraction (runs in thread pool).

        Args:
            file_path: Path to the audio file.

        Returns:
            AudioFeatures containing all extracted features.
        """
        # Suppress librosa warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Load audio
            y, sr = librosa.load(
                str(file_path),
                sr=self.sample_rate,
                duration=self.duration,
                mono=True,
            )

            # Check minimum duration
            if librosa.get_duration(y=y, sr=sr) < 0.5:
                return AudioFeatures()

            # Extract all features
            return AudioFeatures(
                tempo=self._extract_tempo(y, sr),
                key=self._extract_key(y, sr)[0],
                mode=self._extract_key(y, sr)[1],
                energy=self._extract_energy(y, sr),
                danceability=self._extract_danceability(y, sr),
                valence=self._extract_valence(y, sr),
                acousticness=self._extract_acousticness(y, sr),
                instrumentalness=self._extract_instrumentalness(y, sr),
                speechiness=self._extract_speechiness(y, sr),
                loudness=self._extract_loudness(y, sr),
            )

    def _extract_tempo(self, y: np.ndarray, sr: int) -> Optional[float]:
        """Extract BPM using librosa.beat.tempo.

        Args:
            y: Audio time series.
            sr: Sample rate.

        Returns:
            Tempo in BPM or None if detection fails.
        """
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return float(tempo)
        except Exception:
            return None

    def _extract_key(self, y: np.ndarray, sr: int) -> tuple[Optional[str], Optional[str]]:
        """Extract musical key (key + mode: major/minor) using chroma features.

        Args:
            y: Audio time series.
            sr: Sample rate.

        Returns:
            Tuple of (key, mode) where key is like "C", "A#", etc.
            and mode is "major" or "minor".
        """
        try:
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = chroma.mean(axis=1)

            # Get the most prominent pitch class
            key_idx = int(chroma_mean.argmax())
            key = self._KEY_NAMES[key_idx]

            # Determine major/minor by comparing major/minor thirds
            # Major third: 4 semitones, Minor third: 3 semitones
            major_third_idx = (key_idx + 4) % 12
            minor_third_idx = (key_idx + 3) % 12

            if chroma_mean[major_third_idx] > chroma_mean[minor_third_idx]:
                mode = "major"
            else:
                mode = "minor"

            return key, mode
        except Exception:
            return None, None

    def _extract_energy(self, y: np.ndarray, sr: int) -> Optional[float]:
        """Extract RMS energy for loudness dynamics.

        Args:
            y: Audio time series.
            sr: Sample rate.

        Returns:
            Energy value 0.0-1.0 or None.
        """
        try:
            rms = librosa.feature.rms(y=y)
            energy = float(rms.mean())
            # Normalize to 0-1 range (typical RMS values are 0-0.5)
            return min(energy * 2, 1.0)
        except Exception:
            return None

    def _extract_danceability(self, y: np.ndarray, sr: int) -> Optional[float]:
        """Calculate danceability from rhythmic patterns.

        Danceability is based on:
        - Tempo stability
        - Beat strength
        - Rhythmic regularity

        Args:
            y: Audio time series.
            sr: Sample rate.

        Returns:
            Danceability 0.0-1.0 or None.
        """
        try:
            # Get tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

            # Extract onset envelope for rhythmic analysis
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # Danceability factors:
            # 1. Strong rhythmic patterns (high onset variance)
            rhythmic_strength = float(onset_env.std() / (onset_env.mean() + 1e-6))

            # 2. Tempo in danceable range (100-140 BPM is ideal)
            tempo_score = 1.0
            if 100 <= tempo <= 140:
                tempo_score = 1.0
            elif 80 <= tempo < 100 or 140 < tempo <= 160:
                tempo_score = 0.7
            else:
                tempo_score = 0.4

            # Combine factors
            danceability = min(rhythmic_strength * 0.3 + tempo_score * 0.7, 1.0)
            return max(0.0, danceability)
        except Exception:
            return None

    def _extract_valence(self, y: np.ndarray, sr: int) -> Optional[float]:
        """Extract mood (positiveness) from timbral features.

        Valence is estimated from:
        - Spectral centroid (brightness)
        - Spectral contrast
        - Zero crossing rate

        Args:
            y: Audio time series.
            sr: Sample rate.

        Returns:
            Valence 0.0-1.0 (sad to happy) or None.
        """
        try:
            # Spectral features for mood analysis
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)

            # Brighter sounds tend to be happier (higher centroid)
            brightness = float(centroid.mean() / sr)

            # Higher contrast suggests more emotional intensity
            contrast_mean = float(contrast.mean())

            # Lower ZCR suggests cleaner, happier sounds
            zcr_mean = float(zcr.mean())

            # Combine into valence score
            # Brightness and low ZCR -> positive, high contrast -> intense
            valence = (brightness * 0.5 + (1 - zcr_mean) * 0.3 + contrast_mean * 0.2)
            return max(0.0, min(valence, 1.0))
        except Exception:
            return None

    def _extract_acousticness(self, y: np.ndarray, sr: int) -> Optional[float]:
        """Extract acousticness from spectral features.

        Acousticness is indicated by:
        - Lower spectral bandwidth
        - Fewer high-frequency components
        - Less dynamic range compression

        Args:
            y: Audio time series.
            sr: Sample rate.

        Returns:
            Acousticness 0.0-1.0 or None.
        """
        try:
            # Spectral bandwidth (narrower = more acoustic)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            bandwidth_norm = float(bandwidth.mean() / (sr / 2))

            # Spectral rolloff (lower = more acoustic)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            rolloff_norm = float(rolloff.mean() / (sr / 2))

            # Combine: narrow bandwidth + low rolloff = acoustic
            acousticness = 1.0 - (bandwidth_norm * 0.5 + rolloff_norm * 0.5)
            return max(0.0, min(acousticness, 1.0))
        except Exception:
            return None

    def _extract_speechiness(self, y: np.ndarray, sr: int) -> Optional[float]:
        """Extract speechiness using MFCC analysis.

        Speech is characterized by:
        - Specific MFCC patterns
        - Lower rhythmic complexity
        - Different spectral envelope

        Args:
            y: Audio time series.
            sr: Sample rate.

        Returns:
            Speechiness 0.0-1.0 or None.
        """
        try:
            # MFCC analysis
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # Speech has specific MFCC variance patterns
            mfcc_std = mfcc.std(axis=1)

            # Lower MFCC variance in lower coefficients suggests speech
            speech_indicator = float(mfcc_std[:5].mean())

            # Normalize to 0-1 (speech typically has variance 5-15)
            speechiness = max(0.0, min(speech_indicator / 20.0, 1.0))

            return speechiness
        except Exception:
            return None

    def _extract_instrumentalness(self, y: np.ndarray, sr: int) -> Optional[float]:
        """Estimate instrumental vs vocal content.

        Instrumentalness is based on:
        - Vocal presence detection (complex spectral patterns)
        - Harmonic-percussive separation

        Args:
            y: Audio time series.
            sr: Sample rate.

        Returns:
            Instrumentalness 0.0-1.0 or None.
        """
        try:
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)

            # Calculate ratio
            harmonic_energy = float(np.mean(y_harmonic ** 2))
            percussive_energy = float(np.mean(y_percussive ** 2))
            total = harmonic_energy + percussive_energy

            if total < 1e-10:
                return None

            # High harmonic ratio suggests instrumental
            harmonic_ratio = harmonic_energy / total

            # Also consider spectral complexity for vocals
            spectral_complexity = float(librosa.feature.spectral_flatness(y=y).mean())

            # Combine: high harmonic + low complexity = instrumental
            instrumentalness = harmonic_ratio * 0.7 + (1 - spectral_complexity) * 0.3
            return max(0.0, min(instrumentalness, 1.0))
        except Exception:
            return None

    def _extract_loudness(self, y: np.ndarray, sr: int) -> Optional[float]:
        """Extract overall loudness in dB.

        Args:
            y: Audio time series.
            sr: Sample rate.

        Returns:
            Loudness in dB (negative values, typically -60 to 0) or None.
        """
        try:
            # RMS to dB conversion
            rms = librosa.feature.rms(y=y)
            rms_mean = float(rms.mean())

            # Avoid log(0)
            if rms_mean < 1e-10:
                return -60.0

            loudness_db = 20 * np.log10(rms_mean)
            return float(loudness_db)
        except Exception:
            return None

    def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)


@dataclass
class BatchProgress:
    """Progress information for batch extraction."""
    total: int
    completed: int
    failed: int
    current_file: Optional[str] = None


class BatchAudioFeatureExtractor:
    """Extract features from multiple files in parallel with progress tracking.

    This class provides batch processing capabilities for extracting audio
    features from multiple files concurrently, with progress callbacks.
    """

    def __init__(
        self,
        num_workers: int = 4,
        cache: Optional[FeatureCache] = None,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
    ) -> None:
        """Initialize the batch extractor.

        Args:
            num_workers: Number of parallel workers for extraction.
            cache: Optional feature cache.
            progress_callback: Optional callback for progress updates.

        Raises:
            AudioFeatureError: If librosa is not available.
        """
        if not LIBROSA_AVAILABLE:
            raise AudioFeatureError(
                "librosa is not installed. "
                "Install with: pip install librosa numpy"
            )

        self.num_workers = num_workers
        self.cache = cache
        self.progress_callback = progress_callback

    async def extract_batch(
        self,
        file_paths: List[Path],
    ) -> Dict[Path, AudioFeatures]:
        """Extract features from multiple files in parallel.

        Args:
            file_paths: List of paths to audio files.

        Returns:
            Dictionary mapping file paths to their AudioFeatures.
        """
        results: Dict[Path, AudioFeatures] = {}
        progress = BatchProgress(
            total=len(file_paths),
            completed=0,
            failed=0,
        )

        # Create semaphore for parallel processing
        semaphore = asyncio.Semaphore(self.num_workers)

        async def extract_one(path: Path) -> tuple[Path, Optional[AudioFeatures]]:
            async with semaphore:
                progress.current_file = str(path)
                extractor = AudioFeatureExtractor(cache=self.cache)
                try:
                    features = await extractor.extract_features(path)
                    progress.completed += 1
                    return path, features
                except AudioFeatureError:
                    progress.failed += 1
                    return path, None
                finally:
                    extractor.close()
                    if self.progress_callback:
                        self.progress_callback(progress)

        # Run all extractions concurrently
        tasks = [extract_one(path) for path in file_paths]
        for task in asyncio.as_completed(tasks):
            path, features = await task
            if features is not None:
                results[path] = features

        return results


def is_available() -> bool:
    """Check if audio feature extraction is available.

    Returns:
        True if librosa is installed, False otherwise.
    """
    return LIBROSA_AVAILABLE
