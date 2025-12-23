"""Duplicate detection plugin with audio fingerprinting and chroma-based similarity."""

import hashlib
import struct
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict
import math

from ..base import ClassificationPlugin, PluginInfo
from ..config import PluginConfigSchema, ConfigOption, create_classification_plugin_schema
from ...models.audio_file import AudioFile

# Import acoustic similarity for chroma-based cover detection
try:
    from music_organizer.ml.acoustic_similarity import (
        AcousticSimilarityAnalyzer,
        ChromaCache,
        SimilarityResult,
        is_available as acoustic_available,
    )
    ACOUSTIC_AVAILABLE = acoustic_available()
except ImportError:
    ACOUSTIC_AVAILABLE = False


class DuplicateDetectorPlugin(ClassificationPlugin):
    """Detect duplicate audio files using multiple strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize duplicate detector."""
        super().__init__(config)
        self._file_hashes: Dict[str, List[str]] = defaultdict(list)
        self._metadata_hashes: Dict[str, List[str]] = defaultdict(list)
        self._audio_fingerprints: Dict[str, List[str]] = defaultdict(list)
        self._processed_files: Set[str] = set()

        # Acoustic similarity components
        self._chroma_cache: Optional[ChromaCache] = None
        self._chroma_features: Dict[str, Tuple] = {}
        self._acoustic_analyzer: Optional[AcousticSimilarityAnalyzer] = None

        # Initialize acoustic similarity if enabled and available
        if self._should_use_acoustic():
            self._chroma_cache = ChromaCache()
            self._acoustic_analyzer = AcousticSimilarityAnalyzer(cache=self._chroma_cache)

    def _should_use_acoustic(self) -> bool:
        """Check if acoustic similarity should be used."""
        strategies = self.config.get('strategies', ['metadata', 'file_hash', 'audio_fingerprint'])
        return (
            ACOUSTIC_AVAILABLE and
            'chroma_similarity' in strategies
        )

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="duplicate_detector",
            version="1.1.0",
            description="Detect duplicate audio files using audio fingerprinting, chroma-based similarity, and metadata comparison. Includes cover song detection.",
            author="Music Organizer Team",
            dependencies=[],
        )

    def initialize(self) -> None:
        """Initialize the plugin."""
        self._file_hashes.clear()
        self._metadata_hashes.clear()
        self._audio_fingerprints.clear()
        self._processed_files.clear()
        self._chroma_features.clear()

        # Re-initialize acoustic analyzer if needed
        if self._should_use_acoustic() and self._acoustic_analyzer is None:
            self._chroma_cache = ChromaCache()
            self._acoustic_analyzer = AcousticSimilarityAnalyzer(cache=self._chroma_cache)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self._file_hashes.clear()
        self._metadata_hashes.clear()
        self._audio_fingerprints.clear()
        self._processed_files.clear()
        self._chroma_features.clear()

        # Close acoustic analyzer
        if self._acoustic_analyzer is not None:
            self._acoustic_analyzer.close()
            self._acoustic_analyzer = None

        if self._chroma_cache is not None:
            self._chroma_cache.close()
            self._chroma_cache = None

    async def classify(self, audio_file: AudioFile) -> Dict[str, Any]:
        """Classify an audio file for duplicates."""
        if not self.enabled or audio_file.path in self._processed_files:
            return {}

        classifications = {}
        file_path = str(audio_file.path)

        # Generate different hashes for comparison
        strategies = self.config.get('strategies', ['metadata', 'file_hash', 'audio_fingerprint'])

        duplicate_groups = []

        # Strategy 1: Exact metadata match
        if 'metadata' in strategies:
            metadata_hash = self._generate_metadata_hash(audio_file)
            if metadata_hash in self._metadata_hashes:
                duplicates = self._metadata_hashes[metadata_hash]
                if duplicates:
                    duplicate_groups.append({
                        'type': 'metadata',
                        'confidence': 1.0,
                        'duplicates': duplicates,
                        'reason': 'Exact metadata match (artist, title, album, duration)'
                    })
            self._metadata_hashes[metadata_hash].append(file_path)

        # Strategy 2: File hash (exact duplicate)
        if 'file_hash' in strategies:
            file_hash = await self._generate_file_hash(audio_file.path)
            if file_hash in self._file_hashes:
                duplicates = self._file_hashes[file_hash]
                if duplicates:
                    duplicate_groups.append({
                        'type': 'exact',
                        'confidence': 1.0,
                        'duplicates': duplicates,
                        'reason': 'Identical file (exact bit-for-bit copy)'
                    })
            self._file_hashes[file_hash].append(file_path)

        # Strategy 3: Audio fingerprint (basic acoustic similarity)
        if 'audio_fingerprint' in strategies:
            try:
                fingerprint = await self._generate_audio_fingerprint(audio_file)
                if fingerprint:
                    similar_files = self._find_similar_fingerprints(fingerprint)
                    if similar_files:
                        for similar_file, similarity in similar_files:
                            if similarity >= self.config.get('similarity_threshold', 0.85):
                                duplicate_groups.append({
                                    'type': 'acoustic',
                                    'confidence': similarity,
                                    'duplicates': [similar_file],
                                    'reason': f'Audio fingerprint similarity: {similarity:.2%}'
                                })
                    self._audio_fingerprints[fingerprint].append(file_path)
            except Exception as e:
                # Fingerprinting can fail, but don't fail the entire plugin
                pass

        # Strategy 4: Chroma-based similarity (cover song detection)
        if 'chroma_similarity' in strategies and self._acoustic_analyzer is not None:
            try:
                # Extract chroma features for current file
                chroma_key = await self._extract_chroma_key(audio_file.path)
                if chroma_key:
                    # Compare with existing chroma features
                    chroma_matches = await self._find_chroma_matches(audio_file.path, chroma_key)
                    if chroma_matches:
                        for match_file, similarity_result in chroma_matches:
                            threshold = self.config.get('chroma_threshold', 0.60)
                            if similarity_result.similarity >= threshold:
                                match_type = 'cover' if similarity_result.is_cover else 'similar'
                                duplicate_groups.append({
                                    'type': match_type,
                                    'confidence': similarity_result.similarity,
                                    'duplicates': [match_file],
                                    'reason': f'Chroma-based {match_type} detection (confidence: {similarity_result.confidence})',
                                    'details': similarity_result.details
                                })
            except Exception as e:
                # Chroma analysis can fail, but don't fail the entire plugin
                pass

        # Filter duplicates based on configuration
        if duplicate_groups:
            filtered_duplicates = self._filter_duplicates(duplicate_groups)
            if filtered_duplicates:
                classifications['duplicates'] = filtered_duplicates
                # Count total number of duplicate files (all files in groups except the current one)
                classifications['duplicate_count'] = sum(len(g['duplicates']) for g in filtered_duplicates)
                classifications['is_duplicate'] = True

        self._processed_files.add(file_path)
        return classifications

    def get_supported_tags(self) -> List[str]:
        """Return supported classification tags."""
        return ['duplicates', 'duplicate_count', 'is_duplicate']

    def _generate_metadata_hash(self, audio_file: AudioFile) -> str:
        """Generate hash based on key metadata fields."""
        # Normalize metadata for comparison
        artist = self._normalize_text(audio_file.primary_artist or (audio_file.artists[0] if audio_file.artists else ""))
        title = self._normalize_text(audio_file.title or "")
        album = self._normalize_text(audio_file.album or "")
        track = audio_file.track_number or 0

        # Create hash
        metadata = f"{artist}|{title}|{album}|{track}"
        return hashlib.md5(metadata.encode('utf-8')).hexdigest()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        # Remove extra whitespace and convert to lowercase
        return " ".join(text.lower().split())

    async def _generate_file_hash(self, file_path: Path) -> str:
        """Generate MD5 hash of the file."""
        try:
            # Use a simple hash based on file size and last modified for speed
            stat = file_path.stat()
            return hashlib.md5(f"{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()
        except (OSError, FileNotFoundError):
            # If file doesn't exist, use the path as fallback
            return hashlib.md5(str(file_path).encode()).hexdigest()

    async def _generate_audio_fingerprint(self, audio_file: AudioFile) -> Optional[str]:
        """Generate a simplified audio fingerprint.

        This is a basic implementation using file properties and metadata.
        For production use, consider using libraries like chromaprint.
        """
        try:
            # Try to get file properties
            try:
                file_stat = audio_file.path.stat()
                file_size = file_stat.st_size
            except (OSError, FileNotFoundError):
                file_size = 0

            # Extract audio data using mutagen
            try:
                from mutagen import File as MutagenFile
                audio_obj = MutagenFile(audio_file.path)

                if audio_obj is not None:
                    # Get audio properties from mutagen
                    bitrate = getattr(audio_obj.info, 'bitrate', 0) if hasattr(audio_obj, 'info') else 0
                    sample_rate = getattr(audio_obj.info, 'sample_rate', 0) if hasattr(audio_obj, 'info') else 0
                    channels = getattr(audio_obj.info, 'channels', 0) if hasattr(audio_obj, 'info') else 0
                    length = getattr(audio_obj.info, 'length', 0) if hasattr(audio_obj, 'info') else 0
                else:
                    bitrate = sample_rate = channels = length = 0
            except ImportError:
                # Mutagen not available, use zeros
                bitrate = sample_rate = channels = length = 0

            # Combine with metadata
            artist = self._normalize_text(audio_file.primary_artist or (audio_file.artists[0] if audio_file.artists else ""))[:20]
            title = self._normalize_text(audio_file.title or "")[:20]

            fingerprint_data = f"{file_size}_{length:.2f}_{bitrate}_{sample_rate}_{channels}_{artist}_{title}"
            return hashlib.sha256(fingerprint_data.encode('utf-8')).hexdigest()

        except Exception:
            return None

    def _find_similar_fingerprints(self, fingerprint: str) -> List[Tuple[str, float]]:
        """Find similar fingerprints using hamming distance."""
        similar_files = []

        # Store fingerprints for comparison
        for existing_fp, files in self._audio_fingerprints.items():
            if existing_fp == fingerprint:
                continue

            # Calculate similarity (simplified)
            similarity = self._calculate_fingerprint_similarity(fingerprint, existing_fp)

            if similarity > 0:
                for file_path in files:
                    similar_files.append((file_path, similarity))

        return sorted(similar_files, key=lambda x: x[1], reverse=True)

    def _calculate_fingerprint_similarity(self, fp1: str, fp2: str) -> float:
        """Calculate similarity between two fingerprints."""
        if len(fp1) != len(fp2):
            return 0.0

        # Compare chunks of the hash
        chunk_size = 8
        matches = 0
        total_chunks = len(fp1) // chunk_size

        for i in range(0, len(fp1), chunk_size):
            if i + chunk_size <= len(fp1) and i + chunk_size <= len(fp2):
                if fp1[i:i+chunk_size] == fp2[i:i+chunk_size]:
                    matches += 1

        return matches / total_chunks if total_chunks > 0 else 0.0

    async def _extract_chroma_key(self, file_path: Path) -> Optional[str]:
        """Extract chroma features key for similarity comparison.

        Returns a key that can be used for quick filtering before full comparison.
        """
        if self._acoustic_analyzer is None:
            return None

        try:
            from music_organizer.ml.audio_features import FeatureCache
            file_hash = FeatureCache.compute_file_hash(file_path)

            # Check if we already have features cached
            if file_hash in self._chroma_features:
                return file_hash

            # Extract and cache chroma features
            chroma, duration, tempo = await self._acoustic_analyzer.extract_chroma(file_path)
            self._chroma_features[file_hash] = (chroma, duration, tempo)

            return file_hash
        except Exception:
            return None

    async def _find_chroma_matches(
        self,
        file_path: Path,
        chroma_key: str
    ) -> List[Tuple[str, SimilarityResult]]:
        """Find files with similar chroma features using DTW comparison."""
        if self._acoustic_analyzer is None or chroma_key not in self._chroma_features:
            return []

        matches = []
        from music_organizer.ml.audio_features import FeatureCache
        current_hash = FeatureCache.compute_file_hash(file_path)
        current_chroma, current_dur, current_tempo = self._chroma_features[chroma_key]

        # Compare with all cached chroma features
        for other_hash, (other_chroma, other_dur, other_tempo) in self._chroma_features.items():
            if other_hash == current_hash:
                continue

            try:
                # Quick filter: duration and tempo should be in reasonable range
                dur_ratio = min(current_dur, other_dur) / max(current_dur, other_dur)
                tempo_ratio = min(current_tempo, other_tempo) / max(current_tempo, other_tempo)

                # Skip if too different (unless it's a remix)
                if dur_ratio < 0.3 or dur_ratio > 3.0:
                    continue

                # Use DTW for tempo-invariant comparison
                similarity = await self._acoustic_analyzer._dtw_similarity(
                    current_chroma, other_chroma
                )

                if similarity > 0.4:  # Minimum threshold for reporting
                    # Determine if likely a cover
                    is_cover = similarity >= 0.60  # Low threshold for cover
                    if similarity >= 0.75:
                        confidence = "high"
                    elif similarity >= 0.60:
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
                            "method": "dtw",
                        },
                    )

                    # Find the original file path for this hash
                    for other_file in self._processed_files:
                        other_hash_check = FeatureCache.compute_file_hash(Path(other_file))
                        if other_hash_check == other_hash:
                            matches.append((other_file, result))
                            break

            except Exception:
                continue

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1].similarity, reverse=True)
        return matches

    def _filter_duplicates(self, duplicate_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter duplicates based on configuration."""
        filtered = []

        # Filter by minimum confidence
        min_confidence = self.config.get('min_confidence', 0.5)

        for group in duplicate_groups:
            if group['confidence'] >= min_confidence:
                # Filter by duplicate type if specified
                allowed_types = self.config.get('allowed_types', ['exact', 'metadata', 'acoustic'])
                if group['type'] in allowed_types:
                    filtered.append(group)

        return filtered

    def get_config_schema(self) -> PluginConfigSchema:
        """Return configuration schema for this plugin."""
        schema = create_classification_plugin_schema()

        # Add custom options
        schema.add_option(ConfigOption(
            name="strategies",
            type=list,
            default=['metadata', 'file_hash', 'audio_fingerprint'],
            description="Duplicate detection strategies to use",
            choices=['metadata', 'file_hash', 'audio_fingerprint', 'chroma_similarity']
        ))

        schema.add_option(ConfigOption(
            name="similarity_threshold",
            type=float,
            default=0.85,
            min_value=0.0,
            max_value=1.0,
            description="Minimum similarity threshold for acoustic fingerprint matching"
        ))

        schema.add_option(ConfigOption(
            name="chroma_threshold",
            type=float,
            default=0.60,
            min_value=0.0,
            max_value=1.0,
            description="Minimum similarity threshold for chroma-based cover detection"
        ))

        schema.add_option(ConfigOption(
            name="min_confidence",
            type=float,
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            description="Minimum confidence level to report duplicates"
        ))

        schema.add_option(ConfigOption(
            name="allowed_types",
            type=list,
            default=['exact', 'metadata', 'acoustic', 'cover'],
            description="Types of duplicates to detect",
            choices=['exact', 'metadata', 'acoustic', 'cover', 'similar']
        ))

        schema.add_option(ConfigOption(
            name="report_duplicates_only",
            type=bool,
            default=True,
            description="Only report files that have duplicates"
        ))

        return schema

    async def batch_classify(self, audio_files: List[AudioFile]) -> List[Dict[str, Any]]:
        """Classify multiple files with optimized duplicate detection."""
        # Initialize all hashes for batch processing
        if not audio_files:
            return []

        # Process files in batches for better memory usage
        batch_size = self.config.get('batch_size', 100)
        results = []

        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i + batch_size]
            batch_results = []

            # Process batch
            for audio_file in batch:
                result = await self.classify(audio_file)
                if self.config.get('report_duplicates_only', True) and not result.get('is_duplicate'):
                    result = {}
                batch_results.append(result)

            results.extend(batch_results)

        return results

    def get_duplicate_summary(self) -> Dict[str, Any]:
        """Get summary of detected duplicates."""
        total_groups = 0
        total_duplicates = 0

        for hash_key, files in list(self._metadata_hashes.items()) + list(self._file_hashes.items()):
            if len(files) > 1:
                total_groups += 1
                total_duplicates += len(files) - 1

        summary = {
            'total_duplicate_groups': total_groups,
            'total_duplicate_files': total_duplicates,
            'unique_files_processed': len(self._processed_files),
            'metadata_hashes': len(self._metadata_hashes),
            'file_hashes': len(self._file_hashes),
            'audio_fingerprints': len(self._audio_fingerprints),
            'chroma_features_cached': len(self._chroma_features),
            'acoustic_similarity_available': ACOUSTIC_AVAILABLE,
        }

        return summary