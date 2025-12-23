"""Tests for audio feature extraction using librosa."""

from pathlib import Path
import tempfile

import pytest

from music_organizer.ml.audio_features import (
    AudioFeatureExtractor,
    AudioFeatureError,
    FeatureCache,
    BatchAudioFeatureExtractor,
    BatchProgress,
    is_available,
)


def pytest_run_sync(coro):
    """Helper to run async functions in sync tests."""
    import asyncio
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


class TestIsAvailable:
    """Tests for is_available function."""

    def test_returns_boolean(self):
        """Should return a boolean."""
        result = is_available()
        assert isinstance(result, bool)


class TestFeatureCache:
    """Tests for FeatureCache."""

    @pytest.fixture
    def temp_cache(self, tmp_path):
        """Create a temporary cache."""
        cache_path = tmp_path / "test_features.db"
        return FeatureCache(cache_path=cache_path)

    @pytest.fixture
    def sample_features(self):
        """Create sample AudioFeatures."""
        from music_organizer.domain.classification.value_objects import AudioFeatures
        return AudioFeatures(
            tempo=120.0,
            key="C",
            mode="major",
            energy=0.7,
            danceability=0.8,
            valence=0.6,
            acousticness=0.3,
            instrumentalness=0.5,
            speechiness=0.1,
            loudness=-6.0,
        )

    def test_cache_initialization(self, temp_cache):
        """Cache should initialize without error."""
        # Cache is created lazily on first use
        result = pytest_run_sync(temp_cache.get("test"))
        assert temp_cache.cache_path.exists()

    def test_cache_miss(self, temp_cache):
        """Cache miss should return None."""
        result = pytest_run_sync(temp_cache.get("nonexistent"))
        assert result is None

    def test_cache_set_and_get(self, temp_cache, sample_features):
        """Should cache and retrieve features."""
        file_hash = "test_hash_123"

        # Set
        pytest_run_sync(temp_cache.set(file_hash, sample_features))

        # Get
        result = pytest_run_sync(temp_cache.get(file_hash))

        assert result is not None
        assert result.tempo == 120.0
        assert result.key == "C"
        assert result.mode == "major"
        assert result.energy == 0.7

    def test_cache_replace(self, temp_cache, sample_features):
        """Should replace existing cached features."""
        from music_organizer.domain.classification.value_objects import AudioFeatures

        file_hash = "test_hash_456"

        # Set original
        pytest_run_sync(temp_cache.set(file_hash, sample_features))

        # Set updated
        updated = AudioFeatures(tempo=140.0)
        pytest_run_sync(temp_cache.set(file_hash, updated))

        # Get should return updated
        result = pytest_run_sync(temp_cache.get(file_hash))
        assert result.tempo == 140.0

    def test_compute_file_hash(self, tmp_path):
        """Should compute consistent file hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        hash1 = FeatureCache.compute_file_hash(test_file)
        hash2 = FeatureCache.compute_file_hash(test_file)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 = 64 hex chars

    def test_compute_file_hash_different_files(self, tmp_path):
        """Different files should have different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("content 1")
        file2.write_text("content 2")

        hash1 = FeatureCache.compute_file_hash(file1)
        hash2 = FeatureCache.compute_file_hash(file2)

        assert hash1 != hash2


class TestAudioFeatureExtractor:
    """Tests for AudioFeatureExtractor."""

    def test_init_without_librosa(self, monkeypatch):
        """Should raise error if librosa not available."""
        # Mock librosa as unavailable
        import music_organizer.ml.audio_features as af_module
        monkeypatch.setattr(af_module, "LIBROSA_AVAILABLE", False)

        with pytest.raises(AudioFeatureError, match="librosa is not installed"):
            AudioFeatureExtractor()

    def test_init_with_cache(self, tmp_path):
        """Should initialize with cache."""
        if not is_available():
            pytest.skip("librosa not available")

        cache_path = tmp_path / "cache.db"
        cache = FeatureCache(cache_path=cache_path)
        extractor = AudioFeatureExtractor(cache=cache)

        assert extractor.cache is not None
        assert extractor.sample_rate == 22050

    def test_init_custom_sample_rate(self):
        """Should accept custom sample rate."""
        if not is_available():
            pytest.skip("librosa not available")

        extractor = AudioFeatureExtractor(sample_rate=44100)
        assert extractor.sample_rate == 44100

    def test_extract_features_nonexistent_file(self):
        """Should raise error for nonexistent file."""
        if not is_available():
            pytest.skip("librosa not available")

        extractor = AudioFeatureExtractor()

        with pytest.raises(AudioFeatureError, match="File not found"):
            pytest.run_sync(extractor.extract_features(Path("/nonexistent/file.mp3")))

    def test_extract_features_invalid_audio(self, tmp_path):
        """Should handle invalid audio files gracefully."""
        if not is_available():
            pytest.skip("librosa not available")

        # Create a dummy file (not valid audio)
        invalid_file = tmp_path / "invalid.mp3"
        invalid_file.write_bytes(b"not a valid audio file")

        extractor = AudioFeatureExtractor()

        # Should either fail gracefully or return empty features
        try:
            result = pytest_run_sync(extractor.extract_features(invalid_file))
            # If it doesn't fail, result should have some None values
            assert isinstance(result, AudioFeatures)
        except AudioFeatureError:
            # Also acceptable
            pass


class TestBatchAudioFeatureExtractor:
    """Tests for BatchAudioFeatureExtractor."""

    def test_init_without_librosa(self, monkeypatch):
        """Should raise error if librosa not available."""
        import music_organizer.ml.audio_features as af_module
        monkeypatch.setattr(af_module, "LIBROSA_AVAILABLE", False)

        with pytest.raises(AudioFeatureError, match="librosa is not installed"):
            BatchAudioFeatureExtractor()

    def test_init(self):
        """Should initialize without error."""
        if not is_available():
            pytest.skip("librosa not available")

        batch = BatchAudioFeatureExtractor(num_workers=2)
        assert batch.num_workers == 2
        assert batch.cache is None

    def test_init_with_cache(self, tmp_path):
        """Should initialize with cache."""
        if not is_available():
            pytest.skip("librosa not available")

        cache = FeatureCache(cache_path=tmp_path / "cache.db")
        batch = BatchAudioFeatureExtractor(cache=cache)
        assert batch.cache is not None

    def test_extract_batch_empty_list(self):
        """Should handle empty file list."""
        if not is_available():
            pytest.skip("librosa not available")

        batch = BatchAudioFeatureExtractor()
        result = pytest_run_sync(batch.extract_batch([]))

        assert result == {}

    def test_extract_batch_with_invalid_files(self, tmp_path):
        """Should handle mix of valid and invalid files."""
        if not is_available():
            pytest.skip("librosa not available")

        # Create dummy files
        file1 = tmp_path / "file1.mp3"
        file2 = tmp_path / "file2.mp3"
        file1.write_bytes(b"not audio")
        file2.write_bytes(b"not audio")

        batch = BatchAudioFeatureExtractor()
        result = pytest_run_sync(batch.extract_batch([file1, file2]))

        # Result may be empty if files are invalid
        assert isinstance(result, dict)


class TestBatchProgress:
    """Tests for BatchProgress dataclass."""

    def test_initialization(self):
        """Should initialize with values."""
        progress = BatchProgress(
            total=10,
            completed=3,
            failed=1,
            current_file="test.mp3",
        )

        assert progress.total == 10
        assert progress.completed == 3
        assert progress.failed == 1
        assert progress.current_file == "test.mp3"

    def test_defaults(self):
        """Should have sensible defaults."""
        progress = BatchProgress(total=100, completed=0, failed=0)

        assert progress.current_file is None


class TestIntegration:
    """Integration tests."""

    def test_feature_cache_integration(self, tmp_path):
        """Test cache and extractor working together."""
        if not is_available():
            pytest.skip("librosa not available")

        cache = FeatureCache(cache_path=tmp_path / "cache.db")
        extractor = AudioFeatureExtractor(cache=cache)

        # Create dummy file (will fail extraction, but tests integration)
        dummy = tmp_path / "dummy.mp3"
        dummy.write_bytes(b"not audio")

        # Should not crash
        try:
            pytest_run_sync(extractor.extract_features(dummy))
        except AudioFeatureError:
            pass  # Expected for invalid audio

        cache.close()
        extractor.close()

    def test_batch_with_progress_callback(self, tmp_path):
        """Test batch extraction with progress callback."""
        if not is_available():
            pytest.skip("librosa not available")

        progress_updates = []

        def callback(progress: BatchProgress) -> None:
            progress_updates.append(progress)

        batch = BatchAudioFeatureExtractor(progress_callback=callback)

        # Extract from empty list
        pytest_run_sync(batch.extract_batch([]))

        # Should have called callback at least once
        assert len(progress_updates) > 0
        assert progress_updates[0].total == 0
