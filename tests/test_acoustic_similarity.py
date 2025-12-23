"""Tests for acoustic similarity detection using chroma features."""

from pathlib import Path
import tempfile
import asyncio

import pytest

from music_organizer.ml.acoustic_similarity import (
    AcousticSimilarityAnalyzer,
    AcousticSimilarityError,
    ChromaCache,
    SimilarityResult,
    BatchAcousticSimilarityAnalyzer,
    is_available,
)


def pytest_run_sync(coro):
    """Helper to run async functions in sync tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


class TestIsAvailable:
    """Tests for is_available function."""

    def test_returns_boolean(self):
        """Should return a boolean."""
        result = is_available()
        assert isinstance(result, bool)


class TestChromaCache:
    """Tests for ChromaCache."""

    @pytest.fixture
    def temp_cache(self, tmp_path):
        """Create a temporary cache."""
        cache_path = tmp_path / "test_chroma.db"
        return ChromaCache(cache_path=cache_path)

    def test_cache_initialization(self, temp_cache):
        """Cache should initialize without error."""
        # Cache is created lazily on first use
        result = pytest_run_sync(temp_cache.get("test"))
        assert temp_cache.cache_path.exists()

    def test_cache_miss(self, temp_cache):
        """Cache miss should return None."""
        result = pytest_run_sync(temp_cache.get("nonexistent"))
        assert result is None

    def test_cache_set_and_get(self, temp_cache):
        """Should cache and retrieve chroma features."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        file_hash = "test_hash_123"
        chroma = np.random.rand(12, 100).astype(np.float32)
        duration = 180.0
        tempo = 120.0

        # Set
        pytest_run_sync(temp_cache.set(file_hash, chroma, duration, tempo))

        # Get
        result = pytest_run_sync(temp_cache.get(file_hash))

        assert result is not None
        retrieved_chroma, retrieved_dur, retrieved_tempo = result
        assert retrieved_chroma.shape == chroma.shape
        assert retrieved_dur == duration
        assert retrieved_tempo == tempo

    def test_cache_replace(self, temp_cache):
        """Should replace existing cached features."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        file_hash = "test_hash_456"
        chroma1 = np.random.rand(12, 100).astype(np.float32)

        # Set original
        pytest_run_sync(temp_cache.set(file_hash, chroma1, 180.0, 120.0))

        # Set updated
        chroma2 = np.random.rand(12, 200).astype(np.float32)
        pytest_run_sync(temp_cache.set(file_hash, chroma2, 200.0, 140.0))

        # Get should return updated
        result = pytest_run_sync(temp_cache.get(file_hash))
        assert result is not None
        retrieved_chroma, _, _ = result
        assert retrieved_chroma.shape == (12, 200)


class TestSimilarityResult:
    """Tests for SimilarityResult dataclass."""

    def test_initialization(self):
        """Should initialize with values."""
        result = SimilarityResult(
            similarity=0.85,
            is_cover=True,
            confidence="high",
            details={"method": "dtw"},
        )

        assert result.similarity == 0.85
        assert result.is_cover is True
        assert result.confidence == "high"
        assert result.details == {"method": "dtw"}

    def test_is_high_confidence(self):
        """Should correctly identify high confidence results."""
        result = SimilarityResult(
            similarity=0.80,
            is_cover=True,
            confidence="high",
            details={},
        )

        assert result.is_high_confidence is True

    def test_is_not_high_confidence_low_similarity(self):
        """Low similarity should not be high confidence."""
        result = SimilarityResult(
            similarity=0.70,
            is_cover=True,
            confidence="high",
            details={},
        )

        assert result.is_high_confidence is False

    def test_is_not_high_confidence_medium_level(self):
        """Medium confidence should not be high confidence."""
        result = SimilarityResult(
            similarity=0.80,
            is_cover=True,
            confidence="medium",
            details={},
        )

        assert result.is_high_confidence is False


class TestAcousticSimilarityAnalyzer:
    """Tests for AcousticSimilarityAnalyzer."""

    def test_init_without_librosa(self, monkeypatch):
        """Should raise error if librosa not available."""
        import music_organizer.ml.acoustic_similarity as as_module
        monkeypatch.setattr(as_module, "LIBROSA_AVAILABLE", False)

        with pytest.raises(AcousticSimilarityError, match="librosa is not installed"):
            AcousticSimilarityAnalyzer()

    def test_init(self):
        """Should initialize without error."""
        if not is_available():
            pytest.skip("librosa not available")

        analyzer = AcousticSimilarityAnalyzer()
        assert analyzer.sample_rate == 22050
        assert analyzer.n_chroma == 12

    def test_init_with_cache(self, tmp_path):
        """Should initialize with cache."""
        if not is_available():
            pytest.skip("librosa not available")

        cache = ChromaCache(cache_path=tmp_path / "cache.db")
        analyzer = AcousticSimilarityAnalyzer(cache=cache)

        assert analyzer.cache is not None

    def test_extract_chroma_nonexistent_file(self):
        """Should raise error for nonexistent file."""
        if not is_available():
            pytest.skip("librosa not available")

        analyzer = AcousticSimilarityAnalyzer()

        with pytest.raises(AcousticSimilarityError, match="File not found"):
            pytest.run_sync(
                analyzer.extract_chroma(Path("/nonexistent/file.mp3"))
            )

    def test_extract_chroma_invalid_audio(self, tmp_path):
        """Should handle invalid audio files gracefully."""
        if not is_available():
            pytest.skip("librosa not available")

        # Create a dummy file (not valid audio)
        invalid_file = tmp_path / "invalid.mp3"
        invalid_file.write_bytes(b"not a valid audio file")

        analyzer = AcousticSimilarityAnalyzer()

        # Should fail gracefully
        with pytest.raises(AcousticSimilarityError):
            pytest.run_sync(analyzer.extract_chroma(invalid_file))

    def test_compare_similarity_nonexistent_files(self):
        """Should raise error for nonexistent files."""
        if not is_available():
            pytest.skip("librosa not available")

        analyzer = AcousticSimilarityAnalyzer()

        with pytest.raises(AcousticSimilarityError):
            pytest.run_sync(
                analyzer.compare_similarity(
                    Path("/nonexistent/file1.mp3"),
                    Path("/nonexistent/file2.mp3"),
                )
            )


class TestBatchAcousticSimilarityAnalyzer:
    """Tests for BatchAcousticSimilarityAnalyzer."""

    def test_init_without_librosa(self, monkeypatch):
        """Should raise error if librosa not available."""
        import music_organizer.ml.acoustic_similarity as as_module
        monkeypatch.setattr(as_module, "LIBROSA_AVAILABLE", False)

        with pytest.raises(AcousticSimilarityError, match="librosa is not installed"):
            BatchAcousticSimilarityAnalyzer()

    def test_init(self):
        """Should initialize without error."""
        if not is_available():
            pytest.skip("librosa not available")

        batch = BatchAcousticSimilarityAnalyzer(num_workers=2)
        assert batch.num_workers == 2
        assert batch.cache is None

    def test_init_with_cache(self, tmp_path):
        """Should initialize with cache."""
        if not is_available():
            pytest.skip("librosa not available")

        cache = ChromaCache(cache_path=tmp_path / "cache.db")
        batch = BatchAcousticSimilarityAnalyzer(cache=cache)
        assert batch.cache is not None

    def test_batch_find_covers_empty_list(self):
        """Should handle empty file list."""
        if not is_available():
            pytest.skip("librosa not available")

        batch = BatchAcousticSimilarityAnalyzer()
        result = pytest_run_sync(batch.batch_find_covers([], threshold=0.5))

        assert result == {}

    def test_batch_find_covers_with_invalid_files(self, tmp_path):
        """Should handle invalid files gracefully."""
        if not is_available():
            pytest.skip("librosa not available")

        # Create dummy files
        file1 = tmp_path / "file1.mp3"
        file2 = tmp_path / "file2.mp3"
        file1.write_bytes(b"not audio")
        file2.write_bytes(b"not audio")

        batch = BatchAcousticSimilarityAnalyzer()
        result = pytest_run_sync(batch.batch_find_covers([file1, file2], threshold=0.5))

        # Should return empty dict (no valid audio)
        assert result == {}


class TestIntegration:
    """Integration tests."""

    def test_chroma_cache_integration(self, tmp_path):
        """Test cache and analyzer working together."""
        if not is_available():
            pytest.skip("librosa not available")

        cache = ChromaCache(cache_path=tmp_path / "cache.db")
        analyzer = AcousticSimilarityAnalyzer(cache=cache)

        # Create dummy file (will fail extraction, but tests integration)
        dummy = tmp_path / "dummy.mp3"
        dummy.write_bytes(b"not audio")

        # Should not crash
        try:
            pytest.run_sync(analyzer.extract_chroma(dummy))
        except AcousticSimilarityError:
            pass  # Expected for invalid audio

        cache.close()
        analyzer.close()

    def test_batch_with_progress(self, tmp_path):
        """Test batch processing completes without error."""
        if not is_available():
            pytest.skip("librosa not available")

        batch = BatchAcousticSimilarityAnalyzer()

        # Empty list should complete
        pytest_run_sync(batch.batch_find_covers([]))

        batch.close()
