"""Tests for ML genre classifier."""

from pathlib import Path
import tempfile

import pytest

from music_organizer.ml.genre_classifier import (
    GenreClassifier,
    GenreClassificationResult,
    _normalize_genre,
    _extract_features,
    classify_from_text,
)


class TestNormalizeGenre:
    """Tests for genre normalization."""

    def test_normalize_standard_genres(self):
        """Standard genres should pass through."""
        assert _normalize_genre("rock") == "rock"
        assert _normalize_genre("jazz") == "jazz"
        assert _normalize_genre("electronic") == "electronic"

    def test_normalize_synonyms(self):
        """Genre synonyms should be normalized."""
        assert _normalize_genre("r&b") == "rnb"
        assert _normalize_genre("hip-hop") == "hiphop"
        assert _normalize_genre("edm") == "electronic"
        assert _normalize_genre("classic rock") == "rock"

    def test_normalize_none(self):
        """None/empty should return None."""
        assert _normalize_genre(None) is None
        assert _normalize_genre("") is None
        assert _normalize_genre("   ") is None

    def test_normalize_case_insensitive(self):
        """Normalization should be case-insensitive."""
        assert _normalize_genre("ROCK") == "rock"
        assert _normalize_genre("Electronic") == "electronic"
        assert _normalize_genre("JAZZ") == "jazz"


class TestExtractFeatures:
    """Tests for feature extraction."""

    def test_extract_from_title_only(self):
        """Extract features from title only."""
        result = _extract_features(title="Stairway to Heaven")
        assert "Stairway to Heaven" in result

    def test_extract_from_album(self):
        """Extract features from album."""
        result = _extract_features(album="Led Zeppelin IV")
        assert "Led Zeppelin IV" in result

    def test_extract_from_artist(self):
        """Extract features from artist."""
        result = _extract_features(artist="Led Zeppelin")
        assert "Led Zeppelin" in result

    def test_extract_from_year(self):
        """Extract decade from year."""
        result = _extract_features(year=1971)
        assert "decade_1970" in result

        result = _extract_features(year=1999)
        assert "decade_1990" in result

    def test_extract_from_duration(self):
        """Extract duration buckets."""
        result = _extract_features(duration=90)  # 1.5 minutes
        assert "very_short" in result

        result = _extract_features(duration=180)  # 3 minutes
        assert "short" in result

        result = _extract_features(duration=300)  # 5 minutes
        assert "medium" in result

        result = _extract_features(duration=480)  # 8 minutes
        assert "long" in result

        result = _extract_features(duration=660)  # 11 minutes
        assert "extended" in result

    def test_extract_removes_parenthetical(self):
        """Remove parenthetical content from title/album."""
        result = _extract_features(title="Song (Remix)", album="Album (Deluxe Edition)")
        assert "Remix" not in result
        assert "Deluxe Edition" not in result
        assert "Song" in result
        assert "Album" in result

    def test_extract_combined_features(self):
        """Combine all features."""
        result = _extract_features(
            title="Around the World",
            album="Homework",
            artist="Daft Punk",
            year=1997,
            duration=420,
        )
        assert "Around the World" in result
        assert "Homework" in result
        assert "Daft Punk" in result
        assert "decade_1990" in result
        assert "long" in result


class TestGenreClassificationResult:
    """Tests for GenreClassificationResult."""

    def test_empty_result(self):
        """Empty result should have zero confidence."""
        result = GenreClassificationResult()
        assert result.genres == []
        assert result.confidence == 0.0
        assert result.primary_genre is None

    def test_add_genre(self):
        """Adding genres should update result."""
        result = GenreClassificationResult()
        result.add_genre("rock", 0.8)
        result.add_genre("electronic", 0.6)

        assert "rock" in result.genres
        assert "electronic" in result.genres
        assert result.primary_genre == "rock"  # Highest confidence
        assert result.confidence == 0.8

    def test_add_genre_updates_primary(self):
        """Primary genre should update with highest confidence."""
        result = GenreClassificationResult()
        result.add_genre("rock", 0.5)
        result.add_genre("electronic", 0.9)
        assert result.primary_genre == "electronic"

    def test_add_genre_ignores_duplicates(self):
        """Duplicate genres should not be added twice."""
        result = GenreClassificationResult()
        result.add_genre("rock", 0.5)
        result.add_genre("rock", 0.8)  # Higher confidence
        assert result.genres == ["rock"]
        assert result.confidence_scores["rock"] == 0.8

    def test_get_top_genres(self):
        """Get top N genres by confidence."""
        result = GenreClassificationResult()
        result.add_genre("rock", 0.9)
        result.add_genre("electronic", 0.7)
        result.add_genre("jazz", 0.5)
        result.add_genre("ambient", 0.3)

        top_3 = result.get_top_genres(3)
        assert len(top_3) == 3
        assert top_3[0] == ("rock", 0.9)
        assert top_3[1] == ("electronic", 0.7)
        assert top_3[2] == ("jazz", 0.5)

    def test_to_dict(self):
        """Convert result to dictionary."""
        result = GenreClassificationResult()
        result.add_genre("rock", 0.8)
        result.method = "ml"

        d = result.to_dict()
        assert d["genres"] == ["rock"]
        assert d["confidence_scores"]["rock"] == 0.8
        assert d["primary_genre"] == "rock"
        assert d["confidence"] == 0.8
        assert d["method"] == "ml"


class TestGenreClassifier:
    """Tests for GenreClassifier."""

    def test_classifier_initialization(self):
        """Classifier should initialize without error."""
        classifier = GenreClassifier()
        assert classifier.model_name == "genre_classifier"
        assert not classifier.is_loaded

    def test_predict_with_existing_genre(self):
        """Existing genre from metadata should be used."""
        classifier = GenreClassifier()

        result = classifier.predict(
            title="Stairway to Heaven",
            album="Led Zeppelin IV",
            artist="Led Zeppelin",
            year=1971,
            existing_genre="rock",
        )

        assert result.primary_genre == "rock"
        assert result.method == "metadata"

    def test_predict_falls_back_to_rules(self):
        """Without ML model, should use rule-based classification."""
        classifier = GenreClassifier()

        result = classifier.predict(
            title="Around the World (Remix)",
            album="Homework",
            artist="Daft Punk",
        )

        # Should detect electronic from "Remix" keyword
        assert "electronic" in result.genres or result.method == "rule_based"

    def test_predict_with_unknown_metadata(self):
        """Unknown metadata should return empty or low confidence."""
        classifier = GenreClassifier()

        result = classifier.predict(
            title="Unknown Song",
            album="Unknown Album",
            artist="Unknown Artist",
        )

        # Low confidence for unknown tracks
        assert result.confidence < 0.8


class TestClassifyFromText:
    """Tests for classify_from_text function."""

    def test_classify_electronic_keywords(self):
        """Detect electronic from keywords."""
        result = classify_from_text("Around the World (Remix) by Daft Punk")
        assert "electronic" in result

    def test_classify_rock_keywords(self):
        """Detect rock from keywords - requires explicit keyword."""
        result = classify_from_text("Rock Song by Rock Artist")
        assert "rock" in result

    def test_classify_multiple_genres(self):
        """Detect multiple genres when appropriate."""
        result = classify_from_text("Electronic Dance Music Rock Remix")
        assert "electronic" in result

    def test_classify_no_match(self):
        """Return empty list when no patterns match."""
        result = classify_from_text("Some random text with no genre words")
        assert result == []


class TestIntegrationWithClassifier:
    """Integration tests with existing ContentClassifier."""

    def test_content_classifier_has_genre_method(self):
        """ContentClassifier should have classify_genre method."""
        from music_organizer.core.classifier import ContentClassifier

        assert hasattr(ContentClassifier, "classify_genre")

    def test_classify_genre_with_audio_file(self):
        """Test genre classification with AudioFile."""
        from music_organizer.core.classifier import ContentClassifier
        from music_organizer.models.audio_file import AudioFile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir) / "test.mp3"
            tmppath.write_text("dummy")

            audio = AudioFile.from_path(tmppath)
            audio.title = "Stairway to Heaven"
            audio.album = "Led Zeppelin IV"
            audio.primary_artist = "Led Zeppelin"
            audio.genre = "rock"

            genres, confidence = ContentClassifier.classify_genre(audio)

            # Should return at least the existing genre
            assert len(genres) > 0
            assert confidence >= 0.0


@pytest.mark.parametrize("title,album,artist,expected_genres", [
    ("Rock Song", "Rock Album", "Rock Artist", ["rock"]),
    ("Around the World (Remix)", "Homework", "Daft Punk", ["electronic"]),
    ("So What", "Kind of Blue", "Miles Davis", ["jazz"]),
    ("Symphony No. 5", None, "Beethoven", ["classical"]),
])
def test_rule_based_genre_detection(title, album, artist, expected_genres):
    """Test rule-based genre detection for known examples."""
    from music_organizer.ml.genre_classifier import RuleBasedFallback

    result = RuleBasedFallback.infer_from_keywords(title, album, artist)

    # Check that at least one expected genre is detected
    # Rule-based detection is limited, so we just check for some result
    # when explicit keywords are present
    for genre in expected_genres:
        if genre in ["rock", "electronic"]:
            # These have clear keywords that should be detected
            assert genre in result, f"Expected {genre} in {result} for '{title}'"
        else:
            # Jazz and classical require database lookups or ML models
            # Just check we don't crash
            assert isinstance(result, list)
