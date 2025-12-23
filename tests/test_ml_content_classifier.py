"""Tests for ML content type classifier."""

import pytest
from pathlib import Path

from music_organizer.ml.content_classifier import (
    ContentTypeClassifier,
    ContentTypeClassificationResult,
    ContentType,
    classify_from_text,
)
from music_organizer.models.audio_file import AudioFile


class TestContentTypeClassifier:
    """Test ContentTypeClassifier functionality."""

    def test_classifier_init(self):
        """Test classifier initialization."""
        classifier = ContentTypeClassifier()
        assert classifier is not None
        assert classifier.model_name == "content_type_classifier"

    def test_detect_remix_from_title(self):
        """Test REMIX detection from title patterns."""
        classifier = ContentTypeClassifier()

        remix_titles = [
            ("Song Name (DJ Remix)", "Album", "Artist", "Electronic", None),
            ("Track - Club Mix", "Album", "DJ Name", "Electronic", None),
            ("Song (Dub Mix)", "Album", "Artist", "Electronic", None),
            ("Music (Rework)", "Album", "Artist", "Electronic", None),
        ]

        for title, album, artist, genre, duration in remix_titles:
            result = classifier.predict(
                title=title, album=album, artist=artist, genre=genre, duration=duration
            )
            assert ContentType.REMIX in result.content_types or any(
                ct == ContentType.REMIX for ct in result.content_types
            ), f"Failed to detect remix in: {title}"

    def test_detect_podcast(self):
        """Test PODCAST detection."""
        classifier = ContentTypeClassifier()

        podcast_cases = [
            ("Episode 42: The Truth", "My Podcast", "Host Name", "Podcast", 1800),
            ("Interview with Guest", "Podcast Ep. 5", "Host", "Podcast", 1200),
            ("Season 1 Episode 3", "Show Name", "Creator", "Podcast", 900),
        ]

        for title, album, artist, genre, duration in podcast_cases:
            result = classifier.predict(
                title=title, album=album, artist=artist, genre=genre, duration=duration
            )
            assert ContentType.PODCAST in result.content_types or any(
                ct == ContentType.PODCAST for ct in result.content_types
            ), f"Failed to detect podcast in: {title}"

    def test_detect_spoken_word(self):
        """Test SPOKEN_WORD detection."""
        classifier = ContentTypeClassifier()

        spoken_cases = [
            ("Chapter 1", "Audiobook Name", "Narrator Name", "Audiobook", None),
            ("Speech Title", "Album", "Speaker Name", "Speech", None),
            ("Comedy Special", "Live Comedy", "Comedian", "Comedy", None),
            ("Book 2", "The Series", "Author Name", "Audiobook", None),
        ]

        for title, album, artist, genre, duration in spoken_cases:
            result = classifier.predict(
                title=title, album=album, artist=artist, genre=genre, duration=duration
            )
            assert ContentType.SPOKEN_WORD in result.content_types or ContentType.AUDIOBOOK in result.content_types or any(
                ct in (ContentType.SPOKEN_WORD, ContentType.AUDIOBOOK) for ct in result.content_types
            ), f"Failed to detect spoken word in: {title}"

    def test_detect_soundtrack(self):
        """Test SOUNDTRACK detection."""
        classifier = ContentTypeClassifier()

        soundtrack_cases = [
            ("Main Theme", "Movie Soundtrack", "Composer", "Soundtrack", None),
            ("Track Name", "Game OST", "Composer", "Score", None),
            ("End Credits", "Film Score", "John Williams", "Soundtrack", None),
        ]

        for title, album, artist, genre, duration in soundtrack_cases:
            result = classifier.predict(
                title=title, album=album, artist=artist, genre=genre, duration=duration
            )
            assert ContentType.SOUNDTRACK in result.content_types or any(
                ct == ContentType.SOUNDTRACK for ct in result.content_types
            ), f"Failed to detect soundtrack in: {title}"

    def test_unknown_content_type(self):
        """Test UNKNOWN content type for regular music."""
        classifier = ContentTypeClassifier()

        result = classifier.predict(
            title="Regular Song",
            album="Studio Album",
            artist="Band Name",
            genre="Rock",
            duration=240
        )

        # Should not detect special types for regular music
        assert ContentType.REMIX not in result.content_types
        assert ContentType.PODCAST not in result.content_types
        assert ContentType.SOUNDTRACK not in result.content_types

    def test_confidence_scores(self):
        """Test confidence score calculation."""
        classifier = ContentTypeClassifier()

        # High confidence remix
        result = classifier.predict(
            title="Song (DJ Remix) - Club Mix",
            album="Remix Album",
            artist="DJ Producer",
            genre="Electronic",
            duration=300
        )

        if ContentType.REMIX in result.content_types:
            assert result.confidence_scores.get(ContentType.REMIX, 0) > 0.5

    def test_result_to_dict(self):
        """Test result serialization to dict."""
        classifier = ContentTypeClassifier()
        result = classifier.predict(
            title="Song (Remix)",
            album="Album",
            artist="Artist",
            genre="Electronic"
        )

        result_dict = result.to_dict()
        assert "content_types" in result_dict
        assert "confidence_scores" in result_dict
        assert "primary_type" in result_dict
        assert "confidence" in result_dict
        assert "method" in result_dict


class TestClassifyFromText:
    """Test the quick classify_from_text function."""

    def test_classify_remix(self):
        """Test quick remix classification."""
        result = classify_from_text(
            title="Song (DJ Remix)",
            album="Album",
            artist="Artist",
            genre="Electronic"
        )
        assert ContentType.REMIX in result

    def test_classify_podcast(self):
        """Test quick podcast classification."""
        result = classify_from_text(
            title="Episode 42",
            album="My Podcast",
            artist="Host",
            genre="Podcast",
            duration=1200
        )
        assert ContentType.PODCAST in result

    def test_classify_soundtrack(self):
        """Test quick soundtrack classification."""
        result = classify_from_text(
            title="Main Theme",
            album="Movie OST",
            artist="Composer",
            genre="Soundtrack"
        )
        assert ContentType.SOUNDTRACK in result

    def test_classify_no_match(self):
        """Test quick classification with no special type."""
        result = classify_from_text(
            title="Normal Song",
            album="Normal Album",
            artist="Normal Artist",
            genre="Rock"
        )
        # Should return empty or UNKNOWN only
        assert ContentType.REMIX not in result
        assert ContentType.PODCAST not in result


class TestContentTypeResult:
    """Test ContentTypeClassificationResult class."""

    def test_add_type(self):
        """Test adding content types to result."""
        result = ContentTypeClassificationResult()

        result.add_type(ContentType.REMIX, 0.8)
        assert ContentType.REMIX in result.content_types
        assert result.confidence_scores[ContentType.REMIX] == 0.8
        assert result.primary_type == ContentType.REMIX

    def test_add_multiple_types(self):
        """Test adding multiple content types."""
        result = ContentTypeClassificationResult()

        result.add_type(ContentType.REMIX, 0.7)
        result.add_type(ContentType.PODCAST, 0.9)

        assert ContentType.REMIX in result.content_types
        assert ContentType.PODCAST in result.content_types
        assert result.primary_type == ContentType.PODCAST  # Higher confidence

    def test_confidence_property(self):
        """Test confidence property returns max score."""
        result = ContentTypeClassificationResult()

        result.add_type(ContentType.REMIX, 0.6)
        result.add_type(ContentType.SOUNDTRACK, 0.9)

        assert result.confidence == 0.9

    def test_get_top_types(self):
        """Test getting top content types."""
        result = ContentTypeClassificationResult()

        result.add_type(ContentType.REMIX, 0.7)
        result.add_type(ContentType.PODCAST, 0.9)
        result.add_type(ContentType.SOUNDTRACK, 0.5)

        top_types = result.get_top_types(2)
        assert len(top_types) == 2
        assert top_types[0][0] == ContentType.PODCAST
        assert top_types[0][1] == 0.9
