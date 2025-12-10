"""Tests for content classification."""

import pytest
from music_organizer.core.classifier import ContentClassifier
from music_organizer.models.audio_file import AudioFile, ContentType


class TestContentClassifier:
    """Test content classification functionality."""

    def test_classify_live_recording(self):
        """Test classification of live recordings."""
        audio_file = AudioFile(
            path=Path("/test/live album.flac"),
            file_type="FLAC",
            album="Live at Madison Square Garden",
            artists=["Test Artist"],
            title="Test Song"
        )

        content_type, confidence = ContentClassifier.classify(audio_file)

        assert content_type == ContentType.LIVE
        assert confidence > 0.5

    def test_classify_collaboration(self):
        """Test classification of collaborations."""
        audio_file = AudioFile(
            path=Path("/test/collab.flac"),
            file_type="FLAC",
            artists=["Artist1", "Artist2"],
            album="Joint Album",
            title="Collaboration Track"
        )

        content_type, confidence = ContentClassifier.classify(audio_file)

        assert content_type == ContentType.COLLABORATION
        assert confidence > 0.6

    def test_classify_compilation(self):
        """Test classification of compilations."""
        audio_file = AudioFile(
            path=Path("/test/greatest hits.flac"),
            file_type="FLAC",
            artists=["Various Artists"],
            album="Greatest Hits",
            title="Hit Song 1"
        )

        content_type, confidence = ContentClassifier.classify(audio_file)

        assert content_type == ContentType.COMPILATION
        assert confidence > 0.6

    def test_classify_rarity(self):
        """Test classification of rarities."""
        audio_file = AudioFile(
            path=Path("/test/demo.flac"),
            file_type="FLAC",
            artists=["Artist"],
            album="Demo Tracks (Unreleased)",
            title="Demo Song"
        )

        content_type, confidence = ContentClassifier.classify(audio_file)

        assert content_type == ContentType.RARITY
        assert confidence > 0.6

    def test_classify_studio_album(self):
        """Test classification of standard studio albums."""
        audio_file = AudioFile(
            path=Path("/test/album.flac"),
            file_type="FLAC",
            artists=["Artist"],
            album="Studio Album",
            title="Normal Song"
        )

        content_type, confidence = ContentClassifier.classify(audio_file)

        assert content_type == ContentType.STUDIO
        assert confidence == 0.5  # Default confidence

    def test_extract_date_from_string(self):
        """Test date extraction from various formats."""
        test_cases = [
            ("Recorded 2023-12-25 at venue", "2023-12-25"),
            ("12/25/2023 show", "12/25/2023"),
            ("25.12.2023 performance", "25.12.2023"),
            ("Recorded in 2023", "2023"),
            ("No date here", None),
        ]

        for text, expected in test_cases:
            result = ContentClassifier.extract_date_from_string(text)
            assert result == expected

    def test_extract_location_from_string(self):
        """Test location extraction from strings."""
        test_cases = [
            ("2023-12-25 - Madison Square Garden, NY", "Madison Square Garden, NY"),
            ("Live at The Venue, London", "The Venue, London"),
            ("No location here", None),
        ]

        for text, expected in test_cases:
            result = ContentClassifier.extract_location_from_string(text)
            assert result == expected

    def test_is_ambiguous_true(self):
        """Test detection of ambiguous classifications."""
        # File with multiple indicators
        audio_file = AudioFile(
            path=Path("/test/greatest hits live.flac"),
            file_type="FLAC",
            artists=["Artist1", "Artist2"],
            album="Greatest Hits - Live",
            title="Song"
        )

        assert ContentClassifier.is_ambiguous(audio_file) is True

    def test_is_ambiguous_false(self):
        """Test detection of clear classifications."""
        audio_file = AudioFile(
            path=Path("/test/regular album.flac"),
            file_type="FLAC",
            artists=["Single Artist"],
            album="Regular Album",
            title="Normal Song"
        )

        assert ContentClassifier.is_ambiguous(audio_file) is False

    def test_has_indicators(self):
        """Test various indicator detection methods."""
        live_file = AudioFile(
            path=Path("/test/LIVE Concert.flac"),
            file_type="FLAC",
            album="Live Recording",
            title="Song"
        )

        assert ContentClassifier._has_live_indicators(live_file) is True
        assert ContentClassifier._has_compilation_indicators(live_file) is False
        assert ContentClassifier._has_rarity_indicators(live_file) is False

        comp_file = AudioFile(
            path=Path("/test/Greatest Hits.flac"),
            file_type="FLAC",
            album="Greatest Hits Collection",
            title="Song"
        )

        assert ContentClassifier._has_compilation_indicators(comp_file) is True

        rarity_file = AudioFile(
            path=Path("/test/Demo Tracks.flac"),
            file_type="FLAC",
            album="Unreleased Demos",
            title="Demo Song"
        )

        assert ContentClassifier._has_rarity_indicators(rarity_file) is True