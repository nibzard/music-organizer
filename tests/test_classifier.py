"""Tests for content classification."""

import pytest
from pathlib import Path
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


class TestExtendedContentTypeClassification:
    """Test extended content type classification (REMIX, PODCAST, SPOKEN_WORD, SOUNDTRACK)."""

    def test_classify_remix(self):
        """Test REMIX classification."""
        audio_file = AudioFile(
            path=Path("/test/remix.flac"),
            file_type="FLAC",
            artists=["DJ Producer"],
            primary_artist="DJ Producer",
            album="Remix Album",
            title="Song Name (Club Mix)",
            genre="Electronic"
        )

        content_type, confidence = ContentClassifier.classify_extended_content_type(audio_file)

        assert content_type == "remix"
        assert confidence > 0.5

    def test_classify_podcast(self):
        """Test PODCAST classification."""
        audio_file = AudioFile(
            path=Path("/test/podcast.mp3"),
            file_type="MP3",
            artists=["Host Name"],
            primary_artist="Host Name",
            album="My Podcast",
            title="Episode 42: The Topic",
            genre="Podcast"
        )

        content_type, confidence = ContentClassifier.classify_extended_content_type(audio_file)

        assert content_type == "podcast"
        assert confidence > 0.5

    def test_classify_spoken_word(self):
        """Test SPOKEN_WORD classification."""
        audio_file = AudioFile(
            path=Path("/test/audiobook.m4b"),
            file_type="M4B",
            artists=["Narrator Name"],
            primary_artist="Narrator Name",
            album="Audiobook Title",
            title="Chapter 1",
            genre="Audiobook"
        )

        content_type, confidence = ContentClassifier.classify_extended_content_type(audio_file)

        assert content_type == "spoken_word"
        assert confidence > 0.5

    def test_classify_soundtrack(self):
        """Test SOUNDTRACK classification."""
        audio_file = AudioFile(
            path=Path("/test/soundtrack.flac"),
            file_type="FLAC",
            artists=["Composer Name"],
            primary_artist="Composer Name",
            album="Movie Soundtrack (OST)",
            title="Main Theme",
            genre="Soundtrack"
        )

        content_type, confidence = ContentClassifier.classify_extended_content_type(audio_file)

        assert content_type == "soundtrack"
        assert confidence > 0.5

    def test_classify_regular_song_no_extended_type(self):
        """Test that regular songs return None for extended content types."""
        audio_file = AudioFile(
            path=Path("/test/song.flac"),
            file_type="FLAC",
            artists=["Band Name"],
            primary_artist="Band Name",
            album="Studio Album",
            title="Regular Song",
            genre="Rock"
        )

        content_type, confidence = ContentClassifier.classify_extended_content_type(audio_file)

        assert content_type is None
        assert confidence == 0.0

    def test_remix_with_multiple_indicators(self):
        """Test REMIX detection with multiple indicators increases confidence."""
        audio_file = AudioFile(
            path=Path("/test/remix.flac"),
            file_type="FLAC",
            artists=["DJ Producer"],
            primary_artist="DJ Producer",
            album="The Remixes",
            title="Song (Dub Mix) - 2023 Version",
            genre="Electronic"
        )

        content_type, confidence = ContentClassifier.classify_extended_content_type(audio_file)

        assert content_type == "remix"
        # Multiple indicators should give higher confidence
        assert confidence > 0.7

    def test_soundtrack_composer_in_artist(self):
        """Test SOUNDTRACK detection when composer is in artist field."""
        audio_file = AudioFile(
            path=Path("/test/soundtrack.flac"),
            file_type="FLAC",
            artists=["John Powell (composer)"],
            primary_artist="John Powell (composer)",
            album="Motion Picture Score",
            title="Action Theme"
        )

        content_type, confidence = ContentClassifier.classify_extended_content_type(audio_file)

        assert content_type == "soundtrack"
        assert confidence >= 0.5

    def test_audiobook_narrator_in_artist(self):
        """Test SPOKEN_WORD detection when narrator is in artist field."""
        audio_file = AudioFile(
            path=Path("/test/audiobook.m4b"),
            file_type="M4B",
            artists=["Author Name, read by Narrator Name"],
            primary_artist="Author Name, read by Narrator Name",
            album="Book Title",
            title="Chapter 5"
        )

        content_type, confidence = ContentClassifier.classify_extended_content_type(audio_file)

        assert content_type == "spoken_word"
        assert confidence >= 0.5