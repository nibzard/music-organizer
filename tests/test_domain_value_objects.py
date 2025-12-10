"""
Tests for domain value objects.
"""

import pytest
from pathlib import Path
from music_organizer.domain.value_objects import (
    AudioPath,
    ArtistName,
    ContentPattern,
    FileFormat,
    Metadata,
    TrackNumber,
)


class TestAudioPath:
    """Test cases for AudioPath value object."""

    def test_create_with_string(self):
        """Test creating AudioPath with string."""
        path = AudioPath("/music/artist/album/track.flac")
        assert path.path == Path("/music/artist/album/track.flac").resolve()
        assert path.format == FileFormat.FLAC
        assert str(path) == str(Path("/music/artist/album/track.flac").resolve())

    def test_create_with_path(self):
        """Test creating AudioPath with Path object."""
        p = Path("/music/artist/album/track.mp3")
        path = AudioPath(p)
        assert path.path == p.resolve()
        assert path.format == FileFormat.MP3

    def test_unknown_format(self):
        """Test creating AudioPath with unknown format."""
        path = AudioPath("/music/track.xyz")
        assert path.format is None
        assert not path.is_known_format
        assert path.extension == ".xyz"

    def test_properties(self):
        """Test AudioPath properties."""
        # Create a test file
        test_file = Path("/tmp/test_audio.mp3")
        test_file.write_text("dummy content")

        path = AudioPath(test_file)
        assert path.filename == "test_audio.mp3"
        assert path.stem == "test_audio"
        assert path.extension == ".mp3"
        assert path.size_bytes > 0
        assert path.size_mb > 0
        assert path.exists()

        # Cleanup
        test_file.unlink()

    def test_path_operations(self):
        """Test path manipulation operations."""
        path = AudioPath("/music/artist/album/track.flac")

        # Test with_name
        new_path = path.with_name("new_track.flac")
        assert new_path.filename == "new_track.flac"
        assert isinstance(new_path, AudioPath)

        # Test with_suffix
        new_path = path.with_suffix(".mp3")
        assert new_path.extension == ".mp3"
        assert new_path.format == FileFormat.MP3

    def test_validation(self):
        """Test validation errors."""
        # No extension
        with pytest.raises(ValueError, match="must have a file extension"):
            AudioPath("/music/track")

        # Invalid type
        with pytest.raises(TypeError):
            AudioPath(123)

    def test_equality(self):
        """Test equality based on path normalization."""
        path1 = AudioPath("/music/../music/track.flac")
        path2 = AudioPath("/music/track.flac")
        path3 = AudioPath("/music/other.flac")

        assert path1 == path2
        assert path1 != path3

    def test_relative_to(self):
        """Test relative path calculation."""
        # Use absolute paths for relative_to
        path = AudioPath("/home/niko/music/artist/album/track.flac")
        rel = path.relative_to("/home/niko/music")
        # Check that the relative path contains the expected parts
        assert "artist/album/track.flac" in str(rel)


class TestTrackNumber:
    """Test cases for TrackNumber value object."""

    def test_from_integer(self):
        """Test creating from integer."""
        track = TrackNumber(5)
        assert track.number == 5
        assert track.total is None
        assert str(track) == "5"
        assert int(track) == 5

    def test_from_string_simple(self):
        """Test creating from simple string."""
        track = TrackNumber("5")
        assert track.number == 5
        assert str(track) == "5"

    def test_from_string_with_slash(self):
        """Test creating from string with total."""
        track = TrackNumber("5/12")
        assert track.number == 5
        assert track.total == 12
        assert track.has_total
        assert str(track) == "5/12"

    def test_from_string_with_padding(self):
        """Test creating from padded string."""
        track = TrackNumber("05")
        assert track.number == 5
        assert track.formatted() == "05"
        assert track.formatted(3) == "005"

    def test_from_string_with_words(self):
        """Test creating from string with 'of'."""
        track = TrackNumber("5 of 12")
        assert track.number == 5
        assert track.total == 12

        # Test with "of" mixed case
        track = TrackNumber("7 OF 10")
        assert track.number == 7
        assert track.total == 10

    def test_from_tuple(self):
        """Test creating from tuple."""
        track = TrackNumber((7, 15))
        assert track.number == 7
        assert track.total == 15

    def test_formatted(self):
        """Test formatted representations."""
        track = TrackNumber(5)
        assert track.formatted() == "05"
        assert track.formatted(3) == "005"

        # Test with zero width (should default to 2)
        assert track.formatted(0) == "05"

        track_with_total = TrackNumber("5/12")
        assert track_with_total.formatted_with_total() == "05/12"
        assert track_with_total.formatted_with_total(3) == "005/012"

    def test_negative_numbers(self):
        """Test handling of negative numbers."""
        track = TrackNumber(-5)
        assert track.number == 0

        track = TrackNumber((-5, 12))
        assert track.number == 0
        assert track.total == 12

    def test_invalid_input(self):
        """Test invalid input handling."""
        with pytest.raises(ValueError):
            TrackNumber("abc")

        with pytest.raises(TypeError):
            TrackNumber([1, 2, 3])


class TestArtistName:
    """Test cases for ArtistName value object."""

    def test_creation(self):
        """Test basic creation."""
        artist = ArtistName("The Beatles")
        assert artist.name == "The Beatles"
        assert str(artist) == "The Beatles"

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        artist = ArtistName("  The   Beatles  ")
        assert artist.name == "The Beatles"

    def test_empty_name(self):
        """Test empty name validation."""
        with pytest.raises(ValueError):
            ArtistName("")

        with pytest.raises(ValueError):
            ArtistName("   ")

    def test_sortable(self):
        """Test sortable representation."""
        # Artist with "The"
        artist = ArtistName("The Beatles")
        assert artist.sortable == "Beatles, The"

        # Artist without article
        artist = ArtistName("Beatles")
        assert artist.sortable == "Beatles"

        # "The" as part of name (should not be treated as article)
        artist = ArtistName("The The")
        assert artist.sortable == "The The"

        # Artist with "a" article
        artist = ArtistName("A Perfect Circle")
        assert artist.sortable == "Perfect Circle, A"

    def test_first_letter(self):
        """Test first letter extraction."""
        artist = ArtistName("The Beatles")
        assert artist.first_letter == "B"

        artist = ArtistName("#1 Artist")
        assert artist.first_letter == "#"

    def test_string_operations(self):
        """Test string comparison operations."""
        artist = ArtistName("The Beatles")

        assert artist.starts_with("the")
        assert artist.starts_with("THE")
        assert artist.contains("beat")
        assert artist.contains("BEATLES")
        assert not artist.contains("stones")

    def test_equality(self):
        """Test equality (case-insensitive)."""
        artist1 = ArtistName("The Beatles")
        artist2 = ArtistName("the beatles")
        artist3 = ArtistName("The Rolling Stones")

        assert artist1 == artist2
        assert artist1 != artist3

        # Hash should be case-insensitive
        assert hash(artist1) == hash(artist2)

    def test_invalid_type(self):
        """Test invalid input type."""
        with pytest.raises(TypeError):
            ArtistName(123)


class TestMetadata:
    """Test cases for Metadata value object."""

    def test_empty_metadata(self):
        """Test creating empty metadata."""
        meta = Metadata()
        assert meta.title is None
        assert meta.artists == frozenset()
        assert meta.primary_artist is None

    def test_with_artists(self):
        """Test metadata with artists."""
        artists = {ArtistName("Artist 1"), ArtistName("Artist 2")}
        meta = Metadata(artists=artists)

        assert len(meta.artists) == 2
        assert ArtistName("Artist 1") in meta.artists
        assert meta.formatted_artists() in {
            "Artist 1, Artist 2",
            "Artist 2, Artist 1"
        }

    def test_primary_artist(self):
        """Test primary artist selection."""
        # Without albumartist
        artists = {ArtistName("Artist 1"), ArtistName("Artist 2")}
        meta = Metadata(artists=artists)
        assert meta.primary_artist in artists

        # With albumartist
        albumartist = ArtistName("Various Artists")
        meta = Metadata(artists=artists, albumartist=albumartist)
        assert meta.primary_artist == albumartist

    def test_track_number(self):
        """Test track number handling."""
        track = TrackNumber("5/12")
        meta = Metadata(track_number=track)
        assert meta.track_number.number == 5
        assert meta.track_number.total == 12

    def test_year_validation(self):
        """Test year validation."""
        # Valid years
        Metadata(year=2023)
        Metadata(year=0)
        Metadata(year=9999)

        # Invalid years
        with pytest.raises(ValueError):
            Metadata(year=-1)

        with pytest.raises(ValueError):
            Metadata(year=10000)

    def test_is_live(self):
        """Test live recording detection."""
        # With location
        meta = Metadata(title="Song", location="Madison Square Garden")
        assert meta.is_live

        # With date
        meta = Metadata(title="Song", date="2023-07-15")
        assert meta.is_live

        # With "Live" in title
        meta = Metadata(title="Song (Live Version)")
        assert meta.is_live

        # Not live
        meta = Metadata(title="Song")
        assert not meta.is_live

    def test_is_compilation(self):
        """Test compilation detection."""
        # Various Artists albumartist
        meta = Metadata(
            artists=[ArtistName("Artist 1")],
            albumartist=ArtistName("Various Artists")
        )
        assert meta.is_compilation

        # Multiple artists
        meta = Metadata(
            artists=[ArtistName("Artist 1"), ArtistName("Artist 2")]
        )
        assert meta.is_compilation

        # Compilation in album name
        meta = Metadata(album="Greatest Hits Compilation")
        assert meta.is_compilation

    def test_formatted_title(self):
        """Test title formatting."""
        # Basic title
        meta = Metadata(title="Song Title")
        assert meta.formatted_title() == "Song Title"

        # With track number
        track = TrackNumber(5)
        meta = Metadata(title="Song", track_number=track)
        assert meta.formatted_title() == "05 Song"

        # With live info
        meta = Metadata(
            title="Song",
            date="2023-07-15",
            location="Venue"
        )
        assert "Live" in meta.formatted_title()
        assert "2023-07-15" in meta.formatted_title()
        assert "Venue" in meta.formatted_title()

    def test_formatted_duration(self):
        """Test duration formatting."""
        # Minutes and seconds
        meta = Metadata(duration_seconds=185.5)
        assert meta.formatted_duration() == "3:05"

        # Only seconds
        meta = Metadata(duration_seconds=45)
        assert meta.formatted_duration() == "0:45"

        # No duration
        meta = Metadata()
        assert meta.formatted_duration() == ""

    def test_get_hash(self):
        """Test metadata hashing for duplicate detection."""
        meta1 = Metadata(
            title="Song",
            artists=[ArtistName("Artist")],
            album="Album",
            track_number=TrackNumber(5),
            year=2023
        )

        meta2 = Metadata(
            title="song",  # Different case
            artists=[ArtistName("artist")],  # Different case
            album="album",  # Different case
            track_number=TrackNumber(5),
            year=2023
        )

        meta3 = Metadata(
            title="Different Song",
            artists=[ArtistName("Artist")],
            album="Album",
            track_number=TrackNumber(5),
            year=2023
        )

        assert meta1.get_hash() == meta2.get_hash()  # Should match
        assert meta1.get_hash() != meta3.get_hash()  # Should differ

    def test_with_field(self):
        """Test updating fields with with_field method."""
        meta = Metadata(title="Song")
        new_meta = meta.with_field(year=2023)

        assert meta.title == "Song"
        assert meta.year is None

        assert new_meta.title == "Song"
        assert new_meta.year == 2023

        # Test with artists as strings
        new_meta = meta.with_field(artists=["Artist 1", "Artist 2"])
        assert len(new_meta.artists) == 2
        assert ArtistName("Artist 1") in new_meta.artists

        # Test with single artist string
        new_meta = meta.with_field(artists="Single Artist")
        assert len(new_meta.artists) == 1
        assert ArtistName("Single Artist") in new_meta.artists

        # Test with albumartist string
        new_meta = meta.with_field(albumartist="Album Artist")
        assert isinstance(new_meta.albumartist, ArtistName)
        assert new_meta.albumartist.name == "Album Artist"

    def test_bitrate_validation(self):
        """Test bitrate validation."""
        # Valid bitrate
        Metadata(bitrate=320)

        # Invalid bitrate (negative)
        with pytest.raises(ValueError):
            Metadata(bitrate=-100)

    def test_channel_validation(self):
        """Test channel count validation."""
        # Valid channel counts
        for channels in [1, 2, 4, 6, 8]:
            Metadata(channels=channels)

        # Invalid channel count
        with pytest.raises(ValueError):
            Metadata(channels=3)


class TestContentPattern:
    """Test cases for ContentPattern value object."""

    def test_pattern_matching(self):
        """Test pattern matching functionality."""
        pattern = ContentPattern(
            name="Live Patterns",
            patterns=frozenset(["live", "concert", "tour"])
        )

        assert pattern.matches("Live at Budokan")
        assert pattern.matches("In Concert")
        assert pattern.matches("World Tour 2023")
        assert not pattern.matches("Studio Album")

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        pattern = ContentPattern(
            name="Test",
            patterns=frozenset(["pattern"])
        )

        assert pattern.matches("PATTERN")
        assert pattern.matches("Pattern")
        assert pattern.matches("pattern")

    def test_empty_patterns(self):
        """Test pattern with no patterns."""
        pattern = ContentPattern(name="Empty", patterns=frozenset())
        assert not pattern.matches("Anything")

    def test_priority(self):
        """Test priority attribute."""
        pattern1 = ContentPattern(name="Low", priority=1)
        pattern2 = ContentPattern(name="High", priority=10)

        assert pattern1.priority == 1
        assert pattern2.priority == 10

    def test_string_representation(self):
        """Test string representation."""
        pattern = ContentPattern(
            name="Test Pattern",
            patterns=frozenset(["a", "b", "c"])
        )
        str_repr = str(pattern)
        assert "Test Pattern" in str_repr
        assert "3 rules" in str_repr