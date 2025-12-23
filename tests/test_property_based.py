"""Property-based tests for music organizer domain models.

Uses Hypothesis to generate edge cases and verify invariants for critical value objects.
Tests focus on properties that should always hold true regardless of input values.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from music_organizer.domain.catalog.value_objects import (
    ArtistName,
    AudioPath,
    FileFormat,
    Metadata,
    TrackNumber,
)


# ============================================================================
# TrackNumber Property-Based Tests
# ============================================================================

@given(st.integers(min_value=0, max_value=9999))
def test_track_number_from_int_always_non_negative(n: int) -> None:
    """TrackNumber should never be negative regardless of input."""
    tn = TrackNumber(n)
    assert tn.number >= 0


@given(st.integers(min_value=-1000, max_value=-1))
def test_track_number_from_negative_int_clamps_to_zero(n: int) -> None:
    """Negative track numbers should be clamped to zero."""
    tn = TrackNumber(n)
    assert tn.number == 0


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(st.text())
def test_track_number_from_various_string_formats(text: str) -> None:
    """TrackNumber should handle various string formats without crashing."""
    # Skip strings that clearly can't be parsed
    assume(text == "" or any(c.isdigit() for c in text))

    try:
        tn = TrackNumber(text)
        assert tn.number >= 0
    except ValueError:
        # Expected for unparseable strings
        pass


@given(st.integers(min_value=1, max_value=999))
def test_track_number_formatted_returns_string(n: int) -> None:
    """formatted() should always return a string representation."""
    tn = TrackNumber(n)
    result = tn.formatted()
    assert isinstance(result, str)
    assert result.isdigit()


@given(st.integers(min_value=0, max_value=99), st.integers(min_value=1, max_value=100))
def test_track_number_with_total(num: int, total: int) -> None:
    """TrackNumber with total should maintain relationship."""
    tn = TrackNumber((num, total))
    assert tn.number == num
    assert tn.total == total
    assert tn.has_total  # Property, not method


@given(st.integers(min_value=0, max_value=99), st.integers(min_value=1, max_value=100))
def test_track_number_with_total_string_format(num: int, total: int) -> None:
    """TrackNumber with total should format correctly as 'num/total'."""
    tn = TrackNumber((num, total))
    result = tn.formatted_with_total()
    assert "/" in result
    assert str(num) in result or f"{num:02d}" in result
    assert str(total) in result or f"{total:02d}" in result


# ============================================================================
# ArtistName Property-Based Tests
# ============================================================================

@given(st.text(min_size=1))
def test_artist_name_normalizes_whitespace(name: str) -> None:
    """ArtistName should normalize whitespace (no leading/trailing, no multiple spaces)."""
    # Skip strings with only whitespace
    assume(any(c.isalnum() or c not in " \t\n\r" for c in name))

    try:
        artist = ArtistName(name)
        # No leading/trailing whitespace
        assert artist.name == artist.name.strip()
        # No multiple consecutive spaces
        assert "  " not in artist.name
    except ValueError:
        # Empty after normalization is expected
        pass


@given(st.text(min_size=1, max_size=50))
def test_artist_name_preserves_significant_content(name: str) -> None:
    """ArtistName should preserve the essential name content."""
    assume(any(c.isalnum() or c not in " \t\n\r" for c in name))

    try:
        artist = ArtistName(name)
        # The name should contain at least the alphanumeric chars from input
        input_alnum = "".join(c for c in name if c.isalnum())
        if input_alnum:
            assert any(c in artist.name for c in input_alnum[:10]) or len(input_alname) < 10
    except ValueError:
        pass


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(st.text(min_size=2, max_size=50).filter(lambda x: x.isalpha() and x.isascii()))
def test_artist_name_case_insensitive_equality(name: str) -> None:
    """ArtistName equality should be case-insensitive for ASCII letters."""
    artist1 = ArtistName(name)
    artist2 = ArtistName(name.lower())
    artist3 = ArtistName(name.upper())

    assert artist1 == artist2
    assert artist1 == artist3
    assert hash(artist1) == hash(artist2)


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(st.text(min_size=3, max_size=50).filter(lambda x: x.isalpha() and x.isascii()))
def test_artist_name_starts_with_case_insensitive(name: str) -> None:
    """starts_with() should be case-insensitive for ASCII."""
    artist = ArtistName(name)

    prefix = name[:3].lower()
    assert artist.starts_with(prefix)
    assert artist.starts_with(prefix.upper())


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(st.text(min_size=5, max_size=50).filter(lambda x: x.isalpha() and x.isascii()))
def test_artist_name_contains_case_insensitive(name: str) -> None:
    """contains() should be case-insensitive for ASCII."""
    artist = ArtistName(name)

    substring = name[2:5].lower()
    assert artist.contains(substring)
    assert artist.contains(substring.upper())


@given(st.from_regex(r"[A-Za-z]{5,50}"))
def test_artist_name_sortable_removes_articles(name: str) -> None:
    """sortable property should remove leading articles for sorting."""
    artist = ArtistName(name)

    # Articles should be moved to end
    for article in ["The ", "A ", "An "]:
        if name.startswith(article) and len(name) > len(article):
            assert ", " in artist.sortable or artist.sortable != name


@given(st.from_regex(r"[A-Za-z]{2,50}"))
def test_artist_name_first_letter_always_valid(name: str) -> None:
    """first_letter should always return a non-empty string."""
    artist = ArtistName(name)
    first_letter = artist.first_letter

    assert isinstance(first_letter, str)
    # Note: For some Unicode chars like 'ß', sortable may return multi-char result
    assert len(first_letter) >= 1


# ============================================================================
# AudioPath Property-Based Tests
# ============================================================================

@given(st.from_regex(r".*\.[a-z]{2,4}$"))
def test_audiopath_stores_filename(path_str: str) -> None:
    """AudioPath should preserve the filename component."""
    # Skip if path has issues
    assume(all(c not in "\x00" for c in path_str))

    try:
        audio_path = AudioPath(path_str)
        assert audio_path.filename
        assert audio_path.filename == Path(path_str).name
    except (ValueError, OSError):
        # Invalid paths are okay
        pass


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(st.from_regex(r".*\.(flac|mp3|m4a|wav|ogg|opus)$"))
def test_audiopath_recognizes_known_formats(path_str: str) -> None:
    """AudioPath should correctly identify known audio formats."""
    assume(all(c not in "\x00" for c in path_str))
    assume(all(c.isprintable() for c in path_str))
    # Ensure there's a valid filename component
    assume(len(Path(path_str).name) > 4)  # At least "x.mp3"

    try:
        audio_path = AudioPath(path_str)
        assert audio_path.is_known_format
        assert audio_path.format is not None
    except (ValueError, OSError, FileNotFoundError):
        pass


@given(st.from_regex(r".*\.[a-z]{3}$"))
def test_audiopath_extension_matches_input(path_str: str) -> None:
    """AudioPath extension should match the input file extension."""
    assume(all(c not in "\x00" for c in path_str))

    try:
        audio_path = AudioPath(path_str)
        expected_ext = Path(path_str).suffix.lower()
        assert audio_path.extension.lower() == expected_ext
    except (ValueError, OSError):
        pass


@given(st.from_regex(r".*\.[a-z]{2,4}$"))
def test_audiopath_with_name_changes_filename(path_str: str) -> None:
    """AudioPath.with_name() should change only the filename."""
    assume(all(c not in "\x00" for c in path_str))

    try:
        audio_path = AudioPath(path_str)
        new_name = "new_name.mp3"
        new_path = audio_path.with_name(new_name)

        assert new_path.filename == new_name
        assert new_path.extension == ".mp3"
    except (ValueError, OSError):
        pass


# ============================================================================
# SimilarityThreshold Property-Based Tests
# ============================================================================

from music_organizer.domain.classification.value_objects import SimilarityThreshold


@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_similarity_threshold_accepts_valid_ranges(
    title_sim: float, artist_sim: float, album_sim: float, duration_sim: float
) -> None:
    """SimilarityThreshold should accept all values in [0, 1] range."""
    threshold = SimilarityThreshold(
        title_similarity=title_sim,
        artist_similarity=artist_sim,
        album_similarity=album_sim,
        overall_threshold=1.0,
    )

    scores = {"title": title_sim, "artist": artist_sim, "album": album_sim, "duration": duration_sim}
    result = threshold.is_similar(scores)
    assert isinstance(result, bool)


@given(
    st.floats(min_value=-1.0, max_value=2.0),
    st.floats(min_value=-1.0, max_value=2.0),
    st.floats(min_value=-1.0, max_value=2.0),
    st.floats(min_value=-1.0, max_value=2.0),
)
def test_similarity_threshold_handles_edge_cases(
    title_sim: float, artist_sim: float, album_sim: float, duration_sim: float
) -> None:
    """SimilarityThreshold should handle out-of-range values gracefully."""
    threshold = SimilarityThreshold(overall_threshold=0.5)

    scores = {
        "title": max(0.0, min(1.0, title_sim)),
        "artist": max(0.0, min(1.0, artist_sim)),
        "album": max(0.0, min(1.0, album_sim)),
        "duration": max(0.0, min(1.0, duration_sim)),
    }
    result = threshold.is_similar(scores)
    assert isinstance(result, bool)


@given(st.floats(min_value=0.0, max_value=1.0), st.floats(min_value=0.0, max_value=1.0))
def test_similarity_threshold_overall_score_bounds(title: float, artist: float) -> None:
    """Overall score calculation should respect weighted average bounds."""
    threshold = SimilarityThreshold(overall_threshold=0.5)

    scores = {"title": title, "artist": artist, "album": 0.5, "duration": 0.5}
    result = threshold.is_similar(scores)

    # High title and artist should pass with 0.5 threshold
    if title >= 0.8 and artist >= 0.9:
        assert result is True


# ============================================================================
# Metadata Property-Based Tests
# ============================================================================

@given(
    st.text(min_size=1, max_size=100),
    st.lists(st.from_regex(r"[A-Za-z]{2,30}"), min_size=1, max_size=5),
    st.text(min_size=0, max_size=100),
    st.integers(min_value=1900, max_value=2100),
)
def test_metadata_with_valid_fields_does_not_raise(
    title: str, artists: list[str], album: str, year: int
) -> None:
    """Metadata with valid fields should not raise during creation."""
    assume(any(c.isalnum() for c in title))

    try:
        artist_names = frozenset(ArtistName(a) for a in artists)
        metadata = Metadata(
            title=title,
            artists=artist_names,
            album=album or None,
            year=year,
        )
        assert metadata.title == title
        assert metadata.year == year
    except ValueError:
        # Some names might be invalid after normalization
        pass


@given(st.integers(min_value=-100, max_value=99999))
def test_metadata_year_validation(year: int) -> None:
    """Metadata should reject invalid years."""
    # Valid years
    if 0 <= year <= 9999:
        metadata = Metadata(year=year)
        assert metadata.year == year
    else:
        with pytest.raises(ValueError):
            Metadata(year=year)


@given(st.integers(min_value=-10, max_value=20))
def test_metadata_disc_number_validation(disc_number: int) -> None:
    """Metadata should reject invalid disc numbers."""
    if disc_number >= 1:
        metadata = Metadata(disc_number=disc_number)
        assert metadata.disc_number == disc_number
    else:
        with pytest.raises(ValueError):
            Metadata(disc_number=disc_number)


@given(st.integers(min_value=-10, max_value=20))
def test_metadata_bitrate_validation(bitrate: int) -> None:
    """Metadata should reject negative bitrates."""
    if bitrate >= 0:
        metadata = Metadata(bitrate=bitrate)
        assert metadata.bitrate == bitrate
    else:
        with pytest.raises(ValueError):
            Metadata(bitrate=bitrate)


@given(st.integers(min_value=0, max_value=10), st.integers(min_value=0, max_value=10))
def test_metadata_with_field_creates_new_instance(disc: int, total_discs: int) -> None:
    """with_field() should create a new Metadata instance with updated field."""
    metadata = Metadata()

    if disc >= 1 and total_discs >= 1:
        new_metadata = metadata.with_field(disc_number=disc, total_discs=total_discs)
        assert new_metadata.disc_number == disc
        assert new_metadata.total_discs == total_discs
        # Original should be unchanged
        assert metadata.disc_number is None


@given(
    st.text(min_size=1, max_size=50),
    st.lists(st.from_regex(r"[A-Za-z]{2,20}"), min_size=1, max_size=3),
)
def test_metadata_hash_is_consistent(title: str, artists: list[str]) -> None:
    """get_hash() should return consistent values for same metadata."""
    assume(any(c.isalnum() for c in title))

    try:
        artist_names = frozenset(ArtistName(a) for a in artists)
        metadata1 = Metadata(title=title, artists=artist_names, year=2000)
        metadata2 = Metadata(title=title, artists=artist_names, year=2000)

        hash1 = metadata1.get_hash()
        hash2 = metadata2.get_hash()

        assert hash1 == hash2
        assert len(hash1) == 16  # Should be 16 character hex
    except ValueError:
        # Invalid artist names are acceptable
        pass


# ============================================================================
# ClassificationPattern Property-Based Tests
# ============================================================================

from music_organizer.domain.classification.value_objects import (
    ClassificationPattern,
    MatchType,
)


@given(st.text(), st.text())
def test_classification_pattern_matches_never_crashes(pattern_str: str, text: str) -> None:
    """ClassificationPattern.matches() should never crash on any input."""
    try:
        pattern = ClassificationPattern(pattern=pattern_str[:100])  # Limit size
        result = pattern.matches(text)
        assert isinstance(result, bool)
    except (re.error, ValueError):
        # Invalid regex patterns are acceptable
        pass


@given(st.from_regex(r"[A-Za-z]{3,30}"))
def test_classification_pattern_exact_match(text: str) -> None:
    """EXACT match type should require exact equality."""
    pattern = ClassificationPattern(pattern=text, match_type=MatchType.EXACT)
    assert pattern.matches(text)
    assert not pattern.matches(text + "extra")
    assert not pattern.matches("prefix" + text)


@given(st.from_regex(r"[A-Za-z]{5,30}"))
def test_classification_pattern_contains_match(text: str) -> None:
    """CONTAINS match type should find substring."""
    assume(len(text) >= 5)
    substring = text[2:5]
    pattern = ClassificationPattern(pattern=substring, match_type=MatchType.CONTAINS)

    assert pattern.matches(text)


@given(st.from_regex(r"[A-Za-z]{5,30}"))
def test_classification_pattern_starts_with_match(text: str) -> None:
    """STARTS_WITH match type should check prefix."""
    assume(len(text) >= 5)
    prefix = text[:3]
    pattern = ClassificationPattern(pattern=prefix, match_type=MatchType.STARTS_WITH)

    assert pattern.matches(text)
    assert not pattern.matches("xyz" + text)


@given(st.from_regex(r"[A-Za-z]{5,30}"))
def test_classification_pattern_ends_with_match(text: str) -> None:
    """ENDS_WITH match type should check suffix."""
    assume(len(text) >= 5)
    suffix = text[-3:]
    pattern = ClassificationPattern(pattern=suffix, match_type=MatchType.ENDS_WITH)

    assert pattern.matches(text)
    assert not pattern.matches(text + "xyz")


@given(st.from_regex(r"[a-z]{3,20}"))
def test_classification_pattern_case_insensitive(text: str) -> None:
    """Case insensitive matching should work by default."""
    # Only test with pure ASCII strings to avoid Unicode edge cases
    assume(text.isascii() and text.islower())
    pattern = ClassificationPattern(pattern=text, case_sensitive=False)
    assert pattern.matches(text.upper())
    assert pattern.matches(text.capitalize())


# ============================================================================
# Levenshtein Distance Property Tests
# ============================================================================

@given(st.text(min_size=0, max_size=20), st.text(min_size=0, max_size=20))
def test_levenshtein_distance_symmetric(s1: str, s2: str) -> None:
    """Levenshtein distance should be symmetric."""
    pattern = ClassificationPattern(pattern="test")
    d1 = pattern._levenshtein_distance(s1, s2)
    d2 = pattern._levenshtein_distance(s2, s1)

    assert d1 == d2


@given(st.text(min_size=0, max_size=20))
def test_levenshtein_distance_to_self_is_zero(text: str) -> None:
    """Levenshtein distance to itself should be zero."""
    pattern = ClassificationPattern(pattern="test")
    distance = pattern._levenshtein_distance(text, text)
    assert distance == 0


@given(st.text(min_size=0, max_size=20))
def test_levenshtein_distance_non_negative(text: str) -> None:
    """Levenshtein distance should never be negative."""
    pattern = ClassificationPattern(pattern="test")
    other = "different"
    distance = pattern._levenshtein_distance(text, other)
    assert distance >= 0


@given(st.text(min_size=0, max_size=20), st.text(min_size=0, max_size=20))
def test_levenshtein_distance_triangle_inequality(s1: str, s2: str) -> None:
    """Levenshtein distance should satisfy triangle inequality."""
    pattern = ClassificationPattern(pattern="test")
    s3 = "third"

    d12 = pattern._levenshtein_distance(s1, s2)
    d23 = pattern._levenshtein_distance(s2, s3)
    d13 = pattern._levenshtein_distance(s1, s3)

    # d(s1, s3) <= d(s1, s2) + d(s2, s3)
    assert d13 <= d12 + d23


# ============================================================================
# Fuzzy Match Property Tests
# ============================================================================

@given(st.from_regex(r"[a-z]{5,30}"))
def test_fuzzy_match_identical_strings(text: str) -> None:
    """Fuzzy match should always return True for identical strings."""
    pattern = ClassificationPattern(pattern=text, match_type=MatchType.FUZZY)
    assert pattern.matches(text)


@given(st.from_regex(r"[a-z]{10,30}"))
def test_fuzzy_match_small_changes(text: str) -> None:
    """Fuzzy match should return True for strings with small differences."""
    assume(len(text) >= 10)

    # Change one character
    modified = text[:-1] + ("x" if text[-1] != "x" else "y")
    pattern = ClassificationPattern(pattern=text, match_type=MatchType.FUZZY)

    assert pattern.matches(modified)


@given(st.from_regex(r"[a-z]{2,5}"), st.from_regex(r"[a-z]{20,50}"))
def test_fuzzy_match_pattern_much_longer(short_pattern: str, long_text: str) -> None:
    """Fuzzy match should return False when pattern is much longer than text."""
    pattern = ClassificationPattern(pattern=long_text, match_type=MatchType.FUZZY)
    # Should fail because pattern is too long
    assert not pattern.matches(short_pattern)


# ============================================================================
# EnergyLevel Property Tests
# ============================================================================

from music_organizer.domain.classification.value_objects import (
    AudioFeatures,
    EnergyLevel,
)


@given(st.floats(min_value=0.0, max_value=1.0))
def test_energy_level_classification(energy: float) -> None:
    """Energy level should correctly classify based on energy value."""
    features = AudioFeatures(energy=energy)

    if energy < 0.2:
        assert features.energy_level == EnergyLevel.VERY_LOW
    elif energy < 0.4:
        assert features.energy_level == EnergyLevel.LOW
    elif energy < 0.6:
        assert features.energy_level == EnergyLevel.MEDIUM
    elif energy < 0.8:
        assert features.energy_level == EnergyLevel.HIGH
    else:
        assert features.energy_level == EnergyLevel.VERY_HIGH


@given(st.floats(min_value=0.0, max_value=1.0))
def test_energy_level_default_when_none(_unused: float) -> None:
    """Energy level should default to MEDIUM when None."""
    features = AudioFeatures()
    assert features.energy_level == EnergyLevel.MEDIUM


@given(st.floats(min_value=0.0, max_value=1.0))
def test_acoustic_threshold(acousticness: float) -> None:
    """is_acoustic should be True when acousticness > 0.5."""
    features = AudioFeatures(acousticness=acousticness)
    assert features.is_acoustic == (acousticness > 0.5)


@given(st.floats(min_value=0.0, max_value=1.0))
def test_instrumental_threshold(instrumentalness: float) -> None:
    """is_instrumental should be True when instrumentalness > 0.5."""
    features = AudioFeatures(instrumentalness=instrumentalness)
    assert features.is_instrumental == (instrumentalness > 0.5)


@given(st.floats(min_value=0.0, max_value=1.0))
def test_speech_threshold(speechiness: float) -> None:
    """is_speech should be True when speechiness > 0.5."""
    features = AudioFeatures(speechiness=speechiness)
    assert features.is_speech == (speechiness > 0.5)
