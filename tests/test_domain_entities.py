"""Tests for domain entities."""

import pytest
from pathlib import Path
from datetime import datetime, timedelta

from music_organizer.domain.entities import (
    Recording,
    Release,
    Collection,
    AudioLibrary,
    DuplicateResolutionMode,
)
from music_organizer.domain.value_objects import (
    AudioPath,
    ArtistName,
    TrackNumber,
    Metadata,
    FileFormat,
)


@pytest.fixture
def sample_audio_path():
    """Create a sample audio path."""
    return AudioPath("/music/test/artist/album/01.flac")


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return Metadata(
        title="Test Song",
        artists=frozenset([ArtistName("Test Artist")]),
        album="Test Album",
        year=2023,
        track_number=TrackNumber(1),
        duration_seconds=180.5,
        file_hash="abc123",
        acoustic_fingerprint="def456",
    )


@pytest.fixture
def sample_recording(sample_audio_path, sample_metadata):
    """Create a sample recording."""
    return Recording(
        path=sample_audio_path,
        metadata=sample_metadata,
    )


class TestRecording:
    """Test the Recording entity."""

    def test_recording_creation(self, sample_audio_path, sample_metadata):
        """Test creating a recording."""
        recording = Recording(
            path=sample_audio_path,
            metadata=sample_metadata,
        )

        assert recording.path == sample_audio_path
        assert recording.metadata == sample_metadata
        assert not recording.is_processed
        assert not recording.is_duplicate
        assert recording.processing_errors == []
        assert recording.content_type is None
        assert recording.genre_classifications == set()

    def test_recording_properties(self, sample_recording):
        """Test recording properties."""
        assert sample_recording.title == "Test Song"
        assert sample_recording.artists == [ArtistName("Test Artist")]
        assert sample_recording.primary_artist == ArtistName("Test Artist")
        assert sample_recording.album == "Test Album"
        assert sample_recording.year == 2023
        assert sample_recording.track_number == TrackNumber(1)
        assert sample_recording.duration_seconds == 180.5
        assert sample_recording.file_hash == "abc123"
        assert sample_recording.acoustic_fingerprint == "def456"

    def test_get_display_name(self, sample_recording):
        """Test getting display name."""
        assert sample_recording.get_display_name() == "Test Artist - Test Song"

    def test_get_display_name_no_artist(self, sample_audio_path):
        """Test display name with no artist."""
        metadata = Metadata(title="Test Song")
        recording = Recording(path=sample_audio_path, metadata=metadata)
        assert recording.get_display_name() == "Test Song"

    def test_get_display_name_no_title(self, sample_audio_path):
        """Test display name with no title."""
        metadata = Metadata(artists=frozenset([ArtistName("Test Artist")]))
        recording = Recording(path=sample_audio_path, metadata=metadata)
        assert recording.get_display_name() == "01.flac"

    def test_genre_classification(self, sample_recording):
        """Test genre classification."""
        sample_recording.add_genre_classification("Rock")
        sample_recording.add_genre_classification("pop")
        sample_recording.add_genre_classification("ROCK")  # Case insensitive

        assert sample_recording.has_genre("rock")
        assert sample_recording.has_genre("pop")
        assert sample_recording.has_genre("ROCK")
        assert not sample_recording.has_genre("jazz")
        assert sample_recording.genre_classifications == {"rock", "pop"}

    def test_processing_status(self, sample_recording):
        """Test processing status tracking."""
        assert not sample_recording.is_processed

        sample_recording.mark_as_processed()
        assert sample_recording.is_processed

        sample_recording.add_processing_error("Test error")
        assert sample_recording.processing_errors == ["Test error"]

    def test_duplicate_tracking(self, sample_recording, sample_audio_path):
        """Test duplicate tracking."""
        other_metadata = Metadata(title="Test Song", artists=frozenset([ArtistName("Test Artist")]))
        other_recording = Recording(path=sample_audio_path, metadata=other_metadata)

        assert not sample_recording.is_duplicate
        assert sample_recording.duplicate_of is None

        sample_recording.set_duplicate(other_recording)
        assert sample_recording.is_duplicate
        assert sample_recording.duplicate_of == other_recording

    def test_move_history(self, sample_recording):
        """Test move history tracking."""
        from_path = Path("/old/path.flac")
        to_path = Path("/new/path.flac")

        sample_recording.record_move(from_path, to_path)

        assert len(sample_recording.move_history) == 1
        assert sample_recording.move_history[0]["from"] == str(from_path)
        assert sample_recording.move_history[0]["to"] == str(to_path)
        assert "timestamp" in sample_recording.move_history[0]

    def test_similarity_exact_match(self, sample_recording, sample_audio_path):
        """Test similarity calculation for exact match."""
        # Same hash should be 1.0 similarity
        other_metadata = Metadata(
            title="Different Title",
            file_hash="abc123",  # Same hash
        )
        other_recording = Recording(path=sample_audio_path, metadata=other_metadata)

        similarity = sample_recording.calculate_similarity(other_recording)
        assert similarity == 1.0

    def test_similarity_acoustic_fingerprint(self, sample_recording, sample_audio_path):
        """Test similarity calculation for acoustic fingerprint match."""
        other_metadata = Metadata(
            title="Different Title",
            acoustic_fingerprint="def456",  # Same fingerprint
        )
        other_recording = Recording(path=sample_audio_path, metadata=other_metadata)

        similarity = sample_recording.calculate_similarity(other_recording)
        assert similarity == 0.95

    def test_similarity_metadata_based(self, sample_recording, sample_audio_path):
        """Test similarity calculation based on metadata."""
        # Similar but not identical metadata
        other_metadata = Metadata(
            title="Test Song",  # Same title
            artists=frozenset([ArtistName("Test Artist")]),  # Same artist
            album="Test Album",  # Same album
            duration_seconds=185.0,  # Similar duration
        )
        other_recording = Recording(path=sample_audio_path, metadata=other_metadata)

        similarity = sample_recording.calculate_similarity(other_recording)
        assert 0.8 < similarity < 1.0  # High similarity

    def test_similarity_no_match(self, sample_recording, sample_audio_path):
        """Test similarity calculation for no match."""
        other_metadata = Metadata(
            title="Completely Different",
            artists=frozenset([ArtistName("Other Artist")]),
            album="Other Album",
            duration_seconds=120.0,
        )
        other_recording = Recording(path=sample_audio_path, metadata=other_metadata)

        similarity = sample_recording.calculate_similarity(other_recording)
        assert similarity < 0.5  # Low similarity


class TestRelease:
    """Test the Release entity."""

    def test_release_creation(self):
        """Test creating a release."""
        release = Release(
            title="Test Album",
            primary_artist=ArtistName("Test Artist"),
            year=2023,
            release_type="album",
            genre="Rock",
        )

        assert release.title == "Test Album"
        assert release.primary_artist == ArtistName("Test Artist")
        assert release.year == 2023
        assert release.release_type == "album"
        assert release.genre == "Rock"
        assert release.recordings == []
        assert release.source_paths == []

    def test_release_properties(self):
        """Test release properties."""
        release = Release(
            title="Test Album",
            primary_artist=ArtistName("Test Artist"),
            year=2023,
        )

        assert release.display_name == "Test Album (2023)"
        assert release.artist_display == "Test Artist"
        assert release.track_count == 0
        assert release.total_duration_seconds == 0.0
        assert release.total_size_mb == 0.0

    def test_add_recording(self, sample_recording):
        """Test adding recordings to release."""
        release = Release(
            title="Test Album",
            primary_artist=ArtistName("Test Artist"),
        )

        assert release.track_count == 0

        release.add_recording(sample_recording)
        assert release.track_count == 1
        assert sample_recording in release.recordings

        # Adding same recording again should not duplicate
        release.add_recording(sample_recording)
        assert release.track_count == 1

    def test_remove_recording(self, sample_recording):
        """Test removing recordings from release."""
        release = Release(
            title="Test Album",
            primary_artist=ArtistName("Test Artist"),
        )

        release.add_recording(sample_recording)
        assert release.track_count == 1

        release.remove_recording(sample_recording)
        assert release.track_count == 0
        assert sample_recording not in release.recordings

    def test_get_recording_by_track(self, sample_audio_path):
        """Test getting recording by track number."""
        release = Release(
            title="Test Album",
            primary_artist=ArtistName("Test Artist"),
        )

        # Add multiple recordings
        for i in range(1, 4):
            metadata = Metadata(
                title=f"Track {i}",
                artists=frozenset([ArtistName("Test Artist")]),
                track_number=TrackNumber(i),
            )
            recording = Recording(path=sample_audio_path, metadata=metadata)
            release.add_recording(recording)

        # Test getting tracks
        track1 = release.get_recording_by_track(TrackNumber(1))
        assert track1 is not None
        assert track1.title == "Track 1"

        track2 = release.get_recording_by_track(TrackNumber(2))
        assert track2 is not None
        assert track2.title == "Track 2"

        track_nonexistent = release.get_recording_by_track(TrackNumber(99))
        assert track_nonexistent is None

    def test_sort_recordings(self, sample_audio_path):
        """Test sorting recordings."""
        release = Release(
            title="Test Album",
            primary_artist=ArtistName("Test Artist"),
        )

        # Add recordings in random order
        tracks = [
            (3, "Zebra"),
            (1, "Alpha"),
            (2, "Beta"),
            (None, "No Track"),
        ]

        for track_num, title in tracks:
            metadata = Metadata(
                title=title,
                artists=frozenset([ArtistName("Test Artist")]),
                track_number=TrackNumber(track_num) if track_num else None,
            )
            recording = Recording(path=sample_audio_path, metadata=metadata)
            release.add_recording(recording)

        # Sort recordings
        release.sort_recordings()

        # Check order: tracks with numbers first (sorted), then no track
        titles = [r.title for r in release.recordings]
        assert titles == ["Alpha", "Beta", "Zebra", "No Track"]

    def test_get_duplicate_groups(self, sample_audio_path):
        """Test finding duplicate groups in release."""
        release = Release(
            title="Test Album",
            primary_artist=ArtistName("Test Artist"),
        )

        # Add duplicates
        metadata1 = Metadata(
            title="Same Song",
            artists=frozenset([ArtistName("Test Artist")]),
            file_hash="hash1",
            duration_seconds=180.0,
        )
        metadata2 = Metadata(
            title="Same Song",
            artists=frozenset([ArtistName("Test Artist")]),
            file_hash="hash2",
            duration_seconds=180.0,
        )
        metadata3 = Metadata(
            title="Different Song",
            artists=frozenset([ArtistName("Test Artist")]),
            file_hash="hash3",
            duration_seconds=180.0,
        )

        recording1 = Recording(path=sample_audio_path, metadata=metadata1)
        recording2 = Recording(path=sample_audio_path, metadata=metadata2)
        recording3 = Recording(path=sample_audio_path, metadata=metadata3)

        release.add_recording(recording1)
        release.add_recording(recording2)
        release.add_recording(recording3)

        # Get duplicate groups with high threshold (exact matches)
        groups = release.get_duplicate_groups(similarity_threshold=0.99)
        assert len(groups) == 0  # No exact duplicates with different hashes

        # Get duplicate groups with lower threshold
        groups = release.get_duplicate_groups(similarity_threshold=0.8)
        assert len(groups) == 1
        assert len(groups[0]) == 2
        assert recording1 in groups[0]
        assert recording2 in groups[0]
        assert recording3 not in groups[0]

    def test_merge_with(self, sample_audio_path):
        """Test merging releases."""
        release1 = Release(
            title="Album",
            primary_artist=ArtistName("Artist"),
            year=2023,
            genre="Rock",
        )

        release2 = Release(
            title="Album",
            primary_artist=ArtistName("Artist"),
            genre="Pop",  # Different genre
            total_tracks=10,  # Has total tracks
        )

        # Add recordings to both
        metadata1 = Metadata(title="Song 1", artists=frozenset([ArtistName("Artist")]))
        metadata2 = Metadata(title="Song 2", artists=frozenset([ArtistName("Artist")]))
        recording1 = Recording(path=sample_audio_path, metadata=metadata1)
        recording2 = Recording(path=sample_audio_path, metadata=metadata2)

        release1.add_recording(recording1)
        release2.add_recording(recording2)

        # Merge
        release1.merge_with(release2)

        # Check merged properties
        assert release1.genre == "Rock"  # Keeps original
        assert release1.total_tracks == 10  # Gets from release2
        assert release1.track_count == 2  # Both recordings
        assert recording1 in release1.recordings
        assert recording2 in release1.recordings


class TestCollection:
    """Test the Collection entity."""

    def test_collection_creation(self):
        """Test creating a collection."""
        collection = Collection(
            name="My Music",
            description="Personal music collection",
        )

        assert collection.name == "My Music"
        assert collection.description == "Personal music collection"
        assert collection.releases == []
        assert collection.subcollections == []
        assert collection.parent_collection is None

    def test_collection_properties(self, sample_recording):
        """Test collection properties."""
        collection = Collection(name="Test")

        # Add release with recording
        release = Release(
            title="Test Album",
            primary_artist=ArtistName("Test Artist"),
        )
        release.add_recording(sample_recording)
        collection.add_release(release)

        assert collection.release_count == 1
        assert collection.recording_count == 1
        assert collection.total_duration_seconds == sample_recording.duration_seconds
        assert collection.total_size_gb == pytest.approx(sample_recording.path.size_mb / 1024)

    def test_add_remove_release(self):
        """Test adding and removing releases."""
        collection = Collection(name="Test")
        release = Release(title="Test", primary_artist=ArtistName("Artist"))

        collection.add_release(release)
        assert collection.release_count == 1
        assert release in collection.releases

        collection.remove_release(release)
        assert collection.release_count == 0
        assert release not in collection.releases

        # Adding again should not duplicate
        collection.add_release(release)
        collection.add_release(release)
        assert collection.release_count == 1

    def test_subcollections(self):
        """Test subcollection management."""
        parent = Collection(name="Parent")
        child = Collection(name="Child")

        parent.add_subcollection(child)

        assert child in parent.subcollections
        assert child.parent_collection == parent

        # Nested collections
        grandchild = Collection(name="Grandchild")
        child.add_subcollection(grandchild)
        assert grandchild.parent_collection == child

    def test_get_all_releases(self, sample_audio_path):
        """Test getting all releases including nested."""
        root = Collection(name="Root")
        sub = Collection(name="Sub")

        root.add_subcollection(sub)

        # Add releases to both
        release1 = Release(title="Album 1", primary_artist=ArtistName("Artist 1"))
        release2 = Release(title="Album 2", primary_artist=ArtistName("Artist 2"))

        root.add_release(release1)
        sub.add_release(release2)

        # Get all releases
        all_releases = list(root.get_all_releases())
        assert len(all_releases) == 2
        assert release1 in all_releases
        assert release2 in all_releases

    def test_get_all_recordings(self, sample_recording, sample_audio_path):
        """Test getting all recordings including nested."""
        root = Collection(name="Root")
        sub = Collection(name="Sub")

        root.add_subcollection(sub)

        # Add recordings through releases
        release1 = Release(title="Album 1", primary_artist=ArtistName("Artist 1"))
        release2 = Release(title="Album 2", primary_artist=ArtistName("Artist 2"))

        metadata2 = Metadata(title="Song 2", artists=frozenset([ArtistName("Artist 2")]))
        recording2 = Recording(path=sample_audio_path, metadata=metadata2)

        release1.add_recording(sample_recording)
        release2.add_recording(recording2)

        root.add_release(release1)
        sub.add_release(release2)

        # Get all recordings
        all_recordings = list(root.get_all_recordings())
        assert len(all_recordings) == 2
        assert sample_recording in all_recordings
        assert recording2 in all_recordings

    def test_filter_by_genre(self):
        """Test filtering releases by genre."""
        collection = Collection(name="Test")

        rock_release = Release(title="Rock Album", primary_artist=ArtistName("Rock Artist"), genre="Rock")
        pop_release = Release(title="Pop Album", primary_artist=ArtistName("Pop Artist"), genre="Pop")
        no_genre = Release(title="No Genre", primary_artist=ArtistName("Artist"))

        collection.add_release(rock_release)
        collection.add_release(pop_release)
        collection.add_release(no_genre)

        rock_filtered = collection.filter_by_genre("rock")
        assert len(rock_filtered) == 1
        assert rock_release in rock_filtered

        # Case insensitive
        rock_filtered2 = collection.filter_by_genre("ROCK")
        assert len(rock_filtered2) == 1

    def test_filter_by_year(self):
        """Test filtering releases by year."""
        collection = Collection(name="Test")

        release2020 = Release(title="2020 Album", primary_artist=ArtistName("Artist"), year=2020)
        release2021 = Release(title="2021 Album", primary_artist=ArtistName("Artist"), year=2021)
        no_year = Release(title="No Year", primary_artist=ArtistName("Artist"))

        collection.add_release(release2020)
        collection.add_release(release2021)
        collection.add_release(no_year)

        filtered = collection.filter_by_year(2020)
        assert len(filtered) == 1
        assert release2020 in filtered

    def test_filter_by_artist(self):
        """Test filtering releases by artist."""
        collection = Collection(name="Test")

        beatles = Release(title="Abbey Road", primary_artist=ArtistName("The Beatles"))
        solo = Release(title="Imagine", primary_artist=ArtistName("John Lennon"))

        collection.add_release(beatles)
        collection.add_release(solo)

        # Exact match
        filtered = collection.filter_by_artist(ArtistName("The Beatles"))
        assert len(filtered) == 1
        assert beatles in filtered

        # Partial match
        filtered = collection.filter_by_artist(ArtistName("Beatles"))
        assert len(filtered) == 1
        assert beatles in filtered


class TestAudioLibrary:
    """Test the AudioLibrary entity."""

    def test_library_creation(self):
        """Test creating a library."""
        library = AudioLibrary(
            name="My Library",
            root_path=Path("/music"),
        )

        assert library.name == "My Library"
        assert library.root_path == Path("/music")
        assert library.collections == []
        assert library.standalone_recordings == []
        assert library.duplicate_resolution_mode == DuplicateResolutionMode.SKIP
        assert library.scan_count == 0
        assert library.created_at is not None

    def test_library_properties(self, sample_recording):
        """Test library properties."""
        library = AudioLibrary(name="Test", root_path=Path("/test"))

        # Add standalone recording
        library.add_standalone_recording(sample_recording)

        # Add collection with release and recording
        collection = Collection(name="Test Collection")
        release = Release(title="Test Album", primary_artist=ArtistName("Test Artist"))
        release.add_recording(sample_recording)
        collection.add_release(release)
        library.add_collection(collection)

        # Check counts
        assert library.total_recordings == 2  # 1 standalone + 1 in collection
        assert library.total_releases == 1
        assert library.total_size_gb == pytest.approx(
            (sample_recording.path.size_mb * 2) / 1024  # Both recordings
        )

    def test_add_remove_collection(self):
        """Test adding and removing collections."""
        library = AudioLibrary(name="Test", root_path=Path("/test"))
        collection = Collection(name="Test")

        library.add_collection(collection)
        assert len(library.collections) == 1
        assert collection in library.collections

        library.remove_collection(collection)
        assert len(library.collections) == 0

        # Adding again should not duplicate
        library.add_collection(collection)
        library.add_collection(collection)
        assert len(library.collections) == 1

    def test_get_all_recordings(self, sample_audio_path):
        """Test getting all recordings from library."""
        library = AudioLibrary(name="Test", root_path=Path("/test"))

        # Add standalone recording
        standalone_metadata = Metadata(title="Standalone", artists=frozenset([ArtistName("Artist")]))
        standalone = Recording(path=sample_audio_path, metadata=standalone_metadata)
        library.add_standalone_recording(standalone)

        # Add recording in collection
        collection = Collection(name="Test")
        release = Release(title="Album", primary_artist=ArtistName("Artist"))
        album_metadata = Metadata(title="Album Track", artists=frozenset([ArtistName("Artist")]))
        album_track = Recording(path=sample_audio_path, metadata=album_metadata)
        release.add_recording(album_track)
        collection.add_release(release)
        library.add_collection(collection)

        # Get all recordings
        all_recordings = list(library.get_all_recordings())
        assert len(all_recordings) == 2
        assert standalone in all_recordings
        assert album_track in all_recordings

    def test_find_duplicates(self, sample_audio_path):
        """Test finding duplicates in library."""
        library = AudioLibrary(name="Test", root_path=Path("/test"))

        # Add duplicate recordings
        metadata1 = Metadata(
            title="Same Song",
            artists=frozenset([ArtistName("Artist")]),
            file_hash="hash1",
            duration_seconds=180.0,
        )
        metadata2 = Metadata(
            title="Same Song",
            artists=frozenset([ArtistName("Artist")]),
            file_hash="hash2",
            duration_seconds=180.0,
        )
        metadata3 = Metadata(
            title="Different Song",
            artists=frozenset([ArtistName("Artist")]),
            file_hash="hash3",
            duration_seconds=180.0,
        )

        recording1 = Recording(path=sample_audio_path, metadata=metadata1)
        recording2 = Recording(path=sample_audio_path, metadata=metadata2)
        recording3 = Recording(path=sample_audio_path, metadata=metadata3)

        library.add_standalone_recording(recording1)
        library.add_standalone_recording(recording2)
        library.add_standalone_recording(recording3)

        # Find duplicates
        duplicates = library.find_duplicates(similarity_threshold=0.8)

        assert len(duplicates) == 1
        key = list(duplicates.keys())[0]
        assert "Artist" in key and "Same Song" in key
        assert len(duplicates[key]) == 2
        assert recording1 in duplicates[key]
        assert recording2 in duplicates[key]
        assert recording3 not in duplicates[key]

    def test_get_statistics(self, sample_audio_path, tmp_path):
        """Test getting library statistics."""
        library = AudioLibrary(name="Test", root_path=Path("/test"))

        # Add diverse recordings
        recordings = []
        for i in range(5):
            # Create temporary files with enough size to register in GB
            temp_file = tmp_path / f"song_{i}.flac"
            # Write ~1MB per file so total size is ~5MB = 0.005GB
            temp_file.write_bytes(b"fake audio data" * (100000 + i * 10000))

            metadata = Metadata(
                title=f"Song {i}",
                artists=frozenset([ArtistName(f"Artist {i % 2}")]),  # 2 different artists
                year=2020 + (i % 3),  # 3 different years
                duration_seconds=180 + i * 10,
            )
            recording = Recording(path=AudioPath(str(temp_file)), metadata=metadata)
            recording.add_genre_classification("rock" if i % 2 == 0 else "pop")
            recordings.append(recording)
            library.add_standalone_recording(recording)

        # Get statistics
        stats = library.get_statistics()

        assert stats["total_recordings"] == 5
        assert stats["total_releases"] == 0  # No releases yet
        assert stats["total_collections"] == 0
        assert stats["total_size_gb"] > 0
        assert stats["average_duration_seconds"] > 180
        assert "flac" in stats["format_distribution"]
        assert stats["duplicate_count"] == 0
        assert stats["processed_count"] == 0
        assert stats["error_count"] == 0

        # Check genre distribution
        assert "rock" in stats["genre_distribution"]
        assert "pop" in stats["genre_distribution"]
        assert stats["genre_distribution"]["rock"] == 3
        assert stats["genre_distribution"]["pop"] == 2

        # Check artist distribution
        assert len(stats["top_artists"]) == 2
        for artist, count in stats["top_artists"]:
            assert count > 0

    def test_mark_scan_completed(self):
        """Test marking scan completion."""
        library = AudioLibrary(name="Test", root_path=Path("/test"))

        assert library.scan_count == 0
        assert library.last_scanned is None

        library.mark_scan_completed()

        assert library.scan_count == 1
        assert library.last_scanned is not None
        assert library.last_scanned > datetime.now() - timedelta(seconds=1)

        # Multiple scans
        library.mark_scan_completed()
        assert library.scan_count == 2

    def test_get_releases_by_artist(self, sample_audio_path):
        """Test getting releases by artist."""
        library = AudioLibrary(name="Test", root_path=Path("/test"))

        # Create collection with releases
        collection = Collection(name="Test")

        beatles_album = Release(title="Abbey Road", primary_artist=ArtistName("The Beatles"))
        solo_album = Release(title="Imagine", primary_artist=ArtistName("John Lennon"))

        collection.add_release(beatles_album)
        collection.add_release(solo_album)
        library.add_collection(collection)

        # Search for Beatles releases
        beatles_releases = library.get_releases_by_artist(ArtistName("The Beatles"))
        assert len(beatles_releases) == 1
        assert beatles_album in beatles_releases

        # Partial match
        beatles_releases = library.get_releases_by_artist(ArtistName("Beatles"))
        assert len(beatles_releases) == 1

    def test_get_recently_added(self, sample_audio_path, tmp_path):
        """Test getting recently added recordings."""
        library = AudioLibrary(name="Test", root_path=Path("/test"))

        # Create a temporary file to simulate a recent file
        recent_file = tmp_path / "recent.flac"
        recent_file.write_bytes(b"fake audio data")

        # Add recordings
        old_metadata = Metadata(title="Old Song", artists=frozenset([ArtistName("Artist")]))
        old_recording = Recording(
            path=AudioPath(str(sample_audio_path.path)),  # Use existing path
            metadata=old_metadata,
        )

        recent_metadata = Metadata(title="New Song", artists=frozenset([ArtistName("Artist")]))
        recent_recording = Recording(
            path=AudioPath(str(recent_file)),  # Use recent temp file
            metadata=recent_metadata,
        )

        library.add_standalone_recording(old_recording)
        library.add_standalone_recording(recent_recording)

        # Get recently added (last 7 days)
        recent = library.get_recently_added(days=7)

        # Should include the recent recording
        assert len(recent) >= 1
        assert recent_recording in recent

    def test_duplicate_resolution_mode(self):
        """Test duplicate resolution mode."""
        library = AudioLibrary(name="Test", root_path=Path("/test"))
        assert library.duplicate_resolution_mode == DuplicateResolutionMode.SKIP

        library.duplicate_resolution_mode = DuplicateResolutionMode.RENAME
        assert library.duplicate_resolution_mode == DuplicateResolutionMode.RENAME