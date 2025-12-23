"""Tests for Kodi/Jellyfin NFO export functionality."""

import pytest
from pathlib import Path
from xml.etree import ElementTree as ET

from music_organizer.infrastructure.nfo.nfo_generator import (
    NfoGenerator,
    NfoConfig,
    ArtistInfo,
    AlbumInfo,
    TrackInfo
)


class TestArtistInfo:
    """Test ArtistInfo dataclass."""

    def test_create_artist_info_minimal(self):
        """Test creating ArtistInfo with minimal data."""
        artist = ArtistInfo(name="Test Artist")
        assert artist.name == "Test Artist"
        assert artist.musicbrainz_artist_id is None
        assert artist.genre is None

    def test_create_artist_info_full(self):
        """Test creating ArtistInfo with full data."""
        artist = ArtistInfo(
            name="The Beatles",
            musicbrainz_artist_id="b10bbbfc-cf9e-42e0-be17-e2c3e1e2605d",
            type="Group",
            country="GB",
            formed="1960",
            disbanded="1970",
            genre="Rock",
            biography="The Beatles were an English rock band...",
            thumb="https://example.com/thumb.jpg"
        )
        assert artist.name == "The Beatles"
        assert artist.musicbrainz_artist_id == "b10bbbfc-cf9e-42e0-be17-e2c3e1e2605d"
        assert artist.type == "Group"
        assert artist.country == "GB"
        assert artist.formed == "1960"
        assert artist.disbanded == "1970"
        assert artist.genre == "Rock"


class TestAlbumInfo:
    """Test AlbumInfo dataclass."""

    def test_create_album_info_minimal(self):
        """Test creating AlbumInfo with minimal data."""
        album = AlbumInfo(title="Test Album", artist="Test Artist")
        assert album.title == "Test Album"
        assert album.artist == "Test Artist"
        assert album.year is None
        assert album.tracks == []

    def test_create_album_info_with_tracks(self):
        """Test creating AlbumInfo with tracks."""
        tracks = [
            TrackInfo(position=1, title="Song 1", duration=180),
            TrackInfo(position=2, title="Song 2", duration=240)
        ]
        album = AlbumInfo(
            title="Test Album",
            artist="Test Artist",
            year=2023,
            genre="Rock",
            tracks=tracks
        )
        assert album.title == "Test Album"
        assert album.year == 2023
        assert album.genre == "Rock"
        assert len(album.tracks) == 2
        assert album.tracks[0].title == "Song 1"


class TestTrackInfo:
    """Test TrackInfo dataclass."""

    def test_create_track_info_minimal(self):
        """Test creating TrackInfo with minimal data."""
        track = TrackInfo(position=1, title="Test Song")
        assert track.position == 1
        assert track.title == "Test Song"
        assert track.duration is None
        assert track.musicbrainz_track_id is None

    def test_create_track_info_full(self):
        """Test creating TrackInfo with full data."""
        track = TrackInfo(
            position=1,
            title="Test Song",
            duration=245,
            musicbrainz_track_id="test-track-id-123"
        )
        assert track.position == 1
        assert track.title == "Test Song"
        assert track.duration == 245
        assert track.musicbrainz_track_id == "test-track-id-123"


class TestNfoGenerator:
    """Test NFO generator functionality."""

    def test_init_default_config(self):
        """Test NfoGenerator initialization with default config."""
        generator = NfoGenerator()
        assert generator.config.include_bios is True
        assert generator.config.include_reviews is True
        assert generator.config.include_ratings is True
        assert generator.config.indent_xml is True

    def test_init_custom_config(self):
        """Test NfoGenerator initialization with custom config."""
        config = NfoConfig(
            include_bios=False,
            include_reviews=False,
            indent_xml=False
        )
        generator = NfoGenerator(config)
        assert generator.config.include_bios is False
        assert generator.config.include_reviews is False
        assert generator.config.indent_xml is False

    def test_generate_artist_nfo_minimal(self):
        """Test generating artist NFO with minimal data."""
        generator = NfoGenerator()
        artist = ArtistInfo(name="Test Artist")

        xml = generator.generate_artist_nfo(artist)

        assert "<?xml version" in xml
        assert "<artist>" in xml
        assert "<name>Test Artist</name>" in xml

    def test_generate_artist_nfo_with_mbid(self):
        """Test generating artist NFO with MusicBrainz ID."""
        generator = NfoGenerator()
        artist = ArtistInfo(
            name="Test Artist",
            musicbrainz_artist_id="test-mbid-123",
            formed="2000",
            genre="Rock"
        )

        xml = generator.generate_artist_nfo(artist)

        assert "<name>Test Artist</name>" in xml
        assert "<musicBrainzArtistID>test-mbid-123</musicBrainzArtistID>" in xml
        assert "<formed>2000</formed>" in xml
        assert "<genre>Rock</genre>" in xml

    def test_generate_artist_nfo_with_biography(self):
        """Test generating artist NFO with biography."""
        config = NfoConfig(include_bios=True)
        generator = NfoGenerator(config)
        artist = ArtistInfo(
            name="Test Artist",
            biography="This is a test biography."
        )

        xml = generator.generate_artist_nfo(artist)

        assert "<biography>This is a test biography.</biography>" in xml

    def test_generate_artist_nfo_no_biography(self):
        """Test generating artist NFO without biography."""
        config = NfoConfig(include_bios=False)
        generator = NfoGenerator(config)
        artist = ArtistInfo(
            name="Test Artist",
            biography="This should not appear."
        )

        xml = generator.generate_artist_nfo(artist)

        assert "<biography>" not in xml

    def test_generate_artist_nfo_with_albums(self):
        """Test generating artist NFO with album list."""
        generator = NfoGenerator()
        artist = ArtistInfo(
            name="Test Artist",
            albums=[
                {"title": "Album 1", "year": 2020, "musicbrainz_releasegroup_id": "rg1"},
                {"title": "Album 2", "year": 2022, "musicbrainz_releasegroup_id": "rg2"}
            ]
        )

        xml = generator.generate_artist_nfo(artist)

        assert "<album>" in xml
        assert "<title>Album 1</title>" in xml
        assert "<title>Album 2</title>" in xml
        assert "<year>2020</year>" in xml
        assert "<year>2022</year>" in xml

    def test_generate_album_nfo_minimal(self):
        """Test generating album NFO with minimal data."""
        generator = NfoGenerator()
        album = AlbumInfo(title="Test Album", artist="Test Artist")

        xml = generator.generate_album_nfo(album)

        assert "<?xml version" in xml
        assert "<album>" in xml
        assert "<title>Test Album</title>" in xml
        assert "<artist>Test Artist</artist>" in xml

    def test_generate_album_nfo_with_year(self):
        """Test generating album NFO with year."""
        generator = NfoGenerator()
        album = AlbumInfo(
            title="Test Album",
            artist="Test Artist",
            year=2023
        )

        xml = generator.generate_album_nfo(album)

        assert "<year>2023</year>" in xml

    def test_generate_album_nfo_with_mbid(self):
        """Test generating album NFO with MusicBrainz IDs."""
        generator = NfoGenerator()
        album = AlbumInfo(
            title="Test Album",
            artist="Test Artist",
            musicbrainz_release_id="release-123",
            musicbrainz_releasegroup_id="rg-123"
        )

        xml = generator.generate_album_nfo(album)

        assert "<musicBrainzReleaseID>release-123</musicBrainzReleaseID>" in xml
        assert "<musicBrainzReleaseGroupID>rg-123</musicBrainzReleaseGroupID>" in xml

    def test_generate_album_nfo_with_tracks(self):
        """Test generating album NFO with tracks."""
        generator = NfoGenerator()
        album = AlbumInfo(
            title="Test Album",
            artist="Test Artist",
            tracks=[
                TrackInfo(position=1, title="Song 1", duration=180, musicbrainz_track_id="t1"),
                TrackInfo(position=2, title="Song 2", duration=240, musicbrainz_track_id="t2")
            ]
        )

        xml = generator.generate_album_nfo(album)

        assert "<track>" in xml
        assert "<position>1</position>" in xml
        assert "<title>Song 1</title>" in xml
        assert "<duration>180</duration>" in xml
        assert "<musicBrainzTrackID>t1</musicBrainzTrackID>" in xml
        assert "<position>2</position>" in xml
        assert "<title>Song 2</title>" in xml

    def test_generate_album_nfo_with_review(self):
        """Test generating album NFO with review."""
        config = NfoConfig(include_reviews=True)
        generator = NfoGenerator(config)
        album = AlbumInfo(
            title="Test Album",
            artist="Test Artist",
            review="This is a great album!"
        )

        xml = generator.generate_album_nfo(album)

        assert "<review>This is a great album!</review>" in xml

    def test_generate_album_nfo_no_review(self):
        """Test generating album NFO without review."""
        config = NfoConfig(include_reviews=False)
        generator = NfoGenerator(config)
        album = AlbumInfo(
            title="Test Album",
            artist="Test Artist",
            review="This should not appear."
        )

        xml = generator.generate_album_nfo(album)

        assert "<review>" not in xml

    def test_write_artist_nfo_file(self, tmp_path):
        """Test writing artist NFO to file."""
        generator = NfoGenerator()
        nfo_path = tmp_path / "artist.nfo"
        artist = ArtistInfo(name="Test Artist")

        generator.write_artist_nfo(nfo_path, artist)

        assert nfo_path.exists()
        content = nfo_path.read_text()
        assert "Test Artist" in content

    def test_write_album_nfo_file(self, tmp_path):
        """Test writing album NFO to file."""
        generator = NfoGenerator()
        nfo_path = tmp_path / "album.nfo"
        album = AlbumInfo(title="Test Album", artist="Test Artist")

        generator.write_album_nfo(nfo_path, album)

        assert nfo_path.exists()
        content = nfo_path.read_text()
        assert "Test Album" in content
        assert "Test Artist" in content

    def test_write_nfo_creates_directory(self, tmp_path):
        """Test that writing NFO creates parent directories."""
        generator = NfoGenerator()
        nfo_path = tmp_path / "deep" / "nested" / "album.nfo"
        album = AlbumInfo(title="Test Album", artist="Test Artist")

        generator.write_album_nfo(nfo_path, album)

        assert nfo_path.exists()
        assert nfo_path.parent.is_dir()

    def test_create_artist_info_from_dict(self):
        """Test creating ArtistInfo from metadata dictionary."""
        metadata = {
            "musicbrainz_artist_id": "mbid-123",
            "genre": "Rock",
            "formed": "1990",
            "biography": "Test bio"
        }
        artist = NfoGenerator.create_artist_info("Test Artist", metadata)

        assert artist.name == "Test Artist"
        assert artist.musicbrainz_artist_id == "mbid-123"
        assert artist.genre == "Rock"
        assert artist.formed == "1990"
        assert artist.biography == "Test bio"

    def test_create_album_info_from_dict(self):
        """Test creating AlbumInfo from metadata dictionary."""
        metadata = {
            "year": 2023,
            "genre": "Pop",
            "rating": 8.5,
            "musicbrainz_release_id": "rel-123",
            "tracks": [
                {"position": 1, "title": "Song 1", "duration": 180, "musicbrainz_track_id": "t1"}
            ]
        }
        album = NfoGenerator.create_album_info("Test Album", "Test Artist", metadata)

        assert album.title == "Test Album"
        assert album.artist == "Test Artist"
        assert album.year == 2023
        assert album.genre == "Pop"
        assert album.rating == 8.5
        assert album.musicbrainz_release_id == "rel-123"
        assert len(album.tracks) == 1
        assert album.tracks[0].title == "Song 1"


class TestNfoXmlValidation:
    """Test that generated NFO XML is valid."""

    def test_artist_nfo_valid_xml(self):
        """Test that artist NFO generates valid XML."""
        generator = NfoGenerator()
        artist = ArtistInfo(
            name="Test Artist",
            musicbrainz_artist_id="mbid-123",
            genre="Rock"
        )

        xml = generator.generate_artist_nfo(artist)

        # Parse XML to verify validity
        root = ET.fromstring(xml)
        assert root.tag == "artist"
        assert root.find("name").text == "Test Artist"
        assert root.find("musicBrainzArtistID").text == "mbid-123"

    def test_album_nfo_valid_xml(self):
        """Test that album NFO generates valid XML."""
        generator = NfoGenerator()
        album = AlbumInfo(
            title="Test Album",
            artist="Test Artist",
            year=2023,
            tracks=[
                TrackInfo(position=1, title="Song 1")
            ]
        )

        xml = generator.generate_album_nfo(album)

        # Parse XML to verify validity
        root = ET.fromstring(xml)
        assert root.tag == "album"
        assert root.find("title").text == "Test Album"
        assert root.find("artist").text == "Test Artist"
        assert root.find("year").text == "2023"

    def test_album_nfo_tracks_valid_xml(self):
        """Test that album NFO tracks generate valid XML."""
        generator = NfoGenerator()
        album = AlbumInfo(
            title="Test Album",
            artist="Test Artist",
            tracks=[
                TrackInfo(position=1, title="Song 1", duration=180),
                TrackInfo(position=2, title="Song 2", duration=240)
            ]
        )

        xml = generator.generate_album_nfo(album)

        # Parse XML to verify validity
        root = ET.fromstring(xml)
        track_elements = root.findall("track")
        assert len(track_elements) == 2
        assert track_elements[0].find("title").text == "Song 1"
        assert track_elements[1].find("title").text == "Song 2"


class TestKodiNfoExporterPlugin:
    """Test KodiNfoExporterPlugin."""

    def test_plugin_info(self):
        """Test plugin info property."""
        from music_organizer.plugins.builtins.kodi_nfo_exporter import KodiNfoExporterPlugin

        plugin = KodiNfoExporterPlugin()
        info = plugin.info

        assert info.name == "kodi_nfo_exporter"
        assert info.version == "1.0.0"
        assert "Kodi" in info.description
        assert info.author == "Music Organizer Team"

    def test_plugin_initialization(self):
        """Test plugin initialization."""
        from music_organizer.plugins.builtins.kodi_nfo_exporter import KodiNfoExporterPlugin

        config = {
            "include_bios": False,
            "fetch_mbid": False
        }
        plugin = KodiNfoExporterPlugin(config)
        plugin.initialize()

        assert plugin.nfo_generator is not None
        assert plugin.config["include_bios"] is False

    def test_plugin_cleanup(self):
        """Test plugin cleanup."""
        from music_organizer.plugins.builtins.kodi_nfo_exporter import KodiNfoExporterPlugin

        plugin = KodiNfoExporterPlugin()
        plugin.initialize()
        plugin.cleanup()

        assert len(plugin._artist_cache) == 0
        assert len(plugin._album_cache) == 0

    def test_supported_formats(self):
        """Test get_supported_formats."""
        from music_organizer.plugins.builtins.kodi_nfo_exporter import KodiNfoExporterPlugin

        plugin = KodiNfoExporterPlugin()
        assert "nfo" in plugin.get_supported_formats()

    def test_file_extension(self):
        """Test get_file_extension."""
        from music_organizer.plugins.builtins.kodi_nfo_exporter import KodiNfoExporterPlugin

        plugin = KodiNfoExporterPlugin()
        assert plugin.get_file_extension() == "nfo"

    def test_create_plugin_factory(self):
        """Test create_plugin factory function."""
        from music_organizer.plugins.builtins.kodi_nfo_exporter import create_plugin

        plugin = create_plugin({"fetch_mbid": False})
        assert plugin is not None
        assert plugin.config["fetch_mbid"] is False
