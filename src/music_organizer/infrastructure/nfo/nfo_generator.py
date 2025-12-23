"""NFO file generator for Kodi/Jellyfin compatibility."""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NfoConfig:
    """Configuration for NFO generation."""
    include_bios: bool = True
    include_reviews: bool = True
    include_ratings: bool = True
    include_moods: bool = True
    indent_xml: bool = True


@dataclass(slots=True)
class TrackInfo:
    """Track information for NFO generation."""
    position: int
    title: str
    duration: Optional[int] = None
    musicbrainz_track_id: Optional[str] = None


@dataclass(slots=True)
class ArtistInfo:
    """Artist information for NFO generation."""
    name: str
    musicbrainz_artist_id: Optional[str] = None
    type: Optional[str] = None
    country: Optional[str] = None
    formed: Optional[str] = None
    disbanded: Optional[str] = None
    genre: Optional[str] = None
    style: Optional[str] = None
    mood: Optional[str] = None
    biography: Optional[str] = None
    thumb: Optional[str] = None
    fanart: Optional[str] = None
    albums: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class AlbumInfo:
    """Album information for NFO generation."""
    title: str
    artist: str
    year: Optional[int] = None
    genre: Optional[str] = None
    style: Optional[str] = None
    mood: Optional[str] = None
    review: Optional[str] = None
    rating: Optional[float] = None
    musicbrainz_release_id: Optional[str] = None
    musicbrainz_releasegroup_id: Optional[str] = None
    tracks: List[TrackInfo] = field(default_factory=list)


class NfoGenerator:
    """Generate NFO files for Kodi/Jellyfin media centers."""

    def __init__(self, config: Optional[NfoConfig] = None) -> None:
        self.config = config or NfoConfig()

    def generate_artist_nfo(self, artist: ArtistInfo) -> str:
        """Generate artist NFO XML content.

        Args:
            artist: Artist information

        Returns:
            XML string for artist.nfo file
        """
        root = ET.Element("artist")

        # Basic info
        ET.SubElement(root, "name").text = artist.name

        # Dates
        if artist.formed:
            ET.SubElement(root, "formed").text = artist.formed
        if artist.disbanded:
            ET.SubElement(root, "disbanded").text = artist.disbanded

        # Genres
        if artist.genre:
            ET.SubElement(root, "genre").text = artist.genre
        if artist.style:
            ET.SubElement(root, "style").text = artist.style
        if artist.mood:
            ET.SubElement(root, "mood").text = artist.mood

        # Biography
        if self.config.include_bios and artist.biography:
            ET.SubElement(root, "biography").text = artist.biography

        # MusicBrainz ID
        if artist.musicbrainz_artist_id:
            ET.SubElement(root, "musicBrainzArtistID").text = artist.musicbrainz_artist_id

        # Type and country
        if artist.type:
            ET.SubElement(root, "type").text = artist.type
        if artist.country:
            ET.SubElement(root, "country").text = artist.country

        # Thumbnail/Fanart
        if artist.thumb:
            thumb_elem = ET.SubElement(root, "thumb")
            thumb_elem.set("preview", artist.thumb)
            thumb_elem.text = artist.thumb

        if artist.fanart:
            fanart_elem = ET.SubElement(root, "fanart")
            thumb_elem = ET.SubElement(fanart_elem, "thumb")
            thumb_elem.text = artist.fanart

        # Albums
        for album in artist.albums:
            album_elem = ET.SubElement(root, "album")
            ET.SubElement(album_elem, "title").text = album.get("title", "")
            if album.get("year"):
                ET.SubElement(album_elem, "year").text = str(album["year"])
            if album.get("musicbrainz_releasegroup_id"):
                ET.SubElement(album_elem, "musicBrainzReleaseGroupID").text = album["musicbrainz_releasegroup_id"]

        return self._to_xml(root)

    def generate_album_nfo(self, album: AlbumInfo) -> str:
        """Generate album NFO XML content.

        Args:
            album: Album information

        Returns:
            XML string for album.nfo file
        """
        root = ET.Element("album")

        # Basic info
        ET.SubElement(root, "title").text = album.title
        ET.SubElement(root, "artist").text = album.artist

        # Review
        if self.config.include_reviews and album.review:
            ET.SubElement(root, "review").text = album.review

        # MusicBrainz IDs
        if album.musicbrainz_releasegroup_id:
            ET.SubElement(root, "musicBrainzReleaseGroupID").text = album.musicbrainz_releasegroup_id
        if album.musicbrainz_release_id:
            ET.SubElement(root, "musicBrainzReleaseID").text = album.musicbrainz_release_id

        # Year and rating
        if album.year:
            ET.SubElement(root, "year").text = str(album.year)
        if self.config.include_ratings and album.rating:
            ET.SubElement(root, "rating").text = str(album.rating)

        # Genres
        if album.genre:
            ET.SubElement(root, "genre").text = album.genre
        if album.style:
            ET.SubElement(root, "style").text = album.style
        if self.config.include_moods and album.mood:
            ET.SubElement(root, "mood").text = album.mood

        # Tracks
        for track in album.tracks:
            track_elem = ET.SubElement(root, "track")
            ET.SubElement(track_elem, "position").text = str(track.position)
            ET.SubElement(track_elem, "title").text = track.title
            if track.duration:
                ET.SubElement(track_elem, "duration").text = str(track.duration)
            if track.musicbrainz_track_id:
                ET.SubElement(track_elem, "musicBrainzTrackID").text = track.musicbrainz_track_id

        return self._to_xml(root)

    def _to_xml(self, root: ET.Element) -> str:
        """Convert ElementTree to XML string.

        Args:
            root: Root XML element

        Returns:
            Formatted XML string
        """
        if self.config.indent_xml:
            # Pretty print with minidom for compatibility
            from xml.dom import minidom
            rough_string = ET.tostring(root, encoding="utf-8")
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
        else:
            return ET.tostring(root, encoding="utf-8").decode("utf-8")

    def write_artist_nfo(self, path: Path, artist: ArtistInfo) -> None:
        """Write artist.nfo to disk.

        Args:
            path: Destination path for artist.nfo
            artist: Artist information
        """
        content = self.generate_artist_nfo(artist)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.debug(f"Wrote artist.nfo to {path}")

    def write_album_nfo(self, path: Path, album: AlbumInfo) -> None:
        """Write album.nfo to disk.

        Args:
            path: Destination path for album.nfo
            album: Album information
        """
        content = self.generate_album_nfo(album)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.debug(f"Wrote album.nfo to {path}")

    @staticmethod
    def create_artist_info(name: str, metadata: Optional[Dict[str, Any]] = None) -> ArtistInfo:
        """Create ArtistInfo from metadata dictionary.

        Args:
            name: Artist name
            metadata: Optional metadata dictionary

        Returns:
            ArtistInfo instance
        """
        metadata = metadata or {}
        return ArtistInfo(
            name=name,
            musicbrainz_artist_id=metadata.get("musicbrainz_artist_id"),
            type=metadata.get("type"),
            country=metadata.get("country"),
            formed=metadata.get("formed"),
            disbanded=metadata.get("disbanded"),
            genre=metadata.get("genre") or metadata.get("genre_name"),
            style=metadata.get("style"),
            mood=metadata.get("mood"),
            biography=metadata.get("biography") or metadata.get("bio"),
            thumb=metadata.get("thumb"),
            fanart=metadata.get("fanart"),
            albums=metadata.get("albums", [])
        )

    @staticmethod
    def create_album_info(
        title: str,
        artist: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AlbumInfo:
        """Create AlbumInfo from metadata dictionary.

        Args:
            title: Album title
            artist: Artist name
            metadata: Optional metadata dictionary

        Returns:
            AlbumInfo instance
        """
        metadata = metadata or {}

        # Convert tracks to TrackInfo objects
        tracks = []
        for track_data in metadata.get("tracks", []):
            tracks.append(TrackInfo(
                position=track_data.get("position", 0),
                title=track_data.get("title", ""),
                duration=track_data.get("duration"),
                musicbrainz_track_id=track_data.get("musicbrainz_track_id")
            ))

        return AlbumInfo(
            title=title,
            artist=artist,
            year=metadata.get("year"),
            genre=metadata.get("genre") or metadata.get("genre_name"),
            style=metadata.get("style"),
            mood=metadata.get("mood"),
            review=metadata.get("review"),
            rating=metadata.get("rating"),
            musicbrainz_release_id=metadata.get("musicbrainz_release_id"),
            musicbrainz_releasegroup_id=metadata.get("musicbrainz_releasegroup_id"),
            tracks=tracks
        )
