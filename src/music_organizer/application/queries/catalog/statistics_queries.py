"""Statistics-related queries."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

from ...queries.base import Query, QueryHandler, QueryResult
from ....domain.catalog.repositories import RecordingRepository, CatalogRepository


@dataclass(frozen=True, slots=True, kw_only=True)
class GetLibraryStatisticsQuery(Query):
    """Query to get overall library statistics."""

    catalog_id: str = "default"
    include_detailed_breakdown: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class GetArtistStatisticsQuery(Query):
    """Query to get artist-specific statistics."""

    artist_name: str
    include_releases: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class GetGenreDistributionQuery(Query):
    """Query to get genre distribution in the library."""

    catalog_id: str = "default"
    min_count: int = 1


@dataclass
class LibraryStatistics:
    """Comprehensive library statistics."""

    total_recordings: int
    total_artists: int
    total_releases: int
    total_size_gb: float
    total_duration_hours: float
    format_distribution: Dict[str, int]
    genre_distribution: Dict[str, int]
    decade_distribution: Dict[str, int]
    top_artists: List[Tuple[str, int]]
    top_genres: List[Tuple[str, int]]
    recently_added: int  # Count of recordings added in last 30 days
    duplicates_count: int
    average_bitrate: float
    quality_distribution: Dict[str, int]  # Based on bitrate/quality


@dataclass
class ArtistStatistics:
    """Statistics for a specific artist."""

    artist_name: str
    total_recordings: int
    total_releases: int
    total_duration_hours: float
    genres: List[str]
    decade_distribution: Dict[str, int]
    average_year: Optional[float]
    top_releases: List[Tuple[str, int]]  # Release name and track count
    collaborations: List[str]


class GetLibraryStatisticsHandler(QueryHandler[GetLibraryStatisticsQuery, LibraryStatistics]):
    """Handler for getting library statistics."""

    def __init__(self, recording_repo: RecordingRepository, catalog_repo: CatalogRepository):
        self.recording_repo = recording_repo
        self.catalog_repo = catalog_repo

    async def handle(self, query: GetLibraryStatisticsQuery) -> LibraryStatistics:
        """Handle the get library statistics query."""
        # Get all recordings
        recordings = [r async for r in self.recording_repo.find_all()]

        if not recordings:
            return LibraryStatistics(
                total_recordings=0,
                total_artists=0,
                total_releases=0,
                total_size_gb=0.0,
                total_duration_hours=0.0,
                format_distribution={},
                genre_distribution={},
                decade_distribution={},
                top_artists=[],
                top_genres=[],
                recently_added=0,
                duplicates_count=0,
                average_bitrate=0.0,
                quality_distribution={}
            )

        # Calculate statistics
        total_size_mb = sum(r.path.size_mb for r in recordings)
        total_duration_seconds = sum(r.metadata.duration_seconds or 0 for r in recordings)

        # Format distribution
        format_distribution = {}
        for recording in recordings:
            format_name = recording.path.format.name
            format_distribution[format_name] = format_distribution.get(format_name, 0) + 1

        # Genre distribution
        genre_distribution = {}
        for recording in recordings:
            if recording.metadata.genre:
                genres = [g.strip() for g in recording.metadata.genre.split(",")]
                for genre in genres:
                    if genre:
                        genre_distribution[genre] = genre_distribution.get(genre, 0) + 1

        # Decade distribution
        decade_distribution = {}
        for recording in recordings:
            if recording.metadata.year:
                decade = f"{(recording.metadata.year // 10) * 10}s"
                decade_distribution[decade] = decade_distribution.get(decade, 0) + 1

        # Artist counts
        artist_counts = {}
        for recording in recordings:
            for artist in recording.metadata.artists:
                artist_name = str(artist)
                artist_counts[artist_name] = artist_counts.get(artist_name, 0) + 1

        total_artists = len(artist_counts)
        top_artists = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Top genres
        top_genres = sorted(genre_distribution.items(), key=lambda x: x[1], reverse=True)[:10]

        # Recently added (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recently_added = sum(
            1 for r in recordings
            if r.created_at and r.created_at > thirty_days_ago
        )

        # Quality distribution based on bitrate
        quality_distribution = {}
        bitrates = []
        for recording in recordings:
            if recording.metadata.bitrate:
                bitrates.append(recording.metadata.bitrate)
                if recording.metadata.bitrate >= 320:
                    quality = "High (320+ kbps)"
                elif recording.metadata.bitrate >= 256:
                    quality = "Good (256-319 kbps)"
                elif recording.metadata.bitrate >= 192:
                    quality = "Standard (192-255 kbps)"
                elif recording.metadata.bitrate >= 128:
                    quality = "Low (128-191 kbps)"
                else:
                    quality = "Very Low (<128 kbps)"
                quality_distribution[quality] = quality_distribution.get(quality, 0) + 1

        average_bitrate = sum(bitrates) / len(bitrates) if bitrates else 0

        # Count unique releases
        release_count = len(set(r.metadata.album for r in recordings if r.metadata.album))

        return LibraryStatistics(
            total_recordings=len(recordings),
            total_artists=total_artists,
            total_releases=release_count,
            total_size_gb=total_size_mb / 1024,
            total_duration_hours=total_duration_seconds / 3600,
            format_distribution=format_distribution,
            genre_distribution=genre_distribution,
            decade_distribution=decade_distribution,
            top_artists=top_artists,
            top_genres=top_genres,
            recently_added=recently_added,
            duplicates_count=0,  # Would need to calculate duplicates
            average_bitrate=average_bitrate,
            quality_distribution=quality_distribution
        )

    def can_handle(self, query_type: type) -> bool:
        """Check if this handler can handle the given query type."""
        return query_type == GetLibraryStatisticsQuery

    def get_cache_key(self, query: GetLibraryStatisticsQuery) -> Optional[str]:
        """Generate cache key for library statistics."""
        return f"library_stats:{query.catalog_id}:{query.include_detailed_breakdown}"


class GetArtistStatisticsHandler(QueryHandler[GetArtistStatisticsQuery, ArtistStatistics]):
    """Handler for getting artist statistics."""

    def __init__(self, recording_repo: RecordingRepository):
        self.recording_repo = recording_repo

    async def handle(self, query: GetArtistStatisticsQuery) -> ArtistStatistics:
        """Handle the get artist statistics query."""
        from ....domain.value_objects import ArtistName

        artist_name = ArtistName(query.artist_name)
        recordings = await self.recording_repo.find_by_artist(artist_name)

        if not recordings:
            return ArtistStatistics(
                artist_name=query.artist_name,
                total_recordings=0,
                total_releases=0,
                total_duration_hours=0.0,
                genres=[],
                decade_distribution={},
                average_year=None,
                top_releases=[],
                collaborations=[]
            )

        # Calculate statistics
        total_duration_seconds = sum(r.metadata.duration_seconds or 0 for r in recordings)

        # Genres
        all_genres = set()
        for recording in recordings:
            if recording.metadata.genre:
                genres = [g.strip() for g in recording.metadata.genre.split(",")]
                all_genres.update(g for g in genres if g)

        # Decade distribution
        decade_distribution = {}
        years = []
        for recording in recordings:
            if recording.metadata.year:
                years.append(recording.metadata.year)
                decade = f"{(recording.metadata.year // 10) * 10}s"
                decade_distribution[decade] = decade_distribution.get(decade, 0) + 1

        average_year = sum(years) / len(years) if years else None

        # Top releases
        release_counts = {}
        for recording in recordings:
            if recording.metadata.album:
                album = recording.metadata.album
                release_counts[album] = release_counts.get(album, 0) + 1

        top_releases = sorted(release_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Collaborations (other artists appearing with this artist)
        collaborations = set()
        for recording in recordings:
            for artist in recording.metadata.artists:
                if str(artist) != query.artist_name:
                    collaborations.add(str(artist))

        return ArtistStatistics(
            artist_name=query.artist_name,
            total_recordings=len(recordings),
            total_releases=len(release_counts),
            total_duration_hours=total_duration_seconds / 3600,
            genres=sorted(list(all_genres)),
            decade_distribution=decade_distribution,
            average_year=average_year,
            top_releases=top_releases,
            collaborations=sorted(list(collaborations))
        )

    def can_handle(self, query_type: type) -> bool:
        """Check if this handler can handle the given query type."""
        return query_type == GetArtistStatisticsQuery

    def get_cache_key(self, query: GetArtistStatisticsQuery) -> Optional[str]:
        """Generate cache key for artist statistics."""
        return f"artist_stats:{query.artist_name}:{query.include_releases}"


class GetGenreDistributionHandler(QueryHandler[GetGenreDistributionQuery, Dict[str, int]]):
    """Handler for getting genre distribution."""

    def __init__(self, recording_repo: RecordingRepository):
        self.recording_repo = recording_repo

    async def handle(self, query: GetGenreDistributionQuery) -> Dict[str, int]:
        """Handle the get genre distribution query."""
        recordings = [r async for r in self.recording_repo.find_all()]

        genre_distribution = {}
        for recording in recordings:
            if recording.metadata.genre:
                genres = [g.strip() for g in recording.metadata.genre.split(",")]
                for genre in genres:
                    if genre:
                        genre_distribution[genre] = genre_distribution.get(genre, 0) + 1

        # Filter by minimum count
        if query.min_count > 1:
            genre_distribution = {
                genre: count for genre, count in genre_distribution.items()
                if count >= query.min_count
            }

        return genre_distribution

    def can_handle(self, query_type: type) -> bool:
        """Check if this handler can handle the given query type."""
        return query_type == GetGenreDistributionQuery

    def get_cache_key(self, query: GetGenreDistributionQuery) -> Optional[str]:
        """Generate cache key for genre distribution."""
        return f"genre_dist:{query.catalog_id}:{query.min_count}"