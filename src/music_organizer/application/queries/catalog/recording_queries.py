"""Recording-related queries."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from ...queries.base import Query, QueryHandler, QueryResult
from ....domain.catalog.entities import Recording
from ....domain.catalog.repositories import RecordingRepository
from ....domain.value_objects import ArtistName


@dataclass(frozen=True, slots=True, kw_only=True)
class GetRecordingByIdQuery(Query):
    """Query to get a recording by ID."""

    recording_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class GetRecordingsByArtistQuery(Query):
    """Query to get recordings by artist."""

    artist_name: str
    limit: Optional[int] = None
    offset: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class GetRecordingsByGenreQuery(Query):
    """Query to get recordings by genre."""

    genre: str
    limit: Optional[int] = None
    offset: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchRecordingsQuery(Query):
    """Query to search recordings by text."""

    search_term: str
    search_fields: List[str] = field(default_factory=lambda: ['title', 'artist', 'album'])
    limit: Optional[int] = None
    offset: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class GetDuplicateGroupsQuery(Query):
    """Query to get duplicate recording groups."""

    similarity_threshold: float = 0.85
    include_metadata_similarities: bool = True


@dataclass
class RecordingSearchResult:
    """Result of recording search with metadata."""

    recordings: List[Dict[str, Any]]
    total_count: int
    query_time_ms: float

    def __init__(self, recordings: List[Dict[str, Any]], total_count: int, query_time_ms: float):
        self.recordings = recordings
        self.total_count = total_count
        self.query_time_ms = query_time_ms


class RecordingQueries:
    """Service for recording-related queries."""

    def __init__(self, recording_repo: RecordingRepository):
        self.recording_repo = recording_repo


class GetRecordingByIdHandler(QueryHandler[GetRecordingByIdQuery, Optional[Recording]]):
    """Handler for getting recording by ID."""

    def __init__(self, recording_repo: RecordingRepository):
        self.recording_repo = recording_repo

    async def handle(self, query: GetRecordingByIdQuery) -> Optional[Recording]:
        """Handle the get recording by ID query."""
        return await self.recording_repo.find_by_id(query.recording_id)

    def can_handle(self, query_type: type) -> bool:
        """Check if this handler can handle the given query type."""
        return query_type == GetRecordingByIdQuery


class GetRecordingsByArtistHandler(QueryHandler[GetRecordingsByArtistQuery, List[Recording]]):
    """Handler for getting recordings by artist."""

    def __init__(self, recording_repo: RecordingRepository):
        self.recording_repo = recording_repo

    async def handle(self, query: GetRecordingsByArtistQuery) -> List[Recording]:
        """Handle the get recordings by artist query."""
        artist_name = ArtistName(query.artist_name)
        recordings = await self.recording_repo.find_by_artist(artist_name)

        # Apply pagination
        if query.offset > 0 or query.limit:
            start = query.offset
            end = start + query.limit if query.limit else None
            recordings = recordings[start:end]

        return recordings

    def can_handle(self, query_type: type) -> bool:
        """Check if this handler can handle the given query type."""
        return query_type == GetRecordingsByArtistQuery


class GetRecordingsByGenreHandler(QueryHandler[GetRecordingsByGenreQuery, List[Recording]]):
    """Handler for getting recordings by genre."""

    def __init__(self, recording_repo: RecordingRepository):
        self.recording_repo = recording_repo

    async def handle(self, query: GetRecordingsByGenreQuery) -> List[Recording]:
        """Handle the get recordings by genre query."""
        # Get all recordings and filter by genre
        all_recordings = [r async for r in self.recording_repo.find_all()]
        genre_lower = query.genre.lower()

        filtered_recordings = []
        for recording in all_recordings:
            if recording.metadata.genre and genre_lower in recording.metadata.genre.lower():
                filtered_recordings.append(recording)

        # Apply pagination
        if query.offset > 0 or query.limit:
            start = query.offset
            end = start + query.limit if query.limit else None
            filtered_recordings = filtered_recordings[start:end]

        return filtered_recordings

    def can_handle(self, query_type: type) -> bool:
        """Check if this handler can handle the given query type."""
        return query_type == GetRecordingsByGenreQuery


class SearchRecordingsHandler(QueryHandler[SearchRecordingsQuery, RecordingSearchResult]):
    """Handler for searching recordings."""

    def __init__(self, recording_repo: RecordingRepository):
        self.recording_repo = recording_repo

    async def handle(self, query: SearchRecordingsQuery) -> RecordingSearchResult:
        """Handle the search recordings query."""
        start_time = datetime.utcnow()
        search_term = query.search_term.lower()

        # Get all recordings and search
        all_recordings = [r async for r in self.recording_repo.find_all()]
        matched_recordings = []

        for recording in all_recordings:
            recording_dict = recording.to_dict()
            match_score = 0

            # Search in specified fields
            for field in query.search_fields:
                field_value = recording_dict.get(field, "")
                if field_value and search_term in str(field_value).lower():
                    # Simple scoring: exact match gets higher score
                    if search_term == str(field_value).lower():
                        match_score += 100
                    else:
                        match_score += 10

            if match_score > 0:
                recording_dict["match_score"] = match_score
                matched_recordings.append(recording_dict)

        # Sort by match score (descending)
        matched_recordings.sort(key=lambda r: r["match_score"], reverse=True)

        # Apply pagination
        total_count = len(matched_recordings)
        if query.offset > 0 or query.limit:
            start = query.offset
            end = start + query.limit if query.limit else None
            matched_recordings = matched_recordings[start:end]

        query_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return RecordingSearchResult(
            recordings=matched_recordings,
            total_count=total_count,
            query_time_ms=query_time
        )

    def can_handle(self, query_type: type) -> bool:
        """Check if this handler can handle the given query type."""
        return query_type == SearchRecordingsQuery

    def get_cache_key(self, query: SearchRecordingsQuery) -> Optional[str]:
        """Generate cache key for search query."""
        # Create a deterministic cache key
        fields_str = ",".join(sorted(query.search_fields))
        return f"search:{query.search_term}:{fields_str}:{query.limit}:{query.offset}"


class GetDuplicateGroupsHandler(QueryHandler[GetDuplicateGroupsQuery, List[List[Recording]]]):
    """Handler for getting duplicate recording groups."""

    def __init__(self, recording_repo: RecordingRepository):
        self.recording_repo = recording_repo

    async def handle(self, query: GetDuplicateGroupsQuery) -> List[List[Recording]]:
        """Handle the get duplicate groups query."""
        return await self.recording_repo.find_duplicates(
            threshold=query.similarity_threshold,
            include_metadata=query.include_metadata_similarities
        )

    def can_handle(self, query_type: type) -> bool:
        """Check if this handler can handle the given query type."""
        return query_type == GetDuplicateGroupsQuery

    def get_cache_key(self, query: GetDuplicateGroupsQuery) -> Optional[str]:
        """Generate cache key for duplicate groups query."""
        return f"duplicates:{query.similarity_threshold}:{query.include_metadata_similarities}"