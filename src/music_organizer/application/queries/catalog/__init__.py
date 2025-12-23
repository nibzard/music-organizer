"""Catalog queries."""

from .recording_queries import (
    RecordingQueries,
    GetRecordingByIdQuery,
    GetRecordingsByArtistQuery,
    GetRecordingsByGenreQuery,
    SearchRecordingsQuery,
    GetDuplicateGroupsQuery
)
from .statistics_queries import (
    GetLibraryStatisticsQuery,
    GetArtistStatisticsQuery,
    GetGenreDistributionQuery
)

__all__ = [
    "RecordingQueries",
    "GetRecordingByIdQuery",
    "GetRecordingsByArtistQuery",
    "GetRecordingsByGenreQuery",
    "SearchRecordingsQuery",
    "GetDuplicateGroupsQuery",
    "GetLibraryStatisticsQuery",
    "GetArtistStatisticsQuery",
    "GetGenreDistributionQuery",
]