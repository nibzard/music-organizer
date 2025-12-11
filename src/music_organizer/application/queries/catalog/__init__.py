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
    CatalogStatisticsQueries,
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
    "CatalogStatisticsQueries",
    "GetLibraryStatisticsQuery",
    "GetArtistStatisticsQuery",
    "GetGenreDistributionQuery",
]