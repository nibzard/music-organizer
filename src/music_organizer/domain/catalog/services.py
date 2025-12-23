"""Catalog Context Domain Services.

This module defines domain services for the Catalog bounded context.
Domain services contain business logic that doesn't naturally fit in entities or value objects.
"""

from typing import List, Optional, Dict, Any, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..result import Result, success, failure, collect, MetadataError, DuplicateError, NotFoundError
from .entities import Recording, Release, Artist, Catalog
from .value_objects import ArtistName, Metadata, AudioPath
from .repositories import RecordingRepository, ReleaseRepository, ArtistRepository


class MetadataService:
    """Service for handling metadata operations."""

    def __init__(self, recording_repo: RecordingRepository):
        self.recording_repo = recording_repo
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def enhance_metadata(self, recording: Recording, enhanced_metadata: Metadata) -> Result[Recording, MetadataError]:
        """Enhance a recording's metadata with additional information."""
        try:
            # Create new metadata with enhanced fields
            # Note: total_tracks is not in the current Metadata model
            updated_metadata = Metadata(
                title=enhanced_metadata.title or recording.metadata.title,
                artists=enhanced_metadata.artists or recording.metadata.artists,
                album=enhanced_metadata.album or recording.metadata.album,
                year=enhanced_metadata.year or recording.metadata.year,
                genre=enhanced_metadata.genre or recording.metadata.genre,
                track_number=enhanced_metadata.track_number or recording.metadata.track_number,
                # total_tracks not available in current Metadata model
                disc_number=enhanced_metadata.disc_number or recording.metadata.disc_number,
                total_discs=enhanced_metadata.total_discs or recording.metadata.total_discs,
                albumartist=enhanced_metadata.albumartist or recording.metadata.albumartist,
                composer=enhanced_metadata.composer or recording.metadata.composer,
                # Preserve technical metadata
                duration_seconds=recording.metadata.duration_seconds,
                bitrate=recording.metadata.bitrate,
                sample_rate=recording.metadata.sample_rate,
                channels=recording.metadata.channels,
                # Preserve live/recording metadata
                date=recording.metadata.date,
                location=recording.metadata.location,
            )

            # Update recording with new metadata
            recording.metadata = updated_metadata

            # Save changes
            await self.recording_repo.save(recording)

            return success(recording)
        except Exception as e:
            return failure(MetadataError(f"Failed to enhance metadata: {e}"))

    async def batch_enhance_metadata(
        self,
        recordings: List[Recording],
        enhancements: List[Metadata]
    ) -> Result[List[Recording], MetadataError]:
        """Enhance metadata for multiple recordings in parallel."""
        if len(recordings) != len(enhancements):
            return failure(MetadataError("Recordings and enhancements must have same length"))

        tasks = [
            self.enhance_metadata(recording, enhancement)
            for recording, enhancement in zip(recordings, enhancements)
        ]

        results = await asyncio.gather(*tasks)
        return collect(results).map_error(lambda errors: MetadataError(f"Multiple errors: {errors}"))

    async def normalize_artist_names(self, recordings: List[Recording]) -> List[Recording]:
        """Normalize artist names across multiple recordings."""
        # Build artist name mapping (variations -> canonical)
        artist_variations = {}

        for recording in recordings:
            for artist in recording.artists:
                normalized = str(artist).strip().title()
                key = normalized.lower().replace("the ", "").replace(" ", "")

                if key not in artist_variations:
                    artist_variations[key] = artist
                else:
                    # Use the longest name as canonical (includes "The" etc.)
                    if len(str(artist)) > len(str(artist_variations[key])):
                        artist_variations[key] = artist

        # Update recordings with normalized artist names
        updated_recordings = []
        for recording in recordings:
            updated_artists = []
            for artist in recording.artists:
                key = str(artist).lower().replace("the ", "").replace(" ", "")
                canonical = artist_variations.get(key, artist)
                updated_artists.append(canonical)

            recording.metadata.artists = updated_artists
            updated_recordings.append(recording)

            await self.recording_repo.save(recording)

        return updated_recordings

    async def infer_missing_metadata(self, recording: Recording) -> Recording:
        """Infer missing metadata from available information."""
        # Infer year from other tracks in the same album
        if not recording.metadata.year and recording.metadata.album:
            similar = await self._find_similar_album_tracks(recording)
            for other in similar:
                if other.metadata.year:
                    recording.metadata.year = other.metadata.year
                    break

        # Infer album artist from track artist if missing
        if not recording.metadata.albumartist and recording.artists:
            recording.metadata.albumartist = recording.primary_artist

        # Infer genre from artist's other tracks
        if not recording.metadata.genre:
            artist_genres = await self._get_artist_genres(recording.primary_artist)
            if artist_genres:
                # Use most common genre
                recording.metadata.genre = max(set(artist_genres), key=artist_genres.count)

        await self.recording_repo.save(recording)
        return recording

    async def _find_similar_album_tracks(self, recording: Recording) -> List[Recording]:
        """Find other tracks from the same album."""
        if not recording.metadata.album:
            return []

        # This would typically query the repository
        # For now, return empty list
        return []

    async def _get_artist_genres(self, artist: ArtistName) -> List[str]:
        """Get all genres associated with an artist."""
        # This would typically query the repository
        # For now, return empty list
        return []


class CatalogService:
    """Service for catalog operations."""

    def __init__(
        self,
        recording_repo: RecordingRepository,
        release_repo: ReleaseRepository,
        artist_repo: ArtistRepository
    ):
        self.recording_repo = recording_repo
        self.release_repo = release_repo
        self.artist_repo = artist_repo

    async def add_recording_to_catalog(self, catalog: Catalog, recording: Recording) -> Result[None, DuplicateError]:
        """Add a recording to the catalog, maintaining invariants."""
        try:
            # Check for duplicates
            existing = await self.recording_repo.find_by_path(recording.path)
            if existing:
                return failure(DuplicateError(f"Recording already exists at path: {recording.path}"))

            # Find or create artist - artists are in metadata, not directly on Recording
            for artist_name in recording.metadata.artists:
                # artist_name is an ArtistName value object, use .name property
                artist_str = artist_name.name
                artist = await self.artist_repo.find_by_name(artist_str)
                if not artist:
                    artist = Artist(name=artist_str)
                    await self.artist_repo.save(artist)

            # Find or create release
            if recording.metadata.album:
                # Use albumartist if available, otherwise use first artist
                if recording.metadata.albumartist:
                    primary_artist_str = recording.metadata.albumartist.name
                elif recording.metadata.artists:
                    primary_artist_str = list(recording.metadata.artists)[0].name
                else:
                    primary_artist_str = "Unknown"

                release = await self.release_repo.find_by_title_and_artist(
                    recording.metadata.album,
                    primary_artist_str
                )
                if not release:
                    release = Release(
                        title=recording.metadata.album,
                        primary_artist=primary_artist_str,
                        year=recording.metadata.year
                    )
                    await self.release_repo.save(release)

                release.add_recording(recording)
                await self.release_repo.save(release)

            # Add to catalog
            catalog.add_recording(recording)
            await self.recording_repo.save(recording)

            return success(None)
        except Exception as e:
            return failure(DuplicateError(f"Failed to add recording to catalog: {e}"))

    async def remove_recording_from_catalog(self, catalog: Catalog, recording: Recording) -> None:
        """Remove a recording from the catalog, cleaning up references."""
        # Remove from releases
        for release in catalog.releases.values():
            if recording in release.recordings:
                release.remove_recording(recording)
                await self.release_repo.save(release)

                # If release is empty, consider removing it
                if not release.recordings:
                    await self.release_repo.delete(release)

        # Remove from artists (but don't delete artist entities)
        for artist in catalog.artists.values():
            if recording in artist.recordings:
                artist.recordings.remove(recording)

        # Remove from catalog
        catalog.recordings.remove(recording)
        await self.recording_repo.delete(recording)

    async def merge_duplicate_releases(self, catalog: Catalog, threshold: float = 0.9) -> int:
        """Find and merge duplicate releases."""
        merged_count = 0
        processed = set()

        releases = list(catalog.releases.values())

        for i, release1 in enumerate(releases):
            if release1 in processed:
                continue

            for release2 in releases[i+1:]:
                if release2 in processed:
                    continue

                similarity = self._calculate_release_similarity(release1, release2)
                if similarity >= threshold:
                    # Merge release2 into release1
                    release1.merge_with(release2)

                    # Update recordings to point to release1
                    for recording in release2.recordings:
                        if recording.metadata.album == release2.title:
                            recording.metadata.album = release1.title
                            await self.recording_repo.save(recording)

                    await self.release_repo.save(release1)
                    await self.release_repo.delete(release2)

                    processed.add(release2)
                    merged_count += 1

        return merged_count

    async def reorganize_by_genre(self, catalog: Catalog) -> Dict[str, List[Release]]:
        """Organize releases by genre."""
        genre_groups: Dict[str, List[Release]] = {}

        for release in catalog.releases.values():
            # Get genre from release or infer from tracks
            genre = release.genre
            if not genre:
                genres = [track.metadata.genre for track in release.recordings if track.metadata.genre]
                genre = max(set(genres), key=genres.count) if genres else "Unknown"

            if genre not in genre_groups:
                genre_groups[genre] = []
            genre_groups[genre].append(release)

        return genre_groups

    def _calculate_release_similarity(self, release1: Release, release2: Release) -> float:
        """Calculate similarity between two releases."""
        similarity = 0.0

        # Title similarity (50% weight)
        if release1.title and release2.title:
            title_sim = self._string_similarity(release1.title.lower(), release2.title.lower())
            similarity += title_sim * 0.5

        # Artist similarity (30% weight)
        if release1.primary_artist and release2.primary_artist:
            artist_sim = 1.0 if release1.primary_artist == release2.primary_artist else 0.0
            similarity += artist_sim * 0.3

        # Year similarity (20% weight)
        if release1.year and release2.year:
            year_sim = 1.0 if release1.year == release2.year else 0.0
            similarity += year_sim * 0.2

        return similarity

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Jaccard similarity."""
        words1 = set(s1.split())
        words2 = set(s2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0