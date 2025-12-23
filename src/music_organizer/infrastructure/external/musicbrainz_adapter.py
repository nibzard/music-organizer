"""
MusicBrainz Adapter - Anti-Corruption Layer for MusicBrainz API.

This adapter isolates the domain from the MusicBrainz API dependency,
providing a clean interface for metadata enhancement.
"""

import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any
from urllib.parse import quote

from ...domain.catalog.value_objects import Metadata, ArtistName


class MusicBrainzAdapter:
    """
    Adapter for MusicBrainz web service API.

    This adapter implements rate limiting, caching, and data transformation
    to protect the domain from external API details.
    """

    def __init__(
        self,
        base_url: str = "https://musicbrainz.org/ws/2",
        user_agent: str = "MusicOrganizer/1.0 (https://github.com/nibzard/music-organizer)",
        rate_limit: float = 1.0,  # requests per second
        timeout: int = 10
    ):
        self.base_url = base_url
        self.user_agent = user_agent
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request_time = 0
        self._cache: Dict[str, Dict[str, Any]] = {}

    async def search_recording(
        self,
        title: str,
        artist: str,
        release: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Search for a recording in MusicBrainz.

        Returns metadata if found, None otherwise.
        """
        cache_key = f"recording:{title}:{artist}:{release or ''}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        await self._rate_limit()

        # Build query
        query_parts = [f'recording:"{title}"', f'artist:"{artist}"']
        if release:
            query_parts.append(f'release:"{release}"')
        query = " AND ".join(query_parts)

        url = f"{self.base_url}/recording/"
        params = {
            "query": query,
            "fmt": "json",
            "limit": 5
        }

        try:
            async with aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent},
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("recordings"):
                            # Get the best match
                            best_match = self._select_best_recording_match(
                                data["recordings"],
                                title,
                                artist,
                                release
                            )
                            if best_match:
                                self._cache[cache_key] = best_match
                                return best_match
        except Exception as e:
            print(f"Error searching MusicBrainz: {e}")

        return None

    async def search_release(
        self,
        title: str,
        artist: str,
        year: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Search for a release in MusicBrainz.
        """
        cache_key = f"release:{title}:{artist}:{year or ''}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        await self._rate_limit()

        query_parts = [f'release:"{title}"', f'artist:"{artist}"']
        if year:
            query_parts.append(f'date:{year}')
        query = " AND ".join(query_parts)

        url = f"{self.base_url}/release/"
        params = {
            "query": query,
            "fmt": "json",
            "limit": 5
        }

        try:
            async with aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent},
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("releases"):
                            best_match = self._select_best_release_match(
                                data["releases"],
                                title,
                                artist,
                                year
                            )
                            if best_match:
                                self._cache[cache_key] = best_match
                                return best_match
        except Exception as e:
            print(f"Error searching MusicBrainz release: {e}")

        return None

    async def search_artist(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Search for an artist in MusicBrainz.
        """
        cache_key = f"artist:{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        await self._rate_limit()

        url = f"{self.base_url}/artist/"
        params = {
            "query": f'artist:"{name}"',
            "fmt": "json",
            "limit": 5
        }

        try:
            async with aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent},
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("artists"):
                            # Get exact match or first result
                            for artist_data in data["artists"]:
                                if artist_data["name"].lower() == name.lower():
                                    self._cache[cache_key] = artist_data
                                    return artist_data
                            # Return first result if no exact match
                            self._cache[cache_key] = data["artists"][0]
                            return data["artists"][0]
        except Exception as e:
            print(f"Error searching MusicBrainz artist: {e}")

        return None

    def to_metadata(self, mb_data: Dict[str, Any]) -> Metadata:
        """
        Convert MusicBrainz data to domain Metadata.
        """
        metadata_updates = {}

        # Extract basic information
        if "title" in mb_data:
            metadata_updates["title"] = mb_data["title"]

        # Extract artists
        if "artist-credit" in mb_data:
            artists = []
            for credit in mb_data["artist-credit"]:
                if "artist" in credit:
                    artists.append(ArtistName(credit["artist"]["name"]))
            metadata_updates["artists"] = artists

        # Extract release information
        if "releases" in mb_data and mb_data["releases"]:
            release = mb_data["releases"][0]
            metadata_updates["album"] = release.get("title")
            if "date" in release:
                # Extract year from date
                year = self._extract_year(release["date"])
                if year:
                    metadata_updates["year"] = year

        # Extract other metadata
        if "tag-list" in mb_data:
            # Use first genre tag as genre
            for tag in mb_data.get("tag-list", []):
                if tag.get("count", 0) > 0:
                    metadata_updates["genre"] = tag["name"]
                    break

        return metadata_updates

    async def _rate_limit(self) -> None:
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 1.0 / self.rate_limit

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self._last_request_time = time.time()

    def _select_best_recording_match(
        self,
        recordings: List[Dict[str, Any]],
        title: str,
        artist: str,
        release: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Select the best recording match from results.
        """
        if not recordings:
            return None

        # Score each recording
        best_score = 0
        best_match = None

        for recording in recordings:
            score = 0

            # Title match
            if recording.get("title", "").lower() == title.lower():
                score += 40

            # Artist match
            if "artist-credit" in recording:
                for credit in recording["artist-credit"]:
                    if credit.get("artist", {}).get("name", "").lower() == artist.lower():
                        score += 35
                        break

            # Release match
            if release and "releases" in recording:
                for rec_release in recording["releases"]:
                    if rec_release.get("title", "").lower() == release.lower():
                        score += 25
                        break

            if score > best_score:
                best_score = score
                best_match = recording

        return best_match

    def _select_best_release_match(
        self,
        releases: List[Dict[str, Any]],
        title: str,
        artist: str,
        year: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Select the best release match from results.
        """
        if not releases:
            return None

        # Score each release
        best_score = 0
        best_match = None

        for release in releases:
            score = 0

            # Title match
            if release.get("title", "").lower() == title.lower():
                score += 40

            # Artist match
            if "artist-credit" in release:
                for credit in release["artist-credit"]:
                    if credit.get("artist", {}).get("name", "").lower() == artist.lower():
                        score += 35
                        break

            # Year match
            if year and "date" in release:
                release_year = self._extract_year(release["date"])
                if release_year == year:
                    score += 25

            if score > best_score:
                best_score = score
                best_match = release

        return best_match

    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string."""
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return int(year_match.group())
        return None