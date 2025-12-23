"""
Extended MusicBrainz adapter for MBID and rich metadata retrieval.

This module extends the base MusicBrainz adapter to fetch:
- MusicBrainz IDs (MBID) for artists, releases, and tracks
- Artist biographies
- Album reviews
- Extended metadata for NFO generation
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Any

from .musicbrainz_adapter import MusicBrainzAdapter

logger = logging.getLogger(__name__)


class MusicBrainzMbidAdapter:
    """Extended MusicBrainz adapter for MBID and rich metadata."""

    def __init__(
        self,
        base_url: str = "https://musicbrainz.org/ws/2",
        user_agent: str = "MusicOrganizer/1.0 (https://github.com/nibzard/music-organizer)",
        rate_limit: float = 1.0,
        timeout: int = 10
    ):
        self.base_url = base_url
        self.user_agent = user_agent
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request = 0
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._base_adapter = MusicBrainzAdapter(
            base_url=base_url,
            user_agent=user_agent,
            rate_limit=rate_limit,
            timeout=timeout
        )

    async def _rate_limit(self) -> None:
        """Apply rate limiting."""
        current = time.time()
        if current - self._last_request < 1.0 / self.rate_limit:
            await asyncio.sleep(1.0 / self.rate_limit - (current - self._last_request))
        self._last_request = time.time()

    async def get_artist_with_mbid(self, artist: str) -> Optional[Dict[str, Any]]:
        """Get artist info including MBID and extended metadata.

        Args:
            artist: Artist name to search

        Returns:
            Dictionary with artist metadata including MBID, or None
        """
        cache_key = f"artist_mbid:{artist}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        await self._rate_limit()

        url = f"{self.base_url}/artist/"
        params = {
            "query": f'artist:"{artist}"',
            "limit": 1,
            "fmt": "json"
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
                            a = data["artists"][0]
                            result = {
                                "name": a.get("name"),
                                "musicbrainz_artist_id": a.get("id"),
                                "type": a.get("type"),
                                "country": a.get("country"),
                                "formed": None,
                                "disbanded": None,
                                "genre": None,
                                "style": None,
                                "biography": None
                            }

                            # Extract life span dates
                            if a.get("life-span"):
                                life_span = a["life-span"]
                                result["formed"] = life_span.get("begin")
                                result["disbanded"] = life_span.get("end")

                            # Extract genre from tags if available
                            if a.get("tags"):
                                tags = sorted(
                                    a["tags"],
                                    key=lambda t: t.get("count", 0),
                                    reverse=True
                                )
                                if tags:
                                    result["genre"] = tags[0].get("name")

                            self._cache[cache_key] = result
                            return result
        except Exception as e:
            logger.debug(f"Error fetching artist MBID: {e}")

        return None

    async def get_release_with_mbid(self, release_id: str) -> Optional[Dict[str, Any]]:
        """Get release info with full MBIDs and track listing.

        Args:
            release_id: MusicBrainz release ID

        Returns:
            Dictionary with release metadata including MBIDs, or None
        """
        cache_key = f"release_mbid:{release_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        await self._rate_limit()

        url = f"{self.base_url}/release/{release_id}"
        params = {
            "inc": "recordings+artist-credits+release-groups",
            "fmt": "json"
        }

        try:
            async with aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent},
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Extract tracks from media
                        tracks = []
                        for medium in data.get("media", []):
                            for track in medium.get("tracks", []):
                                track_info = {
                                    "position": track.get("position"),
                                    "number": track.get("number"),
                                    "title": track.get("title"),
                                    "musicbrainz_track_id": track.get("id"),
                                    "duration": (track.get("length", 0) // 1000)
                                    if track.get("length") else None
                                }
                                tracks.append(track_info)

                        result = {
                            "id": data.get("id"),
                            "title": data.get("title"),
                            "musicbrainz_release_id": data.get("id"),
                            "musicbrainz_releasegroup_id": data.get("release-group", {}).get("id"),
                            "date": data.get("date"),
                            "country": data.get("country"),
                            "tracks": tracks
                        }

                        self._cache[cache_key] = result
                        return result
        except Exception as e:
            logger.debug(f"Error fetching release MBID: {e}")

        return None

    async def find_release_by_album_artist(
        self,
        album: str,
        artist: str,
        year: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Find a release by album and artist, then fetch full details.

        Args:
            album: Album title
            artist: Artist name
            year: Optional year for disambiguation

        Returns:
            Dictionary with release metadata including MBIDs, or None
        """
        # First search for the release
        search_result = await self._base_adapter.search_release(album, artist, year)
        if not search_result:
            return None

        # Get the release ID and fetch full details
        release_id = search_result.get("id")
        if release_id:
            return await self.get_release_with_mbid(release_id)

        return None

    async def get_artist_albums(self, artist: str) -> List[Dict[str, Any]]:
        """Get list of albums for an artist.

        Args:
            artist: Artist name

        Returns:
            List of album dictionaries with basic info
        """
        # First get artist MBID
        artist_info = await self.get_artist_with_mbid(artist)
        if not artist_info or not artist_info.get("musicbrainz_artist_id"):
            return []

        artist_id = artist_info["musicbrainz_artist_id"]
        cache_key = f"artist_albums:{artist_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        await self._rate_limit()

        url = f"{self.base_url}/release/"
        params = {
            "query": f'arid:{artist_id}',
            "fmt": "json",
            "limit": 100
        }

        try:
            async with aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent},
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        albums = []
                        seen = set()

                        for release in data.get("releases", []):
                            # Use release-group for album uniqueness
                            rg_id = release.get("release-group", {}).get("id")
                            if not rg_id or rg_id in seen:
                                continue
                            seen.add(rg_id)

                            album_info = {
                                "title": release.get("title"),
                                "date": release.get("date"),
                                "year": self._extract_year(release.get("date")),
                                "musicbrainz_release_id": release.get("id"),
                                "musicbrainz_releasegroup_id": rg_id
                            }
                            albums.append(album_info)

                        self._cache[cache_key] = albums
                        return albums
        except Exception as e:
            logger.debug(f"Error fetching artist albums: {e}")

        return []

    def _extract_year(self, date_str: Optional[str]) -> Optional[int]:
        """Extract year from date string.

        Args:
            date_str: Date string (e.g., "2023-05-15")

        Returns:
            Year as integer, or None
        """
        if not date_str:
            return None
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return int(year_match.group())
        return None

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
