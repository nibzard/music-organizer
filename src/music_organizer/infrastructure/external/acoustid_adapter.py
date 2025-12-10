"""
AcoustId Adapter - Anti-Corruption Layer for AcoustId API.

This adapter isolates the domain from the AcoustId API dependency,
providing a clean interface for acoustic fingerprinting.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


class AcoustIdAdapter:
    """
    Adapter for AcoustId web service API.

    This adapter provides acoustic fingerprint lookup functionality
    while protecting the domain from API-specific details.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.acoustid.org/v2/lookup",
        timeout: int = 10
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    async def lookup_by_fingerprint(
        self,
        fingerprint: str,
        duration: int
    ) -> Optional[Dict[str, Any]]:
        """
        Look up metadata by acoustic fingerprint.
        """
        if not self.api_key:
            print("AcoustId API key required for fingerprint lookup")
            return None

        params = {
            "format": "json",
            "client": self.api_key,
            "fingerprint": fingerprint,
            "duration": duration,
            "meta": "recordings+releasegroups+releases+tracks+artists+usermeta"
        }

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "ok" and data.get("results"):
                            # Return the best match (highest score)
                            best_result = max(
                                data["results"],
                                key=lambda r: r.get("score", 0)
                            )
                            return best_result
                    else:
                        print(f"AcoustId API error: {response.status}")
        except Exception as e:
            print(f"Error looking up fingerprint with AcoustId: {e}")

        return None

    async def lookup_by_metadata(
        self,
        title: str,
        artist: str,
        duration: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Look up metadata by title and artist (metadata-based search).
        """
        if not self.api_key:
            print("AcoustId API key required for metadata lookup")
            return None

        params = {
            "format": "json",
            "client": self.api_key,
            "meta": "recordings+releasegroups+releases+tracks+artists"
        }

        # Add title and artist to query
        query_parts = [f'title:"{title}"', f'artist:"{artist}"']
        params["title"] = title
        params["artist"] = artist

        if duration:
            params["duration"] = duration

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "ok" and data.get("results"):
                            return data["results"]
                    else:
                        print(f"AcoustId API error: {response.status}")
        except Exception as e:
            print(f"Error looking up metadata with AcoustId: {e}")

        return None

    def extract_metadata(self, acoustid_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant metadata from AcoustId response.
        """
        extracted = {}

        if "recordings" in acoustid_data and acoustid_data["recordings"]:
            recording = acoustid_data["recordings"][0]

            # Title
            if "title" in recording:
                extracted["title"] = recording["title"]

            # Artists
            if "artists" in recording:
                extracted["artists"] = [artist["name"] for artist in recording["artists"]]

            # Release information
            if "releasegroups" in recording and recording["releasegroups"]:
                release_group = recording["releasegroups"][0]

                # Album title
                if "title" in release_group:
                    extracted["album"] = release_group["title"]

                # Release year
                if "releases" in release_group and release_group["releases"]:
                    releases = release_group["releases"]
                    # Get the earliest release date
                    dates = []
                    for release in releases:
                        if "date" in release:
                            year = self._extract_year(release["date"])
                            if year:
                                dates.append(year)
                    if dates:
                        extracted["year"] = min(dates)

            # Track number
            if "tracks" in recording and recording["tracks"]:
                track = recording["tracks"][0]
                if "position" in track:
                    extracted["track_number"] = str(track["position"])

            # Duration
            if "duration" in recording:
                extracted["duration_seconds"] = recording["duration"]

            # Genres from tags
            if "tags" in recording:
                # Extract most common genre tags
                genre_counts = {}
                for tag in recording["tags"]:
                    name = tag.get("name", "").lower()
                    if name:
                        genre_counts[name] = genre_counts.get(name, 0) + tag.get("count", 1)

                if genre_counts:
                    # Get the most common genre
                    best_genre = max(genre_counts.items(), key=lambda x: x[1])[0]
                    extracted["genre"] = best_genre

        # Include the AcoustId score for confidence
        extracted["acoustid_score"] = acoustid_data.get("score", 0)

        return extracted

    async def submit_fingerprint(
        self,
        fingerprint: str,
        duration: int,
        title: str,
        artist: str,
        release: Optional[str] = None
    ) -> bool:
        """
        Submit a new fingerprint to AcoustId.

        This requires an API key with submission privileges.
        """
        if not self.api_key:
            print("AcoustId API key required for fingerprint submission")
            return False

        # This is a simplified implementation
        # In practice, you'd use the /submit endpoint
        # which requires more complex handling
        print("Fingerprint submission not implemented in this adapter")
        return False

    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string."""
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return int(year_match.group())
        return None

    async def batch_lookup(
        self,
        fingerprints: List[Tuple[str, int]]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Look up multiple fingerprints concurrently.
        """
        tasks = [
            self.lookup_by_fingerprint(fingerprint, duration)
            for fingerprint, duration in fingerprints
        ]
        return await asyncio.gather(*tasks)