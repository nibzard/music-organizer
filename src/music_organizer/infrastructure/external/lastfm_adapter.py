"""
Last.fm Adapter - Anti-Corruption Layer for Last.fm API.

This adapter isolates the domain from the Last.fm API dependency,
providing a clean interface for scrobbling and metadata enhancement.
"""

import asyncio
import hashlib
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode


class LastFmAdapter:
    """
    Adapter for Last.fm API v2.0.

    This adapter implements rate limiting, authentication, and data transformation
    to protect the domain from external API details.

    API Documentation: https://www.last.fm/api/intro
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        session_key: Optional[str] = None,
        base_url: str = "https://ws.audioscrobbler.com/2.0/",
        timeout: int = 10,
        rate_limit: float = 2.0
    ):
        """Initialize Last.fm adapter.

        Args:
            api_key: Last.fm API key
            api_secret: Last.fm API secret
            session_key: Optional authenticated session key
            base_url: Last.fm API base URL
            timeout: Request timeout in seconds
            rate_limit: Maximum requests per second
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_key = session_key
        self.base_url = base_url
        self.timeout = timeout
        self.rate_limit = rate_limit
        self._session: Optional[Any] = None
        self._last_request_time = 0.0

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            try:
                import aiohttp
            except ImportError:
                raise RuntimeError("aiohttp is required for Last.fm integration")

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 1.0 / self.rate_limit

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self._last_request_time = time.time()

    def _sign_params(self, params: Dict[str, str]) -> str:
        """Generate API signature for authenticated requests.

        Args:
            params: Parameters to sign

        Returns:
            MD5 hash of sorted parameters + secret
        """
        # Sort parameters alphabetically
        sorted_params = sorted(params.items())
        # Create string to sign
        to_sign = "".join(f"{k}{v}" for k, v in sorted_params) + self.api_secret
        return hashlib.md5(to_sign.encode()).hexdigest()

    def _parse_response(self, xml_text: str) -> Optional[ET.Element]:
        """Parse Last.fm XML response.

        Args:
            xml_text: XML response text

        Returns:
            Root element or None if parsing fails
        """
        try:
            return ET.fromstring(xml_text)
        except ET.ParseError as e:
            print(f"Error parsing Last.fm response: {e}")
            return None

    def _check_error(self, root: ET.Element) -> Optional[Dict[str, Any]]:
        """Check if Last.fm response contains an error.

        Args:
            root: Parsed XML root element

        Returns:
            Error dict with code and message, or None if no error
        """
        error = root.find("error")
        if error is not None:
            return {
                "code": error.get("code"),
                "message": error.text
            }
        return None

    async def scrobble(
        self,
        artist: str,
        track: str,
        timestamp: int,
        album: Optional[str] = None,
        track_number: Optional[int] = None,
        duration: Optional[int] = None
    ) -> bool:
        """Submit a scrobble to Last.fm.

        Args:
            artist: Artist name
            track: Track title
            timestamp: Unix timestamp of when track started playing
            album: Optional album name
            track_number: Optional track number
            duration: Optional track duration in seconds

        Returns:
            True if scrobble was successful
        """
        if not self.session_key:
            print("Last.fm session key required for scrobbling")
            return False

        await self._rate_limit()

        params = {
            "method": "track.scrobble",
            "artist": artist,
            "track": track,
            "timestamp": str(timestamp),
            "api_key": self.api_key,
            "sk": self.session_key
        }

        if album:
            params["album"] = album
        if track_number is not None:
            params["trackNumber"] = str(track_number)
        if duration is not None:
            params["duration"] = str(duration)

        params["api_sig"] = self._sign_params(params)

        try:
            session = await self._get_session()
            async with session.post(self.base_url, data=params) as response:
                if response.status == 200:
                    data = await response.text()
                    root = self._parse_response(data)
                    if root:
                        error = self._check_error(root)
                        if error:
                            print(f"Last.fm scrobble error: {error['code']} - {error['message']}")
                            return False
                        return True
                else:
                    print(f"Last.fm API error: {response.status}")
                    return False
        except Exception as e:
            print(f"Error scrobbling to Last.fm: {e}")
            return False

        return False

    async def update_now_playing(
        self,
        artist: str,
        track: str,
        album: Optional[str] = None,
        duration: Optional[int] = None
    ) -> bool:
        """Update the "now playing" status on Last.fm.

        Args:
            artist: Artist name
            track: Track title
            album: Optional album name
            duration: Optional track duration in seconds

        Returns:
            True if update was successful
        """
        if not self.session_key:
            return False

        await self._rate_limit()

        params = {
            "method": "track.updateNowPlaying",
            "artist": artist,
            "track": track,
            "api_key": self.api_key,
            "sk": self.session_key
        }

        if album:
            params["album"] = album
        if duration is not None:
            params["duration"] = str(duration)

        params["api_sig"] = self._sign_params(params)

        try:
            session = await self._get_session()
            async with session.post(self.base_url, data=params) as response:
                if response.status == 200:
                    data = await response.text()
                    root = self._parse_response(data)
                    if root:
                        error = self._check_error(root)
                        if error:
                            print(f"Last.fm now playing error: {error['code']} - {error['message']}")
                            return False
                        return True
                return False
        except Exception as e:
            print(f"Error updating now playing: {e}")
            return False

    async def get_artist_info(self, artist: str) -> Optional[Dict[str, Any]]:
        """Get artist information from Last.fm.

        Args:
            artist: Artist name

        Returns:
            Artist info dict or None
        """
        await self._rate_limit()

        params = {
            "method": "artist.getInfo",
            "artist": artist,
            "api_key": self.api_key
        }

        try:
            session = await self._get_session()
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.text()
                    root = self._parse_response(data)
                    if root:
                        error = self._check_error(root)
                        if error:
                            return None

                        artist_elem = root.find(".//artist")
                        if artist_elem is not None:
                            return self._parse_artist_info(artist_elem)
        except Exception as e:
            print(f"Error getting artist info: {e}")

        return None

    def _parse_artist_info(self, artist_elem: ET.Element) -> Dict[str, Any]:
        """Parse artist info from XML element.

        Args:
            artist_elem: Artist XML element

        Returns:
            Parsed artist info dict
        """
        info = {
            "name": artist_elem.findtext("name"),
            "mbid": artist_elem.findtext("mbid"),
            "url": artist_elem.findtext("url"),
        }

        # Parse stats
        stats = artist_elem.find("stats")
        if stats is not None:
            info["listeners"] = int(stats.findtext("listeners") or 0)
            info["playcount"] = int(stats.findtext("playcount") or 0)

        # Parse tags
        tags = artist_elem.find("tags")
        if tags is not None:
            info["tags"] = [
                tag.findtext("name")
                for tag in tags.findall("tag")
                if tag.findtext("name")
            ]

        # Parse similar artists
        similar = artist_elem.find("similar")
        if similar is not None:
            info["similar_artists"] = [
                artist.findtext("name")
                for artist in similar.findall("artist")
                if artist.findtext("name")
            ]

        return info

    async def get_album_info(self, artist: str, album: str) -> Optional[Dict[str, Any]]:
        """Get album information from Last.fm.

        Args:
            artist: Artist name
            album: Album name

        Returns:
            Album info dict or None
        """
        await self._rate_limit()

        params = {
            "method": "album.getInfo",
            "artist": artist,
            "album": album,
            "api_key": self.api_key
        }

        try:
            session = await self._get_session()
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.text()
                    root = self._parse_response(data)
                    if root:
                        error = self._check_error(root)
                        if error:
                            return None

                        album_elem = root.find(".//album")
                        if album_elem is not None:
                            return self._parse_album_info(album_elem)
        except Exception as e:
            print(f"Error getting album info: {e}")

        return None

    def _parse_album_info(self, album_elem: ET.Element) -> Dict[str, Any]:
        """Parse album info from XML element.

        Args:
            album_elem: Album XML element

        Returns:
            Parsed album info dict
        """
        info = {
            "name": album_elem.findtext("name"),
            "artist": album_elem.findtext("artist"),
            "mbid": album_elem.findtext("mbid"),
            "url": album_elem.findtext("url"),
        }

        # Parse stats
        stats = album_elem.find("stats")
        if stats is not None:
            info["listeners"] = int(stats.findtext("listeners") or 0)
            info["playcount"] = int(stats.findtext("playcount") or 0)

        # Parse tags
        tags = album_elem.find("tags")
        if tags is not None:
            info["tags"] = [
                tag.findtext("name")
                for tag in tags.findall("tag")
                if tag.findtext("name")
            ]

        # Parse tracks
        tracks = album_elem.find("tracks")
        if tracks is not None:
            info["tracks"] = []
            for track in tracks.findall("track"):
                track_info = {
                    "name": track.findtext("name"),
                    "duration": int(track.findtext("duration") or 0) // 1000,  # ms to seconds
                    "artist": track.findtext("artist/name"),
                    "url": track.findtext("url")
                }
                if track_info["name"]:
                    info["tracks"].append(track_info)

        return info

    async def get_track_info(
        self,
        artist: str,
        track: str
    ) -> Optional[Dict[str, Any]]:
        """Get track information from Last.fm.

        Args:
            artist: Artist name
            track: Track title

        Returns:
            Track info dict or None
        """
        await self._rate_limit()

        params = {
            "method": "track.getInfo",
            "artist": artist,
            "track": track,
            "api_key": self.api_key,
            "autocorrect": "1"
        }

        try:
            session = await self._get_session()
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.text()
                    root = self._parse_response(data)
                    if root:
                        error = self._check_error(root)
                        if error:
                            return None

                        track_elem = root.find(".//track")
                        if track_elem is not None:
                            return self._parse_track_info(track_elem)
        except Exception as e:
            print(f"Error getting track info: {e}")

        return None

    def _parse_track_info(self, track_elem: ET.Element) -> Dict[str, Any]:
        """Parse track info from XML element.

        Args:
            track_elem: Track XML element

        Returns:
            Parsed track info dict
        """
        info = {
            "name": track_elem.findtext("name"),
            "artist": track_elem.findtext("artist/name"),
            "album": track_elem.findtext("album/album"),
            "mbid": track_elem.findtext("mbid"),
            "url": track_elem.findtext("url"),
            "duration": int(track_elem.findtext("duration") or 0) // 1000,
        }

        # Parse stats
        stats = track_elem.find("toptags")
        if stats is not None:
            info["tags"] = [
                tag.findtext("name")
                for tag in stats.findall("tag")
                if tag.findtext("name")
            ]

        return info

    def get_auth_url(self, callback_url: Optional[str] = None) -> str:
        """Get the URL for user authentication.

        Args:
            callback_url: Optional callback URL after authentication

        Returns:
            Full URL user should visit to authorize
        """
        params = {
            "api_key": self.api_key
        }
        if callback_url:
            params["cb"] = callback_url
        return f"https://www.last.fm/api/auth/?{urlencode(params)}"

    async def get_session(self, token: str) -> Optional[str]:
        """Exchange auth token for session key.

        Args:
            token: Authentication token from callback

        Returns:
            Session key or None if failed
        """
        await self._rate_limit()

        params = {
            "method": "auth.getsession",
            "token": token,
            "api_key": self.api_key
        }
        params["api_sig"] = self._sign_params(params)

        try:
            session = await self._get_session()
            async with session.post(self.base_url, data=params) as response:
                if response.status == 200:
                    data = await response.text()
                    root = self._parse_response(data)
                    if root:
                        error = self._check_error(root)
                        if error:
                            print(f"Last.fm session error: {error['code']} - {error['message']}")
                            return None

                        session_key = root.findtext(".//key")
                        if session_key:
                            self.session_key = session_key
                            return session_key
        except Exception as e:
            print(f"Error getting session: {e}")

        return None

    async def get_token(self) -> Optional[str]:
        """Get an authentication token.

        Returns:
            Token string or None if failed
        """
        await self._rate_limit()

        params = {
            "method": "auth.gettoken",
            "api_key": self.api_key
        }
        params["api_sig"] = self._sign_params(params)

        try:
            session = await self._get_session()
            async with session.post(self.base_url, data=params) as response:
                if response.status == 200:
                    data = await response.text()
                    root = self._parse_response(data)
                    if root:
                        error = self._check_error(root)
                        if error:
                            print(f"Last.fm token error: {error['code']} - {error['message']}")
                            return None

                        token = root.findtext(".//token")
                        return token
        except Exception as e:
            print(f"Error getting token: {e}")

        return None

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
