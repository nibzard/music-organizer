"""Spotify Web API Adapter - Anti-Corruption Layer for Spotify integration."""

import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None


@dataclass
class SpotifyTrack:
    """Spotify track data."""
    id: str
    uri: str
    name: str
    artists: List[str]
    album: str
    duration_ms: int
    isrc: Optional[str] = None
    track_number: Optional[int] = None
    disc_number: Optional[int] = None
    external_urls: Dict[str, str] = None
    release_date: Optional[str] = None

    def __post_init__(self):
        if self.external_urls is None:
            self.external_urls = {}


@dataclass
class SpotifyAudioFeatures:
    """Audio features from Spotify."""
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    time_signature: int


@dataclass
class SpotifyPlaylist:
    """Spotify playlist data."""
    id: str
    uri: str
    name: str
    owner: str
    tracks: List[SpotifyTrack]
    description: Optional[str] = None
    public: bool = False
    snapshot_id: Optional[str] = None


class SpotifyAdapter:
    """Adapter for Spotify Web API with rate limiting and auth."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "http://localhost:8080/callback",
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        timeout: int = 10,
        rate_limit: float = 10.0
    ):
        if aiohttp is None:
            raise ImportError("aiohttp required for Spotify integration")

        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.timeout = timeout
        self.rate_limit = rate_limit
        self._session: Optional[aiohttp.ClientSession] = None
        self._token_expires_at = 0
        self._last_request_time = 0.0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            headers = {}
            if self.access_token:
                headers["Authorization"] = f"Bearer {self.access_token}"
            self._session = aiohttp.ClientSession(
                headers=headers,
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

    async def _ensure_token(self) -> bool:
        """Refresh access token if expired."""
        current_time = time.time()

        if self.access_token and current_time < self._token_expires_at:
            return True

        if self.refresh_token:
            return await self._refresh_access_token()

        return await self._get_client_credentials_token()

    async def _get_client_credentials_token(self) -> bool:
        """Get access token using client credentials flow."""
        await self._rate_limit()

        auth_str = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        url = "https://accounts.spotify.com/api/token"
        data = {"grant_type": "client_credentials"}
        headers = {"Authorization": f"Basic {auth_str}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data["access_token"]
                        self._token_expires_at = time.time() + data["expires_in"] - 60
                        await self._update_session_auth()
                        return True
        except Exception as e:
            print(f"Error getting Spotify token: {e}")

        return False

    async def _refresh_access_token(self) -> bool:
        """Refresh access token using refresh token."""
        await self._rate_limit()

        url = "https://accounts.spotify.com/api/token"
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token
        }
        auth_str = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        headers = {"Authorization": f"Basic {auth_str}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as response:
                    if response.status == 200:
                        resp_data = await response.json()
                        self.access_token = resp_data["access_token"]
                        self._token_expires_at = time.time() + resp_data.get("expires_in", 3600) - 60
                        if "refresh_token" in resp_data:
                            self.refresh_token = resp_data["refresh_token"]
                        await self._update_session_auth()
                        return True
        except Exception as e:
            print(f"Error refreshing Spotify token: {e}")

        return False

    async def _update_session_auth(self) -> None:
        """Update session authorization header."""
        if self._session and self.access_token:
            self._session.headers["Authorization"] = f"Bearer {self.access_token}"

    async def search_track(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[SpotifyTrack]:
        """Search for a track on Spotify."""
        if not await self._ensure_token():
            return []

        await self._rate_limit()

        url = "https://api.spotify.com/v1/search"
        params = {
            "q": query,
            "type": "track",
            "limit": min(limit, 50),
            "offset": offset
        }

        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    tracks = []
                    for item in data.get("tracks", {}).get("items", []):
                        tracks.append(SpotifyTrack(
                            id=item["id"],
                            uri=item["uri"],
                            name=item["name"],
                            artists=[a["name"] for a in item["artists"]],
                            album=item["album"]["name"],
                            duration_ms=item["duration_ms"],
                            isrc=item.get("external_ids", {}).get("isrc"),
                            track_number=item.get("track_number"),
                            disc_number=item.get("disc_number"),
                            external_urls=item.get("external_urls", {}),
                            release_date=item.get("album", {}).get("release_date")
                        ))
                    return tracks
                elif response.status == 401:
                    await self._get_client_credentials_token()
                    return await self.search_track(query, limit, offset)
                elif response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 10))
                    await asyncio.sleep(retry_after)
                    return await self.search_track(query, limit, offset)
        except Exception as e:
            print(f"Error searching Spotify: {e}")

        return []

    async def get_track(self, track_id: str) -> Optional[SpotifyTrack]:
        """Get detailed track metadata by ID."""
        if not await self._ensure_token():
            return None

        await self._rate_limit()

        url = f"https://api.spotify.com/v1/tracks/{track_id}"

        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return SpotifyTrack(
                        id=data["id"],
                        uri=data["uri"],
                        name=data["name"],
                        artists=[a["name"] for a in data["artists"]],
                        album=data["album"]["name"],
                        duration_ms=data["duration_ms"],
                        isrc=data.get("external_ids", {}).get("isrc"),
                        track_number=data.get("track_number"),
                        disc_number=data.get("disc_number"),
                        external_urls=data.get("external_urls", {}),
                        release_date=data.get("album", {}).get("release_date")
                    )
                elif response.status == 401:
                    await self._get_client_credentials_token()
                    return await self.get_track(track_id)
        except Exception as e:
            print(f"Error getting track: {e}")

        return None

    async def get_audio_features(self, track_id: str) -> Optional[SpotifyAudioFeatures]:
        """Get audio features for a track."""
        if not await self._ensure_token():
            return None

        await self._rate_limit()

        url = f"https://api.spotify.com/v1/audio-features/{track_id}"

        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return SpotifyAudioFeatures(
                        danceability=data["danceability"],
                        energy=data["energy"],
                        key=data["key"],
                        loudness=data["loudness"],
                        mode=data["mode"],
                        speechiness=data["speechiness"],
                        acousticness=data["acousticness"],
                        instrumentalness=data["instrumentalness"],
                        liveness=data["liveness"],
                        valence=data["valence"],
                        tempo=data["tempo"],
                        time_signature=data["time_signature"]
                    )
                elif response.status == 401:
                    await self._get_client_credentials_token()
                    return await self.get_audio_features(track_id)
        except Exception as e:
            print(f"Error getting audio features: {e}")

        return None

    async def get_playlist(
        self,
        playlist_id: str,
        limit: int = 100
    ) -> Optional[SpotifyPlaylist]:
        """Get playlist tracks by ID (requires user auth)."""
        if not await self._ensure_token():
            return None

        await self._rate_limit()

        url = f"https://api.spotify.com/v1/playlists/{playlist_id}"
        params = {"limit": min(limit, 100)}

        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    tracks = []
                    for item in data.get("tracks", {}).get("items", []):
                        track_data = item.get("track")
                        if track_data:
                            tracks.append(SpotifyTrack(
                                id=track_data["id"],
                                uri=track_data["uri"],
                                name=track_data["name"],
                                artists=[a["name"] for a in track_data["artists"]],
                                album=track_data["album"]["name"],
                                duration_ms=track_data["duration_ms"],
                                isrc=track_data.get("external_ids", {}).get("isrc"),
                                track_number=track_data.get("track_number"),
                                disc_number=track_data.get("disc_number"),
                                external_urls=track_data.get("external_urls", {}),
                                release_date=track_data.get("album", {}).get("release_date")
                            ))

                    return SpotifyPlaylist(
                        id=data["id"],
                        uri=data["uri"],
                        name=data["name"],
                        owner=data.get("owner", {}).get("id", ""),
                        tracks=tracks,
                        description=data.get("description"),
                        public=data.get("public", False),
                        snapshot_id=data.get("snapshot_id")
                    )
                elif response.status == 401:
                    await self._refresh_access_token()
                    return await self.get_playlist(playlist_id, limit)
        except Exception as e:
            print(f"Error getting playlist: {e}")

        return None

    async def create_playlist(
        self,
        user_id: str,
        name: str,
        description: str = "",
        public: bool = False
    ) -> Optional[SpotifyPlaylist]:
        """Create a new playlist (requires user auth)."""
        if not await self._ensure_token():
            return None

        await self._rate_limit()

        url = f"https://api.spotify.com/v1/users/{user_id}/playlists"
        data = {
            "name": name,
            "description": description,
            "public": public
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=data) as response:
                if response.status == 201:
                    resp_data = await response.json()
                    return SpotifyPlaylist(
                        id=resp_data["id"],
                        uri=resp_data["uri"],
                        name=resp_data["name"],
                        owner=resp_data.get("owner", {}).get("id", ""),
                        tracks=[],
                        description=resp_data.get("description"),
                        public=resp_data.get("public", False),
                        snapshot_id=resp_data.get("snapshot_id")
                    )
                elif response.status == 401:
                    await self._refresh_access_token()
                    return await self.create_playlist(user_id, name, description, public)
        except Exception as e:
            print(f"Error creating playlist: {e}")

        return None

    async def add_to_playlist(
        self,
        playlist_id: str,
        track_uris: List[str]
    ) -> bool:
        """Add tracks to a playlist (requires user auth)."""
        if not await self._ensure_token():
            return False

        await self._rate_limit()

        url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
        data = {"uris": track_uris[:100]}

        try:
            session = await self._get_session()
            async with session.post(url, json=data) as response:
                if response.status == 201:
                    return True
                elif response.status == 401:
                    await self._refresh_access_token()
                    return await self.add_to_playlist(playlist_id, track_uris)
        except Exception as e:
            print(f"Error adding to playlist: {e}")

        return False

    def find_best_match(
        self,
        spotify_tracks: List[SpotifyTrack],
        title: str,
        artist: str,
        duration_ms: Optional[int] = None
    ) -> Optional[tuple[SpotifyTrack, float]]:
        """Find best matching Spotify track using similarity scoring."""
        if not spotify_tracks:
            return None

        try:
            from ...utils.string_similarity import music_metadata_similarity
        except ImportError:
            def music_metadata_similarity(a: str, b: str) -> float:
                """Fallback simple similarity."""
                a_lower = a.lower().strip()
                b_lower = b.lower().strip()
                if a_lower == b_lower:
                    return 1.0
                if a_lower in b_lower or b_lower in a_lower:
                    return 0.8
                return 0.0

        best_match = None
        best_score = 0.0

        for track in spotify_tracks:
            title_similarity = music_metadata_similarity(track.name, title)
            artist_similarity = max(
                music_metadata_similarity(track.artists[0], artist)
                if track.artists else 0.0,
                *(music_metadata_similarity(a, artist) for a in track.artists)
            )

            score = (title_similarity * 0.6 + artist_similarity * 0.4)

            if duration_ms:
                duration_diff = abs(track.duration_ms - duration_ms)
                if duration_diff < 5000:
                    score *= 1.1
                elif duration_diff > 30000:
                    score *= 0.5

            if score > best_score:
                best_score = score
                best_match = track

        if best_score > 0.7:
            return (best_match, best_score)

        return None

    def get_auth_url(self) -> str:
        """Get OAuth authorization URL."""
        scopes = "playlist-read-private playlist-modify-public playlist-modify-private"
        return (
            f"https://accounts.spotify.com/authorize"
            f"?client_id={self.client_id}"
            f"&response_type=code"
            f"&redirect_uri={self.redirect_uri}"
            f"&scope={scopes}"
        )

    async def exchange_code_for_tokens(self, code: str) -> bool:
        """Exchange auth code for access and refresh tokens."""
        auth_str = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        url = "https://accounts.spotify.com/api/token"
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri
        }
        headers = {"Authorization": f"Basic {auth_str}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as response:
                    if response.status == 200:
                        resp_data = await response.json()
                        self.access_token = resp_data["access_token"]
                        self.refresh_token = resp_data.get("refresh_token", self.refresh_token)
                        self._token_expires_at = time.time() + resp_data.get("expires_in", 3600) - 60
                        await self._update_session_auth()
                        return True
        except Exception as e:
            print(f"Error exchanging code for tokens: {e}")

        return False

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
