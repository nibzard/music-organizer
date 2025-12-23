# Spotify Integration Research

**Research Date**: 2025-12-23
**Status**: Technical Investigation Complete - Implementation Plan Ready

## Executive Summary

The music-organizer has excellent foundations for Spotify integration through existing metadata extraction, plugin architecture, M3U export capability, and string similarity algorithms. The main work is Spotify API client and track matching.

**Recommendation**: Implement Spotify integration as optional plugins for metadata enhancement and playlist import/export. Leverage existing string similarity for track matching.

**Estimated Effort**: 16-20 hours for full playlist sync + metadata enhancement

## Current Capabilities

### Existing Playlist Export

**M3U Export Plugin** (`plugins/builtins/m3u_exporter.py`):
- ✅ Working M3U playlist export
- ✅ Extended M3U format with metadata (#EXTINF)
- ✅ Artist/title/duration support
- ✅ Relative/absolute path handling

**Gaps**:
- No PLS format support
- No Spotify-specific playlist format

### Available Metadata for Track Matching

| Spotify Field | Local Source | Match Quality |
|---------------|--------------|---------------|
| track_name | `audio_file.title` | ✅ Excellent |
| artists | `audio_file.artists` | ✅ Excellent |
| album | `audio_file.album` | ✅ Good |
| duration_ms | `metadata.duration_seconds` | ✅ Excellent |
| isrc | (new) | ⚠️ Could add from MusicBrainz |
| track_number | `audio_file.track_number` | ✅ Good |
| disc_number | `metadata.disc_number` | ✅ Good |

**String Similarity** (already available):
- Rust extension for fast matching
- Levenshtein, Jaro-Winkler, Jaccard algorithms
- Music-specific `music_metadata_similarity()` function

## Spotify Web API Capabilities

### Endpoints for Integration

**Search** (`/v1/search`):
- Search tracks by artist, album, ISRC
- Returns Spotify URIs and metadata
- Limited to API rate limits

**Tracks** (`/v1/tracks/{id}`):
- Get detailed track metadata
- ISRC, album info, audio features

**Playlists** (`/v1/playlists`):
- Create/read/update playlists
- Add/remove tracks
- User playlist management

**Audio Features** (`/v1/audio-features`):
- Danceability, energy, valence
- Tempo, key, mode, time signature
- Acousticness, instrumentalness, liveness, speechiness

### Rate Limits

- **Free tier**: 10 requests/10 seconds
- **Premium tier**: Higher limits for approved apps
- Need token refresh logic (OAuth2 1-hour expiry)

## Proposed Implementation

### Component 1: Spotify Adapter

```python
# src/music_organizer/infrastructure/external/spotify_adapter.py
"""
Spotify Web API adapter - anti-corruption layer for Spotify integration.
"""

import asyncio
import time
import base64
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

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


class SpotifyAdapter:
    """Adapter for Spotify Web API."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "http://localhost:8080/callback",
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None
    ):
        if aiohttp is None:
            raise ImportError("aiohttp required for Spotify integration")

        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.access_token = access_token
        self.refresh_token = refresh_token
        self._session: Optional[aiohttp.ClientSession] = None
        self._token_expires_at = 0
        self._rate_limit_until = 0
        self._rate_limit_remaining = 10

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            headers = {}
            if self.access_token:
                headers["Authorization"] = f"Bearer {self.access_token}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def _ensure_token(self) -> bool:
        """Refresh access token if expired."""
        current_time = time.time()

        if self.access_token and current_time < self._token_expires_at:
            return True

        if self.refresh_token:
            # Use refresh token
            return await self._refresh_access_token()

        # Need to get new token via client credentials flow
        return await self._get_client_credentials_token()

    async def _get_client_credentials_token(self) -> bool:
        """Get access token using client credentials flow."""
        await self._rate_limit()

        auth_str = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        url = "https://accounts.spotify.com/api/token"
        data = {
            "grant_type": "client_credentials"
        }
        headers = {"Authorization": f"Basic {auth_str}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data["access_token"]
                        self._token_expires_at = time.time() + data["expires_in"] - 60
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
                        return True
        except Exception as e:
            print(f"Error refreshing Spotify token: {e}")

        return False

    async def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()

        if current_time < self._rate_limit_until:
            await asyncio.sleep(self._rate_limit_until - current_time)
            self._rate_limit_remaining = 10

        if self._rate_limit_remaining <= 1:
            # Wait for reset
            self._rate_limit_until = current_time + 10
            self._rate_limit_remaining = 10
            await asyncio.sleep(10)

        self._rate_limit_remaining -= 1

    async def search_track(
        self,
        query: str,
        limit: int = 10
    ) -> List[SpotifyTrack]:
        """Search for a track on Spotify."""
        if not await self._ensure_token():
            return []

        await self._rate_limit()

        url = "https://api.spotify.com/v1/search"
        params = {
            "q": query,
            "type": "track",
            "limit": limit
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
                            track_number=item.get("track_number"),
                            disc_number=item.get("disc_number"),
                            external_urls=item.get("external_urls", {})
                        ))
                    return tracks
                elif response.status == 401:
                    # Token expired, refresh and retry
                    await self._get_client_credentials_token()
                    return await self.search_track(query, limit)
                elif response.status == 429:
                    # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 10))
                    self._rate_limit_until = time.time() + retry_after
                    await asyncio.sleep(retry_after)
                    return await self.search_track(query, limit)
        except Exception as e:
            print(f"Error searching Spotify: {e}")

        return []

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
        except Exception as e:
            print(f"Error getting audio features: {e}")

        return None

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

        from ...utils.string_similarity import music_metadata_similarity

        best_match = None
        best_score = 0.0

        for track in spotify_tracks:
            # Calculate similarity score
            title_similarity = music_metadata_similarity(track.name, title)
            artist_similarity = max(
                music_metadata_similarity(track.artists[0], artist)
                if track.artists else 0.0,
                *(music_metadata_similarity(a, artist) for a in track.artists)
            )

            # Combined score
            score = (title_similarity * 0.6 + artist_similarity * 0.4)

            # Duration bonus
            if duration_ms:
                duration_diff = abs(track.duration_ms - duration_ms)
                if duration_diff < 5000:  # Within 5 seconds
                    score *= 1.1
                elif duration_diff > 30000:  # More than 30 seconds
                    score *= 0.5

            if score > best_score:
                best_score = score
                best_match = track

        if best_score > 0.7:  # Confidence threshold
            return (best_match, best_score)

        return None

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
```

### Component 2: Track Matcher

```python
# src/music_organizer/infrastructure/spotify/track_matcher.py
"""
Track matcher for mapping local files to Spotify tracks.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..external.spotify_adapter import SpotifyAdapter, SpotifyTrack
from ...models.audio_file import AudioFile


class SpotifyTrackMatcher:
    """Match local audio files to Spotify tracks."""

    def __init__(self, adapter: SpotifyAdapter):
        self.adapter = adapter
        self._cache: Dict[str, Optional[SpotifyTrack]] = {}

    def _cache_key(self, artist: str, title: str, album: Optional[str] = None) -> str:
        key = f"{artist.lower()}:{title.lower()}"
        if album:
            key += f":{album.lower()}"
        return key

    async def match_track(
        self,
        audio_file: AudioFile,
        min_confidence: float = 0.7
    ) -> Optional[Tuple[SpotifyTrack, float]]:
        """Match an audio file to a Spotify track."""
        if not audio_file.title or not audio_file.artists:
            return None

        artist = audio_file.artists[0]
        title = audio_file.title
        album = audio_file.album

        # Check cache
        cache_key = self._cache_key(artist, title, album)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return (cached, 1.0) if cached else None

        # Build search query
        query = f"track:{title} artist:{artist}"
        if album:
            query += f" album:{album}"

        # Search Spotify
        spotify_tracks = await self.adapter.search_track(query, limit=10)

        if not spotify_tracks:
            self._cache[cache_key] = None
            return None

        # Find best match
        duration_ms = int(audio_file.metadata.get("duration_seconds", 0) * 1000)
        result = self.adapter.find_best_match(spotify_tracks, title, artist, duration_ms)

        if result and result[1] >= min_confidence:
            self._cache[cache_key] = result[0]
            return result

        self._cache[cache_key] = None
        return None

    async def match_tracks(
        self,
        audio_files: List[AudioFile],
        min_confidence: float = 0.7
    ) -> Dict[AudioFile, Optional[Tuple[SpotifyTrack, float]]]:
        """Match multiple audio files to Spotify tracks."""
        results = {}

        # Batch processing with rate limit awareness
        for audio_file in audio_files:
            result = await self.match_track(audio_file, min_confidence)
            results[audio_file] = result

        return results
```

### Component 3: Playlist Import Plugin

```python
# src/music_organizer/plugins/builtins/spotify_playlist_import.py
"""
Spotify playlist import plugin.
"""

import logging
from typing import List, Optional
from pathlib import Path

from ..base import MetadataPlugin, PluginInfo
from ...plugins.config import PluginConfigSchema, ConfigOption
from ...infrastructure.external.spotify_adapter import SpotifyAdapter
from ...infrastructure.spotify.track_matcher import SpotifyTrackMatcher
from ...models.audio_file import AudioFile

logger = logging.getLogger(__name__)


class SpotifyPlaylistImportPlugin(MetadataPlugin):
    """Plugin to import Spotify playlists and match local files."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="spotify_playlist_import",
            version="1.0.0",
            description="Import Spotify playlists and match to local files",
            author="Music Organizer Team",
            dependencies=["aiohttp"],
        )

    def get_config_schema(self) -> PluginConfigSchema:
        return PluginConfigSchema([
            ConfigOption(
                name="enabled",
                type=bool,
                default=False,
                description="Enable Spotify playlist import"
            ),
            ConfigOption(
                name="client_id",
                type=str,
                default="",
                description="Spotify client ID"
            ),
            ConfigOption(
                name="client_secret",
                type=str,
                default="",
                description="Spotify client secret"
            ),
            ConfigOption(
                name="user_id",
                type=str,
                default="",
                description="Spotify user ID (optional, for user playlists)"
            ),
            ConfigOption(
                name="min_confidence",
                type=float,
                default=0.7,
                description="Minimum confidence for track matching"
            ),
            ConfigOption(
                name="export_m3u",
                type=bool,
                default=True,
                description="Export matched playlists as M3U"
            ),
        ])

    def initialize(self) -> None:
        self._adapter: Optional[SpotifyAdapter] = None
        self._matcher: Optional[SpotifyTrackMatcher] = None

    def _get_adapter(self) -> Optional[SpotifyAdapter]:
        if not self.config.get("enabled"):
            return None

        client_id = self.config.get("client_id")
        client_secret = self.config.get("client_secret")

        if not client_id or not client_secret:
            logger.warning("Spotify credentials not configured")
            return None

        if self._adapter is None:
            self._adapter = SpotifyAdapter(
                client_id=client_id,
                client_secret=client_secret
            )

        return self._adapter

    def cleanup(self) -> None:
        if self._adapter:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._adapter.close())
            except RuntimeError:
                pass

    async def import_playlist(
        self,
        playlist_id: str,
        local_library: List[AudioFile],
        output_path: Optional[Path] = None
    ) -> List[AudioFile]:
        """Import a Spotify playlist and match to local files."""
        adapter = self._get_adapter()
        if not adapter:
            return []

        # Get playlist tracks from Spotify
        # (This would need to be implemented in the adapter)
        # For now, placeholder

        matched = []
        for audio_file in local_library:
            # Match and add if found
            pass

        return matched
```

### Component 4: Metadata Enhancement Plugin

```python
# src/music_organizer/plugins/builtins/spotify_enhancer.py
"""
Spotify metadata enhancement plugin.
"""

import logging
from typing import List, Optional

from ..base import MetadataPlugin, PluginInfo
from ...plugins.config import PluginConfigSchema, ConfigOption
from ...infrastructure.external.spotify_adapter import SpotifyAdapter
from ...infrastructure.spotify.track_matcher import SpotifyTrackMatcher
from ...models.audio_file import AudioFile

logger = logging.getLogger(__name__)


class SpotifyEnhancerPlugin(MetadataPlugin):
    """Plugin to enhance metadata using Spotify data."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="spotify_enhancer",
            version="1.0.0",
            description="Enhances metadata using Spotify database",
            author="Music Organizer Team",
            dependencies=["aiohttp"],
        )

    def get_config_schema(self) -> PluginConfigSchema:
        return PluginConfigSchema([
            ConfigOption(
                name="enabled",
                type=bool,
                default=False,
                description="Enable Spotify metadata enhancement"
            ),
            ConfigOption(
                name="client_id",
                type=str,
                default="",
                description="Spotify client ID"
            ),
            ConfigOption(
                name="client_secret",
                type=str,
                default="",
                description="Spotify client secret"
            ),
            ConfigOption(
                name="add_audio_features",
                type=bool,
                default=False,
                description="Add Spotify audio features to metadata"
            ),
            ConfigOption(
                name="min_confidence",
                type=float,
                default=0.8,
                description="Minimum confidence for metadata enhancement"
            ),
        ])

    def initialize(self) -> None:
        self._adapter: Optional[SpotifyAdapter] = None
        self._matcher: Optional[SpotifyTrackMatcher] = None

    def cleanup(self) -> None:
        if self._adapter:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._adapter.close())
            except RuntimeError:
                pass

    async def enhance_metadata(self, audio_file: AudioFile) -> AudioFile:
        """Enhance metadata using Spotify."""
        if not self._adapter:
            client_id = self.config.get("client_id")
            client_secret = self.config.get("client_secret")
            if not client_id or not client_secret:
                return audio_file

            self._adapter = SpotifyAdapter(
                client_id=client_id,
                client_secret=client_secret
            )
            self._matcher = SpotifyTrackMatcher(self._adapter)

        # Find matching track
        result = await self._matcher.match_track(
            audio_file,
            min_confidence=self.config.get("min_confidence", 0.8)
        )

        if result:
            spotify_track, confidence = result

            # Enhance with Spotify metadata
            if not audio_file.year and spotify_track.album:
                # Could extract year from album (if available)
                pass

            if self.config.get("add_audio_features", False):
                features = await self._adapter.get_audio_features(spotify_track.id)
                if features:
                    audio_file.metadata["spotify_audio_features"] = {
                        "danceability": features.danceability,
                        "energy": features.energy,
                        "valence": features.valence,
                        "tempo": features.tempo
                    }

            audio_file.metadata["spotify_uri"] = spotify_track.uri
            audio_file.metadata["spotify_id"] = spotify_track.id

        return audio_file

    async def batch_enhance(self, audio_files: List[AudioFile]) -> List[AudioFile]:
        """Enhance metadata for multiple files."""
        enhanced = []
        for audio_file in audio_files:
            enhanced.append(await self.enhance_metadata(audio_file))
        return enhanced
```

## Configuration

### Plugin Configuration

```json
// config/plugins/spotify_enhancer.json
{
    "enabled": false,
    "client_id": "",
    "client_secret": "",
    "add_audio_features": false,
    "min_confidence": 0.8
}
```

### Global Configuration

```yaml
# config/spotify.yaml
spotify:
  # API credentials from https://developer.spotify.com/dashboard
  client_id: ""
  client_secret: ""
  redirect_uri: "http://localhost:8080/callback"

  # Playlist import
  import:
    enabled: false
    user_id: ""
    export_m3u: true
    min_confidence: 0.7

  # Metadata enhancement
  enhance:
    enabled: false
    add_audio_features: false
    min_confidence: 0.8
```

## CLI Integration

```bash
# Authenticate with Spotify
music-organize-async spotify auth

# Import a Spotify playlist
music-organize-async spotify import-playlist spotify:playlist:37i9dQZF1DXcBWIGoYBM5M

# Match local library to Spotify
music-organize-async spotify match-library /music/library

# Enhance metadata with Spotify
music-organize-async enhance /music/library --spotify

# Export local playlist to Spotify format
music-organize-async export-playlist local.m3u --format spotify
```

## Testing

```python
# tests/test_spotify_integration.py
import pytest
from music_organizer.infrastructure.external.spotify_adapter import SpotifyAdapter

@pytest.mark.asyncio
async def test_search_track():
    adapter = SpotifyAdapter(
        client_id="test",
        client_secret="test"
    )

    # Mock the HTTP requests
    # ...

    tracks = await adapter.search_track("artist:Beatles track:Hey Jude")

    assert len(tracks) > 0
    assert tracks[0].name == "Hey Jude"

def test_find_best_match():
    from music_organizer.infrastructure.external.spotify_adapter import SpotifyAdapter, SpotifyTrack

    adapter = SpotifyAdapter("test", "test")

    spotify_tracks = [
        SpotifyTrack(
            id="1",
            uri="spotify:track:1",
            name="Hey Jude",
            artists=["The Beatles"],
            album="The Beatles 1967-1970",
            duration_ms=431000
        ),
        SpotifyTrack(
            id="2",
            uri="spotify:track:2",
            name="Hey Jude",
            artists=["The Beatles"],
            album="1 (Remastered)",
            duration_ms=258000
        )
    ]

    # Should match closer duration
    result = adapter.find_best_match(
        spotify_tracks,
        "Hey Jude",
        "The Beatles",
        duration_ms=259000  # About 4:19
    )

    assert result is not None
    assert result[0].id == "2"  # Closer duration match
```

## Dependency Impact

### New Dependencies

| Package | Version | Size | License | Required |
|---------|---------|------|--------|----------|
| aiohttp | >=3.8.0 | ~500KB | Apache 2.0 | Yes (already used for MusicBrainz) |
| spotipy | >=2.23.0 | ~100KB | MIT | Optional (we use custom adapter) |

**No additional dependencies** - aiohttp is already in use.

## Implementation Roadmap

### Phase 1: Core Adapter (6-8 hours)

1. Create `SpotifyAdapter` class
2. Implement client credentials flow
3. Implement search endpoint
4. Add audio features endpoint
5. Implement track matching algorithm
6. Unit tests

### Phase 2: Track Matcher (2-3 hours)

1. Create `SpotifyTrackMatcher` class
2. Implement caching
3. Add confidence scoring
4. Handle batch matching

### Phase 3: Metadata Enhancer (3-4 hours)

1. Create `SpotifyEnhancerPlugin`
2. Wire up metadata enhancement
3. Add audio features support
4. Configuration management

### Phase 4: Playlist Import/Export (4-5 hours)

1. Create `SpotifyPlaylistImportPlugin`
2. Implement playlist fetching
3. Match tracks to local library
4. Export to M3U format
5. CLI commands

**Total: 15-20 hours**

## Limitations

1. **API Rate Limits**: Strict rate limiting on free tier
2. **Authentication**: User auth required for playlist access
3. **Track Availability**: Not all tracks on Spotify (especially local/obscure)
4. **Matching Accuracy**: Fuzzy matching can produce false positives
5. **ISRC Access**: Requires track lookup (not available in search)

## Conclusion

Spotify integration is feasible with existing infrastructure. The main work is API client and track matching.

**Recommendation**: Implement as optional plugin with clear opt-in flow.

**MVP Scope**: Metadata enhancement + basic playlist sync (15-18 hours)

**Future Enhancements**:
- Real-time playlist sync
- Spotify Connect for playback
- Discover Weekly integration
- Playlist recommendations based on library

---

*Research by task-master agent on 2025-12-23*
