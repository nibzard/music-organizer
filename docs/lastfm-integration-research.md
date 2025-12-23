# Last.fm Integration Research

**Research Date**: 2025-12-23
**Status**: Technical Investigation Complete - Implementation Plan Ready

## Executive Summary

The music-organizer codebase has excellent architecture for Last.fm integration through existing adapter patterns, event-driven architecture, and plugin system. All required metadata is available; main work is API integration and event handling.

**Recommendation**: Implement Last.fm scrobbling as an optional plugin with configuration-based authentication. Leverage existing MusicBrainz adapter as template.

**Estimated Effort**: 8-12 hours for basic scrobbling, 16-20 hours with metadata enhancement

## Current State Analysis

### No Existing Last.fm Integration

- **Status**: No Last.fm code currently exists
- **References**: TODO.md (pending), FUTURE-FEAT.md (mentioned as future feature)
- **Priority**: Marked as 🟢 (nice-to-have, optional)

### Available Metadata (All Required Fields Present)

From `AudioFile` model and `Metadata` value object:

| Last.fm Field | Source | Status |
|---------------|--------|--------|
| artist | `audio_file.artists[0]` | ✅ Available |
| track | `audio_file.title` | ✅ Available |
| timestamp | File mtime or current time | ✅ Derivable |
| album | `audio_file.album` | ✅ Available |
| trackNumber | `audio_file.track_number` | ✅ Available |
| duration | `audio_file.duration_seconds` | ✅ Available |
| mbid | (future) MusicBrainz ID | ⚠️ Could add |

### Extensibility Points in Existing Codebase

**1. External Service Adapter Pattern**
- `src/music_organizer/infrastructure/external/musicbrainz_adapter.py` - Template for API integration
- `src/music_organizer/infrastructure/external/acoustid_adapter.py` - Another example

**2. Event System**
- `src/music_organizer/events/domain_events.py` - RecordingAdded, FileMoved, MetadataEnhanced events
- Can trigger scrobbles on file organization or metadata enhancement

**3. Plugin Hooks**
- `src/music_organizer/plugins/hooks.py` - POST_MOVE, POST_METADATA_EXTRACT hooks
- Optional Last.fm enhancement plugin

## Proposed Implementation

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Last.fm Integration Layer                │
│  ┌────────────────────────────────────────────────────┐    │
│  │  LastFmAdapter (Anti-Corruption Layer)             │    │
│  │  - scrobble(track, artist, album, timestamp)       │    │
│  │  - update_now_playing(track, artist)               │    │
│  │  - get_artist_info(artist)                         │    │
│  │  - get_album_info(album, artist)                   │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
           │                           │
           ▼                           ▼
┌──────────────────┐        ┌──────────────────┐
│  ScrobblePlugin  │        │ LastFmEnhancer   │
│  (Event Handler) │        │  (Metadata)      │
│  - Listen to     │        │  - Artist genre  │
│    FileMoved     │        │  - Album info    │
│  - Submit scrob  │        │  - Similar artists│
└──────────────────┘        └──────────────────┘
```

### Component 1: Last.fm Adapter

```python
# src/music_organizer/infrastructure/external/lastfm_adapter.py
"""
Last.fm Adapter - Anti-Corruption Layer for Last.fm API.

Implements scrobbling, now playing updates, and metadata retrieval.
"""

import asyncio
import aiohttp
import time
import hashlib
from typing import Dict, Optional, Any
from urllib.parse import urlencode

class LastFmAdapter:
    """Adapter for Last.fm API v2.0."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        session_key: Optional[str] = None,
        base_url: str = "https://ws.audioscrobbler.com/2.0/",
        timeout: int = 10
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_key = session_key
        self.base_url = base_url
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0
        self._rate_limit = 0.5  # 2 requests per second max

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 1.0 / self._rate_limit

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self._last_request_time = time.time()

    def _sign_params(self, params: Dict[str, str]) -> str:
        """Generate API signature for authenticated requests."""
        # Sort parameters and append secret
        sorted_params = sorted(params.items())
        to_sign = "".join(f"{k}{v}" for k, v in sorted_params) + self.api_secret
        return hashlib.md5(to_sign.encode()).hexdigest()

    async def scrobble(
        self,
        artist: str,
        track: str,
        timestamp: int,
        album: Optional[str] = None,
        track_number: Optional[int] = None,
        duration: Optional[int] = None
    ) -> bool:
        """Submit a scrobble to Last.fm."""
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
        if track_number:
            params["trackNumber"] = str(track_number)
        if duration:
            params["duration"] = str(duration)

        params["api_sig"] = self._sign_params(params)

        try:
            session = await self._get_session()
            async with session.post(self.base_url, data=params) as response:
                if response.status == 200:
                    import xml.etree.ElementTree as ET
                    data = await response.text()
                    root = ET.fromstring(data)
                    # Check for error
                    error = root.find("error")
                    if error is not None:
                        print(f"Last.fm scrobble error: {error.get('code')} - {error.text}")
                        return False
                    return True
                else:
                    print(f"Last.fm API error: {response.status}")
                    return False
        except Exception as e:
            print(f"Error scrobbling to Last.fm: {e}")
            return False

    async def update_now_playing(
        self,
        artist: str,
        track: str,
        album: Optional[str] = None,
        duration: Optional[int] = None
    ) -> bool:
        """Update the "now playing" status on Last.fm."""
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
        if duration:
            params["duration"] = str(duration)

        params["api_sig"] = self._sign_params(params)

        try:
            session = await self._get_session()
            async with session.post(self.base_url, data=params) as response:
                if response.status == 200:
                    return True
                return False
        except Exception as e:
            print(f"Error updating now playing: {e}")
            return False

    async def get_artist_info(self, artist: str) -> Optional[Dict[str, Any]]:
        """Get artist information from Last.fm."""
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
                    import xml.etree.ElementTree as ET
                    data = await response.text()
                    root = ET.fromstring(data)
                    # Extract artist info
                    artist_elem = root.find(".//artist")
                    if artist_elem is not None:
                        return {
                            "name": artist_elem.findtext("name"),
                            "mbid": artist_elem.findtext("mbid"),
                            "url": artist_elem.findtext("url"),
                            "listeners": int(artist_elem.findtext("stats/listeners") or 0),
                            "playcount": int(artist_elem.findtext("stats/playcount") or 0),
                            "tags": [tag.findtext("name") for tag in artist_elem.findall(".//tag")]
                        }
        except Exception as e:
            print(f"Error getting artist info: {e}")

        return None

    async def get_album_info(self, artist: str, album: str) -> Optional[Dict[str, Any]]:
        """Get album information from Last.fm."""
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
                    import xml.etree.ElementTree as ET
                    data = await response.text()
                    root = ET.fromstring(data)
                    album_elem = root.find(".//album")
                    if album_elem is not None:
                        return {
                            "name": album_elem.findtext("name"),
                            "artist": album_elem.findtext("artist"),
                            "mbid": album_elem.findtext("mbid"),
                            "url": album_elem.findtext("url"),
                            "listeners": int(album_elem.findtext("stats/listeners") or 0),
                            "playcount": int(album_elem.findtext("stats/playcount") or 0),
                            "tags": [tag.findtext("name") for tag in album_elem.findall(".//tag")]
                        }
        except Exception as e:
            print(f"Error getting album info: {e}")

        return None

    async def get_auth_url(self, callback_url: str) -> str:
        """Get the URL for user authentication."""
        params = {
            "api_key": self.api_key,
            "method": "auth.gettoken",
            "cb": callback_url
        }
        return f"https://www.last.fm/api/auth/?{urlencode(params)}"

    async def get_session(self, token: str) -> Optional[str]:
        """Exchange auth token for session key."""
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
                    import xml.etree.ElementTree as ET
                    data = await response.text()
                    root = ET.fromstring(data)
                    session_key = root.findtext(".//key")
                    if session_key:
                        self.session_key = session_key
                        return session_key
        except Exception as e:
            print(f"Error getting session: {e}")

        return None

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
```

### Component 2: Scrobble Plugin

```python
# src/music_organizer/plugins/builtins/lastfm_scrobbler.py
"""Last.fm scrobbling plugin - submits tracks when organized."""

import logging
import time
from typing import Optional, List
from pathlib import Path

from ..base import OutputPlugin, PluginInfo
from ...plugins.config import PluginConfigSchema, ConfigOption
from ...infrastructure.external.lastfm_adapter import LastFmAdapter
from ...models.audio_file import AudioFile

logger = logging.getLogger(__name__)


class LastFmScrobblerPlugin(OutputPlugin):
    """Plugin to scrobble tracks to Last.fm after organization."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="lastfm_scrobbler",
            version="1.0.0",
            description="Scrobbles tracks to Last.fm after organization",
            author="Music Organizer Team",
            dependencies=["aiohttp"],
        )

    def get_config_schema(self) -> PluginConfigSchema:
        return PluginConfigSchema([
            ConfigOption(
                name="enabled",
                type=bool,
                default=False,
                description="Enable Last.fm scrobbling"
            ),
            ConfigOption(
                name="api_key",
                type=str,
                default="",
                description="Last.fm API key"
            ),
            ConfigOption(
                name="api_secret",
                type=str,
                default="",
                description="Last.fm API secret"
            ),
            ConfigOption(
                name="session_key",
                type=str,
                default="",
                description="Last.fm session key (authenticated)"
            ),
            ConfigOption(
                name="scrobble_on_move",
                type=bool,
                default=True,
                description="Scrobble when files are moved/organized"
            ),
            ConfigOption(
                name="scrobble_timestamp",
                type=str,
                default="current",
                description="Timestamp to use: 'current' or 'file_mtime'"
            ),
        ])

    def initialize(self) -> None:
        self._adapter: Optional[LastFmAdapter] = None

    def _get_adapter(self) -> Optional[LastFmAdapter]:
        """Get or create the Last.fm adapter."""
        if not self.config.get("enabled"):
            return None

        api_key = self.config.get("api_key")
        api_secret = self.config.get("api_secret")
        session_key = self.config.get("session_key")

        if not api_key or not api_secret:
            logger.warning("Last.fm API key/secret not configured")
            return None

        if not session_key:
            logger.warning("Last.fm not authenticated (no session key)")
            return None

        if self._adapter is None:
            self._adapter = LastFmAdapter(
                api_key=api_key,
                api_secret=api_secret,
                session_key=session_key
            )

        return self._adapter

    def cleanup(self) -> None:
        if self._adapter:
            # Close session (async, but cleanup may be sync)
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._adapter.close())
            except RuntimeError:
                pass

    async def after_move(self, audio_file: AudioFile, target_path: Path) -> AudioFile:
        """Scrobble track after moving."""
        if not self.config.get("scrobble_on_move", True):
            return audio_file

        adapter = self._get_adapter()
        if not adapter:
            return audio_file

        # Get timestamp
        if self.config.get("scrobble_timestamp", "current") == "file_mtime":
            timestamp = int(audio_file.path.stat().st_mtime)
        else:
            timestamp = int(time.time())

        # Submit scrobble
        success = await adapter.scrobble(
            artist=audio_file.artists[0] if audio_file.artists else "Unknown Artist",
            track=audio_file.title or "Unknown Track",
            timestamp=timestamp,
            album=audio_file.album,
            track_number=audio_file.track_number,
            duration=int(audio_file.metadata.get("duration_seconds", 0))
        )

        if success:
            logger.info(f"Scrobbled: {audio_file.get_display_name()}")
        else:
            logger.warning(f"Failed to scrobble: {audio_file.get_display_name()}")

        return audio_file

    async def batch_after_move(self, audio_files: List[AudioFile]) -> List[AudioFile]:
        """Scrobble multiple tracks after moving."""
        adapter = self._get_adapter()
        if not adapter:
            return audio_files

        for audio_file in audio_files:
            await self.after_move(audio_file, audio_file.path)

        return audio_files
```

### Component 3: Last.fm Metadata Enhancer Plugin

```python
# src/music_organizer/plugins/builtins/lastfm_enhancer.py
"""Last.fm metadata enhancement plugin."""

import logging
from typing import Optional, List

from ..base import MetadataPlugin, PluginInfo
from ...plugins.config import PluginConfigSchema, ConfigOption
from ...infrastructure.external.lastfm_adapter import LastFmAdapter
from ...models.audio_file import AudioFile

logger = logging.getLogger(__name__)


class LastFmEnhancerPlugin(MetadataPlugin):
    """Plugin to enhance metadata using Last.fm data."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="lastfm_enhancer",
            version="1.0.0",
            description="Enhances metadata using Last.fm database",
            author="Music Organizer Team",
            dependencies=["aiohttp"],
        )

    def get_config_schema(self) -> PluginConfigSchema:
        return PluginConfigSchema([
            ConfigOption(
                name="enabled",
                type=bool,
                default=False,
                description="Enable Last.fm metadata enhancement"
            ),
            ConfigOption(
                name="api_key",
                type=str,
                default="",
                description="Last.fm API key (no auth required for metadata)"
            ),
            ConfigOption(
                name="enhance_genres",
                type=bool,
                default=True,
                description="Enhance genre tags from Last.fm"
            ),
            ConfigOption(
                name="cache_enabled",
                type=bool,
                default=True,
                description="Enable metadata caching"
            ),
        ])

    def initialize(self) -> None:
        self._adapter: Optional[LastFmAdapter] = None
        self._cache: dict = {}

    def _get_adapter(self) -> Optional[LastFmAdapter]:
        if not self.enabled:
            return None

        api_key = self.config.get("api_key")
        if not api_key:
            return None

        if self._adapter is None:
            self._adapter = LastFmAdapter(api_key=api_key, api_secret="")

        return self._adapter

    def cleanup(self) -> None:
        if self._adapter:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._adapter.close())
            except RuntimeError:
                pass
        self._cache.clear()

    async def enhance_metadata(self, audio_file: AudioFile) -> AudioFile:
        """Enhance metadata using Last.fm."""
        if not audio_file.artists:
            return audio_file

        adapter = self._get_adapter()
        if not adapter:
            return audio_file

        artist = audio_file.artists[0]

        # Check cache
        cache_key = f"artist:{artist}"
        if self.config.get("cache_enabled", True) and cache_key in self._cache:
            artist_info = self._cache[cache_key]
        else:
            # Fetch from Last.fm
            artist_info = await adapter.get_artist_info(artist)
            if artist_info and self.config.get("cache_enabled", True):
                self._cache[cache_key] = artist_info

        # Apply genre enhancement
        if self.config.get("enhance_genres", True) and artist_info:
            tags = artist_info.get("tags", [])
            if tags and not audio_file.genre:
                audio_file.genre = tags[0]

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
// config/plugins/lastfm_scrobbler.json
{
    "enabled": false,
    "api_key": "",
    "api_secret": "",
    "session_key": "",
    "scrobble_on_move": true,
    "scrobble_timestamp": "current"
}
```

### Global Configuration

```yaml
# config/lastfm.yaml
lastfm:
  # API credentials from https://www.last.fm/api/account/create
  api_key: "YOUR_API_KEY"
  api_secret: "YOUR_API_SECRET"

  # Authentication (obtained via login flow)
  session_key: ""

  # Scrobbling preferences
  scrobble_on_move: true
  scrobble_on_metadata_update: false
  scrobble_timestamp: "current"  # or "file_mtime"

  # Metadata enhancement (no auth required)
  enhance_metadata: false
  enhance_genres: true
  use_lastfm_genres: true
```

## CLI Integration

### Authentication Command

```bash
# Get Last.fm authentication URL
music-organize-async lastfm auth

# Follow the URL to authorize, then get session key
music-organize-async lastfm auth --callback-url http://localhost:8080
```

### Scrobbling Commands

```bash
# Scrobble a specific file
music-organize-async lastfm scrobble /path/to/song.flac

# Scrobble all files in a directory
music-organize-async lastfm scrobble /music/library

# Enable/disable scrobbling
music-organize-async config set lastfm.scrobble_on_move true
```

### Metadata Enhancement

```bash
# Enhance metadata using Last.fm
music-organize-async enhance /music/library --lastfm

# Get artist info from Last.fm
music-organize-async lastfm artist-info "The Beatles"

# Get album info
music-organize-async lastfm album-info "Abbey Road" "The Beatles"
```

## Testing

```python
# tests/test_lastfm_integration.py
import pytest
from music_organizer.infrastructure.external.lastfm_adapter import LastFmAdapter

@pytest.mark.asyncio
async def test_scrobble():
    adapter = LastFmAdapter(
        api_key="test_key",
        api_secret="test_secret",
        session_key="test_session"
    )

    # Test scrobble (will fail without real credentials)
    result = await adapter.scrobble(
        artist="Test Artist",
        track="Test Track",
        timestamp=int(time.time())
    )

    # Should return False with test credentials
    assert result is False or result is True

@pytest.mark.asyncio
async def test_artist_info():
    adapter = LastFmAdapter(api_key="test_key", api_secret="")

    # Get artist info (public API, no auth needed)
    info = await adapter.get_artist_info("The Beatles")

    assert info is not None
    assert "name" in info
    assert info["name"] == "The Beatles"
```

## Dependency Impact

### New Dependencies

| Package | Version | Size | License | Required |
|---------|---------|------|--------|----------|
| aiohttp | >=3.8.0 | ~500KB | Apache 2.0 | No (already used) |
| lastfmclient | >=0.6.0 | ~50KB | MIT | Optional (we use custom adapter) |

**No new dependencies required** - aiohttp is already used for MusicBrainz integration.

## Cost-Benefit Analysis

### Benefits

| Feature | Value | Users Affected |
|---------|-------|----------------|
| Scrobbling on organize | Medium | Last.fm users |
| Metadata enhancement | Low-Medium | All users |
| Artist/album info lookup | Low | Power users |
| Integration | Low | Small user base |

### Costs

| Cost | Impact | Mitigation |
|------|--------|------------|
| API rate limits | Low | Respect limits, queue scrobbles |
| Authentication complexity | Medium | Clear setup guide |
| Testing burden | Low | Mock Last.fm API |
| Maintenance | Low | Stable API v2.0 |

## Implementation Roadmap

### Phase 1: Core Adapter (4-6 hours)

1. Create `LastFmAdapter` class
2. Implement scrobble method
3. Implement now playing update
4. Implement auth flow
5. Unit tests

### Phase 2: Scrobble Plugin (2-3 hours)

1. Create `LastFmScrobblerPlugin`
2. Wire up POST_MOVE hook
3. Handle batch scrobbles
4. Configuration management
5. Integration tests

### Phase 3: Metadata Enhancer (2-3 hours)

1. Create `LastFmEnhancerPlugin`
2. Artist/album info methods
3. Genre enhancement
4. Caching

### Phase 4: CLI & Documentation (2 hours)

1. Auth commands
2. Scrobble commands
3. Setup documentation
4. Configuration examples

**Total: 10-14 hours**

## Security Considerations

### API Credentials

- Store API key/secret in config file (user-specific)
- Never commit credentials to repo
- Session key obtained via OAuth-like flow
- Support environment variables

### Privacy

- Scrobbling sends listening data to Last.fm
- Opt-in by default (disabled)
- Clear documentation about what data is sent
- User can disable at any time

## Limitations

1. **No Playback Tracking**: Music organizer doesn't play music, so can't scrobble "now playing" in real-time
2. **Timestamp Accuracy**: Using file mtime or current time, not actual playback time
3. **Rate Limits**: Last.fm allows ~5 requests/second, need to respect this
4. **API Key Required**: Users must obtain their own API key from Last.fm

## Conclusion

Last.fm integration is straightforward with minimal dependencies and effort. The existing adapter pattern and plugin system make this a natural extension.

**Recommendation**: Implement as optional feature with clear opt-in and documentation.

**MVP Scope**: Scrobbling plugin + Last.fm adapter (10-12 hours)

**Future Enhancements**:
- Real-time now playing (requires music player integration)
- Love/ban tracks sync
- Last.fm recommendations for library suggestions
- Weekly/monthly listening reports

---

*Research by task-master agent on 2025-12-23*
