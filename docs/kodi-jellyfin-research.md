# Kodi/Jellyfin Compatibility Mode Research

**Research Date**: 2025-12-23
**Status**: Technical Investigation Complete - Implementation Plan Ready

## Executive Summary

The music-organizer has excellent metadata and cover art handling capabilities that can be extended to generate Kodi/Jellyfin-compatible NFO files and directory structures. The existing plugin system makes this a natural extension.

**Recommendation**: Implement as an optional OutputPlugin that generates NFO files alongside music organization. Leverage existing MusicBrainz integration for rich metadata.

**Estimated Effort**: 12-16 hours for full NFO generation with MBID support

## Current Capabilities

### Available Metadata

The system has comprehensive metadata suitable for NFO generation:

| Kodi/Jellyfin Field | Source | Status |
|---------------------|--------|--------|
| title | `audio_file.title` | ✅ Available |
| artist | `audio_file.artists` | ✅ Available |
| album artist | `audio_file.primary_artist` | ✅ Available |
| album | `audio_file.album` | ✅ Available |
| year | `audio_file.year` | ✅ Available |
| track number | `audio_file.track_number` | ✅ Available |
| genre | `audio_file.genre` | ✅ Available |
| duration | `metadata.duration_seconds` | ✅ Available |
| rating | (new) | ⚠️ Could add from MusicBrainz |
| mood | (new) | ⚠️ Could add from Last.fm |
| style | (new) | ⚠️ Could derive from genre |
| biography | (new) | ⚠️ From MusicBrainz |
| review | (new) | ⚠️ From MusicBrainz |
| MBID | (new) | ⚠️ From MusicBrainz adapter |

### Cover Art Support

Existing cover art capabilities:
- `CoverArt` model with types: front, back, disc, other
- Automatic detection from filenames (folder, cover, front, etc.)
- Formats: jpg, jpeg, png, gif
- Cover art movement during organization
- `has_cover_art` flag in AudioFile

## Kodi/Jellyfin Requirements

### NFO File Structure

**Artist NFO** (`artist.nfo`):
```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<artist>
  <name>The Beatles</name>
  <born>1960</born>
  <formed>1960</formed>
  <disbanded>1970</disbanded>
  <genre>Rock</genre>
  <style>Pop Rock</style>
  <mood>Upbeat</mood>
  <biography>The Beatles were an English rock band...</biography>
  <musicBrainzArtistID>b10bbbfc-cf9e-42e0-be17-e2c3e1e2605d</musicBrainzArtistID>
  <thumb preview="https://last.fm/...">https://last.fm/...</thumb>
  <album>
    <title>Abbey Road</title>
    <year>1969</year>
    <musicBrainzReleaseGroupID>...</musicBrainzReleaseGroupID>
  </album>
</artist>
```

**Album NFO** (`album.nfo`):
```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<album>
  <title>Abbey Road</title>
  <artist>The Beatles</artist>
  <review>Abbey Road is the eleventh studio album...</review>
  <musicBrainzReleaseGroupID>...</musicBrainzReleaseGroupID>
  <musicBrainzReleaseID>...</musicBrainzReleaseID>
  <year>1969</year>
  <rating>8.5</rating>
  <genre>Rock</genre>
  <style>Pop Rock</style>
  <mood>Classic</mood>
  <track>
    <position>1</position>
    <title>Come Together</title>
    <duration>259</duration>
    <musicBrainzTrackID>...</musicBrainzTrackID>
  </track>
  <track>
    <position>2</position>
    <title>Something</title>
    <duration>183</duration>
  </track>
</album>
```

### Directory Structure

```
Music/
├── Artists/
│   └── The Beatles/
│       ├── artist.nfo
│       ├── fanart.jpg
│       ├── folder.jpg
│       └── Albums/
│           └── Abbey Road (1969)/
│               ├── album.nfo
│               ├── folder.jpg
│               ├── discart.jpg
│               ├── 01. Come Together.flac
│               └── 02. Something.flac
```

## Proposed Implementation

### Component 1: NFO Generator

```python
# src/music_organizer/infrastructure/nfo/nfo_generator.py
"""
NFO file generator for Kodi/Jellyfin compatibility.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

from ...domain.catalog import Metadata, Recording, Release


@dataclass
class NfoConfig:
    """Configuration for NFO generation."""
    include_bios: bool = True
    include_reviews: bool = True
    include_ratings: bool = True
    include_moods: bool = True
    indent_xml: bool = True


class NfoGenerator:
    """Generate NFO files for Kodi/Jellyfin media centers."""

    def __init__(self, config: NfoConfig = None):
        self.config = config or NfoConfig()

    def generate_artist_nfo(
        self,
        name: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Generate artist NFO XML content."""
        root = ET.Element("artist")

        # Basic info
        ET.SubElement(root, "name").text = name

        # Dates
        if metadata.get("born"):
            ET.SubElement(root, "born").text = str(metadata["born"])
        if metadata.get("formed"):
            ET.SubElement(root, "formed").text = str(metadata["formed"])
        if metadata.get("disbanded"):
            ET.SubElement(root, "disbanded").text = str(metadata["disbanded"])

        # Genres
        if metadata.get("genre"):
            ET.SubElement(root, "genre").text = metadata["genre"]
        if metadata.get("style"):
            ET.SubElement(root, "style").text = metadata["style"]
        if metadata.get("mood"):
            ET.SubElement(root, "mood").text = metadata["mood"]

        # Biography
        if self.config.include_bios and metadata.get("biography"):
            ET.SubElement(root, "biography").text = metadata["biography"]

        # MusicBrainz ID
        if metadata.get("musicbrainz_artist_id"):
            ET.SubElement(root, "musicBrainzArtistID").text = metadata["musicbrainz_artist_id"]

        # Thumbnail/Fanart
        if metadata.get("thumb"):
            thumb = ET.SubElement(root, "thumb")
            thumb.set("preview", metadata["thumb"])
            thumb.text = metadata["thumb"]

        # Albums
        if metadata.get("albums"):
            for album in metadata["albums"]:
                album_elem = ET.SubElement(root, "album")
                ET.SubElement(album_elem, "title").text = album.get("title", "")
                ET.SubElement(album_elem, "year").text = str(album.get("year", ""))
                if album.get("musicbrainz_releasegroup_id"):
                    ET.SubElement(album_elem, "musicBrainzReleaseGroupID").text = album["musicbrainz_releasegroup_id"]

        # Fanart
        if metadata.get("fanart"):
            fanart = ET.SubElement(root, "fanart")
            thumb = ET.SubElement(fanart, "thumb")
            thumb.text = metadata["fanart"]

        return self._to_xml(root)

    def generate_album_nfo(
        self,
        title: str,
        artist: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Generate album NFO XML content."""
        root = ET.Element("album")

        # Basic info
        ET.SubElement(root, "title").text = title
        ET.SubElement(root, "artist").text = artist

        # Review
        if self.config.include_reviews and metadata.get("review"):
            ET.SubElement(root, "review").text = metadata["review"]

        # MusicBrainz IDs
        if metadata.get("musicbrainz_releasegroup_id"):
            ET.SubElement(root, "musicBrainzReleaseGroupID").text = metadata["musicbrainz_releasegroup_id"]
        if metadata.get("musicbrainz_release_id"):
            ET.SubElement(root, "musicBrainzReleaseID").text = metadata["musicbrainz_release_id"]

        # Year and rating
        if metadata.get("year"):
            ET.SubElement(root, "year").text = str(metadata["year"])
        if self.config.include_ratings and metadata.get("rating"):
            ET.SubElement(root, "rating").text = str(metadata["rating"])

        # Genres
        if metadata.get("genre"):
            ET.SubElement(root, "genre").text = metadata["genre"]
        if metadata.get("style"):
            ET.SubElement(root, "style").text = metadata["style"]
        if self.config.include_moods and metadata.get("mood"):
            ET.SubElement(root, "mood").text = metadata["mood"]

        # Tracks
        if metadata.get("tracks"):
            for track in metadata["tracks"]:
                track_elem = ET.SubElement(root, "track")
                ET.SubElement(track_elem, "position").text = str(track.get("position", ""))
                ET.SubElement(track_elem, "title").text = track.get("title", "")
                if track.get("duration"):
                    ET.SubElement(track_elem, "duration").text = str(track["duration"])
                if track.get("musicbrainz_track_id"):
                    ET.SubElement(track_elem, "musicBrainzTrackID").text = track["musicbrainz_track_id"]

        return self._to_xml(root)

    def _to_xml(self, root: ET.Element) -> str:
        """Convert ElementTree to XML string."""
        if self.config.indent_xml:
            # Pretty print
            from xml.dom import minidom
            rough_string = ET.tostring(root, encoding="utf-8")
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
        else:
            return ET.tostring(root, encoding="utf-8").decode("utf-8")

    def write_artist_nfo(self, path: Path, name: str, metadata: Dict[str, Any]) -> None:
        """Write artist.nfo to disk."""
        content = self.generate_artist_nfo(name, metadata)
        path.write_text(content, encoding="utf-8")

    def write_album_nfo(self, path: Path, title: str, artist: str, metadata: Dict[str, Any]) -> None:
        """Write album.nfo to disk."""
        content = self.generate_album_nfo(title, artist, metadata)
        path.write_text(content, encoding="utf-8")
```

### Component 2: MusicBrainz MBID Extension

```python
# src/music_organizer/infrastructure/external/musicbrainz_mbid.py
"""
Extended MusicBrainz adapter for MBID and rich metadata.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any


class MusicBrainzMbidAdapter:
    """Extended MusicBrainz adapter for MBID and rich metadata."""

    def __init__(
        self,
        base_url: str = "https://musicbrainz.org/ws/2",
        user_agent: str = "MusicOrganizer/1.0",
        rate_limit: float = 1.0
    ):
        self.base_url = base_url
        self.user_agent = user_agent
        self.rate_limit = rate_limit
        self._last_request = 0

    async def _rate_limit(self):
        """Apply rate limiting."""
        import time
        current = time.time()
        if current - self._last_request < 1.0 / self.rate_limit:
            await asyncio.sleep(1.0 / self.rate_limit - (current - self._last_request))
        self._last_request = time.time()

    async def get_artist_with_mbid(
        self,
        artist: str
    ) -> Optional[Dict[str, Any]]:
        """Get artist info including MBID."""
        await self._rate_limit()

        url = f"{self.base_url}/artist/"
        params = {
            "query": f'artist:"{artist}"',
            "limit": 1,
            "fmt": "json"
        }

        try:
            async with aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent}
            ) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("artists"):
                            a = data["artists"][0]
                            return {
                                "name": a.get("name"),
                                "musicbrainz_artist_id": a.get("id"),
                                "type": a.get("type"),
                                "country": a.get("country"),
                                "life_span": a.get("life-span"),
                                "genres": [g.get("name") for g in a.get("genres", [])]
                            }
        except Exception as e:
            print(f"Error fetching artist MBID: {e}")

        return None

    async def get_release_with_mbid(
        self,
        release_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get release info with full MBIDs."""
        await self._rate_limit()

        url = f"{self.base_url}/release/{release_id}"
        params = {
            "inc": "recordings+artist-credits+release-groups",
            "fmt": "json"
        }

        try:
            async with aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent}
            ) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "id": data.get("id"),
                            "title": data.get("title"),
                            "musicbrainz_release_id": data.get("id"),
                            "musicbrainz_releasegroup_id": data.get("release-group", {}).get("id"),
                            "date": data.get("date"),
                            "country": data.get("country"),
                            "tracks": [
                                {
                                    "position": t.get("position"),
                                    "number": t.get("number"),
                                    "title": t.get("title"),
                                    "id": t.get("id"),
                                    "musicbrainz_track_id": t.get("id"),
                                    "duration": t.get("length", 0) // 1000 if t.get("length") else 0
                                }
                                for t in data.get("media", [{}])[0].get("tracks", [])
                            ]
                        }
        except Exception as e:
            print(f"Error fetching release MBID: {e}")

        return None
```

### Component 3: Kodi NFO Plugin

```python
# src/music_organizer/plugins/builtins/kodi_nfo_exporter.py
"""
Kodi/Jellyfin NFO export plugin.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

from ..base import OutputPlugin, PluginInfo
from ...plugins.config import PluginConfigSchema, ConfigOption
from ...infrastructure.nfo.nfo_generator import NfoGenerator, NfoConfig
from ...infrastructure.external.musicbrainz_mbid import MusicBrainzMbidAdapter
from ...models.audio_file import AudioFile

logger = logging.getLogger(__name__)


class KodiNfoExporterPlugin(OutputPlugin):
    """Plugin to generate Kodi/Jellyfin NFO files during organization."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="kodi_nfo_exporter",
            version="1.0.0",
            description="Generates NFO files for Kodi/Jellyfin compatibility",
            author="Music Organizer Team",
            dependencies=[],
        )

    def get_config_schema(self) -> PluginConfigSchema:
        return PluginConfigSchema([
            ConfigOption(
                name="enabled",
                type=bool,
                default=False,
                description="Enable NFO generation"
            ),
            ConfigOption(
                name="generate_artist_nfo",
                type=bool,
                default=True,
                description="Generate artist.nfo files"
            ),
            ConfigOption(
                name="generate_album_nfo",
                type=bool,
                default=True,
                description="Generate album.nfo files"
            ),
            ConfigOption(
                name="include_bios",
                type=bool,
                default=True,
                description="Include artist biographies"
            ),
            ConfigOption(
                name="include_reviews",
                type=bool,
                default=True,
                description="Include album reviews"
            ),
            ConfigOption(
                name="fetch_mbid",
                type=bool,
                default=True,
                description="Fetch MusicBrainz IDs"
            ),
            ConfigOption(
                name="use_kodi_structure",
                type=bool,
                default=False,
                description="Use Kodi-specific directory structure"
            ),
        ])

    def initialize(self) -> None:
        config = NfoConfig(
            include_bios=self.config.get("include_bios", True),
            include_reviews=self.config.get("include_reviews", True),
            include_moods=True,
            indent_xml=True
        )
        self.nfo_generator = NfoGenerator(config)
        self.mbid_adapter: Any = None
        self._artist_cache: Dict[str, Dict] = {}
        self._album_cache: Dict[str, Dict] = {}

    def cleanup(self) -> None:
        self._artist_cache.clear()
        self._album_cache.clear()

    async def _get_artist_metadata(self, artist: str) -> Dict[str, Any]:
        """Get metadata for artist NFO."""
        cache_key = f"artist:{artist}"

        if cache_key in self._artist_cache:
            return self._artist_cache[cache_key]

        metadata = {
            "genre": "Unknown",
            "style": "Unknown",
            "mood": "Unknown"
        }

        # Fetch from MusicBrainz if enabled
        if self.config.get("fetch_mbid", True):
            if self.mbid_adapter is None:
                self.mbid_adapter = MusicBrainzMbidAdapter()

            mb_data = await self.mbid_adapter.get_artist_with_mbid(artist)
            if mb_data:
                metadata.update({
                    "musicbrainz_artist_id": mb_data.get("musicbrainz_artist_id"),
                    "type": mb_data.get("type"),
                    "country": mb_data.get("country"),
                    "formed": mb_data.get("life_span", {}).get("begin"),
                    "disbanded": mb_data.get("life_span", {}).get("end"),
                    "genre": ", ".join(mb_data.get("genres", []) or ["Unknown"])
                })

        self._artist_cache[cache_key] = metadata
        return metadata

    async def _get_album_metadata(self, album: str, artist: str) -> Dict[str, Any]:
        """Get metadata for album NFO."""
        cache_key = f"album:{artist}:{album}"

        if cache_key in self._album_cache:
            return self._album_cache[cache_key]

        metadata = {
            "genre": "Unknown",
            "style": "Unknown",
            "mood": "Unknown",
            "tracks": []
        }

        self._album_cache[cache_key] = metadata
        return metadata

    async def after_move(self, audio_file: AudioFile, target_path: Path) -> AudioFile:
        """Generate NFO files after moving a file."""
        target_dir = target_path.parent

        # Generate album.nfo
        if self.config.get("generate_album_nfo", True):
            if audio_file.album and audio_file.artists:
                artist = audio_file.artists[0]
                album_metadata = await self._get_album_metadata(audio_file.album, artist)
                album_metadata["year"] = audio_file.year
                album_metadata["genre"] = audio_file.genre or "Unknown"

                nfo_path = target_dir / "album.nfo"
                if not nfo_path.exists():
                    self.nfo_generator.write_album_nfo(
                        nfo_path,
                        audio_file.album,
                        artist,
                        album_metadata
                    )
                    logger.info(f"Generated album.nfo for {audio_file.album}")

        # Generate artist.nfo (if using Kodi structure)
        if self.config.get("use_kodi_structure", False) and self.config.get("generate_artist_nfo", True):
            if audio_file.artists:
                # Navigate to artist directory
                artist_dir = target_dir.parent.parent if "Albums" in str(target_dir) else target_dir.parent

                if artist_dir.name != audio_file.artists[0]:
                    artist_dir = artist_dir / audio_file.artists[0]

                artist_metadata = await self._get_artist_metadata(audio_file.artists[0])
                nfo_path = artist_dir / "artist.nfo"
                if not nfo_path.exists():
                    self.nfo_generator.write_artist_nfo(
                        nfo_path,
                        audio_file.artists[0],
                        artist_metadata
                    )
                    logger.info(f"Generated artist.nfo for {audio_file.artists[0]}")

        return audio_file

    async def batch_after_move(self, audio_files: List[AudioFile]) -> List[AudioFile]:
        """Generate NFO files for batch operations."""
        for audio_file in audio_files:
            await self.after_move(audio_file, audio_file.path)
        return audio_files
```

## Configuration

### Plugin Configuration

```json
// config/plugins/kodi_nfo_exporter.json
{
    "enabled": false,
    "generate_artist_nfo": true,
    "generate_album_nfo": true,
    "include_bios": true,
    "include_reviews": true,
    "fetch_mbid": true,
    "use_kodi_structure": false
}
```

### Global Configuration

```yaml
# config/kodi.yaml
kodi:
  enabled: false

  # NFO generation
  nfo:
    artist: true
    album: true
    include_bios: true
    include_reviews: true
    include_ratings: true

  # MusicBrainz integration
  musicbrainz:
    fetch_mbid: true
    fetch_bios: true
    fetch_reviews: true

  # Cover art
  cover_art:
    copy_to_folder: true
    generate_fanart: false
    generate_discart: false
    max_resolution: 1024

  # Directory structure
  structure:
    use_kodi_layout: false
    separate_artists: false
    artist_folder_name: "Artists"
    album_folder_name: "Albums"
```

## CLI Integration

```bash
# Enable NFO generation
music-organize-async organize /source /target --kodi-nfo

# Generate NFOs for existing library
music-organize-async kodi generate-nfo /music/library

# Generate only artist NFOs
music-organize-async kodi generate-nfo /music/library --artist-only

# Fetch all MusicBrainz data
music-organize-async kodi generate-nfo /music/library --fetch-mbid --bios --reviews

# Use Kodi directory structure
music-organize-async organize /source /target --kodi-structure --kodi-nfo
```

## Testing

```python
# tests/test_kodi_nfo.py
import pytest
from music_organizer.infrastructure.nfo.nfo_generator import NfoGenerator, NfoConfig
from pathlib import Path

def test_artist_nfo_generation():
    generator = NfoGenerator()
    metadata = {
        "born": "1940",
        "genre": "Rock",
        "biography": "Test biography",
        "musicbrainz_artist_id": "test-id-123"
    }

    xml = generator.generate_artist_nfo("Test Artist", metadata)

    assert "<?xml version" in xml
    assert "<artist>" in xml
    assert "<name>Test Artist</name>" in xml
    assert "<musicBrainzArtistID>test-id-123</musicBrainzArtistID>" in xml

def test_album_nfo_generation():
    generator = NfoGenerator()
    metadata = {
        "year": 1969,
        "genre": "Rock",
        "review": "Great album",
        "tracks": [
            {"position": 1, "title": "Song 1", "duration": 180}
        ]
    }

    xml = generator.generate_album_nfo("Test Album", "Test Artist", metadata)

    assert "<?xml version" in xml
    assert "<album>" in xml
    assert "<title>Test Album</title>" in xml
    assert "<year>1969</year>" in xml

@pytest.mark.asyncio
async def test_nfo_file_writing(tmp_path):
    generator = NfoGenerator()
    nfo_path = tmp_path / "test.nfo"

    generator.write_album_nfo(
        nfo_path,
        "Test Album",
        "Test Artist",
        {"year": 2023}
    )

    assert nfo_path.exists()
    content = nfo_path.read_text()
    assert "Test Album" in content
```

## Dependency Impact

**No new dependencies required** - uses Python standard library `xml.etree.ElementTree`.

## Implementation Roadmap

### Phase 1: Core NFO Generator (4-6 hours)

1. Create `NfoGenerator` class
2. Implement artist NFO generation
3. Implement album NFO generation
4. Add XML pretty printing
5. Unit tests

### Phase 2: MusicBrainz MBID (3-4 hours)

1. Extend MusicBrainz adapter for MBID
2. Fetch artist biographies
3. Fetch album reviews
4. Cache results

### Phase 3: Plugin Integration (3-4 hours)

1. Create `KodiNfoExporterPlugin`
2. Wire up POST_MOVE hook
3. Handle batch operations
4. Configuration management

### Phase 4: Kodi Structure (2 hours)

1. Optional Kodi directory layout
2. Artist folder separation
3. Cover art handling
4. Integration tests

**Total: 12-16 hours**

## Limitations

1. **Biography Source**: Requires MusicBrainz/Wikipedia API (rate limited)
2. **Ratings**: No native rating system in music-organizer (could add)
3. **Mood/Style**: Derived from genre (no ML mood detection yet)
4. **Fanart**: Requires external image sources (Last.fm, Fanart.tv)

## Conclusion

Kodi/Jellyfin NFO generation is straightforward with minimal dependencies. The existing metadata and plugin system provide excellent foundation.

**Recommendation**: Implement as optional OutputPlugin with clear documentation.

**MVP Scope**: Basic album/artist NFO generation (8-10 hours)

**Future Enhancements**:
- Fanart.tv integration for artwork
- Kodi-style directory layout option
- Rating system integration
- Mood detection from audio features

---

*Research by task-master agent on 2025-12-23*
