"""Tests for Last.fm integration including adapter and plugins."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from pathlib import Path

from music_organizer.infrastructure.external.lastfm_adapter import LastFmAdapter
from music_organizer.plugins.builtins.lastfm_enhancer import LastFmEnhancerPlugin
from music_organizer.plugins.builtins.lastfm_scrobbler import LastFmScrobblerPlugin
from music_organizer.models.audio_file import AudioFile


class MockAsyncContextManager:
    """Helper class for mocking async context managers."""
    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def lastfm_adapter():
    """Create a LastFmAdapter instance for testing."""
    return LastFmAdapter(
        api_key="test_key",
        api_secret="test_secret",
        session_key="test_session"
    )


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file for testing."""
    # Create a dummy file
    audio_path = tmp_path / "test_track.flac"
    audio_path.write_bytes(b"dummy flac data")

    audio_file = AudioFile.from_path(audio_path)
    audio_file.artists = ["Test Artist"]
    audio_file.title = "Test Track"
    audio_file.album = "Test Album"
    audio_file.track_number = 1
    audio_file.metadata["duration_seconds"] = 180

    return audio_file


class TestLastFmAdapter:
    """Tests for LastFmAdapter."""

    def test_init(self):
        """Test adapter initialization."""
        adapter = LastFmAdapter(
            api_key="key",
            api_secret="secret",
            session_key="session"
        )

        assert adapter.api_key == "key"
        assert adapter.api_secret == "secret"
        assert adapter.session_key == "session"
        assert adapter._session is None

    def test_sign_params(self, lastfm_adapter):
        """Test API signature generation."""
        params = {
            "method": "track.scrobble",
            "artist": "Test Artist",
            "track": "Test Track",
            "api_key": "test_key"
        }

        sig = lastfm_adapter._sign_params(params)

        # Signature should be an MD5 hex string
        assert len(sig) == 32
        assert all(c in "0123456789abcdef" for c in sig)

    def test_get_auth_url(self, lastfm_adapter):
        """Test authentication URL generation."""
        url = lastfm_adapter.get_auth_url()

        assert "https://www.last.fm/api/auth/" in url
        assert "api_key=test_key" in url

    def test_get_auth_url_with_callback(self, lastfm_adapter):
        """Test authentication URL with callback."""
        url = lastfm_adapter.get_auth_url(callback_url="http://localhost:8080")

        assert "https://www.last.fm/api/auth/" in url
        assert "api_key=test_key" in url
        # URL encoding will encode the callback URL
        assert "localhost" in url or "cb=" in url

    @pytest.mark.asyncio
    async def test_get_session(self, lastfm_adapter):
        """Test session creation."""
        # Skip this test in CI since mocking dynamic imports is tricky
        # The adapter creates sessions correctly in real usage
        import builtins
        original_import = builtins.__import__

        def selective_import(name, *args, **kwargs):
            if name == 'aiohttp':
                mock_aiohttp = MagicMock()
                mock_aiohttp.ClientSession = MagicMock(return_value=AsyncMock())
                mock_aiohttp.ClientTimeout = Mock(return_value=Mock())
                return mock_aiohttp
            return original_import(name, *args, **kwargs)

        builtins.__import__ = selective_import
        try:
            session = await lastfm_adapter._get_session()
            assert session is not None
        finally:
            builtins.__import__ = original_import

    @pytest.mark.asyncio
    async def test_rate_limit(self, lastfm_adapter):
        """Test rate limiting."""
        import time

        lastfm_adapter.rate_limit = 10.0  # 10 requests per second

        # First call should not delay much
        start = time.time()
        await lastfm_adapter._rate_limit()
        elapsed = time.time() - start

        assert elapsed < 0.2  # Should be nearly instant

        # Immediate second call should delay
        start = time.time()
        await lastfm_adapter._rate_limit()
        elapsed = time.time() - start

        assert elapsed >= 0.05  # Should have some delay

    @pytest.mark.asyncio
    async def test_scrobble_no_session_key(self, lastfm_adapter):
        """Test scrobble without session key."""
        lastfm_adapter.session_key = None

        result = await lastfm_adapter.scrobble(
            artist="Test Artist",
            track="Test Track",
            timestamp=1234567890
        )

        assert result is False

    def test_parse_response(self, lastfm_adapter):
        """Test XML response parsing."""
        xml = '<lfm status="ok"><scrobbles/></lfm>'
        root = lastfm_adapter._parse_response(xml)

        assert root is not None
        assert root.tag == "lfm"

    def test_parse_response_invalid(self, lastfm_adapter):
        """Test parsing invalid XML."""
        root = lastfm_adapter._parse_response("not xml")

        assert root is None

    def test_check_error_no_error(self, lastfm_adapter):
        """Test error checking when no error."""
        xml = '<lfm status="ok"><scrobbles/></lfm>'
        root = lastfm_adapter._parse_response(xml)

        error = lastfm_adapter._check_error(root)
        assert error is None

    def test_check_error_with_error(self, lastfm_adapter):
        """Test error checking with error."""
        xml = '<lfm status="failed"><error code="10">Invalid API key</error></lfm>'
        root = lastfm_adapter._parse_response(xml)

        error = lastfm_adapter._check_error(root)
        assert error is not None
        assert error["code"] == "10"

    def test_parse_artist_info(self, lastfm_adapter):
        """Test parsing artist info from XML."""
        xml = """
        <artist>
            <name>Test Artist</name>
            <mbid>12345678-1234-1234-1234-123456789012</mbid>
            <url>https://www.last.fm/music/Test+Artist</url>
            <stats>
                <listeners>1000</listeners>
                <playcount>10000</playcount>
            </stats>
            <tags>
                <tag><name>rock</name></tag>
                <tag><name>alternative</name></tag>
            </tags>
        </artist>
        """
        root = lastfm_adapter._parse_response(xml)
        info = lastfm_adapter._parse_artist_info(root)

        assert info["name"] == "Test Artist"
        assert info["listeners"] == 1000
        assert info["playcount"] == 10000
        assert "rock" in info["tags"]

    def test_parse_album_info(self, lastfm_adapter):
        """Test parsing album info from XML."""
        xml = """
        <album>
            <name>Test Album</name>
            <artist>Test Artist</artist>
            <mbid>87654321-4321-4321-4321-210987654321</mbid>
            <url>https://www.last.fm/music/Test+Artist/Test+Album</url>
            <stats>
                <listeners>500</listeners>
                <playcount>5000</playcount>
            </stats>
            <tags>
                <tag><name>rock</name></tag>
            </tags>
        </album>
        """
        root = lastfm_adapter._parse_response(xml)
        info = lastfm_adapter._parse_album_info(root)

        assert info["name"] == "Test Album"
        assert info["artist"] == "Test Artist"

    def test_parse_track_info(self, lastfm_adapter):
        """Test parsing track info from XML."""
        xml = """
        <track>
            <name>Test Track</name>
            <artist>
                <name>Test Artist</name>
            </artist>
            <album>
                <album>Test Album</album>
            </album>
            <duration>180000</duration>
            <mbid>11111111-2222-3333-4444-555555555555</mbid>
            <toptags>
                <tag><name>rock</name></tag>
            </toptags>
        </track>
        """
        root = lastfm_adapter._parse_response(xml)
        info = lastfm_adapter._parse_track_info(root)

        assert info["name"] == "Test Track"
        assert info["artist"] == "Test Artist"
        assert info["duration"] == 180  # Converted from ms


class TestLastFmEnhancerPlugin:
    """Tests for LastFmEnhancerPlugin."""

    def test_init(self):
        """Test plugin initialization."""
        plugin = LastFmEnhancerPlugin()

        assert plugin.info.name == "lastfm_enhancer"
        assert plugin.info.version == "1.0.0"
        assert plugin.enabled is True

    def test_info(self):
        """Test plugin info."""
        plugin = LastFmEnhancerPlugin()
        info = plugin.info

        assert info.name == "lastfm_enhancer"
        assert "Last.fm" in info.description
        assert "aiohttp" in info.dependencies

    def test_get_config_schema(self):
        """Test config schema."""
        plugin = LastFmEnhancerPlugin()
        schema = plugin.get_config_schema()

        assert "enabled" in schema._options
        assert "api_key" in schema._options
        assert "enhance_genres" in schema._options
        assert "cache_enabled" in schema._options

    def test_initialize(self):
        """Test plugin initialization."""
        plugin = LastFmEnhancerPlugin()
        plugin.initialize()

        assert plugin._adapter is None
        assert plugin._cache == {}

    def test_cleanup(self):
        """Test plugin cleanup."""
        plugin = LastFmEnhancerPlugin()
        plugin.initialize()
        plugin._cache["test"] = {"data": "test"}

        plugin.cleanup()

        assert plugin._cache == {}

    @pytest.mark.asyncio
    async def test_enhance_metadata_no_api_key(self, sample_audio_file):
        """Test enhancement without API key."""
        plugin = LastFmEnhancerPlugin({"api_key": ""})

        result = await plugin.enhance_metadata(sample_audio_file)

        # Should return file unchanged
        assert result == sample_audio_file

    @pytest.mark.asyncio
    async def test_enhance_metadata_no_artists(self, sample_audio_file):
        """Test enhancement without artist info."""
        plugin = LastFmEnhancerPlugin({"api_key": "test_key"})
        sample_audio_file.artists = []

        result = await plugin.enhance_metadata(sample_audio_file)

        assert result == sample_audio_file

    @pytest.mark.asyncio
    async def test_enhance_metadata_success(self, sample_audio_file):
        """Test successful metadata enhancement."""
        plugin = LastFmEnhancerPlugin({
            "api_key": "test_key",
            "enhance_genres": True
        })
        plugin.initialize()

        with patch.object(plugin, "_get_adapter") as mock_get_adapter:
            mock_adapter = AsyncMock()
            mock_adapter.get_artist_info = AsyncMock(return_value={
                "name": "Test Artist",
                "tags": ["rock", "alternative"]
            })
            mock_get_adapter.return_value = mock_adapter

            result = await plugin.enhance_metadata(sample_audio_file)

            assert result.genre == "rock"
            assert "artist:test artist" in plugin._cache

    @pytest.mark.asyncio
    async def test_batch_enhance(self, sample_audio_file):
        """Test batch enhancement."""
        plugin = LastFmEnhancerPlugin({
            "api_key": "test_key",
            "enhance_genres": True
        })
        plugin.initialize()

        files = [sample_audio_file]

        with patch.object(plugin, "_get_adapter") as mock_get_adapter:
            mock_adapter = AsyncMock()
            mock_adapter.get_artist_info = AsyncMock(return_value={
                "name": "Test Artist",
                "tags": ["rock"]
            })
            mock_get_adapter.return_value = mock_adapter

            results = await plugin.batch_enhance(files)

            assert len(results) == 1
            assert results[0].genre == "rock"


class TestLastFmScrobblerPlugin:
    """Tests for LastFmScrobblerPlugin."""

    def test_init(self):
        """Test plugin initialization."""
        plugin = LastFmScrobblerPlugin()

        assert plugin.info.name == "lastfm_scrobbler"
        assert plugin._scrobble_queue == []
        assert plugin._failed_scrobbles == []

    def test_info(self):
        """Test plugin info."""
        plugin = LastFmScrobblerPlugin()
        info = plugin.info

        assert info.name == "lastfm_scrobbler"
        assert "scrobble" in info.description.lower()
        assert "aiohttp" in info.dependencies

    def test_get_config_schema(self):
        """Test config schema."""
        plugin = LastFmScrobblerPlugin()
        schema = plugin.get_config_schema()

        assert "enabled" in schema._options
        assert "api_key" in schema._options
        assert "api_secret" in schema._options
        assert "session_key" in schema._options
        assert "batch_size" in schema._options
        assert "retry_failed" in schema._options

    def test_initialize(self):
        """Test plugin initialization."""
        plugin = LastFmScrobblerPlugin()
        plugin.initialize()

        assert plugin._scrobble_queue == []
        assert plugin._failed_scrobbles == []

    def test_cleanup(self):
        """Test plugin cleanup."""
        plugin = LastFmScrobblerPlugin()
        plugin.initialize()
        plugin._scrobble_queue.append({"test": "data"})

        plugin.cleanup()

        assert plugin._scrobble_queue == []
        assert plugin._failed_scrobbles == []

    def test_get_timestamp_current(self, sample_audio_file):
        """Test getting current timestamp."""
        plugin = LastFmScrobblerPlugin({
            "scrobble_timestamp": "current"
        })

        import time
        before = int(time.time())
        timestamp = plugin._get_timestamp(sample_audio_file)
        after = int(time.time())

        assert before <= timestamp <= after

    def test_get_timestamp_file_mtime(self, sample_audio_file):
        """Test getting file mtime timestamp."""
        plugin = LastFmScrobblerPlugin({
            "scrobble_timestamp": "file_mtime"
        })

        timestamp = plugin._get_timestamp(sample_audio_file)

        # Should match file's mtime
        assert timestamp == int(sample_audio_file.path.stat().st_mtime)

    def test_get_supported_formats(self):
        """Test supported formats."""
        plugin = LastFmScrobblerPlugin()

        assert "lastfm" in plugin.get_supported_formats()

    def test_get_file_extension(self):
        """Test file extension (empty for scrobbler)."""
        plugin = LastFmScrobblerPlugin()

        assert plugin.get_file_extension() == ""

    @pytest.mark.asyncio
    async def test_scrobble_single_no_adapter(self, sample_audio_file):
        """Test scrobbling without adapter."""
        plugin = LastFmScrobblerPlugin({"api_key": ""})

        result = await plugin.scrobble_single(sample_audio_file)

        assert result is False

    @pytest.mark.asyncio
    async def test_scrobble_single_success(self, sample_audio_file):
        """Test successful single scrobble."""
        plugin = LastFmScrobblerPlugin({
            "api_key": "test_key",
            "api_secret": "test_secret",
            "session_key": "test_session"
        })
        plugin.initialize()

        with patch.object(plugin, "_get_adapter") as mock_get_adapter:
            mock_adapter = AsyncMock()
            mock_adapter.scrobble = AsyncMock(return_value=True)
            mock_get_adapter.return_value = mock_adapter

            result = await plugin.scrobble_single(sample_audio_file)

            assert result is True
            mock_adapter.scrobble.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_disabled(self, sample_audio_file):
        """Test export when disabled."""
        plugin = LastFmScrobblerPlugin({
            "scrobble_on_export": False
        })

        result = await plugin.export([sample_audio_file], Path("/tmp/output"))

        assert result is True

    @pytest.mark.asyncio
    async def test_export_success(self, sample_audio_file):
        """Test successful export (scrobble)."""
        plugin = LastFmScrobblerPlugin({
            "api_key": "test_key",
            "api_secret": "test_secret",
            "session_key": "test_session",
            "scrobble_on_export": True
        })
        plugin.initialize()

        with patch.object(plugin, "_get_adapter") as mock_get_adapter:
            mock_adapter = AsyncMock()
            mock_adapter.scrobble = AsyncMock(return_value=True)
            mock_get_adapter.return_value = mock_adapter

            result = await plugin.export([sample_audio_file], Path("/tmp/output"))

            assert result is True
            mock_adapter.scrobble.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_with_failure(self, sample_audio_file):
        """Test export with some failures."""
        plugin = LastFmScrobblerPlugin({
            "api_key": "test_key",
            "api_secret": "test_secret",
            "session_key": "test_session",
            "scrobble_on_export": True,
            "retry_failed": False
        })
        plugin.initialize()

        with patch.object(plugin, "_get_adapter") as mock_get_adapter:
            mock_adapter = AsyncMock()
            mock_adapter.scrobble = AsyncMock(return_value=False)
            mock_get_adapter.return_value = mock_adapter

            result = await plugin.export([sample_audio_file], Path("/tmp/output"))

            assert result is False
            assert len(plugin._failed_scrobbles) == 1

    def test_get_failed_scrobbles(self, sample_audio_file):
        """Test getting failed scrobbles."""
        plugin = LastFmScrobblerPlugin()
        plugin._failed_scrobbles = [{"audio_file": sample_audio_file, "retries": 1}]

        failed = plugin.get_failed_scrobbles()

        assert len(failed) == 1
        assert failed[0]["audio_file"] == sample_audio_file

    def test_clear_failed_scrobbles(self, sample_audio_file):
        """Test clearing failed scrobbles."""
        plugin = LastFmScrobblerPlugin()
        plugin._failed_scrobbles = [{"audio_file": sample_audio_file, "retries": 1}]

        plugin.clear_failed_scrobbles()

        assert plugin._failed_scrobbles == []
