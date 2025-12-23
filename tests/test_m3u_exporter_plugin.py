"""Tests for the M3U exporter plugin."""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import tempfile

from music_organizer.plugins.builtins.m3u_exporter import M3UExporterPlugin
from music_organizer.models.audio_file import AudioFile


class TestM3UExporterPlugin:
    """Tests for the M3UExporterPlugin class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = M3UExporterPlugin()
        self.sample_audio_files = [
            AudioFile(
                path=Path("/music/artist1/album1/01 - song1.mp3"),
                file_type="mp3",
                artists=["Artist 1"],
                title="Song 1"
            ),
            AudioFile(
                path=Path("/music/artist2/album2/02 - song2.flac"),
                file_type="flac",
                artists=["Artist 2"],
                title="Song 2"
            ),
            AudioFile(
                path=Path("/music/artist3/album3/03 - song3.wav"),
                file_type="wav",
                title="Song 3"  # No artist
            ),
        ]

    def test_plugin_info(self):
        """Test plugin information."""
        info = self.plugin.info
        assert info.name == "m3u_exporter"
        assert info.version == "1.0.0"
        assert "m3u" in info.description.lower()
        assert info.author == "Music Organizer Team"
        assert info.dependencies == []

    def test_initialization(self):
        """Test plugin initialization."""
        with patch('builtins.print') as mock_print:
            self.plugin.initialize()
            mock_print.assert_called_once_with("M3U exporter plugin initialized")

    def test_cleanup(self):
        """Test plugin cleanup."""
        with patch('builtins.print') as mock_print:
            self.plugin.cleanup()
            mock_print.assert_called_once_with("M3U exporter plugin cleaned up")

    def test_get_supported_formats(self):
        """Test getting supported export formats."""
        formats = self.plugin.get_supported_formats()
        assert isinstance(formats, list)
        assert 'm3u' in formats

    def test_get_file_extension(self):
        """Test getting file extension."""
        extension = self.plugin.get_file_extension()
        assert extension == '.m3u'

    def test_get_duration_estimate(self):
        """Test duration estimate for M3U extended info."""
        duration = self.plugin._get_duration_estimate(
            AudioFile(
                path=Path("/test/song.mp3"),
                file_type="mp3"
            )
        )
        assert duration == -1  # Unknown duration placeholder

    @pytest.mark.asyncio
    async def test_export_basic_playlist(self):
        """Test basic playlist export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            result = await self.plugin.export(self.sample_audio_files, output_path)

            assert result is True
            assert output_path.exists()

            # Read and verify content
            content = output_path.read_text(encoding='utf-8')
            assert "#EXTM3U" in content
            assert "song1.mp3" in content
            assert "song2.flac" in content
            assert "song3.wav" in content

    @pytest.mark.asyncio
    async def test_export_creates_directory(self):
        """Test that export creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "playlist.m3u"

            result = await self.plugin.export(self.sample_audio_files, output_path)

            assert result is True
            assert output_path.exists()
            assert output_path.parent.exists()

    @pytest.mark.asyncio
    async def test_export_with_extended_info(self):
        """Test export with extended M3U info (#EXTINF)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            await self.plugin.export(self.sample_audio_files, output_path)

            content = output_path.read_text(encoding='utf-8')

            # Check for EXTINF entries
            assert "#EXTINF:-1,Artist 1 - Song 1" in content
            assert "#EXTINF:-1,Artist 2 - Song 2" in content
            assert "#EXTINF:-1,Song 3" in content  # No artist

    @pytest.mark.asyncio
    async def test_export_with_relative_paths(self):
        """Test export with relative file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlists" / "playlist.m3u"

            # Create audio files in subdirectory of output parent
            audio_files = [
                AudioFile(
                    path=Path(tmpdir) / "music" / "song1.mp3",
                    file_type="mp3",
                    artists=["Artist"],
                    title="Song"
                )
            ]

            await self.plugin.export(audio_files, output_path)

            content = output_path.read_text(encoding='utf-8')

            # Should use relative path if possible
            assert "../music/song1.mp3" in content or "music/song1.mp3" in content

    @pytest.mark.asyncio
    async def test_export_with_absolute_paths(self):
        """Test export with absolute file paths when relative not possible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            # Audio file on completely different path
            audio_files = [
                AudioFile(
                    path=Path("/completely/different/path/song.mp3"),
                    file_type="mp3",
                    artists=["Artist"],
                    title="Song"
                )
            ]

            await self.plugin.export(audio_files, output_path)

            content = output_path.read_text(encoding='utf-8')

            # Should use absolute path
            assert "/completely/different/path/song.mp3" in content

    @pytest.mark.asyncio
    async def test_export_empty_list(self):
        """Test export with empty audio file list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            result = await self.plugin.export([], output_path)

            assert result is True
            content = output_path.read_text(encoding='utf-8')
            assert content == "#EXTM3U\n"

    @pytest.mark.asyncio
    async def test_export_unicode_characters(self):
        """Test export with Unicode characters in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            unicode_files = [
                AudioFile(
                    path=Path("/music/Björk/album/song.mp3"),
                    file_type="mp3",
                    artists=["Björk"],
                    title="Sögur"
                ),
                AudioFile(
                    path=Path("/music/日本語/album/song.mp3"),
                    file_type="mp3",
                    artists=["日本語"],
                    title="タイトル"
                )
            ]

            result = await self.plugin.export(unicode_files, output_path)

            assert result is True
            content = output_path.read_text(encoding='utf-8')
            assert "Björk" in content
            assert "Sögur" in content
            assert "日本語" in content
            assert "タイトル" in content

    @pytest.mark.asyncio
    async def test_export_handles_errors(self):
        """Test export error handling."""
        # Use an invalid path (e.g., in a directory that doesn't exist and can't be created)
        output_path = Path("/root/nonexistent/playlist.m3u")  # No permissions

        result = await self.plugin.export(self.sample_audio_files, output_path)

        # Should handle error gracefully and return False
        assert result is False

    @pytest.mark.asyncio
    async def test_export_multiple_files(self):
        """Test export with multiple audio files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "large_playlist.m3u"

            # Create many files
            many_files = []
            for i in range(50):
                many_files.append(
                    AudioFile(
                        path=Path(f"/music/artist/album/{i:02d} - song.mp3"),
                        file_type="mp3",
                        artists=[f"Artist {i}"],
                        title=f"Song {i}"
                    )
                )

            result = await self.plugin.export(many_files, output_path)

            assert result is True
            content = output_path.read_text(encoding='utf-8')

            # Check header
            assert content.startswith("#EXTM3U")

            # Check that all files are included
            for i in range(50):
                assert f"Song {i}" in content

    @pytest.mark.asyncio
    async def test_export_overwrites_existing(self):
        """Test that export overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            # Create existing file
            output_path.write_text("old content")

            # Export should overwrite
            result = await self.plugin.export(self.sample_audio_files, output_path)

            assert result is True
            content = output_path.read_text(encoding='utf-8')
            assert "old content" not in content
            assert "#EXTM3U" in content

    @pytest.mark.asyncio
    async def test_export_preserves_file_order(self):
        """Test that export preserves the order of audio files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            # Files in specific order
            ordered_files = [
                AudioFile(
                    path=Path("/music/first.mp3"),
                    file_type="mp3",
                    artists=["A"],
                    title="First"
                ),
                AudioFile(
                    path=Path("/music/second.mp3"),
                    file_type="mp3",
                    artists=["B"],
                    title="Second"
                ),
                AudioFile(
                    path=Path("/music/third.mp3"),
                    file_type="mp3",
                    artists=["C"],
                    title="Third"
                ),
            ]

            await self.plugin.export(ordered_files, output_path)

            content = output_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            # Find lines with song titles
            first_idx = next(i for i, line in enumerate(lines) if "First" in line)
            second_idx = next(i for i, line in enumerate(lines) if "Second" in line)
            third_idx = next(i for i, line in enumerate(lines) if "Third" in line)

            assert first_idx < second_idx < third_idx

    @pytest.mark.asyncio
    async def test_export_files_without_metadata(self):
        """Test export with files missing some metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            minimal_files = [
                AudioFile(
                    path=Path("/music/no_metadata.mp3"),
                    file_type="mp3"
                    # No artists, title, etc.
                )
            ]

            result = await self.plugin.export(minimal_files, output_path)

            assert result is True
            content = output_path.read_text(encoding='utf-8')

            # Should still include file path
            assert "no_metadata.mp3" in content

    @pytest.mark.asyncio
    async def test_export_files_with_special_chars(self):
        """Test export with special characters in paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            special_files = [
                AudioFile(
                    path=Path("/music/artist/album/01 - song (feat. artist) [remix].mp3"),
                    file_type="mp3",
                    artists=["Artist"],
                    title="Song (feat. artist) [Remix]"
                )
            ]

            result = await self.plugin.export(special_files, output_path)

            assert result is True
            content = output_path.read_text(encoding='utf-8')
            assert "song (feat. artist) [remix].mp3" in content

    def test_enable_disable(self):
        """Test enabling and disabling plugin."""
        assert self.plugin.enabled is True

        self.plugin.disable()
        assert self.plugin.enabled is False

        self.plugin.enable()
        assert self.plugin.enabled is True

    def test_config_property(self):
        """Test plugin config property."""
        plugin = M3UExporterPlugin({'custom_config': 'value'})
        assert plugin.config == {'custom_config': 'value'}

    def test_plugin_is_output_plugin(self):
        """Test that plugin implements OutputPlugin interface."""
        from music_organizer.plugins.base import OutputPlugin
        assert isinstance(self.plugin, OutputPlugin)

    @pytest.mark.asyncio
    async def test_export_creates_nested_directories(self):
        """Test that export creates deeply nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "a" / "b" / "c" / "d" / "playlist.m3u"

            result = await self.plugin.export(self.sample_audio_files, output_path)

            assert result is True
            assert output_path.exists()


class TestM3UExporterEdgeCases:
    """Test edge cases and special scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = M3UExporterPlugin()

    @pytest.mark.asyncio
    async def test_export_with_windows_paths(self):
        """Test export with Windows-style paths (if applicable)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            # Simulate Windows path
            windows_file = AudioFile(
                path=Path("C:/Music/artist/album/song.mp3"),
                file_type="mp3",
                artists=["Artist"],
                title="Song"
            )

            result = await self.plugin.export([windows_file], output_path)

            assert result is True
            content = output_path.read_text(encoding='utf-8')
            assert "song.mp3" in content

    @pytest.mark.asyncio
    async def test_export_filename_without_title(self):
        """Test export when file has no title (uses filename)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            file_no_title = AudioFile(
                path=Path("/music/album/track01.mp3"),
                file_type="mp3",
                artists=["Artist"]
                # No title
            )

            await self.plugin.export([file_no_title], output_path)

            content = output_path.read_text(encoding='utf-8')
            assert "track01.mp3" in content

    @pytest.mark.asyncio
    async def test_export_title_only(self):
        """Test EXTINF entry when only title is available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            file_title_only = AudioFile(
                path=Path("/music/song.mp3"),
                file_type="mp3",
                title="Just a Title"
            )

            await self.plugin.export([file_title_only], output_path)

            content = output_path.read_text(encoding='utf-8')
            assert "#EXTINF:-1,Just a Title" in content

    @pytest.mark.asyncio
    async def test_export_empty_artists_list(self):
        """Test export when artists list is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "playlist.m3u"

            file_no_artists = AudioFile(
                path=Path("/music/song.mp3"),
                file_type="mp3",
                artists=[],
                title="Song Title"
            )

            await self.plugin.export([file_no_artists], output_path)

            content = output_path.read_text(encoding='utf-8')
            # Should use title only
            assert "#EXTINF:-1,Song Title" in content
