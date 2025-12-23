"""Tests for Interactive Duplicate Resolution System."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from src.music_organizer.core.interactive_duplicate_resolver import (
    InteractiveDuplicateResolver,
    DuplicateAction,
    ResolutionStrategy,
    DuplicateQualityScorer,
    DuplicateGroup,
    DuplicatePair,
    ResolutionDecision,
    ResolutionSummary
)
# quick_duplicate_resolution function doesn't exist - removed import
from src.music_organizer.ui.duplicate_resolver_ui import DuplicateResolverUI
from src.music_organizer.models.audio_file import AudioFile
from src.music_organizer.domain.value_objects import Metadata
from src.music_organizer.domain.value_objects import FileFormat, ArtistName
from src.music_organizer.exceptions import MusicOrganizerError


class TestDuplicateQualityScorer:
    """Test the DuplicateQualityScorer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = DuplicateQualityScorer()

    def test_score_file_flac_high_quality(self):
        """Test scoring a high-quality FLAC file."""
        audio_file = AudioFile(
            path=Path("/test/song.flac"),
            file_type='flac',
            metadata={'bitrate': 1000, 'sample_rate': 96000},
            title="Test Song",
            artists=["Test Artist"],
            album="Test Album",
            year=2023,
            genre="Rock",
            track_number=1
        )

        # Mock path.stat() to return file size
        with patch.object(Path, 'stat', return_value=MagicMock(st_size=50 * 1024 * 1024)):
            score = self.scorer.score_file(audio_file)

        assert score > 7.0  # Should be high quality

    def test_score_file_mp3_low_quality(self):
        """Test scoring a low-quality MP3 file."""
        audio_file = AudioFile(
            path=Path("/test/song.mp3"),
            file_type='mp3',
            metadata={'bitrate': 128, 'sample_rate': 44100},
            title="Test Song",
            artists=["Test Artist"]
            # Missing: album, year, genre, track_number
        )

        # Mock path.stat() to return file size
        with patch.object(Path, 'stat', return_value=MagicMock(st_size=3 * 1024 * 1024)):
            score = self.scorer.score_file(audio_file)

        assert score < 6.0  # Should be lower quality

    def test_choose_best_single_file(self):
        """Test choosing best from a single file."""
        audio_file = AudioFile(
            path=Path("/test/song.flac"),
            file_type='flac',
            metadata={}
        )
        files = [audio_file]

        best_file, best_score = self.scorer.choose_best(files)

        assert best_file == audio_file
        assert isinstance(best_score, float)

    def test_choose_best_multiple_files(self):
        """Test choosing best from multiple files."""
        # Create mock files with different qualities
        flac_file = AudioFile(
            path=Path("/test/song.flac"),
            file_type='flac',
            metadata={'bitrate': 1000, 'sample_rate': 96000},
            title="Test",
            artists=["Artist"],
            album="Album",
            year=2023,
            genre="Rock",
            track_number=1
        )

        mp3_file = AudioFile(
            path=Path("/test/song.mp3"),
            file_type='mp3',
            metadata={'bitrate': 192, 'sample_rate': 44100},
            title="Test",
            artists=["Artist"],
            album="Album",
            year=2023,
            genre="Rock",
            track_number=1
        )

        files = [mp3_file, flac_file]

        # Mock path.stat() to return file sizes
        with patch.object(Path, 'stat', return_value=MagicMock(st_size=50 * 1024 * 1024)):
            best_file, best_score = self.scorer.choose_best(files)

        assert best_file == flac_file
        # Note: We can't compare scores directly due to mocking, so just verify the choice


class TestInteractiveDuplicateResolver:
    """Test the InteractiveDuplicateResolver class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.duplicate_dir = self.temp_dir / "duplicates"
        self.resolver = InteractiveDuplicateResolver(
            strategy=ResolutionStrategy.AUTO_KEEP_BEST,
            duplicate_dir=self.duplicate_dir,
            dry_run=False
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_audio_file(self, name: str, format: str = 'mp3', quality: str = 'medium') -> AudioFile:
        """Create a mock audio file for testing."""
        audio_file_path = self.temp_dir / f"{name}.{format}"

        # Set quality based on parameters
        if quality == 'high':
            bitrate = 1000 if format == 'flac' else 320
            sample_rate = 96000 if format == 'flac' else 48000
            file_size = 50 * 1024 * 1024
        elif quality == 'low':
            bitrate = 128
            sample_rate = 22050
            file_size = 2 * 1024 * 1024
        else:  # medium
            bitrate = 192
            sample_rate = 44100
            file_size = 5 * 1024 * 1024

        audio_file = AudioFile(
            path=audio_file_path,
            file_type=format,
            metadata={'bitrate': bitrate, 'sample_rate': sample_rate},
            title=f"Test Song {name}",
            artists=[f"Test Artist {name}"],
            album=f"Test Album {name}",
            year=2023,
            genre="Rock",
            track_number=1
        )

        return audio_file

    def create_duplicate_group(self, num_files: int = 2) -> DuplicateGroup:
        """Create a duplicate group for testing."""
        files = []
        for i in range(num_files):
            # Create a file, make the second one lower quality
            quality = 'low' if i == 1 else 'high'
            file = self.create_mock_audio_file(f"song_{i}", quality=quality)
            files.append(file)

        return DuplicateGroup(
            files=files,
            duplicate_type='metadata',
            confidence=0.9,
            reason='Same artist and title'
        )

    @pytest.mark.asyncio
    async def test_resolve_empty_duplicate_groups(self):
        """Test resolving empty duplicate groups."""
        groups = []
        summary = await self.resolver.resolve_duplicates(groups)

        assert summary.total_groups == 0
        assert summary.resolved_groups == 0
        assert summary.kept_files == 0
        assert summary.moved_files == 0
        assert summary.deleted_files == 0

    @pytest.mark.asyncio
    async def test_resolve_auto_best_strategy(self):
        """Test auto-best resolution strategy."""
        # Create resolver with auto-best strategy
        resolver = InteractiveDuplicateResolver(
            strategy=ResolutionStrategy.AUTO_KEEP_BEST,
            dry_run=True  # Use dry run to avoid actual file operations
        )

        # Create duplicate group with different quality files
        group = self.create_duplicate_group(3)
        groups = [group]

        summary = await resolver.resolve_duplicates(groups)

        assert summary.total_groups == 1
        assert summary.resolved_groups == 1
        assert summary.kept_files == 1  # Should keep the best one
        assert len(summary.decisions) == 1
        assert summary.decisions[0].action == DuplicateAction.KEEP_BEST

    @pytest.mark.asyncio
    async def test_resolve_auto_first_strategy(self):
        """Test auto-first resolution strategy."""
        resolver = InteractiveDuplicateResolver(
            strategy=ResolutionStrategy.AUTO_FIRST,
            dry_run=True
        )

        group = self.create_duplicate_group(3)
        groups = [group]

        summary = await resolver.resolve_duplicates(groups)

        assert summary.total_groups == 1
        assert summary.resolved_groups == 1
        assert len(summary.decisions) == 1

    @pytest.mark.asyncio
    async def test_resolve_auto_smart_strategy(self):
        """Test auto-smart resolution strategy."""
        resolver = InteractiveDuplicateResolver(
            strategy=ResolutionStrategy.AUTO_SMART,
            dry_run=True
        )

        group = self.create_duplicate_group(2)
        groups = [group]

        summary = await resolver.resolve_duplicates(groups)

        assert summary.total_groups == 1
        assert summary.resolved_groups == 1
        assert len(summary.decisions) == 1

    @pytest.mark.asyncio
    async def test_unknown_strategy_raises_error(self):
        """Test that unknown strategy raises an error."""
        resolver = InteractiveDuplicateResolver(strategy=ResolutionStrategy.INTERACTIVE)
        resolver.strategy = "unknown"  # type: ignore

        with pytest.raises(MusicOrganizerError):
            await resolver.resolve_duplicates([])

    def test_get_summary_dict(self):
        """Test getting summary as dictionary."""
        # Add some mock decisions
        self.resolver.summary.decisions = [
            ResolutionDecision(
                action=DuplicateAction.KEEP_BEST,
                target_file=Path("/test/file1.mp3"),
                reason="Best quality"
            )
        ]

        summary_dict = self.resolver.get_summary()

        assert isinstance(summary_dict, dict)
        assert 'total_groups' in summary_dict
        assert 'resolved_groups' in summary_dict
        assert 'kept_files' in summary_dict
        assert 'decisions' in summary_dict
        assert len(summary_dict['decisions']) == 1
        assert summary_dict['decisions'][0]['action'] == 'keep_best_quality'

    def test_save_report(self):
        """Test saving a resolution report."""
        report_path = self.temp_dir / "report.json"

        # Add mock data
        self.resolver.summary.total_groups = 5
        self.resolver.summary.resolved_groups = 3
        self.resolver.summary.kept_files = 3
        self.resolver.summary.space_saved_mb = 100.5

        self.resolver.save_report(report_path)

        assert report_path.exists()
        import json
        with open(report_path) as f:
            report = json.load(f)

        assert report['strategy'] == 'auto_keep_best'
        assert report['summary']['total_groups'] == 5
        assert report['summary']['space_saved_mb'] == 100.5


class TestDuplicateResolverUI:
    """Test the DuplicateResolverUI class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ui = DuplicateResolverUI(dry_run=True)

    @patch('builtins.input')
    @pytest.mark.asyncio
    async def test_show_duplicate_group_keep_first(self, mock_input):
        """Test showing duplicate group and choosing to keep first file."""
        mock_input.return_value = "1"

        # Create mock files using AudioFile
        file1 = AudioFile(
            path=Path("/test/file1.mp3"),
            file_type='mp3',
            metadata={'bitrate': 320, 'sample_rate': 48000},
            title="Test Song",
            artists=["Test Artist"],
            album="Test Album",
            year=2023,
            genre="Rock",
            track_number=1
        )

        file2 = AudioFile(
            path=Path("/test/file2.mp3"),
            file_type='mp3',
            metadata={'bitrate': 192, 'sample_rate': 44100},
            title="Test Song",
            artists=["Test Artist"],
            album="Test Album",
            year=2023,
            genre="Rock",
            track_number=1
        )

        group = DuplicateGroup(
            files=[file1, file2],
            duplicate_type='metadata',
            confidence=0.9,
            reason='Same artist and title'
        )

        # Mock path.stat() to return file sizes
        with patch('builtins.print'), patch.object(Path, 'stat', return_value=MagicMock(st_size=5 * 1024 * 1024)):
            decision = await self.ui.show_duplicate_group(group)

        assert decision is not None
        assert decision.action == DuplicateAction.KEEP_FIRST
        assert decision.target_file == file1.path
        assert "User chose to keep first file" in decision.reason

    @patch('builtins.input')
    @pytest.mark.asyncio
    async def test_show_duplicate_group_keyboard_interrupt(self, mock_input):
        """Test keyboard interrupt during duplicate resolution."""
        mock_input.side_effect = KeyboardInterrupt()

        file1 = AudioFile(
            path=Path("/test/file1.mp3"),
            file_type='mp3',
            metadata={}
        )

        group = DuplicateGroup(
            files=[file1],
            duplicate_type='metadata',
            confidence=0.9,
            reason='Test'
        )

        with patch('builtins.print'):
            decision = await self.ui.show_duplicate_group(group)

        assert decision is None

    @patch('builtins.input')
    def test_confirm_action_yes(self, mock_input):
        """Test confirming action with yes."""
        mock_input.return_value = "y"

        result = self.ui.confirm_action("Delete file", "This will delete file.mp3")

        assert result is True

    @patch('builtins.input')
    def test_confirm_action_no(self, mock_input):
        """Test confirming action with no."""
        mock_input.return_value = "n"

        result = self.ui.confirm_action("Delete file", "This will delete file.mp3")

        assert result is False

    @patch('builtins.input')
    def test_confirm_action_invalid_then_yes(self, mock_input):
        """Test confirming action with invalid input then yes."""
        mock_input.side_effect = ["maybe", "y"]

        result = self.ui.confirm_action("Delete file")

        assert result is True


class TestDuplicateResolverIntegration:
    """Integration tests for duplicate resolution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # Note: test_quick_duplicate_resolution_function was removed because
    # quick_duplicate_resolution function doesn't exist in the codebase
    # This test was commented out at line 20 in the imports section

    def test_duplicate_action_enum(self):
        """Test DuplicateAction enum values."""
        assert DuplicateAction.KEEP_FIRST.value == "keep_first"
        assert DuplicateAction.KEEP_SECOND.value == "keep_second"
        assert DuplicateAction.KEEP_BOTH.value == "keep_both"
        assert DuplicateAction.KEEP_BEST.value == "keep_best_quality"
        assert DuplicateAction.MOVE_DUPLICATE.value == "move_duplicate"
        assert DuplicateAction.DELETE_DUPLICATE.value == "delete_duplicate"

    def test_resolution_strategy_enum(self):
        """Test ResolutionStrategy enum values."""
        assert ResolutionStrategy.INTERACTIVE.value == "interactive"
        assert ResolutionStrategy.AUTO_KEEP_BEST.value == "auto_keep_best"
        assert ResolutionStrategy.AUTO_FIRST.value == "auto_first"
        assert ResolutionStrategy.AUTO_SMART.value == "auto_smart"


if __name__ == '__main__':
    pytest.main([__file__])