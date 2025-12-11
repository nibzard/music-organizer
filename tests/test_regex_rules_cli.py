"""Tests for the regex rules CLI."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from music_organizer.regex_rules_cli import (
    cmd_rules_list, cmd_rules_test, cmd_rules_validate, cmd_rules_init,
    cmd_rules_add, cmd_rules_remove, cmd_rules_enable_disable,
    cmd_rules_preview, cmd_rules_fields, cmd_rules_operators,
    setup_rules_parser, handle_rules_command
)
from music_organizer.core.regex_rule_engine import Rule, RuleCondition, ComparisonOperator
from music_organizer.models.audio_file import AudioFile
from music_organizer.models.content_type import ContentType


@pytest.fixture
def mock_args():
    """Create mock arguments for CLI commands."""
    class MockArgs:
        def __init__(self):
            self.rules_command = None
            self.rules_file = None
            self.enabled_only = False
            self.stats = False
            self.directory = None
            self.limit = 10
            self.show_matches = False
            self.verbose = False
            self.file = None
            self.extended = False
            self.output = "rules.json"
            self.builtin = False
            self.name = None
            self.pattern = None
            self.description = None
            self.priority = 0
            self.genre = None
            self.artist = None
            self.year = None
            self.album = None
            self.source = None
            self.target = None
            self.show_unmatched = False
            self.examples = False

    return MockArgs()


@pytest.fixture
def sample_rules_file():
    """Create a sample rules file for testing."""
    rules_data = {
        "rules": [
            {
                "name": "Rock Music",
                "description": "Organize rock music",
                "conditions": [
                    {
                        "field": "genre",
                        "operator": "eq",
                        "value": "Rock"
                    }
                ],
                "pattern": "Rock/{artist}/{album}",
                "priority": 100,
                "enabled": True,
                "tags": ["genre"]
            },
            {
                "name": "Disabled Rule",
                "description": "This rule is disabled",
                "conditions": [
                    {
                        "field": "genre",
                        "operator": "eq",
                        "value": "Jazz"
                    }
                ],
                "pattern": "Jazz/{artist}/{album}",
                "priority": 50,
                "enabled": False
            }
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(rules_data, f)
        return Path(f.name)


@pytest.fixture
def mock_console():
    """Create a mock console."""
    console = Mock()
    console.info = Mock()
    console.print = Mock()
    console.success = Mock()
    console.warning = Mock()
    console.error = Mock()
    console.header = Mock()
    console.confirm = Mock(return_value=True)
    return console


class TestCliCommands:
    """Test individual CLI commands."""

    @pytest.mark.asyncio
    async def test_cmd_rules_list(self, mock_args, sample_rules_file):
        """Test the rules list command."""
        mock_args.rules_file = str(sample_rules_file)
        mock_args.enabled_only = False
        mock_args.stats = True

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_list(mock_args)
            assert result == 0
            mock_console.header.assert_called()
            mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_cmd_rules_list_enabled_only(self, mock_args, sample_rules_file):
        """Test the rules list command with enabled_only filter."""
        mock_args.rules_file = str(sample_rules_file)
        mock_args.enabled_only = True
        mock_args.stats = False

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_list(mock_args)
            assert result == 0
            # Should only show the enabled rule
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert "Rock Music" in str(calls)
            # Check that "Disabled Rule" is not shown
            disabled_found = any("Disabled Rule" in str(call) for call in calls)
            assert not disabled_found

    @pytest.mark.asyncio
    async def test_cmd_rules_validate_valid(self, mock_args, sample_rules_file):
        """Test the rules validate command with a valid file."""
        mock_args.file = str(sample_rules_file)
        mock_args.extended = False

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_validate(mock_args)
            assert result == 0
            mock_console.success.assert_called()

    @pytest.mark.asyncio
    async def test_cmd_rules_validate_invalid(self, mock_args):
        """Test the rules validate command with an invalid file."""
        # Create invalid rules file
        invalid_rules = {
            "rules": [
                {
                    # Missing required 'name' field
                    "pattern": "{artist}/{album}"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_rules, f)
            invalid_file = Path(f.name)

        try:
            mock_args.file = str(invalid_file)
            mock_args.extended = False

            with patch('music_organizer.regex_rules_cli.console') as mock_console:
                result = await cmd_rules_validate(mock_args)
                assert result == 1
                mock_console.error.assert_called()
        finally:
            invalid_file.unlink()

    @pytest.mark.asyncio
    async def test_cmd_rules_init(self, mock_args):
        """Test the rules init command."""
        mock_args.output = "/tmp/test_rules.json"
        mock_args.builtin = False

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_init(mock_args)
            assert result == 0
            mock_console.success.assert_called()

            # Check file was created
            output_path = Path(mock_args.output)
            assert output_path.exists()
            output_path.unlink()

    @pytest.mark.asyncio
    async def test_cmd_rules_add(self, mock_args):
        """Test the rules add command."""
        mock_args.name = "Test Rule"
        mock_args.pattern = "Test/{artist}/{album}"
        mock_args.description = "A test rule"
        mock_args.priority = 50
        mock_args.genre = "Rock|Pop"
        mock_args.artist = None
        mock_args.year = None
        mock_args.album = None
        mock_args.rules_file = None  # In-memory only

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_add(mock_args)
            assert result == 0
            mock_console.success.assert_called()

    @pytest.mark.asyncio
    async def test_cmd_rules_add_with_year_range(self, mock_args):
        """Test the rules add command with a year range."""
        mock_args.name = "90s Music"
        mock_args.pattern = "90s/{artist}/{album}"
        mock_args.year = "1990-1999"
        mock_args.rules_file = None

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_add(mock_args)
            assert result == 0
            mock_console.success.assert_called()

    @pytest.mark.asyncio
    async def test_cmd_rules_remove(self, mock_args, sample_rules_file):
        """Test the rules remove command."""
        mock_args.name = "Rock Music"
        mock_args.rules_file = str(sample_rules_file)

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_remove(mock_args)
            assert result == 0
            mock_console.success.assert_called()

    @pytest.mark.asyncio
    async def test_cmd_rules_remove_not_found(self, mock_args, sample_rules_file):
        """Test the rules remove command with a non-existent rule."""
        mock_args.name = "Non-existent Rule"
        mock_args.rules_file = str(sample_rules_file)

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_remove(mock_args)
            assert result == 0
            mock_console.warning.assert_called()

    @pytest.mark.asyncio
    async def test_cmd_rules_enable_disable(self, mock_args, sample_rules_file):
        """Test the rules enable/disable commands."""
        mock_args.name = "Disabled Rule"
        mock_args.rules_file = str(sample_rules_file)

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            # Enable
            result = await cmd_rules_enable_disable(mock_args, enable=True)
            assert result == 0
            mock_console.success.assert_called()

            # Reset mock
            mock_console.reset_mock()

            # Disable
            result = await cmd_rules_enable_disable(mock_args, enable=False)
            assert result == 0
            mock_console.success.assert_called()

    @pytest.mark.asyncio
    async def test_cmd_rules_fields(self, mock_args):
        """Test the rules fields command."""
        mock_args.examples = False

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_fields(mock_args)
            assert result == 0
            mock_console.header.assert_called()
            mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_cmd_rules_fields_with_examples(self, mock_args):
        """Test the rules fields command with examples."""
        mock_args.examples = True

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_fields(mock_args)
            assert result == 0
            mock_console.header.assert_called()
            # Should print examples
            assert mock_console.print.call_count > 5  # More than just the basic output

    @pytest.mark.asyncio
    async def test_cmd_rules_operators(self, mock_args):
        """Test the rules operators command."""
        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_operators(mock_args)
            assert result == 0
            mock_console.header.assert_called()
            mock_console.print.assert_called()

    @pytest.mark.asyncio
    @patch('music_organizer.regex_rules_cli.AsyncFileScanner')
    @patch('music_organizer.regex_rules_cli.MetadataHandler')
    async def test_cmd_rules_test(self, mock_metadata_handler, mock_file_scanner, mock_args):
        """Test the rules test command."""
        # Mock file scanner
        mock_scanner = Mock()
        mock_scanner.scan_directory = AsyncMock(return_value=[
            Path("/test/track1.mp3"),
            Path("/test/track2.mp3")
        ])
        mock_file_scanner.return_value = mock_scanner

        # Mock metadata handler
        mock_handler = Mock()
        mock_handler.extract_metadata = AsyncMock(return_value=AudioFile(
            path=Path("/test/track1.mp3"),
            title="Test Song",
            artist="Test Artist",
            album="Test Album",
            year=2020,
            genre="Rock"
        ))
        mock_metadata_handler.return_value = mock_handler

        mock_args.directory = "/test"
        mock_args.limit = 2
        mock_args.show_matches = True
        mock_args.verbose = True
        mock_args.show_unmatched = True

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_test(mock_args)
            assert result == 0
            mock_console.header.assert_called()
            mock_console.print.assert_called()


class TestCliIntegration:
    """Test CLI integration."""

    def test_setup_rules_parser(self):
        """Test that the rules parser is set up correctly."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        rules_parser = setup_rules_parser(subparsers)

        assert rules_parser is not None
        assert rules_parser.name == "rules"

        # Test that subcommands exist
        subcommands = rules_parser._subparsers._group_actions[0].choices
        assert "list" in subcommands
        assert "test" in subcommands
        assert "validate" in subcommands
        assert "init" in subcommands
        assert "add" in subcommands
        assert "remove" in subcommands
        assert "enable" in subcommands
        assert "disable" in subcommands
        assert "preview" in subcommands
        assert "fields" in subcommands
        assert "operators" in subcommands

    @pytest.mark.asyncio
    async def test_handle_rules_command_unknown(self, mock_args):
        """Test handling an unknown rules command."""
        mock_args.rules_command = "unknown"

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await handle_rules_command(mock_args)
            assert result == 1
            mock_console.error.assert_called()

    @pytest.mark.asyncio
    async def test_handle_rules_command_list(self, mock_args):
        """Test handling the list command."""
        mock_args.rules_command = "list"
        mock_args.rules_file = None
        mock_args.enabled_only = False
        mock_args.stats = False

        with patch('music_organizer.regex_rules_cli.cmd_rules_list', return_value=0) as mock_cmd:
            result = await handle_rules_command(mock_args)
            assert result == 0
            mock_cmd.assert_called_once_with(mock_args)

    @pytest.mark.asyncio
    async def test_handle_rules_command_validate(self, mock_args):
        """Test handling the validate command."""
        mock_args.rules_command = "validate"
        mock_args.file = "test.json"
        mock_args.extended = False

        with patch('music_organizer.regex_rules_cli.cmd_rules_validate', return_value=0) as mock_cmd:
            result = await handle_rules_command(mock_args)
            assert result == 0
            mock_cmd.assert_called_once_with(mock_args)


class TestCliEdgeCases:
    """Test CLI edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_cmd_rules_add_no_rules_file(self, mock_args):
        """Test adding a rule without specifying a file."""
        mock_args.name = "Test Rule"
        mock_args.pattern = "Test/{artist}"
        mock_args.rules_file = None

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            result = await cmd_rules_add(mock_args)
            assert result == 0
            mock_console.success.assert_called()
            # Should succeed but mention it's in-memory only

    @pytest.mark.asyncio
    async def test_cmd_rules_init_file_exists(self, mock_args):
        """Test init command when file already exists."""
        # Create existing file
        existing_file = Path("/tmp/existing_rules.json")
        existing_file.touch()

        mock_args.output = str(existing_file)

        with patch('music_organizer.regex_rules_cli.console') as mock_console:
            mock_console.confirm.return_value = False  # User says no to overwrite
            result = await cmd_rules_init(mock_args)
            assert result == 0  # Should exit cleanly
            mock_console.confirm.assert_called_once()

        existing_file.unlink()

    @pytest.mark.asyncio
    async def test_cmd_rules_preview_no_files(self, mock_args):
        """Test preview command when no files are found."""
        mock_args.source = "/nonexistent"
        mock_args.target = "/tmp/target"
        mock_args.limit = 10

        with patch('music_organizer.regex_rules_cli.AsyncFileScanner') as mock_scanner:
            scanner = Mock()
            scanner.scan_directory = AsyncMock(return_value=[])  # No files
            mock_scanner.return_value = scanner

            with patch('music_organizer.regex_rules_cli.console') as mock_console:
                result = await cmd_rules_preview(mock_args)
                assert result == 0
                # Should not error, just process 0 files

    @pytest.mark.asyncio
    async def test_cmd_rules_test_with_metadata_error(self, mock_args):
        """Test rules test command when metadata extraction fails."""
        mock_args.directory = "/test"
        mock_args.limit = 1

        with patch('music_organizer.regex_rules_cli.AsyncFileScanner') as mock_scanner:
            scanner = Mock()
            scanner.scan_directory = AsyncMock(return_value=[Path("/test/track.mp3")])
            mock_scanner.return_value = scanner

            with patch('music_organizer.regex_rules_cli.MetadataHandler') as mock_handler:
                handler = Mock()
                handler.extract_metadata = AsyncMock(side_effect=Exception("Metadata error"))
                mock_handler.return_value = handler

                mock_args.verbose = False  # Don't show error details

                with patch('music_organizer.regex_rules_cli.console') as mock_console:
                    result = await cmd_rules_test(mock_args)
                    assert result == 0  # Should continue with other files
                    mock_console.warning.assert_not_called()  # Since verbose is False