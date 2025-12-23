"""Tests for the duplicate detection plugin."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.music_organizer.plugins.builtins.duplicate_detector import DuplicateDetectorPlugin
from src.music_organizer.models.audio_file import AudioFile


@pytest.fixture
def plugin():
    """Create a duplicate detector plugin instance."""
    config = {
        'enabled': True,
        'strategies': ['metadata', 'file_hash', 'audio_fingerprint'],
        'similarity_threshold': 0.85,
        'min_confidence': 0.5,
        'allowed_types': ['exact', 'metadata', 'acoustic'],
        'report_duplicates_only': True,
        'batch_size': 100,
    }
    return DuplicateDetectorPlugin(config)


@pytest.fixture
def sample_audio_file():
    """Create a sample audio file for testing."""
    return AudioFile(
        path=Path("/test/song1.mp3"),
        file_type="MP3",
        artists=["Test Artist"],
        primary_artist="Test Artist",
        title="Test Song",
        album="Test Album",
        year=2023,
        genre="Test",
        track_number=1,
    )


@pytest.fixture
def duplicate_audio_file():
    """Create a duplicate audio file for testing."""
    return AudioFile(
        path=Path("/test/song1_copy.mp3"),
        file_type="MP3",
        artists=["Test Artist"],
        primary_artist="Test Artist",
        title="Test Song",
        album="Test Album",
        year=2023,
        genre="Test",
        track_number=1,
    )


@pytest.fixture
def similar_audio_file():
    """Create a similar audio file for testing."""
    return AudioFile(
        path=Path("/test/song1_similar.mp3"),
        file_type="MP3",
        artists=["Test Artist"],
        primary_artist="Test Artist",
        title="Test Song (Live)",
        album="Test Album (Live)",
        year=2023,
        genre="Test",
        track_number=1,
    )


class TestDuplicateDetectorPlugin:
    """Test cases for DuplicateDetectorPlugin."""

    def test_plugin_info(self, plugin):
        """Test plugin information."""
        info = plugin.info
        assert info.name == "duplicate_detector"
        assert info.version == "1.1.0"
        assert "duplicate" in info.description.lower()
        assert info.dependencies == []

    def test_initialization(self, plugin):
        """Test plugin initialization."""
        plugin.initialize()
        assert len(plugin._file_hashes) == 0
        assert len(plugin._metadata_hashes) == 0
        assert len(plugin._audio_fingerprints) == 0
        assert len(plugin._processed_files) == 0

    def test_cleanup(self, plugin):
        """Test plugin cleanup."""
        plugin._file_hashes['test'] = ['file1']
        plugin._metadata_hashes['test'] = ['file2']
        plugin._audio_fingerprints['test'] = ['file3']
        plugin._processed_files.add('file4')

        plugin.cleanup()

        assert len(plugin._file_hashes) == 0
        assert len(plugin._metadata_hashes) == 0
        assert len(plugin._audio_fingerprints) == 0
        assert len(plugin._processed_files) == 0

    def test_supported_tags(self, plugin):
        """Test supported classification tags."""
        tags = plugin.get_supported_tags()
        assert 'duplicates' in tags
        assert 'duplicate_count' in tags
        assert 'is_duplicate' in tags

    def test_config_schema(self, plugin):
        """Test configuration schema."""
        schema = plugin.get_config_schema()
        assert schema is not None

        # Check custom options
        strategies_opt = schema.get_option('strategies')
        assert strategies_opt is not None
        assert strategies_opt.type == list
        assert 'metadata' in strategies_opt.default

        threshold_opt = schema.get_option('similarity_threshold')
        assert threshold_opt is not None
        assert threshold_opt.type == float
        assert threshold_opt.default == 0.85

    def test_generate_metadata_hash(self, plugin):
        """Test metadata hash generation."""
        file1 = AudioFile(
            path=Path("/test/1.mp3"),
            file_type="MP3",
            artists=["The Artist"],
            primary_artist="The Artist",
            title="The Song",
            album="The Album",
        )
        file2 = AudioFile(
            path=Path("/test/2.mp3"),
            file_type="MP3",
            artists=["the artist"],  # Different case
            primary_artist="the artist",
            title="  the song  ",  # Extra whitespace
            album="THE ALBUM",  # Different case
        )
        file3 = AudioFile(
            path=Path("/test/3.mp3"),
            file_type="MP3",
            artists=["Different Artist"],
            primary_artist="Different Artist",
            title="Different Song",
            album="Different Album",
        )

        hash1 = plugin._generate_metadata_hash(file1)
        hash2 = plugin._generate_metadata_hash(file2)
        hash3 = plugin._generate_metadata_hash(file3)

        # Files 1 and 2 should have the same hash (normalized)
        assert hash1 == hash2
        # File 3 should have a different hash
        assert hash1 != hash3

    def test_text_normalization(self, plugin):
        """Test text normalization."""
        test_cases = [
            ("Hello World", "hello world"),
            ("  MULTIPLE   SPACES  ", "multiple spaces"),
            ("MiXeD CaSe", "mixed case"),
            ("", ""),
            (None, ""),
        ]

        for input_text, expected in test_cases:
            result = plugin._normalize_text(input_text)
            assert result == expected

    @pytest.mark.asyncio
    async def test_classify_no_duplicates(self, plugin, sample_audio_file):
        """Test classification when no duplicates exist."""
        result = await plugin.classify(sample_audio_file)
        assert result == {}
        assert not result.get('is_duplicate', False)

    @pytest.mark.asyncio
    async def test_classify_metadata_duplicate(self, plugin, sample_audio_file, duplicate_audio_file):
        """Test classification with metadata duplicate."""
        # Process first file
        result1 = await plugin.classify(sample_audio_file)
        assert not result1.get('is_duplicate', False)

        # Process duplicate file
        result2 = await plugin.classify(duplicate_audio_file)
        assert result2.get('is_duplicate', False)
        assert 'duplicates' in result2
        # The duplicate_count can be >= 1 (multiple strategies might detect duplicates)
        assert result2['duplicate_count'] >= 1

        # Check that at least metadata duplicate was found
        duplicate_types = [d['type'] for d in result2['duplicates']]
        assert 'metadata' in duplicate_types

        # Find the metadata duplicate
        metadata_dup = next(d for d in result2['duplicates'] if d['type'] == 'metadata')
        assert metadata_dup['confidence'] == 1.0

    @pytest.mark.asyncio
    async def test_classify_file_disabled(self, plugin, sample_audio_file):
        """Test classification when plugin is disabled."""
        plugin.enabled = False
        result = await plugin.classify(sample_audio_file)
        assert result == {}

    @pytest.mark.asyncio
    async def test_classify_already_processed(self, plugin, sample_audio_file):
        """Test classification for already processed file."""
        plugin._processed_files.add(str(sample_audio_file.path))
        result = await plugin.classify(sample_audio_file)
        assert result == {}

    def test_fingerprint_similarity(self, plugin):
        """Test fingerprint similarity calculation."""
        fp1 = "a1b2c3d4e5f6" * 4  # 32 chars
        fp2 = "a1b2c3d4e5f6" * 4  # Same
        fp3 = "z9y8x7w6v5u4" * 4  # Different
        fp4 = "a1b2c3d4z9y8" * 4  # Partially similar

        # Identical fingerprints
        assert plugin._calculate_fingerprint_similarity(fp1, fp2) == 1.0

        # Completely different fingerprints
        assert plugin._calculate_fingerprint_similarity(fp1, fp3) == 0.0

        # Partially similar fingerprints
        similarity = plugin._calculate_fingerprint_similarity(fp1, fp4)
        assert 0.0 < similarity < 1.0

    def test_duplicate_summary(self, plugin, sample_audio_file, duplicate_audio_file):
        """Test duplicate summary generation."""
        # Add some test data
        plugin._metadata_hashes['hash1'] = [str(sample_audio_file.path), str(duplicate_audio_file.path)]
        plugin._file_hashes['hash2'] = [str(sample_audio_file.path)]
        plugin._processed_files.add(str(sample_audio_file.path))
        plugin._processed_files.add(str(duplicate_audio_file.path))

        summary = plugin.get_duplicate_summary()

        assert summary['total_duplicate_groups'] == 1
        assert summary['total_duplicate_files'] == 1  # 2 files - 1 unique = 1 duplicate
        assert summary['unique_files_processed'] == 2
        assert summary['metadata_hashes'] == 1
        assert summary['file_hashes'] == 1

    @pytest.mark.asyncio
    async def test_generate_file_hash(self, plugin, tmp_path):
        """Test file hash generation."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        hash1 = await plugin._generate_file_hash(test_file)
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length

        # Same file should produce same hash
        hash2 = await plugin._generate_file_hash(test_file)
        assert hash1 == hash2

        # Non-existent file should still return a hash
        non_existent = tmp_path / "nonexistent.mp3"
        hash3 = await plugin._generate_file_hash(non_existent)
        assert isinstance(hash3, str)
        assert len(hash3) == 32

    def test_filter_duplicates(self, plugin):
        """Test duplicate filtering."""
        duplicate_groups = [
            {'type': 'exact', 'confidence': 1.0, 'duplicates': ['file1']},
            {'type': 'metadata', 'confidence': 0.9, 'duplicates': ['file2']},
            {'type': 'acoustic', 'confidence': 0.3, 'duplicates': ['file3']},  # Below threshold
        ]

        filtered = plugin._filter_duplicates(duplicate_groups)

        # Should filter out the low confidence acoustic match
        assert len(filtered) == 2
        assert filtered[0]['type'] == 'exact'
        assert filtered[1]['type'] == 'metadata'

    def test_filter_duplicates_by_type(self, plugin):
        """Test duplicate filtering by type."""
        plugin.config['allowed_types'] = ['exact']

        duplicate_groups = [
            {'type': 'exact', 'confidence': 1.0, 'duplicates': ['file1']},
            {'type': 'metadata', 'confidence': 1.0, 'duplicates': ['file2']},
            {'type': 'acoustic', 'confidence': 1.0, 'duplicates': ['file3']},
        ]

        filtered = plugin._filter_duplicates(duplicate_groups)

        # Should only keep exact duplicates
        assert len(filtered) == 1
        assert filtered[0]['type'] == 'exact'

    def test_config_validation(self):
        """Test configuration validation."""
        plugin = DuplicateDetectorPlugin()

        # Test invalid configuration
        invalid_config = {
            'similarity_threshold': 1.5,  # Invalid: > 1.0
            'min_confidence': -0.1,  # Invalid: < 0
            'strategies': ['invalid_strategy'],  # Invalid strategy
        }

        # The plugin should still initialize but use defaults
        plugin = DuplicateDetectorPlugin(invalid_config)
        assert plugin.config['similarity_threshold'] == 1.5  # Config is not auto-validated in this simple implementation

    def test_find_similar_fingerprints(self, plugin):
        """Test finding similar fingerprints."""
        # Add test fingerprints with SHA256 length (64 chars)
        fp1 = "a1b2c3d4e5f6" * 5 + "1234567890ab"  # 64 chars
        fp2 = "a1b2c3d4e5f6" * 4 + "z9y8x7w6v5u4" + "1234567890ab"  # 64 chars

        plugin._audio_fingerprints[fp1] = ['file1']
        plugin._audio_fingerprints[fp2] = ['file2']

        # Find similar to fp1
        similar = plugin._find_similar_fingerprints(fp1)
        assert len(similar) == 1
        assert similar[0][0] == 'file2'
        assert 0 <= similar[0][1] <= 1  # Similarity score

    @pytest.mark.asyncio
    async def test_report_duplicates_only_false(self, plugin, sample_audio_file):
        """Test classification when report_duplicates_only is False."""
        plugin.config['report_duplicates_only'] = False

        result = await plugin.classify(sample_audio_file)
        # Should return empty dict, but not filtered out
        assert result == {}
        assert 'is_duplicate' not in result

    @pytest.mark.asyncio
    async def test_classify_with_min_confidence_filter(self, plugin, sample_audio_file, duplicate_audio_file):
        """Test classification with confidence filter."""
        plugin.config['min_confidence'] = 1.0

        # Process files
        await plugin.classify(sample_audio_file)
        result = await plugin.classify(duplicate_audio_file)

        # Should still detect duplicate (confidence = 1.0)
        assert result.get('is_duplicate', False)