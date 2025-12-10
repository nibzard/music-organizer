"""
Test fixtures for plugin testing.

This module provides ready-to-use test data and fixtures for testing
different types of plugins with realistic scenarios.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import shutil
import json

from .mocks import MockAudioFile, create_mock_audio_file
from ...models.audio_file import AudioFile


class AudioTestFixture:
    """
    Provides test fixtures for audio files covering various scenarios.

    Includes files with different metadata completeness, formats, and edge cases.
    """

    def __init__(self):
        """Initialize audio file test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self._setup_files()

    def _setup_files(self):
        """Setup test audio files."""
        # Complete metadata file
        self.complete_file = create_mock_audio_file(
            path=self.temp_dir / "complete.mp3",
            metadata={
                'title': 'Complete Song',
                'artist': 'Complete Artist',
                'album': 'Complete Album',
                'year': '2023',
                'track_number': '1',
                'genre': 'Rock',
                'duration': 210.5,
                'bitrate': 320,
                'albumartist': 'Complete Artist',
                'discnumber': '1',
                'totaldiscs': '1',
                'totaltracks': '12',
                'comment': 'A complete test file',
                'composer': 'Test Composer',
                'lyrics': 'Test lyrics here'
            }
        )

        # Minimal metadata file
        self.minimal_file = create_mock_audio_file(
            path=self.temp_dir / "minimal.mp3",
            metadata={
                'title': 'Minimal Song'
            },
            duration=120.0,
            bitrate=128
        )

        # Compilation file
        self.compilation_file = create_mock_audio_file(
            path=self.temp_dir / "compilation.mp3",
            metadata={
                'title': 'Compilation Track',
                'artist': 'Various Artists',
                'album': 'Various Compilation',
                'year': '2023',
                'track_number': '5',
                'genre': 'Pop',
                'compilation': True,
                'albumartist': 'Various Artists'
            }
        )

        # File with special characters
        self.special_chars_file = create_mock_audio_file(
            path=self.temp_dir / "special.mp3",
            metadata={
                'title': 'Special "Characters" & More',
                'artist': 'Artist/Special',
                'album': 'Album: Special?'
            }
        )

        # Long title file
        self.long_title_file = create_mock_audio_file(
            path=self.temp_dir / "long.mp3",
            metadata={
                'title': 'This is a very long song title that might cause issues in some file systems or export formats',
                'artist': 'Artist with a very long name',
                'album': 'Album with an exceptionally long name that could cause problems'
            }
        )

        # Non-English characters
        self.international_file = create_mock_audio_file(
            path=self.temp_dir / "international.mp3",
            metadata={
                'title': 'Cañón español',
                'artist': 'François',
                'album': 'альбом русский',
                'year': '2023',
                'genre': '世界音乐'
            }
        )

        # Missing year file
        self.no_year_file = create_mock_audio_file(
            path=self.temp_dir / "no_year.mp3",
            metadata={
                'title': 'No Year Song',
                'artist': 'No Year Artist',
                'album': 'No Year Album'
            }
        )

        # Various genres
        self.genres = [
            create_mock_audio_file(
                path=self.temp_dir / f"rock_{i}.mp3",
                metadata={
                    'title': f'Rock Song {i}',
                    'artist': f'Rock Artist {i}',
                    'album': f'Rock Album {i}',
                    'genre': 'Rock'
                }
            )
            for i in range(5)
        ] + [
            create_mock_audio_file(
                path=self.temp_dir / f"jazz_{i}.mp3",
                metadata={
                    'title': f'Jazz Song {i}',
                    'artist': f'Jazz Artist {i}',
                    'album': f'Jazz Album {i}',
                    'genre': 'Jazz'
                }
            )
            for i in range(3)
        ]

    def get_all_files(self) -> List[MockAudioFile]:
        """Get all test files."""
        return [
            self.complete_file,
            self.minimal_file,
            self.compilation_file,
            self.special_chars_file,
            self.long_title_file,
            self.international_file,
            self.no_year_file
        ] + self.genres

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class MetadataTestFixture:
    """
    Provides test fixtures specifically for metadata plugin testing.

    Includes scenarios with different metadata enhancement needs.
    """

    def __init__(self):
        """Initialize metadata test fixtures."""
        self.test_cases = self._create_test_cases()

    def _create_test_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create test cases for metadata enhancement."""
        return {
            'missing_year': {
                'input': create_mock_audio_file(
                    metadata={
                        'title': 'Song without year',
                        'artist': 'Test Artist',
                        'album': 'Test Album'
                    }
                ),
                'expected_enhancement': {'year': '2023'},
                'description': 'File missing year information'
            },
            'missing_genre': {
                'input': create_mock_audio_file(
                    metadata={
                        'title': 'Song without genre',
                        'artist': 'Test Artist',
                        'album': 'Test Album',
                        'year': '2023'
                    }
                ),
                'expected_enhancement': {'genre': 'Unknown'},
                'description': 'File missing genre information'
            },
            'incomplete_album': {
                'input': create_mock_audio_file(
                    metadata={
                        'title': 'Track 1',
                        'artist': 'Test Artist'
                    }
                ),
                'expected_enhancement': {
                    'album': 'Unknown Album',
                    'track_number': '1'
                },
                'description': 'File missing album information'
            },
            'enhanced_already': {
                'input': create_mock_audio_file(
                    metadata={
                        'title': 'Already Enhanced',
                        'artist': 'Test Artist',
                        'album': 'Test Album',
                        'enhanced_by': 'previous-plugin',
                        'enhanced_at': '2023-01-01T00:00:00'
                    }
                ),
                'expected_enhancement': {},
                'description': 'File already enhanced by another plugin'
            },
            'normalization_needed': {
                'input': create_mock_audio_file(
                    metadata={
                        'title': '  messy title  ',
                        'artist': 'ARTIST IN CAPS',
                        'album': 'Album_with_Underscores',
                        'genre': 'rock'
                    }
                ),
                'expected_enhancement': {
                    'title': 'Messy Title',
                    'artist': 'Artist In Caps',
                    'genre': 'Rock'
                },
                'description': 'File needing metadata normalization'
            }
        }

    def get_test_case(self, name: str) -> Dict[str, Any]:
        """Get a specific test case."""
        return self.test_cases.get(name, {})

    def get_all_test_cases(self) -> Dict[str, Dict[str, Any]]:
        """Get all test cases."""
        return self.test_cases


class ClassificationTestFixture:
    """
    Provides test fixtures specifically for classification plugin testing.

    Includes scenarios with different classification challenges.
    """

    def __init__(self):
        """Initialize classification test fixtures."""
        self.test_files = self._create_test_files()
        self.expected_classifications = self._create_expected_classifications()

    def _create_test_files(self) -> Dict[str, MockAudioFile]:
        """Create test files for classification."""
        return {
            'rock_song': create_mock_audio_file(
                metadata={
                    'title': 'Rock Anthem',
                    'artist': 'The Rockers',
                    'album': 'Guitar Heavy',
                    'genre': 'Rock',
                    'comment': 'Electric guitars, drums, heavy'
                },
                duration=240.0
            ),
            'classical_piece': create_mock_audio_file(
                metadata={
                    'title': 'Symphony No. 5',
                    'artist': 'Classical Orchestra',
                    'album': 'Classical Collection',
                    'genre': 'Classical',
                    'comment': 'orchestra strings violin'
                },
                duration=1800.0
            ),
            'electronic_track': create_mock_audio_file(
                metadata={
                    'title': 'Digital Dreams',
                    'artist': 'DJ Electron',
                    'album': 'Synth Wave',
                    'genre': 'Electronic',
                    'comment': 'synth beat digital loop'
                },
                duration=300.0
            ),
            'jazz_standard': create_mock_audio_file(
                metadata={
                    'title': 'Blue Notes',
                    'artist': 'Jazz Quartet',
                    'album': 'Jazz Sessions',
                    'genre': 'Jazz',
                    'comment': 'saxophone trumpet swing'
                },
                duration=420.0
            ),
            'ambiguous_file': create_mock_audio_file(
                metadata={
                    'title': 'Unknown Style',
                    'artist': 'Mystery Artist',
                    'album': 'Various',
                    'comment': 'experimental avant-garde'
                },
                duration=180.0
            )
        }

    def _create_expected_classifications(self) -> Dict[str, Dict[str, Any]]:
        """Create expected classification results."""
        return {
            'rock_song': {
                'genre': 'Rock',
                'energy': 'high',
                'mood': 'energetic',
                'era': 'modern',
                'confidence': 0.9
            },
            'classical_piece': {
                'genre': 'Classical',
                'energy': 'low',
                'mood': 'calm',
                'era': 'classical',
                'confidence': 0.95
            },
            'electronic_track': {
                'genre': 'Electronic',
                'energy': 'medium',
                'mood': 'upbeat',
                'era': 'modern',
                'confidence': 0.85
            },
            'jazz_standard': {
                'genre': 'Jazz',
                'energy': 'medium',
                'mood': 'relaxed',
                'era': 'mid-century',
                'confidence': 0.9
            },
            'ambiguous_file': {
                'genre': 'Experimental',
                'energy': 'unknown',
                'mood': 'unknown',
                'era': 'unknown',
                'confidence': 0.3
            }
        }

    def get_test_file(self, name: str) -> Optional[MockAudioFile]:
        """Get a specific test file."""
        return self.test_files.get(name)

    def get_expected_classification(self, name: str) -> Optional[Dict[str, Any]]:
        """Get expected classification for a file."""
        return self.expected_classifications.get(name)

    def get_all_files(self) -> List[MockAudioFile]:
        """Get all test files."""
        return list(self.test_files.values())


def get_test_audio_files(count: int = 10) -> List[MockAudioFile]:
    """
    Get a list of test audio files.

    Args:
        count: Number of test files to generate

    Returns:
        List of mock audio files with varied metadata
    """
    files = []
    artists = ['Artist A', 'Artist B', 'Artist C']
    albums = ['Album 1', 'Album 2', 'Album 3']
    genres = ['Rock', 'Pop', 'Jazz', 'Classical', 'Electronic']

    for i in range(count):
        file = create_mock_audio_file(
            path=f"/test/test_file_{i:03d}.mp3",
            metadata={
                'title': f'Test Song {i + 1}',
                'artist': artists[i % len(artists)],
                'album': albums[i % len(albums)],
                'year': str(2020 + (i % 4)),
                'track_number': str((i % 10) + 1),
                'genre': genres[i % len(genres)]
            },
            duration=180.0 + (i * 10),
            bitrate=128 + (i % 5) * 64
        )
        files.append(file)

    return files


def get_test_metadata() -> Dict[str, Any]:
    """
    Get sample metadata for testing.

    Returns:
        Dictionary with various metadata fields
    """
    return {
        'basic': {
            'title': 'Test Song',
            'artist': 'Test Artist',
            'album': 'Test Album',
            'year': '2023',
            'track_number': '1',
            'genre': 'Rock'
        },
        'complete': {
            'title': 'Complete Test Song',
            'artist': 'Complete Test Artist',
            'albumartist': 'Complete Test Artist',
            'album': 'Complete Test Album',
            'year': '2023',
            'track_number': '1',
            'totaltracks': '12',
            'discnumber': '1',
            'totaldiscs': '1',
            'genre': 'Rock',
            'style': 'Alternative Rock',
            'mood': 'Energetic',
            'composer': 'Test Composer',
            'comment': 'A complete test file with all metadata',
            'lyrics': 'Test lyrics go here',
            'bpm': '120',
            'key': 'C Major',
            'language': 'English',
            'country': 'USA',
            'originalyear': '2022',
            'release_date': '2023-01-15',
            'encoder': 'Test Encoder',
            'copyright': '© 2023 Test Label',
            'isrc': 'TEST123456789',
            'barcode': '1234567890123',
            'catalognumber': 'TEST-001',
            'label': 'Test Label',
            'publisher': 'Test Publisher',
            'mediatype': 'Digital',
            'originalfilename': 'original_test.mp3'
        },
        'minimal': {
            'title': 'Minimal Song'
        },
        'problematic': {
            'title': '  Leading and trailing spaces  ',
            'artist': 'ARTIST_IN_ALL_CAPS',
            'album': 'album_with_underscores',
            'genre': 'genre-in-hyphens',
            'year': 'year-not-a-number',
            'track_number': 'not-a-track-number'
        },
        'international': {
            'title': 'Título con acentos',
            'artist': 'Артист русский',
            'album': '中文专辑',
            'genre': '世界音乐'
        },
        'compilation': {
            'title': 'Compilation Track',
            'artist': 'Original Artist',
            'album': 'Various Artists Compilation',
            'albumartist': 'Various Artists',
            'compilation': True,
            'genre': 'Various'
        }
    }


def create_test_audio_file(
    title: str = "Test Song",
    artist: str = "Test Artist",
    **metadata
) -> MockAudioFile:
    """
    Create a single test audio file with specified metadata.

    Args:
        title: Song title
        artist: Artist name
        **metadata: Additional metadata fields

    Returns:
        MockAudioFile with specified metadata
    """
    base_metadata = {
        'title': title,
        'artist': artist,
        'album': 'Test Album',
        'year': '2023',
        'track_number': '1',
        'genre': 'Test Genre'
    }
    base_metadata.update(metadata)

    return create_mock_audio_file(metadata=base_metadata)


def save_test_fixtures(output_dir: Path):
    """
    Save test fixtures to files for reuse.

    Args:
        output_dir: Directory to save fixtures to
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata fixtures
    metadata = get_test_metadata()
    with open(output_dir / 'test_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save audio file fixtures
    audio_fixture = AudioTestFixture()
    audio_files_data = []
    for audio_file in audio_fixture.get_all_files():
        audio_files_data.append({
            'path': str(audio_file.path),
            'metadata': audio_file.metadata,
            'duration': audio_file.duration,
            'bitrate': audio_file.bitrate,
            'format': audio_file.format
        })

    with open(output_dir / 'test_audio_files.json', 'w', encoding='utf-8') as f:
        json.dump(audio_files_data, f, indent=2, ensure_ascii=False)

    # Save classification fixtures
    classification_fixture = ClassificationTestFixture()
    classification_data = {}
    for name, file in classification_fixture.test_files.items():
        classification_data[name] = {
            'metadata': file.metadata,
            'duration': file.duration,
            'expected_classification': classification_fixture.expected_classifications[name]
        }

    with open(output_dir / 'test_classifications.json', 'w', encoding='utf-8') as f:
        json.dump(classification_data, f, indent=2, ensure_ascii=False)

    # Clean up
    audio_fixture.cleanup()