"""
Adapters - Infrastructure Layer

This package contains adapters that isolate the domain from external concerns,
implementing the Anti-Corruption Layer pattern to protect domain integrity.
"""

from .audio_file_adapter import AudioFileToRecordingAdapter
from .mutagen_adapter import MutagenMetadataAdapter
from .filesystem_adapter import FilesystemAdapter

__all__ = [
    "AudioFileToRecordingAdapter",
    "MutagenMetadataAdapter",
    "FilesystemAdapter",
]