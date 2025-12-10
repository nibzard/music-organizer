"""
External Services - Infrastructure Layer

This package contains adapters for external services and APIs,
implementing the Anti-Corruption Layer pattern to protect domain integrity.
"""

from .musicbrainz_adapter import MusicBrainzAdapter
from .acoustid_adapter import AcoustIdAdapter

__all__ = [
    "MusicBrainzAdapter",
    "AcoustIdAdapter",
]