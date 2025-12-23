"""Built-in plugins for the music organizer."""

from .lastfm_enhancer import LastFmEnhancerPlugin
from .lastfm_scrobbler import LastFmScrobblerPlugin

__all__ = [
    "LastFmEnhancerPlugin",
    "LastFmScrobblerPlugin",
]