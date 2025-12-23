"""Built-in plugins for the music organizer."""

from .lastfm_enhancer import LastFmEnhancerPlugin
from .lastfm_scrobbler import LastFmScrobblerPlugin
from .kodi_nfo_exporter import KodiNfoExporterPlugin

__all__ = [
    "LastFmEnhancerPlugin",
    "LastFmScrobblerPlugin",
    "KodiNfoExporterPlugin",
]
