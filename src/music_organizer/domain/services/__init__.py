"""
Domain Services - Cross-context Business Logic.

This package contains domain services that operate across multiple bounded contexts.
These services orchestrate complex workflows that involve multiple contexts.
"""

from .orchestration import MusicLibraryOrchestrator
from .integration import ContextIntegrationService

__all__ = [
    "MusicLibraryOrchestrator",
    "ContextIntegrationService",
]