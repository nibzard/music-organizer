"""
AudioFile Adapter - Anti-Corruption Layer for AudioFile model.

This adapter converts between the legacy AudioFile model and the new domain entities,
protecting the domain from external model dependencies.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from ...domain.catalog import Recording as DomainRecording, Metadata as DomainMetadata, ArtistName, TrackNumber, AudioPath
from ...domain.organization import TargetPath, OrganizationPattern


class AudioFileToRecordingAdapter:
    """
    Adapter for converting between AudioFile and domain Recording entities.

    This implements the Anti-Corruption Layer pattern to ensure the domain
    remains isolated from external model dependencies.
    """

    def to_recording(self, audio_file: Any) -> DomainRecording:
        """
        Convert an AudioFile instance to a domain Recording entity.
        """
        # Convert path
        audio_path = AudioPath(audio_file.path)

        # Convert metadata
        metadata = self._convert_metadata(audio_file.metadata if hasattr(audio_file, 'metadata') else {})

        # Create recording
        recording = DomainRecording(
            path=audio_path,
            metadata=metadata
        )

        # Copy additional attributes if they exist
        if hasattr(audio_file, 'is_processed'):
            recording.is_processed = audio_file.is_processed
        if hasattr(audio_file, 'target_path'):
            recording.target_path = audio_file.target_path

        return recording

    def to_audio_file(self, recording: DomainRecording) -> Dict[str, Any]:
        """
        Convert a domain Recording entity back to AudioFile-compatible format.
        """
        return {
            "path": str(recording.path.path),
            "metadata": self._metadata_to_dict(recording.metadata),
            "is_processed": recording.is_processed,
            "target_path": str(recording.target_path) if recording.target_path else None,
            "genre_classifications": list(recording.genre_classifications),
            "content_type": recording.content_type,
            "energy_level": recording.energy_level,
        }

    def _convert_metadata(self, metadata: Any) -> DomainMetadata:
        """Convert legacy metadata to domain Metadata."""
        # Handle different metadata formats
        if hasattr(metadata, '__dict__'):
            metadata_dict = {
                k: v for k, v in metadata.__dict__.items()
                if not k.startswith('_')
            }
        elif isinstance(metadata, dict):
            metadata_dict = metadata.copy()
        else:
            metadata_dict = {}

        # Convert artist names
        artists = []
        if 'artists' in metadata_dict and metadata_dict['artists']:
            artists = [ArtistName(a) for a in metadata_dict['artists']]
        elif 'artist' in metadata_dict and metadata_dict['artist']:
            artists = [ArtistName(metadata_dict['artist'])]

        # Convert track number
        track_number = None
        if 'track_number' in metadata_dict and metadata_dict['track_number']:
            track_number = TrackNumber(metadata_dict['track_number'])

        # Create domain metadata
        return DomainMetadata(
            title=metadata_dict.get('title'),
            artists=artists,
            album=metadata_dict.get('album'),
            year=metadata_dict.get('year'),
            genre=metadata_dict.get('genre'),
            track_number=track_number,
            total_tracks=metadata_dict.get('total_tracks'),
            disc_number=metadata_dict.get('disc_number'),
            total_discs=metadata_dict.get('total_discs'),
            albumartist=metadata_dict.get('albumartist'),
            composer=metadata_dict.get('composer'),
            # Technical metadata
            duration_seconds=metadata_dict.get('duration_seconds'),
            bitrate=metadata_dict.get('bitrate'),
            sample_rate=metadata_dict.get('sample_rate'),
            channels=metadata_dict.get('channels'),
            file_hash=metadata_dict.get('file_hash'),
            acoustic_fingerprint=metadata_dict.get('acoustic_fingerprint'),
            format_metadata=metadata_dict.get('format_metadata', {}),
        )

    def _metadata_to_dict(self, metadata: DomainMetadata) -> Dict[str, Any]:
        """Convert domain Metadata to dictionary format."""
        return {
            "title": metadata.title,
            "artists": [str(a) for a in metadata.artists] if metadata.artists else [],
            "album": metadata.album,
            "year": metadata.year,
            "genre": metadata.genre,
            "track_number": metadata.track_number.formatted() if metadata.track_number else None,
            "total_tracks": metadata.total_tracks,
            "disc_number": metadata.disc_number,
            "total_discs": metadata.total_discs,
            "albumartist": str(metadata.albumartist) if metadata.albumartist else None,
            "composer": metadata.composer,
            "duration_seconds": metadata.duration_seconds,
            "bitrate": metadata.bitrate,
            "sample_rate": metadata.sample_rate,
            "channels": metadata.channels,
            "file_hash": metadata.file_hash,
            "acoustic_fingerprint": metadata.acoustic_fingerprint,
            "format_metadata": metadata.format_metadata,
            "is_live": metadata.is_live,
            "is_compilation": metadata.is_compilation,
            "has_multiple_artists": metadata.has_multiple_artists,
        }

    def batch_to_recordings(self, audio_files: List[Any]) -> List[DomainRecording]:
        """Convert multiple AudioFiles to domain Recordings."""
        return [self.to_recording(audio_file) for audio_file in audio_files]

    def batch_to_audio_files(self, recordings: List[DomainRecording]) -> List[Dict[str, Any]]:
        """Convert multiple domain Recordings to AudioFile format."""
        return [self.to_audio_file(recording) for recording in recordings]


class OrganizationTargetPathAdapter:
    """
    Adapter for converting between domain TargetPath and external path representations.
    """

    def from_domain(self, target_path: TargetPath) -> Dict[str, Any]:
        """Convert domain TargetPath to external representation."""
        return {
            "path": str(target_path.path),
            "original_path": str(target_path.original_path) if target_path.original_path else None,
            "exists": target_path.exists,
            "conflict_strategy": target_path.conflict_strategy.value if target_path.conflict_strategy else None,
        }

    def to_domain(self, path_dict: Dict[str, Any]) -> TargetPath:
        """Convert external representation to domain TargetPath."""
        from ..domain.organization import ConflictStrategy

        conflict_strategy = ConflictStrategy.SKIP
        if path_dict.get("conflict_strategy"):
            conflict_strategy = ConflictStrategy(path_dict["conflict_strategy"])

        return TargetPath(
            path=Path(path_dict["path"]),
            original_path=Path(path_dict["original_path"]) if path_dict.get("original_path") else None,
            conflict_strategy=conflict_strategy
        )


class PluginResultAdapter:
    """
    Adapter for converting plugin results to domain entities.
    """

    def to_metadata_updates(self, plugin_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert plugin enhancement results to domain metadata updates."""
        updates = {}

        # Map common plugin fields to domain fields
        field_mappings = {
            "year": "year",
            "genre": "genre",
            "albumartist": "albumartist",
            "composer": "composer",
            "title": "title",
            "artists": "artists",
            "album": "album",
        }

        for plugin_field, domain_field in field_mappings.items():
            if plugin_field in plugin_result and plugin_result[plugin_field] is not None:
                updates[domain_field] = plugin_result[plugin_field]

        return updates

    def to_classification_result(self, plugin_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert plugin classification to domain classification format."""
        return {
            "content_type": plugin_result.get("content_type", "unknown"),
            "genres": plugin_result.get("genres", []),
            "confidence": plugin_result.get("confidence", 0.0),
            "source": "plugin",
            "plugin_name": plugin_result.get("plugin_name"),
        }