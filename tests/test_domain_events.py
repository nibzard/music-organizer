"""
Comprehensive tests for domain events.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from music_organizer.events.domain_events import (
    RecordingAdded,
    RecordingModified,
    RecordingDeleted,
    DuplicateDetected,
    DuplicateResolved,
    ClassificationCompleted,
    OrganizationCompleted,
    FileMoved,
    MetadataEnhanced,
    LibraryScanned,
    PluginExecuted,
    UserCorrectionApplied,
    PerformanceWarning,
)


class TestRecordingAdded:
    """Test RecordingAdded event."""

    def test_recording_added_creation(self):
        """Test creating RecordingAdded event."""
        event = RecordingAdded(
            recording_id="rec_123",
            file_path="/path/to/file.mp3",
            title="Test Song",
            artist="Test Artist",
            album="Test Album",
            year=2023
        )

        assert event.recording_id == "rec_123"
        assert event.file_path == "/path/to/file.mp3"
        assert event.title == "Test Song"
        assert event.artist == "Test Artist"
        assert event.album == "Test Album"
        assert event.year == 2023

    def test_recording_added_optional_fields(self):
        """Test RecordingAdded with optional fields."""
        event = RecordingAdded(
            recording_id="rec_123",
            file_path="/path/to/file.mp3",
            title="Test Song",
            artist="Test Artist"
        )

        assert event.album is None
        assert event.year is None

    def test_recording_added_to_dict(self):
        """Test RecordingAdded serialization."""
        event = RecordingAdded(
            recording_id="rec_123",
            file_path="/path/to/file.mp3",
            title="Test Song",
            artist="Test Artist",
            album="Test Album",
            year=2023
        )

        data = event.to_dict()
        assert data["event_type"] == "RecordingAdded"
        assert data["data"]["recording_id"] == "rec_123"
        assert data["data"]["title"] == "Test Song"
        assert data["data"]["year"] == 2023

    def test_recording_added_event_data(self):
        """Test _get_event_data method."""
        event = RecordingAdded(
            recording_id="rec_123",
            file_path="/path/to/file.mp3",
            title="Test Song",
            artist="Test Artist"
        )

        event_data = event._get_event_data()
        assert event_data["recording_id"] == "rec_123"
        assert event_data["file_path"] == "/path/to/file.mp3"
        assert event_data["title"] == "Test Song"
        assert event_data["artist"] == "Test Artist"


class TestRecordingModified:
    """Test RecordingModified event."""

    def test_recording_modified_creation(self):
        """Test creating RecordingModified event."""
        event = RecordingModified(
            recording_id="rec_123",
            modified_fields=["title", "artist"],
            old_values={"title": "Old Title", "artist": "Old Artist"},
            new_values={"title": "New Title", "artist": "New Artist"},
            modification_source="user"
        )

        assert event.recording_id == "rec_123"
        assert event.modified_fields == ["title", "artist"]
        assert event.old_values == {"title": "Old Title", "artist": "Old Artist"}
        assert event.new_values == {"title": "New Title", "artist": "New Artist"}
        assert event.modification_source == "user"

    def test_recording_modified_defaults(self):
        """Test RecordingModified default values."""
        event = RecordingModified(
            recording_id="rec_123",
            modified_fields=["title"]
        )

        assert event.old_values == {}
        assert event.new_values == {}
        assert event.modification_source == "user"

    def test_recording_modified_sources(self):
        """Test different modification sources."""
        sources = ["user", "plugin", "auto"]
        for source in sources:
            event = RecordingModified(
                recording_id="rec_123",
                modified_fields=["title"],
                modification_source=source
            )
            assert event.modification_source == source

    def test_recording_modified_to_dict(self):
        """Test RecordingModified serialization."""
        event = RecordingModified(
            recording_id="rec_123",
            modified_fields=["title"],
            old_values={"title": "Old"},
            new_values={"title": "New"}
        )

        data = event.to_dict()
        assert data["event_type"] == "RecordingModified"
        assert data["data"]["modified_fields"] == ["title"]
        assert data["data"]["old_values"] == {"title": "Old"}
        assert data["data"]["new_values"] == {"title": "New"}


class TestRecordingDeleted:
    """Test RecordingDeleted event."""

    def test_recording_deleted_creation(self):
        """Test creating RecordingDeleted event."""
        event = RecordingDeleted(
            recording_id="rec_123",
            file_path="/path/to/file.mp3",
            title="Test Song",
            artist="Test Artist",
            reason="user_deleted"
        )

        assert event.recording_id == "rec_123"
        assert event.file_path == "/path/to/file.mp3"
        assert event.title == "Test Song"
        assert event.artist == "Test Artist"
        assert event.reason == "user_deleted"

    def test_recording_deleted_reasons(self):
        """Test different deletion reasons."""
        reasons = ["user_deleted", "duplicate_removed", "error"]
        for reason in reasons:
            event = RecordingDeleted(
                recording_id="rec_123",
                file_path="/path/to/file.mp3",
                title="Test",
                artist="Artist",
                reason=reason
            )
            assert event.reason == reason

    def test_recording_deleted_default_reason(self):
        """Test default deletion reason."""
        event = RecordingDeleted(
            recording_id="rec_123",
            file_path="/path/to/file.mp3",
            title="Test",
            artist="Artist"
        )
        assert event.reason == "user_deleted"

    def test_recording_deleted_to_dict(self):
        """Test RecordingDeleted serialization."""
        event = RecordingDeleted(
            recording_id="rec_123",
            file_path="/path/to/file.mp3",
            title="Test",
            artist="Artist",
            reason="duplicate_removed"
        )

        data = event.to_dict()
        assert data["event_type"] == "RecordingDeleted"
        assert data["data"]["reason"] == "duplicate_removed"


class TestDuplicateDetected:
    """Test DuplicateDetected event."""

    def test_duplicate_detected_creation(self):
        """Test creating DuplicateDetected event."""
        event = DuplicateDetected(
            duplicate_group_id="group_123",
            recording_ids=["rec_1", "rec_2", "rec_3"],
            similarity_scores={"rec_1": 0.95, "rec_2": 0.92},
            similarity_threshold=0.90,
            detection_method="metadata"
        )

        assert event.duplicate_group_id == "group_123"
        assert event.recording_ids == ["rec_1", "rec_2", "rec_3"]
        assert event.similarity_scores == {"rec_1": 0.95, "rec_2": 0.92}
        assert event.similarity_threshold == 0.90
        assert event.detection_method == "metadata"

    def test_duplicate_detected_methods(self):
        """Test different detection methods."""
        methods = ["metadata", "fingerprint", "hybrid"]
        for method in methods:
            event = DuplicateDetected(
                duplicate_group_id="group_123",
                recording_ids=["rec_1"],
                similarity_scores={},
                similarity_threshold=0.90,
                detection_method=method
            )
            assert event.detection_method == method

    def test_duplicate_detected_to_dict(self):
        """Test DuplicateDetected serialization."""
        event = DuplicateDetected(
            duplicate_group_id="group_123",
            recording_ids=["rec_1", "rec_2"],
            similarity_scores={"rec_1": 0.95},
            similarity_threshold=0.90
        )

        data = event.to_dict()
        assert data["event_type"] == "DuplicateDetected"
        assert data["data"]["duplicate_group_id"] == "group_123"
        assert data["data"]["recording_ids"] == ["rec_1", "rec_2"]


class TestDuplicateResolved:
    """Test DuplicateResolved event."""

    def test_duplicate_resolved_creation(self):
        """Test creating DuplicateResolved event."""
        event = DuplicateResolved(
            duplicate_group_id="group_123",
            kept_recording_id="rec_1",
            removed_recording_ids=["rec_2", "rec_3"],
            resolution_strategy="keep_best"
        )

        assert event.duplicate_group_id == "group_123"
        assert event.kept_recording_id == "rec_1"
        assert event.removed_recording_ids == ["rec_2", "rec_3"]
        assert event.resolution_strategy == "keep_best"

    def test_duplicate_resolved_keep_all(self):
        """Test resolution when all recordings kept."""
        event = DuplicateResolved(
            duplicate_group_id="group_123",
            kept_recording_id=None,
            removed_recording_ids=[],
            resolution_strategy="keep_all"
        )

        assert event.kept_recording_id is None
        assert event.removed_recording_ids == []

    def test_duplicate_resolved_strategies(self):
        """Test different resolution strategies."""
        strategies = ["keep_best", "keep_all", "remove_duplicates"]
        for strategy in strategies:
            event = DuplicateResolved(
                duplicate_group_id="group_123",
                kept_recording_id="rec_1",
                removed_recording_ids=[],
                resolution_strategy=strategy
            )
            assert event.resolution_strategy == strategy

    def test_duplicate_resolved_to_dict(self):
        """Test DuplicateResolved serialization."""
        event = DuplicateResolved(
            duplicate_group_id="group_123",
            kept_recording_id="rec_1",
            removed_recording_ids=["rec_2"],
            resolution_strategy="keep_best"
        )

        data = event.to_dict()
        assert data["event_type"] == "DuplicateResolved"
        assert data["data"]["kept_recording_id"] == "rec_1"
        assert data["data"]["resolution_strategy"] == "keep_best"


class TestClassificationCompleted:
    """Test ClassificationCompleted event."""

    def test_classification_completed_creation(self):
        """Test creating ClassificationCompleted event."""
        event = ClassificationCompleted(
            recording_id="rec_123",
            content_type="music",
            genres=["Rock", "Alternative"],
            confidence=0.95,
            classification_source="automatic"
        )

        assert event.recording_id == "rec_123"
        assert event.content_type == "music"
        assert event.genres == ["Rock", "Alternative"]
        assert event.confidence == 0.95
        assert event.classification_source == "automatic"

    def test_classification_completed_single_genre(self):
        """Test classification with single genre."""
        event = ClassificationCompleted(
            recording_id="rec_123",
            content_type="speech",
            genres=["Podcast"],
            confidence=0.88
        )

        assert len(event.genres) == 1
        assert event.genres == ["Podcast"]

    def test_classification_completed_sources(self):
        """Test different classification sources."""
        sources = ["automatic", "user", "plugin"]
        for source in sources:
            event = ClassificationCompleted(
                recording_id="rec_123",
                content_type="music",
                genres=["Rock"],
                confidence=0.90,
                classification_source=source
            )
            assert event.classification_source == source

    def test_classification_completed_confidence_range(self):
        """Test confidence values in valid range."""
        for conf in [0.0, 0.5, 0.99, 1.0]:
            event = ClassificationCompleted(
                recording_id="rec_123",
                content_type="music",
                genres=["Rock"],
                confidence=conf
            )
            assert event.confidence == conf

    def test_classification_completed_to_dict(self):
        """Test ClassificationCompleted serialization."""
        event = ClassificationCompleted(
            recording_id="rec_123",
            content_type="music",
            genres=["Jazz"],
            confidence=0.92
        )

        data = event.to_dict()
        assert data["event_type"] == "ClassificationCompleted"
        assert data["data"]["genres"] == ["Jazz"]
        assert data["data"]["confidence"] == 0.92


class TestOrganizationCompleted:
    """Test OrganizationCompleted event."""

    def test_organization_completed_creation(self):
        """Test creating OrganizationCompleted event."""
        event = OrganizationCompleted(
            source_directory="/source",
            target_directory="/target",
            organized_count=100,
            moved_count=80,
            skipped_count=15,
            error_count=5,
            conflicts_resolved=3
        )

        assert event.source_directory == "/source"
        assert event.target_directory == "/target"
        assert event.organized_count == 100
        assert event.moved_count == 80
        assert event.skipped_count == 15
        assert event.error_count == 5
        assert event.conflicts_resolved == 3

    def test_organization_completed_no_errors(self):
        """Test organization with no errors."""
        event = OrganizationCompleted(
            source_directory="/source",
            target_directory="/target",
            organized_count=50,
            moved_count=50,
            skipped_count=0,
            error_count=0,
            conflicts_resolved=0
        )

        assert event.error_count == 0
        assert event.skipped_count == 0

    def test_organization_completed_to_dict(self):
        """Test OrganizationCompleted serialization."""
        event = OrganizationCompleted(
            source_directory="/source",
            target_directory="/target",
            organized_count=10,
            moved_count=8,
            skipped_count=1,
            error_count=1,
            conflicts_resolved=2
        )

        data = event.to_dict()
        assert data["event_type"] == "OrganizationCompleted"
        assert data["data"]["organized_count"] == 10
        assert data["data"]["moved_count"] == 8


class TestFileMoved:
    """Test FileMoved event."""

    def test_file_moved_creation(self):
        """Test creating FileMoved event."""
        event = FileMoved(
            recording_id="rec_123",
            source_path="/source/file.mp3",
            target_path="/target/file.mp3",
            conflict_resolved=True,
            conflict_strategy="rename"
        )

        assert event.recording_id == "rec_123"
        assert event.source_path == "/source/file.mp3"
        assert event.target_path == "/target/file.mp3"
        assert event.conflict_resolved is True
        assert event.conflict_strategy == "rename"

    def test_file_moved_no_conflict(self):
        """Test file moved without conflict."""
        event = FileMoved(
            recording_id="rec_123",
            source_path="/source/file.mp3",
            target_path="/target/file.mp3"
        )

        assert event.conflict_resolved is False
        assert event.conflict_strategy is None

    def test_file_moved_to_dict(self):
        """Test FileMoved serialization."""
        event = FileMoved(
            recording_id="rec_123",
            source_path="/source/file.mp3",
            target_path="/target/renamed.mp3",
            conflict_resolved=True,
            conflict_strategy="rename"
        )

        data = event.to_dict()
        assert data["event_type"] == "FileMoved"
        assert data["data"]["conflict_resolved"] is True
        assert data["data"]["conflict_strategy"] == "rename"


class TestMetadataEnhanced:
    """Test MetadataEnhanced event."""

    def test_metadata_enhanced_creation(self):
        """Test creating MetadataEnhanced event."""
        event = MetadataEnhanced(
            recording_id="rec_123",
            enhanced_fields=["title", "artist", "album"],
            source="musicbrainz",
            enhancement_confidence=0.98,
            old_values={"title": "Unknown"},
            new_values={"title": "Real Title", "artist": "Real Artist"}
        )

        assert event.recording_id == "rec_123"
        assert event.enhanced_fields == ["title", "artist", "album"]
        assert event.source == "musicbrainz"
        assert event.enhancement_confidence == 0.98
        assert event.old_values == {"title": "Unknown"}
        assert event.new_values == {"title": "Real Title", "artist": "Real Artist"}

    def test_metadata_enhanced_sources(self):
        """Test different metadata sources."""
        sources = ["musicbrainz", "acoustid", "user_input"]
        for source in sources:
            event = MetadataEnhanced(
                recording_id="rec_123",
                enhanced_fields=["title"],
                source=source,
                enhancement_confidence=0.90
            )
            assert event.source == source

    def test_metadata_enhanced_empty_values(self):
        """Test metadata enhancement with empty old/new values."""
        event = MetadataEnhanced(
            recording_id="rec_123",
            enhanced_fields=["title"],
            source="user_input",
            enhancement_confidence=1.0
        )

        assert event.old_values == {}
        assert event.new_values == {}

    def test_metadata_enhanced_to_dict(self):
        """Test MetadataEnhanced serialization."""
        event = MetadataEnhanced(
            recording_id="rec_123",
            enhanced_fields=["genre"],
            source="acoustid",
            enhancement_confidence=0.85,
            new_values={"genre": "Rock"}
        )

        data = event.to_dict()
        assert data["event_type"] == "MetadataEnhanced"
        assert data["data"]["enhanced_fields"] == ["genre"]
        assert data["data"]["source"] == "acoustid"


class TestLibraryScanned:
    """Test LibraryScanned event."""

    def test_library_scanned_creation(self):
        """Test creating LibraryScanned event."""
        event = LibraryScanned(
            source_directory="/music",
            total_files_found=1000,
            total_files_imported=950,
            duplicates_found=50,
            errors=["file1.mp3: corrupt", "file2.mp3: unreadable"],
            scan_duration_seconds=120.5
        )

        assert event.source_directory == "/music"
        assert event.total_files_found == 1000
        assert event.total_files_imported == 950
        assert event.duplicates_found == 50
        assert event.errors == ["file1.mp3: corrupt", "file2.mp3: unreadable"]
        assert event.scan_duration_seconds == 120.5

    def test_library_scanned_no_errors(self):
        """Test library scan with no errors."""
        event = LibraryScanned(
            source_directory="/music",
            total_files_found=100,
            total_files_imported=100,
            duplicates_found=0
        )

        assert event.errors == []
        assert event.scan_duration_seconds is None

    def test_library_scanned_to_dict(self):
        """Test LibraryScanned serialization."""
        event = LibraryScanned(
            source_directory="/music",
            total_files_found=500,
            total_files_imported=450,
            duplicates_found=50,
            errors=["Error 1"],
            scan_duration_seconds=60.0
        )

        data = event.to_dict()
        assert data["event_type"] == "LibraryScanned"
        assert data["data"]["total_files_found"] == 500
        assert len(data["data"]["errors"]) == 1


class TestPluginExecuted:
    """Test PluginExecuted event."""

    def test_plugin_executed_creation(self):
        """Test creating PluginExecuted event."""
        event = PluginExecuted(
            plugin_name="metadata_enhancer",
            plugin_type="metadata",
            recording_ids=["rec_1", "rec_2", "rec_3"],
            execution_duration_ms=150.5,
            success=True
        )

        assert event.plugin_name == "metadata_enhancer"
        assert event.plugin_type == "metadata"
        assert event.recording_ids == ["rec_1", "rec_2", "rec_3"]
        assert event.execution_duration_ms == 150.5
        assert event.success is True

    def test_plugin_executed_failure(self):
        """Test plugin execution failure."""
        event = PluginExecuted(
            plugin_name="failing_plugin",
            plugin_type="classification",
            recording_ids=["rec_1"],
            success=False,
            error_message="Failed to connect to service"
        )

        assert event.success is False
        assert event.error_message == "Failed to connect to service"

    def test_plugin_executed_types(self):
        """Test different plugin types."""
        types = ["metadata", "classification", "organization", "output"]
        for plugin_type in types:
            event = PluginExecuted(
                plugin_name="test_plugin",
                plugin_type=plugin_type,
                recording_ids=["rec_1"]
            )
            assert event.plugin_type == plugin_type

    def test_plugin_executed_no_duration(self):
        """Test plugin execution without duration tracking."""
        event = PluginExecuted(
            plugin_name="quick_plugin",
            plugin_type="metadata",
            recording_ids=["rec_1"]
        )

        assert event.execution_duration_ms is None

    def test_plugin_executed_to_dict(self):
        """Test PluginExecuted serialization."""
        event = PluginExecuted(
            plugin_name="test_plugin",
            plugin_type="organization",
            recording_ids=["rec_1"],
            execution_duration_ms=100.0,
            success=True
        )

        data = event.to_dict()
        assert data["event_type"] == "PluginExecuted"
        assert data["data"]["plugin_name"] == "test_plugin"
        assert data["data"]["success"] is True


class TestUserCorrectionApplied:
    """Test UserCorrectionApplied event."""

    def test_user_correction_creation(self):
        """Test creating UserCorrectionApplied event."""
        event = UserCorrectionApplied(
            recording_id="rec_123",
            correction_type="genre",
            original_value="Pop",
            corrected_value="Rock",
            correction_source="manual_ui"
        )

        assert event.recording_id == "rec_123"
        assert event.correction_type == "genre"
        assert event.original_value == "Pop"
        assert event.corrected_value == "Rock"
        assert event.correction_source == "manual_ui"

    def test_user_correction_types(self):
        """Test different correction types."""
        types = ["genre", "content_type", "organization_path"]
        for corr_type in types:
            event = UserCorrectionApplied(
                recording_id="rec_123",
                correction_type=corr_type,
                original_value="old",
                corrected_value="new",
                correction_source="manual_ui"
            )
            assert event.correction_type == corr_type

    def test_user_correction_sources(self):
        """Test different correction sources."""
        sources = ["manual_ui", "bulk_edit"]
        for source in sources:
            event = UserCorrectionApplied(
                recording_id="rec_123",
                correction_type="genre",
                original_value="old",
                corrected_value="new",
                correction_source=source
            )
            assert event.correction_source == source

    def test_user_correction_complex_values(self):
        """Test correction with complex values."""
        event = UserCorrectionApplied(
            recording_id="rec_123",
            correction_type="organization_path",
            original_value="/old/path/file.mp3",
            corrected_value="/new/path/file.mp3",
            correction_source="bulk_edit"
        )

        assert event.original_value == "/old/path/file.mp3"
        assert event.corrected_value == "/new/path/file.mp3"

    def test_user_correction_to_dict(self):
        """Test UserCorrectionApplied serialization."""
        event = UserCorrectionApplied(
            recording_id="rec_123",
            correction_type="content_type",
            original_value="speech",
            corrected_value="music",
            correction_source="manual_ui"
        )

        data = event.to_dict()
        assert data["event_type"] == "UserCorrectionApplied"
        assert data["data"]["correction_type"] == "content_type"
        assert data["data"]["original_value"] == "speech"
        assert data["data"]["corrected_value"] == "music"


class TestPerformanceWarning:
    """Test PerformanceWarning event."""

    def test_performance_warning_creation(self):
        """Test creating PerformanceWarning event."""
        event = PerformanceWarning(
            operation="batch_scan",
            duration_ms=5000.0,
            threshold_ms=3000.0,
            details={"file_count": 10000, "avg_time_per_file": 0.5}
        )

        assert event.operation == "batch_scan"
        assert event.duration_ms == 5000.0
        assert event.threshold_ms == 3000.0
        assert event.details == {"file_count": 10000, "avg_time_per_file": 0.5}

    def test_performance_warning_no_details(self):
        """Test performance warning without details."""
        event = PerformanceWarning(
            operation="quick_operation",
            duration_ms=150.0,
            threshold_ms=100.0
        )

        assert event.details == {}

    def test_performance_warning_to_dict(self):
        """Test PerformanceWarning serialization."""
        event = PerformanceWarning(
            operation="metadata_fetch",
            duration_ms=2500.0,
            threshold_ms=2000.0,
            details={"api_calls": 100}
        )

        data = event.to_dict()
        assert data["event_type"] == "PerformanceWarning"
        assert data["data"]["operation"] == "metadata_fetch"
        assert data["data"]["duration_ms"] == 2500.0


class TestEventInheritance:
    """Test that all events inherit from DomainEvent properly."""

    def test_all_events_have_base_fields(self):
        """Test all events have base DomainEvent fields."""
        event_classes = [
            RecordingAdded,
            RecordingModified,
            RecordingDeleted,
            DuplicateDetected,
            DuplicateResolved,
            ClassificationCompleted,
            OrganizationCompleted,
            FileMoved,
            MetadataEnhanced,
            LibraryScanned,
            PluginExecuted,
            UserCorrectionApplied,
            PerformanceWarning,
        ]

        for event_class in event_classes:
            # Create minimal instance
            if event_class == RecordingAdded:
                event = event_class(
                    recording_id="test",
                    file_path="/path",
                    title="Test",
                    artist="Artist"
                )
            elif event_class == RecordingModified:
                event = event_class(
                    recording_id="test",
                    modified_fields=["test"]
                )
            elif event_class == RecordingDeleted:
                event = event_class(
                    recording_id="test",
                    file_path="/path",
                    title="Test",
                    artist="Artist"
                )
            elif event_class == DuplicateDetected:
                event = event_class(
                    duplicate_group_id="test",
                    recording_ids=["rec1"],
                    similarity_scores={},
                    similarity_threshold=0.9
                )
            elif event_class == DuplicateResolved:
                event = event_class(
                    duplicate_group_id="test",
                    kept_recording_id="rec1",
                    removed_recording_ids=[],
                    resolution_strategy="keep_best"
                )
            elif event_class == ClassificationCompleted:
                event = event_class(
                    recording_id="test",
                    content_type="music",
                    genres=["Rock"],
                    confidence=0.9
                )
            elif event_class == OrganizationCompleted:
                event = event_class(
                    source_directory="/src",
                    target_directory="/dst",
                    organized_count=10,
                    moved_count=10,
                    skipped_count=0,
                    error_count=0,
                    conflicts_resolved=0
                )
            elif event_class == FileMoved:
                event = event_class(
                    recording_id="test",
                    source_path="/src",
                    target_path="/dst"
                )
            elif event_class == MetadataEnhanced:
                event = event_class(
                    recording_id="test",
                    enhanced_fields=["title"],
                    source="musicbrainz",
                    enhancement_confidence=0.9
                )
            elif event_class == LibraryScanned:
                event = event_class(
                    source_directory="/src",
                    total_files_found=100,
                    total_files_imported=100,
                    duplicates_found=0
                )
            elif event_class == PluginExecuted:
                event = event_class(
                    plugin_name="test",
                    plugin_type="metadata",
                    recording_ids=["rec1"]
                )
            elif event_class == UserCorrectionApplied:
                event = event_class(
                    recording_id="test",
                    correction_type="genre",
                    original_value="old",
                    corrected_value="new",
                    correction_source="manual_ui"
                )
            elif event_class == PerformanceWarning:
                event = event_class(
                    operation="test",
                    duration_ms=100.0,
                    threshold_ms=50.0
                )

            # Check base fields
            assert hasattr(event, 'event_id')
            assert hasattr(event, 'timestamp')
            assert hasattr(event, 'aggregate_id')
            assert hasattr(event, 'aggregate_type')
            assert hasattr(event, 'version')
            assert hasattr(event, 'metadata')
            assert hasattr(event, 'to_dict')
            assert hasattr(event, '_get_event_data')

            # Check to_dict works
            data = event.to_dict()
            assert 'event_type' in data
            assert 'event_id' in data
            assert 'timestamp' in data
            assert 'data' in data


class TestEventTimestamps:
    """Test event timestamps are set correctly."""

    def test_events_have_timestamps(self):
        """Test events get timestamps at creation."""
        before = datetime.now()
        event = RecordingAdded(
            recording_id="test",
            file_path="/path",
            title="Test",
            artist="Artist"
        )
        after = datetime.now()

        assert before <= event.timestamp <= after


class TestEventMetadata:
    """Test event metadata handling."""

    def test_events_can_have_metadata(self):
        """Test events can store metadata."""
        event = RecordingAdded(
            recording_id="test",
            file_path="/path",
            title="Test",
            artist="Artist",
            metadata={"source": "test", "batch_id": "123"}
        )

        assert event.metadata == {"source": "test", "batch_id": "123"}


class TestEventAggregateInfo:
    """Test event aggregate information."""

    def test_event_with_aggregate_info(self):
        """Test events can have aggregate info."""
        event = RecordingAdded(
            recording_id="test",
            file_path="/path",
            title="Test",
            artist="Artist",
            aggregate_id="agg_123",
            aggregate_type="Recording"
        )

        assert event.aggregate_id == "agg_123"
        assert event.aggregate_type == "Recording"

    def test_event_to_dict_includes_aggregate(self):
        """Test aggregate info in serialized event."""
        event = RecordingAdded(
            recording_id="test",
            file_path="/path",
            title="Test",
            artist="Artist",
            aggregate_id="agg_123",
            aggregate_type="Recording"
        )

        data = event.to_dict()
        assert data["aggregate_id"] == "agg_123"
        assert data["aggregate_type"] == "Recording"
