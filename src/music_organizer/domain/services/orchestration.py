"""
Orchestration Service - Cross-context workflow coordination.

This service orchestrates complex workflows that span multiple bounded contexts.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..catalog import Catalog, CatalogService, MetadataService
from ..organization import OrganizationService, OrganizationRule, FolderStructure
from ..classification import ClassificationService, DuplicateService, Classifier


class MusicLibraryOrchestrator:
    """
    Orchestrates complex workflows across bounded contexts.

    This service coordinates operations that require multiple contexts to work together,
    such as importing new music, organizing the library, and maintaining consistency.
    """

    def __init__(
        self,
        catalog: Catalog,
        catalog_service: CatalogService,
        metadata_service: MetadataService,
        organization_service: OrganizationService,
        classification_service: ClassificationService,
        duplicate_service: DuplicateService
    ):
        self.catalog = catalog
        self.catalog_service = catalog_service
        self.metadata_service = metadata_service
        self.organization_service = organization_service
        self.classification_service = classification_service
        self.duplicate_service = duplicate_service

    async def import_music(
        self,
        source_paths: List[Path],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Import music from source paths into the catalog.

        This workflow spans Catalog, Classification, and potentially Organization contexts.
        """
        options = options or {}
        import_results = {
            "imported_count": 0,
            "skipped_count": 0,
            "error_count": 0,
            "duplicates_found": 0,
            "classifications": {},
            "errors": []
        }

        try:
            # Phase 1: Scan and extract metadata
            recordings_to_import = await self._scan_and_extract_metadata(source_paths)

            # Phase 2: Classify content
            classified_recordings = await self._classify_content(recordings_to_import)

            # Phase 3: Detect duplicates
            duplicate_groups = await self._detect_import_duplicates(classified_recordings)

            # Phase 4: Add to catalog
            for recording in classified_recordings:
                try:
                    # Check if it's a duplicate
                    is_duplicate = any(
                        id(recording) in group.recordings
                        for group in duplicate_groups
                    )

                    if is_duplicate:
                        import_results["duplicates_found"] += 1
                        if options.get("skip_duplicates", True):
                            import_results["skipped_count"] += 1
                            continue

                    # Add to catalog
                    await self.catalog_service.add_recording_to_catalog(self.catalog, recording)
                    import_results["imported_count"] += 1

                except Exception as e:
                    import_results["error_count"] += 1
                    import_results["errors"].append(f"Error adding {recording.path.path}: {str(e)}")

            # Phase 5: Optional organization
            if options.get("auto_organize", False):
                organization_results = await self._auto_organize_imported(classified_recordings)
                import_results["organization"] = organization_results

        except Exception as e:
            import_results["errors"].append(f"Import failed: {str(e)}")

        return import_results

    async def organize_library(
        self,
        organization_rules: List[OrganizationRule],
        folder_structure: Optional[FolderStructure] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Organize the entire music library.

        This workflow uses Classification to determine organization strategies
        and Organization to execute the file moves.
        """
        options = options or {}
        organization_results = {
            "organized_count": 0,
            "moved_count": 0,
            "skipped_count": 0,
            "error_count": 0,
            "conflicts_resolved": 0,
            "errors": []
        }

        try:
            # Phase 1: Analyze library organization needs
            all_recordings = list(self.catalog.get_all_recordings())
            classification_results = await self.classification_service.batch_classify(all_recordings)

            # Phase 2: Generate organization plan
            organization_plan = await self._generate_organization_plan(
                all_recordings,
                classification_results,
                organization_rules,
                folder_structure
            )

            # Phase 3: Preview changes (if requested)
            if options.get("preview_only", False):
                organization_results["preview"] = organization_plan
                return organization_results

            # Phase 4: Execute organization
            for item in organization_plan:
                try:
                    # Apply organization rule
                    moved_file = await self.organization_service.organize_file(
                        item["source_path"],
                        item["target_path"],
                        options.get("conflict_strategy", "skip"),
                        dry_run=options.get("dry_run", False)
                    )

                    if moved_file.is_successful:
                        organization_results["moved_count"] += 1
                        # Update catalog with new path
                        await self._update_catalog_after_move(moved_file)
                    elif moved_file.status.value == "skipped":
                        organization_results["skipped_count"] += 1
                    else:
                        organization_results["error_count"] += 1
                        organization_results["errors"].append(
                            f"Failed to move {moved_file.source_path}: {moved_file.error_message}"
                        )

                    if moved_file.has_conflict:
                        organization_results["conflicts_resolved"] += 1

                except Exception as e:
                    organization_results["error_count"] += 1
                    organization_results["errors"].append(str(e))

            organization_results["organized_count"] = len(organization_plan)

        except Exception as e:
            organization_results["errors"].append(f"Organization failed: {str(e)}")

        return organization_results

    async def enhance_metadata(
        self,
        criteria: Optional[Dict[str, Any]] = None,
        enhancement_sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Enhance metadata across the library.

        This workflow uses Classification to identify what needs enhancement
        and Catalog services to apply the enhancements.
        """
        criteria = criteria or {}
        enhancement_results = {
            "enhanced_count": 0,
            "no_change_count": 0,
            "error_count": 0,
            "enhancements": {},
            "errors": []
        }

        try:
            # Phase 1: Find recordings needing enhancement
            recordings_to_enhance = await self._find_recordings_needing_enhancement(criteria)

            # Phase 2: Enhance metadata
            for recording in recordings_to_enhance:
                try:
                    # Check what metadata can be enhanced
                    enhancement_needs = await self._analyze_enhancement_needs(recording)

                    if enhancement_needs:
                        # Apply enhancements from various sources
                        enhanced_metadata = await self._gather_enhancements(
                            recording,
                            enhancement_needs,
                            enhancement_sources or []
                        )

                        # Apply to recording
                        updated_recording = await self.metadata_service.enhance_metadata(
                            recording,
                            enhanced_metadata
                        )

                        enhancement_results["enhanced_count"] += 1
                        enhancement_results["enhancements"][str(recording.path.path)] = enhancement_needs
                    else:
                        enhancement_results["no_change_count"] += 1

                except Exception as e:
                    enhancement_results["error_count"] += 1
                    enhancement_results["errors"].append(
                        f"Error enhancing {recording.path.path}: {str(e)}"
                    )

        except Exception as e:
            enhancement_results["errors"].append(f"Metadata enhancement failed: {str(e)}")

        return enhancement_results

    async def _scan_and_extract_metadata(self, source_paths: List[Path]) -> List[Any]:
        """Scan paths and extract metadata from audio files."""
        # This would integrate with the file scanning and metadata extraction logic
        # For now, return empty list as placeholder
        return []

    async def _classify_content(self, recordings: List[Any]) -> List[Any]:
        """Classify the content type and metadata of recordings."""
        return await self.classification_service.batch_classify(recordings)

    async def _detect_import_duplicates(self, recordings: List[Any]) -> List[Any]:
        """Detect duplicates among new recordings."""
        # Get existing recordings from catalog
        existing_recordings = list(self.catalog.get_all_recordings())
        all_recordings = existing_recordings + recordings

        return await self.duplicate_service.find_duplicates(
            all_recordings,
            threshold=0.85
        )

    async def _auto_organize_imported(self, recordings: List[Any]) -> Dict[str, Any]:
        """Auto-organize newly imported recordings."""
        # Implementation would apply default organization rules
        return {"auto_organized": len(recordings)}

    async def _generate_organization_plan(
        self,
        recordings: List[Any],
        classifications: List[Dict[str, Any]],
        rules: List[OrganizationRule],
        folder_structure: Optional[FolderStructure]
    ) -> List[Dict[str, Any]]:
        """Generate a plan for organizing files."""
        plan = []
        for recording, classification in zip(recordings, classifications):
            # Find matching rule
            for rule in rules:
                if rule.matches(recording.metadata, recording.path):
                    # Generate target path
                    target_path = await self.organization_service.path_service.generate_target_path(
                        recording.path,
                        recording.metadata,
                        [rule],
                        folder_structure
                    )
                    if target_path:
                        plan.append({
                            "source_path": recording.path.path,
                            "target_path": target_path,
                            "rule": rule.name,
                            "classification": classification
                        })
                    break
        return plan

    async def _update_catalog_after_move(self, moved_file: Any) -> None:
        """Update catalog entries after file moves."""
        # This would update the recording's path in the catalog
        pass

    async def _find_recordings_needing_enhancement(self, criteria: Dict[str, Any]) -> List[Any]:
        """Find recordings that need metadata enhancement."""
        # Filter recordings based on criteria
        all_recordings = list(self.catalog.get_all_recordings())
        needing_enhancement = []

        for recording in all_recordings:
            needs_enhancement = False

            # Check various criteria
            if criteria.get("missing_year") and not recording.metadata.year:
                needs_enhancement = True
            if criteria.get("missing_genre") and not recording.metadata.genre:
                needs_enhancement = True
            if criteria.get("missing_albumartist") and not recording.metadata.albumartist:
                needs_enhancement = True

            if needs_enhancement:
                needing_enhancement.append(recording)

        return needing_enhancement

    async def _analyze_enhancement_needs(self, recording: Any) -> List[str]:
        """Analyze what metadata needs enhancement for a recording."""
        needs = []
        metadata = recording.metadata

        if not metadata.year:
            needs.append("year")
        if not metadata.genre:
            needs.append("genre")
        if not metadata.albumartist and metadata.artists:
            needs.append("albumartist")
        if not metadata.composer:
            needs.append("composer")

        return needs

    async def _gather_enhancements(
        self,
        recording: Any,
        needs: List[str],
        sources: List[str]
    ) -> Any:
        """Gather metadata enhancements from various sources."""
        # This would integrate with external services like MusicBrainz
        # For now, return the existing metadata
        return recording.metadata