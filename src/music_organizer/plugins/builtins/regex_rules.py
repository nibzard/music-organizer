"""Regex Rules Plugin - Advanced music organization with regex-based rules.

This plugin extends the basic custom naming pattern functionality with
powerful regex-based rule matching and conditional logic.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..base import PathPlugin, PluginInfo
from ...models.audio_file import AudioFile
from ...core.regex_rules import RegexRuleEngine, Rule

logger = logging.getLogger(__name__)


def create_plugin(config: Optional[Dict[str, Any]] = None) -> PathPlugin:
    """Factory function to create the plugin."""
    return RegexRulesPlugin(config)


@dataclass
class RuleMatchResult:
    """Result of a rule match operation."""
    rule: Optional[Rule]
    matched: bool
    pattern: str
    reason: str = ""


class RegexRulesPlugin(PathPlugin):
    """Plugin that provides regex-based organization rules for music files."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the plugin with configuration."""
        super().__init__(config)

        # Initialize the rule engine
        rules_file = None
        if self.config.get('rules_file'):
            rules_file = Path(self.config['rules_file'])

        self.rule_engine = RegexRuleEngine(rules_file)

        # Fallback settings
        self.fallback_pattern = self.config.get('fallback_pattern',
            "{content_type}/{artist}/{album} ({year})")
        self.enable_fallback = self.config.get('enable_fallback', True)
        self.log_matches = self.config.get('log_matches', False)
        self.debug_mode = self.config.get('debug_mode', False)

        # Statistics
        self.match_stats = {
            'total_processed': 0,
            'rules_matched': 0,
            'fallback_used': 0,
            'rule_usage': {}
        }

    @property
    def info(self) -> PluginInfo:
        """Return plugin information."""
        return PluginInfo(
            name="regex_rules",
            version="1.0.0",
            description="Advanced music organization with regex-based rule matching",
            author="Music Organizer Team",
            dependencies=[]
        )

    def initialize(self) -> None:
        """Initialize the plugin."""
        # Validate rules if specified
        if self.rules_file and self.rules_file.exists():
            errors = self.rule_engine.validate_rules()
            if errors:
                logger.warning(f"Rule validation warnings: {'; '.join(errors)}")

        # Log statistics
        stats = self.rule_engine.get_rule_statistics()
        logger.info(f"Regex Rules Plugin initialized with {stats['total_rules']} rules "
                   f"({stats['enabled_rules']} enabled)")

    def cleanup(self) -> None:
        """Cleanup resources when plugin is disabled."""
        # Log final statistics
        if self.match_stats['total_processed'] > 0:
            logger.info(f"Regex Rules Plugin stats: "
                       f"{self.match_stats['total_processed']} files processed, "
                       f"{self.match_stats['rules_matched']} matched by rules, "
                       f"{self.match_stats['fallback_used']} used fallback")

    async def generate_target_path(self, audio_file: AudioFile, base_dir: Path) -> Path:
        """Generate target directory path using regex rules."""
        self.match_stats['total_processed'] += 1

        # Find matching rule
        matching_rule = self.rule_engine.find_matching_rule(audio_file)

        if matching_rule:
            # Use the rule's pattern
            self.match_stats['rules_matched'] += 1
            rule_name = matching_rule.name
            self.match_stats['rule_usage'][rule_name] = \
                self.match_stats['rule_usage'].get(rule_name, 0) + 1

            if self.log_matches or self.debug_mode:
                logger.debug(f"Rule '{rule_name}' matched for: {audio_file.path.name}")

            # Use the existing template engine from the parent class
            from .custom_naming_pattern import PatternTemplate
            template_engine = PatternTemplate()
            rendered_path = template_engine.render(matching_rule.pattern, audio_file)

            return base_dir / rendered_path

        # No rule matched, use fallback if enabled
        if self.enable_fallback:
            self.match_stats['fallback_used'] += 1
            if self.debug_mode:
                logger.debug(f"No rule matched for: {audio_file.path.name}, using fallback pattern")

            from .custom_naming_pattern import PatternTemplate
            template_engine = PatternTemplate()
            rendered_path = template_engine.render(self.fallback_pattern, audio_file)

            return base_dir / rendered_path

        # If no fallback, return None to let other plugins handle it
        return None

    async def generate_filename(self, audio_file: AudioFile) -> str:
        """Generate filename using regex rules."""
        # For now, use the standard filename pattern
        # This could be extended to support rule-specific filename patterns
        default_pattern = "{track_number} {title}{file_extension}"

        from .custom_naming_pattern import PatternTemplate
        template_engine = PatternTemplate()
        rendered = template_engine.render(default_pattern, audio_file)

        # Ensure the filename has an extension
        if not Path(rendered).suffix:
            rendered += audio_file.path.suffix

        return rendered

    def test_rules_for_file(self, audio_file: AudioFile) -> List[RuleMatchResult]:
        """Test all rules against an audio file.

        Args:
            audio_file: The audio file to test

        Returns:
            List of rule match results
        """
        results = []

        for rule in self.rule_engine.rules:
            matched = rule.matches(audio_file)

            result = RuleMatchResult(
                rule=rule,
                matched=matched,
                pattern=rule.pattern if matched else "",
                reason=f"Rule {'matched' if matched else 'did not match'}"
            )

            results.append(result)

        return results

    def get_rule_engine(self) -> RegexRuleEngine:
        """Get the underlying rule engine instance."""
        return self.rule_engine

    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin usage statistics."""
        return {
            **self.match_stats,
            'engine_stats': self.rule_engine.get_rule_statistics()
        }

    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self.match_stats = {
            'total_processed': 0,
            'rules_matched': 0,
            'fallback_used': 0,
            'rule_usage': {}
        }

    def export_current_rules(self, output_path: Path) -> None:
        """Export current rules to a file.

        Args:
            output_path: Path to export rules to
        """
        self.rule_engine.export_rules(output_path)

    def add_rule(self, rule: Rule) -> None:
        """Add a new rule.

        Args:
            rule: The rule to add
        """
        self.rule_engine.add_rule(rule)
        logger.info(f"Added rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name.

        Args:
            rule_name: Name of the rule to remove

        Returns:
            True if the rule was removed
        """
        removed = self.rule_engine.remove_rule(rule_name)
        if removed:
            logger.info(f"Removed rule: {rule_name}")
        return removed

    def enable_rule(self, rule_name: str) -> bool:
        """Enable a rule.

        Args:
            rule_name: Name of the rule to enable

        Returns:
            True if the rule was enabled
        """
        enabled = self.rule_engine.enable_rule(rule_name)
        if enabled:
            logger.info(f"Enabled rule: {rule_name}")
        return enabled

    def disable_rule(self, rule_name: str) -> bool:
        """Disable a rule.

        Args:
            rule_name: Name of the rule to disable

        Returns:
            True if the rule was disabled
        """
        disabled = self.rule_engine.disable_rule(rule_name)
        if disabled:
            logger.info(f"Disabled rule: {rule_name}")
        return disabled

    def reload_rules(self) -> None:
        """Reload rules from the configured file."""
        if self.config.get('rules_file'):
            rules_file = Path(self.config['rules_file'])
            if rules_file.exists():
                self.rule_engine.load_rules(rules_file)
                logger.info(f"Reloaded rules from {rules_file}")
            else:
                logger.warning(f"Rules file not found: {rules_file}")

    def create_rule_template(self, template_type: str, **kwargs) -> Rule:
        """Create a rule from a template.

        Args:
            template_type: Type of template to create
            **kwargs: Additional parameters for the template

        Returns:
            Created Rule object
        """
        templates = {
            'genre': self._create_genre_rule,
            'artist': self._create_artist_rule,
            'decade': self._create_decade_rule,
            'soundtrack': self._create_soundtrack_rule,
            'compilation': self._create_compilation_rule,
            'multi_disc': self._create_multi_disc_rule,
        }

        if template_type not in templates:
            raise ValueError(f"Unknown template type: {template_type}")

        return templates[template_type](**kwargs)

    def _create_genre_rule(self, genres: List[str], pattern: str, **kwargs) -> Rule:
        """Create a genre-based rule."""
        from ...core.regex_rule_engine import RuleCondition, ComparisonOperator

        conditions = [
            RuleCondition(
                field="genre",
                operator=ComparisonOperator.MATCHES,
                value="|".join(genres),
                case_sensitive=False
            )
        ]

        return Rule(
            name=f"Genre: {', '.join(genres)}",
            description=f"Organize {', '.join(genres)} music",
            conditions=conditions,
            pattern=pattern,
            **kwargs
        )

    def _create_artist_rule(self, artists: List[str], pattern: str, **kwargs) -> Rule:
        """Create an artist-based rule."""
        from ...core.regex_rule_engine import RuleCondition, ComparisonOperator

        conditions = [
            RuleCondition(
                field="artist",
                operator=ComparisonOperator.IN,
                value=artists,
                case_sensitive=False
            )
        ]

        return Rule(
            name=f"Artists: {', '.join(artists[:3])}{'...' if len(artists) > 3 else ''}",
            description=f"Organize specific artists",
            conditions=conditions,
            pattern=pattern,
            **kwargs
        )

    def _create_decade_rule(self, decade: str, pattern: str, **kwargs) -> Rule:
        """Create a decade-based rule."""
        from ...core.regex_rule_engine import RuleCondition, ComparisonOperator

        conditions = [
            RuleCondition(
                field="year",
                operator=ComparisonOperator.GREATER_EQUAL,
                value=int(decade[:4])
            ),
            RuleCondition(
                field="year",
                operator=ComparisonOperator.LESS_THAN,
                value=int(decade[:4]) + 10
            )
        ]

        return Rule(
            name=f"Decade: {decade}",
            description=f"Organize {decade} music",
            conditions=conditions,
            pattern=pattern,
            **kwargs
        )

    def _create_soundtrack_rule(self, pattern: str = "Soundtracks/{album} ({year})", **kwargs) -> Rule:
        """Create a soundtrack rule."""
        from ...core.regex_rule_engine import RuleCondition, ComparisonOperator

        conditions = [
            RuleCondition(
                field="genre",
                operator=ComparisonOperator.MATCHES,
                value="soundtrack|score|film|motion picture",
                case_sensitive=False
            )
        ]

        return Rule(
            name="Soundtracks",
            description="Organize soundtrack and score albums",
            conditions=conditions,
            pattern=pattern,
            priority=100,
            **kwargs
        )

    def _create_compilation_rule(self, pattern: str = "Compilations/{album} ({year})", **kwargs) -> Rule:
        """Create a compilation rule."""
        from ...core.regex_rule_engine import RuleCondition, ComparisonOperator

        conditions = [
            RuleCondition(
                field="albumartist",
                operator=ComparisonOperator.EQUALS,
                value="Various Artists"
            )
        ]

        return Rule(
            name="Compilations",
            description="Organize various artist compilations",
            conditions=conditions,
            pattern=pattern,
            priority=80,
            **kwargs
        )

    def _create_multi_disc_rule(self, pattern: str = "{artist}/{album} (Disc {disc_number})/{track_number:02} {title}", **kwargs) -> Rule:
        """Create a multi-disc album rule."""
        from ...core.regex_rule_engine import RuleCondition, ComparisonOperator

        conditions = [
            RuleCondition(
                field="disc_number",
                operator=ComparisonOperator.GREATER_THAN,
                value=1
            )
        ]

        return Rule(
            name="Multi-disc Albums",
            description="Handle multi-disc albums",
            conditions=conditions,
            pattern=pattern,
            priority=110,
            **kwargs
        )