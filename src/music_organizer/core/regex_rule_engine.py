"""Regex Rules Engine for advanced music organization patterns.

This module provides a powerful regex-based rule engine that allows users
to define complex organization patterns with conditional logic and regex matching.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..models.audio_file import AudioFile, ContentType

logger = logging.getLogger(__name__)


class LogicalOperator(Enum):
    """Logical operators for rule conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"


class ComparisonOperator(Enum):
    """Comparison operators for rule conditions."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"
    NOT_MATCHES = "not_matches"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "ge"
    LESS_EQUAL = "le"
    IN = "in"
    NOT_IN = "not_in"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"


@dataclass
class RuleCondition:
    """A single condition in a rule."""
    field: str  # The AudioFile field to check
    operator: ComparisonOperator  # The comparison operator
    value: Any = None  # The value to compare against (optional for some operators)
    case_sensitive: bool = True  # Whether the comparison is case sensitive

    def __post_init__(self) -> None:
        """Validate the condition after creation."""
        # Convert string operators to enum if needed
        if isinstance(self.operator, str):
            self.operator = ComparisonOperator(self.operator.lower())

    def matches(self, audio_file: AudioFile) -> bool:
        """Check if this condition matches the given audio file."""
        # Get the field value from the audio file
        field_value = self._get_field_value(audio_file)

        # Handle empty check operators
        if self.operator == ComparisonOperator.IS_EMPTY:
            return self._is_empty(field_value)
        elif self.operator == ComparisonOperator.IS_NOT_EMPTY:
            return not self._is_empty(field_value)

        # Convert both values to strings for text comparisons
        if field_value is not None:
            field_str = str(field_value)
            value_str = str(self.value) if self.value is not None else ""

            if not self.case_sensitive:
                field_str = field_str.lower()
                value_str = value_str.lower()
        else:
            field_str = ""
            value_str = str(self.value) if self.value is not None else ""

        # Apply the comparison operator
        if self.operator == ComparisonOperator.EQUALS:
            return field_str == value_str
        elif self.operator == ComparisonOperator.NOT_EQUALS:
            return field_str != value_str
        elif self.operator == ComparisonOperator.CONTAINS:
            return value_str in field_str
        elif self.operator == ComparisonOperator.NOT_CONTAINS:
            return value_str not in field_str
        elif self.operator == ComparisonOperator.MATCHES:
            try:
                pattern = re.compile(value_str, re.IGNORECASE if not self.case_sensitive else 0)
                return bool(pattern.search(field_str))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{value_str}': {e}")
                return False
        elif self.operator == ComparisonOperator.NOT_MATCHES:
            try:
                pattern = re.compile(value_str, re.IGNORECASE if not self.case_sensitive else 0)
                return not bool(pattern.search(field_str))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{value_str}': {e}")
                return True  # Treat non-matching patterns as not matching
        elif self.operator == ComparisonOperator.STARTS_WITH:
            return field_str.startswith(value_str)
        elif self.operator == ComparisonOperator.ENDS_WITH:
            return field_str.endswith(value_str)
        elif self.operator == ComparisonOperator.IN:
            if isinstance(self.value, (list, tuple)):
                return field_str in [str(v) for v in self.value]
            return field_str in value_str.split(',') if isinstance(value_str, str) else False
        elif self.operator == ComparisonOperator.NOT_IN:
            if isinstance(self.value, (list, tuple)):
                return field_str not in [str(v) for v in self.value]
            return field_str not in value_str.split(',') if isinstance(value_str, str) else True
        elif self.operator in [ComparisonOperator.GREATER_THAN, ComparisonOperator.LESS_THAN,
                               ComparisonOperator.GREATER_EQUAL, ComparisonOperator.LESS_EQUAL]:
            try:
                field_num = float(field_str)
                value_num = float(value_str)
                if self.operator == ComparisonOperator.GREATER_THAN:
                    return field_num > value_num
                elif self.operator == ComparisonOperator.LESS_THAN:
                    return field_num < value_num
                elif self.operator == ComparisonOperator.GREATER_EQUAL:
                    return field_num >= value_num
                elif self.operator == ComparisonOperator.LESS_EQUAL:
                    return field_num <= value_num
            except ValueError:
                return False

        return False

    def _get_field_value(self, audio_file: AudioFile) -> Any:
        """Get the value of a field from an AudioFile."""
        # Handle field aliases
        if self.field == "artist":
            # Alias for primary_artist
            return audio_file.primary_artist or (audio_file.artists[0] if audio_file.artists else None)

        # Direct attributes
        if hasattr(audio_file, self.field):
            return getattr(audio_file, self.field)

        # Metadata fields
        if self.field in audio_file.metadata:
            return audio_file.metadata[self.field]

        # Special computed fields
        if self.field == 'albumartist':
            return audio_file.metadata.get('albumartist', '') or audio_file.primary_artist or ''
        elif self.field == 'first_letter':
            return (audio_file.primary_artist or audio_file.artists[0] if audio_file.artists else 'Unknown')[0].upper()
        elif self.field == 'decade':
            return f"{(audio_file.year // 10) * 10}s" if audio_file.year else ''
        elif self.field == 'file_extension':
            return audio_file.path.suffix.lstrip('.')
        elif self.field == 'duration':
            return str(audio_file.metadata.get('duration', ''))
        elif self.field == 'duration_minutes':
            duration = audio_file.metadata.get('duration', 0)
            return f"{duration / 60:.1f}" if duration else ''
        elif self.field == 'bitrate':
            return str(audio_file.metadata.get('bitrate', ''))
        elif self.field == 'format':
            return audio_file.file_type.upper()
        elif self.field == 'disc_number':
            return str(audio_file.metadata.get('disc_number', ''))
        elif self.field == 'total_discs':
            return str(audio_file.metadata.get('total_discs', ''))
        elif self.field == 'total_tracks':
            return str(audio_file.metadata.get('total_tracks', ''))

        return None

    def _is_empty(self, value: Any) -> bool:
        """Check if a value is empty."""
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple)):
            return len(value) == 0
        return False


@dataclass
class Rule:
    """A complete organization rule with conditions and actions."""
    name: str
    description: str = ""
    conditions: List[Union[RuleCondition, Dict[str, Any]]] = field(default_factory=list)
    condition_operator: LogicalOperator = LogicalOperator.AND
    pattern: str = ""  # The path/filename pattern
    priority: int = 0  # Higher priority rules are checked first
    enabled: bool = True
    tags: List[str] = field(default_factory=list)  # For categorizing rules

    def __post_init__(self) -> None:
        """Convert dict conditions to RuleCondition objects."""
        for i, condition in enumerate(self.conditions):
            if isinstance(condition, dict):
                self.conditions[i] = RuleCondition(**condition)

        # Convert string operator to enum if needed
        if isinstance(self.condition_operator, str):
            self.condition_operator = LogicalOperator(self.condition_operator.lower())

    def matches(self, audio_file: AudioFile) -> bool:
        """Check if this rule matches the given audio file."""
        if not self.enabled:
            return False

        # Rules with no conditions match everything (fallback rules)
        if not self.conditions:
            return True

        # Evaluate each condition
        condition_results = []
        for condition in self.conditions:
            if isinstance(condition, RuleCondition):
                condition_results.append(condition.matches(audio_file))

        # Apply logical operator
        if self.condition_operator == LogicalOperator.AND:
            return all(condition_results)
        elif self.condition_operator == LogicalOperator.OR:
            return any(condition_results)
        elif self.condition_operator == LogicalOperator.NOT:
            return not all(condition_results)

        return False


class RegexRuleEngine:
    """Main engine for processing regex-based organization rules."""

    def __init__(self, rules_file: Optional[Path] = None):
        """Initialize the rule engine.

        Args:
            rules_file: Path to a JSON file containing rule definitions
        """
        self.rules: List[Rule] = []
        self.rules_file = rules_file
        self._compiled_patterns: Dict[str, re.Pattern] = {}

        # Initialize with built-in rules if no file specified
        if rules_file and rules_file.exists():
            self.load_rules(rules_file)
        else:
            self._load_builtin_rules()

    def load_rules(self, rules_file: Path) -> None:
        """Load rules from a JSON file.

        Args:
            rules_file: Path to the rules JSON file
        """
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)

            self.rules = []
            for rule_dict in rules_data.get('rules', []):
                rule = Rule(**rule_dict)
                self.rules.append(rule)

            # Sort by priority (highest first)
            self.rules.sort(key=lambda r: r.priority, reverse=True)

            # Pre-compile regex patterns for performance
            self._compile_patterns()

            logger.info(f"Loaded {len(self.rules)} rules from {rules_file}")

        except Exception as e:
            logger.error(f"Failed to load rules from {rules_file}: {e}")
            raise

    def _load_builtin_rules(self) -> None:
        """Load built-in default rules."""
        builtin_rules = [
            Rule(
                name="Soundtracks",
                description="Organize soundtrack and score albums",
                conditions=[
                    RuleCondition(
                        field="genre",
                        operator=ComparisonOperator.MATCHES,
                        value=r"soundtrack|score|film|motion picture"
                    )
                ],
                pattern="Soundtracks/{album} ({year})",
                priority=100
            ),
            Rule(
                name="Classical by Composer",
                description="Organize classical music by composer",
                conditions=[
                    RuleCondition(
                        field="genre",
                        operator=ComparisonOperator.MATCHES,
                        value=r"classical|orchestra|symphony"
                    ),
                    RuleCondition(
                        field="albumartist",
                        operator=ComparisonOperator.NOT_EQUALS,
                        value="Various Artists"
                    )
                ],
                pattern="Classical/{albumartist}/{album}/{track_number:02} - {title} - {artist}",
                priority=90
            ),
            Rule(
                name="Multi-disc Albums",
                description="Handle multi-disc albums with disc organization",
                conditions=[
                    RuleCondition(
                        field="disc_number",
                        operator=ComparisonOperator.GREATER_THAN,
                        value="1"
                    )
                ],
                pattern="{artist}/{album} (Disc {disc_number})/{track_number:02} {title}",
                priority=110
            ),
            Rule(
                name="Compilations",
                description="Organize various artist compilations",
                conditions=[
                    RuleCondition(
                        field="albumartist",
                        operator=ComparisonOperator.EQUALS,
                        value="Various Artists"
                    )
                ],
                pattern="Compilations/{album} ({year})",
                priority=80
            ),
            Rule(
                name="Live Albums",
                description="Organize live recordings",
                conditions=[
                    RuleCondition(
                        field="album",
                        operator=ComparisonOperator.MATCHES,
                        value=r".*\b(live|live at|in concert)\b.*"
                    )
                ],
                pattern="Live/{artist}/{album} ({year})",
                priority=85
            ),
            Rule(
                name="EPs and Singles",
                description="Organize EPs and singles separately",
                conditions=[
                    RuleCondition(
                        field="album",
                        operator=ComparisonOperator.MATCHES,
                        value=r".*\b(EP|Single)\b.*"
                    )
                ],
                pattern="{artist}/EPs and Singles/{album} ({year})",
                priority=70
            ),
            Rule(
                name="Remix Albums",
                description="Organize remix and DJ mix albums",
                conditions=[
                    RuleCondition(
                        field="album",
                        operator=ComparisonOperator.MATCHES,
                        value=r".*\b(remix|remixed|dj mix|mix)\b.*"
                    )
                ],
                pattern="Remixes/{artist}/{album} ({year})",
                priority=75
            ),
            Rule(
                name="Artist First Letter",
                description="Organize by artist's first letter",
                conditions=[],
                pattern="{first_letter}/{artist}/{album} ({year})",
                priority=10
            )
        ]

        self.rules = builtin_rules
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for better performance."""
        self._compiled_patterns = {}
        for rule in self.rules:
            for condition in rule.conditions:
                if isinstance(condition, RuleCondition) and condition.operator == ComparisonOperator.MATCHES:
                    pattern_key = f"{rule.name}:{condition.field}"
                    try:
                        pattern = re.compile(
                            str(condition.value),
                            re.IGNORECASE if not condition.case_sensitive else 0
                        )
                        self._compiled_patterns[pattern_key] = pattern
                    except re.error as e:
                        logger.warning(f"Failed to compile regex for {pattern_key}: {e}")

    def find_matching_rule(self, audio_file: AudioFile) -> Optional[Rule]:
        """Find the first rule that matches the given audio file.

        Args:
            audio_file: The audio file to check

        Returns:
            The matching rule or None if no rule matches
        """
        for rule in self.rules:
            if rule.matches(audio_file):
                return rule
        return None

    def get_all_matching_rules(self, audio_file: AudioFile) -> List[Rule]:
        """Get all rules that match the given audio file.

        Args:
            audio_file: The audio file to check

        Returns:
            List of matching rules ordered by priority
        """
        matching_rules = []
        for rule in self.rules:
            if rule.matches(audio_file):
                matching_rules.append(rule)
        return matching_rules

    def test_rule(self, rule_name: str, audio_file: AudioFile) -> bool:
        """Test if a specific rule matches an audio file.

        Args:
            rule_name: Name of the rule to test
            audio_file: The audio file to test against

        Returns:
            True if the rule matches
        """
        rule = self.get_rule_by_name(rule_name)
        if rule:
            return rule.matches(audio_file)
        return False

    def get_rule_by_name(self, name: str) -> Optional[Rule]:
        """Get a rule by its name.

        Args:
            name: The rule name to find

        Returns:
            The rule or None if not found
        """
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None

    def add_rule(self, rule: Rule) -> None:
        """Add a new rule to the engine.

        Args:
            rule: The rule to add
        """
        self.rules.append(rule)
        # Re-sort by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        # Re-compile patterns
        self._compile_patterns()

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name.

        Args:
            rule_name: Name of the rule to remove

        Returns:
            True if the rule was removed
        """
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                self._compile_patterns()
                return True
        return False

    def enable_rule(self, rule_name: str) -> bool:
        """Enable a rule.

        Args:
            rule_name: Name of the rule to enable

        Returns:
            True if the rule was found and enabled
        """
        rule = self.get_rule_by_name(rule_name)
        if rule:
            rule.enabled = True
            return True
        return False

    def disable_rule(self, rule_name: str) -> bool:
        """Disable a rule.

        Args:
            rule_name: Name of the rule to disable

        Returns:
            True if the rule was found and disabled
        """
        rule = self.get_rule_by_name(rule_name)
        if rule:
            rule.enabled = False
            return True
        return False

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded rules.

        Returns:
            Dictionary with rule statistics
        """
        total = len(self.rules)
        enabled = sum(1 for r in self.rules if r.enabled)
        disabled = total - enabled

        # Count rules by tags
        tag_counts = {}
        for rule in self.rules:
            for tag in rule.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            'total_rules': total,
            'enabled_rules': enabled,
            'disabled_rules': disabled,
            'tag_distribution': tag_counts,
            'compiled_patterns': len(self._compiled_patterns)
        }

    def validate_rules(self) -> List[str]:
        """Validate all loaded rules.

        Returns:
            List of validation errors
        """
        errors = []

        for rule in self.rules:
            # Check for duplicate names
            if sum(1 for r in self.rules if r.name == rule.name) > 1:
                errors.append(f"Duplicate rule name: {rule.name}")

            # Check pattern is not empty
            if not rule.pattern:
                errors.append(f"Rule '{rule.name}' has empty pattern")

            # Validate conditions
            for condition in rule.conditions:
                if isinstance(condition, RuleCondition):
                    # Check if field exists
                    if not condition.field:
                        errors.append(f"Rule '{rule.name}' has condition with empty field")

                    # Check regex patterns
                    if condition.operator == ComparisonOperator.MATCHES:
                        try:
                            re.compile(str(condition.value))
                        except re.error as e:
                            errors.append(f"Rule '{rule.name}' has invalid regex: {e}")

        return errors

    def export_rules(self, output_file: Path) -> None:
        """Export current rules to a JSON file.

        Args:
            output_file: Path to write the rules to
        """
        rules_data = {
            'rules': []
        }

        for rule in self.rules:
            rule_dict = {
                'name': rule.name,
                'description': rule.description,
                'conditions': [],
                'condition_operator': rule.condition_operator.value,
                'pattern': rule.pattern,
                'priority': rule.priority,
                'enabled': rule.enabled,
                'tags': rule.tags
            }

            # Convert conditions to dicts
            for condition in rule.conditions:
                if isinstance(condition, RuleCondition):
                    condition_dict = {
                        'field': condition.field,
                        'operator': condition.operator.value,
                        'value': condition.value,
                        'case_sensitive': condition.case_sensitive
                    }
                    rule_dict['conditions'].append(condition_dict)

            rules_data['rules'].append(rule_dict)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(self.rules)} rules to {output_file}")