"""Command line interface for regex rules management."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .core.regex_rule_engine import RegexRuleEngine, Rule, RuleCondition, ComparisonOperator
from .core.rule_schema import validate_rule_file, validate_rule_json, create_example_rules_file, get_field_examples, get_operator_examples
from .plugins.builtins.regex_rules import RegexRulesPlugin
from .models.audio_file import AudioFile
from .models.config import load_config
from .metadata import MetadataHandler
from .file_scanner import AsyncFileScanner
from .console_utils import SimpleConsole

logger = logging.getLogger(__name__)
console = SimpleConsole()


def setup_rules_parser(subparsers) -> argparse.ArgumentParser:
    """Setup the rules subcommand parser."""
    rules_parser = subparsers.add_parser(
        'rules',
        help='Manage regex-based organization rules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all rules
  %(prog)s rules list

  # Test rules against a directory
  %(prog)s rules test /music/unsorted --rules-file my-rules.json

  # Create example rules file
  %(prog)s rules init --output my-rules.json

  # Validate a rules file
  %(prog)s rules validate my-rules.json

  # Add a new rule
  %(prog)s rules add --name "My Rule" --pattern "{genre}/{artist}" --genre "rock|pop"

  # Preview organization with rules
  %(prog)s rules preview /music/unsorted /music/organized --rules-file my-rules.json
        """
    )

    rules_subparsers = rules_parser.add_subparsers(dest='rules_command', help='Rules commands')

    # List command
    list_parser = rules_subparsers.add_parser('list', help='List all rules')
    list_parser.add_argument('--rules-file', help='Path to rules file (default: use built-in rules)')
    list_parser.add_argument('--enabled-only', action='store_true', help='Show only enabled rules')
    list_parser.add_argument('--stats', action='store_true', help='Show rule statistics')

    # Test command
    test_parser = rules_subparsers.add_parser('test', help='Test rules against music files')
    test_parser.add_argument('directory', help='Directory with music files to test')
    test_parser.add_argument('--rules-file', help='Path to rules file')
    test_parser.add_argument('--limit', type=int, default=10, help='Number of files to test (default: 10)')
    test_parser.add_argument('--show-matches', action='store_true', help='Show which rules matched')
    test_parser.add_argument('--verbose', action='store_true', help='Show detailed match information')

    # Validate command
    validate_parser = rules_subparsers.add_parser('validate', help='Validate a rules file')
    validate_parser.add_argument('file', help='Rules file to validate')
    validate_parser.add_argument('--extended', action='store_true', help='Use extended schema with metadata')

    # Init command
    init_parser = rules_subparsers.add_parser('init', help='Create an example rules file')
    init_parser.add_argument('--output', default='rules.json', help='Output file path (default: rules.json)')
    init_parser.add_argument('--builtin', action='store_true', help='Include only built-in rules')

    # Add command
    add_parser = rules_subparsers.add_parser('add', help='Add a new rule (interactive)')
    add_parser.add_argument('--name', required=True, help='Rule name')
    add_parser.add_argument('--pattern', required=True, help='Path pattern')
    add_parser.add_argument('--description', help='Rule description')
    add_parser.add_argument('--priority', type=int, default=0, help='Rule priority (default: 0)')
    add_parser.add_argument('--rules-file', help='Rules file to add to')
    add_parser.add_argument('--genre', help='Genre condition (regex pattern)')
    add_parser.add_argument('--artist', help='Artist condition (regex pattern)')
    add_parser.add_argument('--year', help='Year condition (e.g., ">2000", "1990-1999")')
    add_parser.add_argument('--album', help='Album condition (regex pattern)')

    # Remove command
    remove_parser = rules_subparsers.add_parser('remove', help='Remove a rule')
    remove_parser.add_argument('name', help='Name of rule to remove')
    remove_parser.add_argument('--rules-file', help='Rules file to modify')

    # Enable/Disable commands
    enable_parser = rules_subparsers.add_parser('enable', help='Enable a rule')
    enable_parser.add_argument('name', help='Name of rule to enable')
    enable_parser.add_argument('--rules-file', help='Rules file to modify')

    disable_parser = rules_subparsers.add_parser('disable', help='Disable a rule')
    disable_parser.add_argument('name', help='Name of rule to disable')
    disable_parser.add_argument('--rules-file', help='Rules file to modify')

    # Preview command
    preview_parser = rules_subparsers.add_parser('preview', help='Preview organization with rules')
    preview_parser.add_argument('source', help='Source directory')
    preview_parser.add_argument('target', help='Target directory')
    preview_parser.add_argument('--rules-file', help='Path to rules file')
    preview_parser.add_argument('--limit', type=int, default=20, help='Number of files to preview (default: 20)')
    preview_parser.add_argument('--show-unmatched', action='store_true', help='Show files that don\'t match any rules')

    # Fields command
    fields_parser = rules_subparsers.add_parser('fields', help='Show available fields for conditions')
    fields_parser.add_argument('--examples', action='store_true', help='Show example values for each field')

    # Operators command
    operators_parser = rules_subparsers.add_parser('operators', help='Show available comparison operators')

    return rules_parser


async def cmd_rules_list(args) -> int:
    """Handle the rules list command."""
    try:
        # Load rules
        rules_file = Path(args.rules_file) if args.rules_file else None
        engine = RegexRuleEngine(rules_file)

        # Get rules
        rules = engine.rules
        if args.enabled_only:
            rules = [r for r in rules if r.enabled]

        if not rules:
            console.info("No rules found")
            return 0

        # Display rules
        console.header(f"Rules ({len(rules)} total)")
        for rule in rules:
            status = "✓" if rule.enabled else "✗"
            console.print(f"{status} {rule.name} (priority: {rule.priority})")
            if rule.description:
                console.print(f"    {rule.description}")
            console.print(f"    Pattern: {rule.pattern}")
            if rule.conditions:
                cond_str = " AND ".join([f"{c.field} {c.operator.value} {c.value}" for c in rule.conditions[:2]])
                if len(rule.conditions) > 2:
                    cond_str += "..."
                console.print(f"    Conditions: {cond_str}")
            console.print()

        if args.stats:
            stats = engine.get_rule_statistics()
            console.header("Statistics")
            console.print(f"Total rules: {stats['total_rules']}")
            console.print(f"Enabled: {stats['enabled_rules']}")
            console.print(f"Disabled: {stats['disabled_rules']}")
            if stats['tag_distribution']:
                console.print("\nTags:")
                for tag, count in stats['tag_distribution'].items():
                    console.print(f"  {tag}: {count}")
            console.print(f"\nCompiled patterns: {stats['compiled_patterns']}")

        return 0

    except Exception as e:
        console.error(f"Error listing rules: {e}")
        return 1


async def cmd_rules_test(args) -> int:
    """Handle the rules test command."""
    try:
        # Load rules
        rules_file = Path(args.rules_file) if args.rules_file else None
        engine = RegexRuleEngine(rules_file)

        # Scan for music files
        scanner = AsyncFileScanner()
        files = await scanner.scan_directory(Path(args.directory), recursive=True)
        audio_files = []

        # Load metadata for a sample of files
        metadata_handler = MetadataHandler()
        for i, file_path in enumerate(files[:args.limit]):
            try:
                audio_file = await metadata_handler.extract_metadata(file_path)
                audio_files.append(audio_file)
            except Exception as e:
                if args.verbose:
                    console.warning(f"Failed to load metadata for {file_path.name}: {e}")

        if not audio_files:
            console.warning("No audio files found to test")
            return 0

        console.header(f"Testing {len(audio_files)} files against {len(engine.rules)} rules")
        console.print()

        matches = 0
        for audio_file in audio_files:
            matching_rule = engine.find_matching_rule(audio_file)
            if matching_rule:
                matches += 1
                console.print(f"✓ {audio_file.path.name}")
                console.print(f"  Matched rule: {matching_rule.name}")
                console.print(f"  Pattern: {matching_rule.pattern}")

                if args.verbose:
                    console.print(f"  Metadata: {audio_file.artist} - {audio_file.album}")
                    if matching_rule.conditions:
                        for condition in matching_rule.conditions:
                            if isinstance(condition, RuleCondition):
                                value = condition._get_field_value(audio_file)
                                console.print(f"  Condition: {condition.field} = {value}")
                console.print()
            else:
                console.print(f"✗ {audio_file.path.name} (no rule matched)")
                if args.show_unmatched:
                    console.print(f"  Metadata: {audio_file.artist} - {audio_file.album}")
                console.print()

        # Summary
        console.header("Summary")
        console.print(f"Files tested: {len(audio_files)}")
        console.print(f"Files matched: {matches}")
        console.print(f"Match rate: {matches / len(audio_files) * 100:.1f}%")

        # Rule usage statistics
        if args.show_matches:
            rule_usage = {}
            for audio_file in audio_files:
                matching_rules = engine.get_all_matching_rules(audio_file)
                for rule in matching_rules:
                    rule_usage[rule.name] = rule_usage.get(rule.name, 0) + 1

            if rule_usage:
                console.print("\nRule usage:")
                for rule_name, count in sorted(rule_usage.items(), key=lambda x: x[1], reverse=True):
                    console.print(f"  {rule_name}: {count}")

        return 0

    except Exception as e:
        console.error(f"Error testing rules: {e}")
        return 1


async def cmd_rules_validate(args) -> int:
    """Handle the rules validate command."""
    try:
        errors = validate_rule_file(Path(args.file), extended=args.extended)

        if errors:
            console.error(f"Validation failed for {args.file}:")
            for error in errors:
                console.error(f"  • {error}")
            return 1
        else:
            console.success(f"✓ {args.file} is valid")
            return 0

    except Exception as e:
        console.error(f"Error validating rules: {e}")
        return 1


async def cmd_rules_init(args) -> int:
    """Handle the rules init command."""
    try:
        output_path = Path(args.output)

        if output_path.exists():
            if not console.confirm(f"File {output_path} already exists. Overwrite?"):
                console.print("Cancelled")
                return 0

        # Create rules data
        if args.builtin:
            # Just the basic rules structure
            engine = RegexRuleEngine()
            engine.export_rules(output_path)
        else:
            # Full example with metadata and templates
            rules_data = create_example_rules_file()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)

        console.success(f"Created {output_path}")
        console.print(f"Edit this file to customize your organization rules")
        return 0

    except Exception as e:
        console.error(f"Error creating rules file: {e}")
        return 1


async def cmd_rules_add(args) -> int:
    """Handle the rules add command."""
    try:
        # Create conditions from command line args
        conditions = []

        if args.genre:
            conditions.append(RuleCondition(
                field="genre",
                operator=ComparisonOperator.MATCHES,
                value=args.genre,
                case_sensitive=False
            ))

        if args.artist:
            conditions.append(RuleCondition(
                field="artist",
                operator=ComparisonOperator.MATCHES,
                value=args.artist,
                case_sensitive=False
            ))

        if args.year:
            # Parse year condition (e.g., ">2000", "1990-1999")
            if args.year.startswith('>'):
                conditions.append(RuleCondition(
                    field="year",
                    operator=ComparisonOperator.GREATER_THAN,
                    value=args.year[1:]
                ))
            elif args.year.startswith('<'):
                conditions.append(RuleCondition(
                    field="year",
                    operator=ComparisonOperator.LESS_THAN,
                    value=args.year[1:]
                ))
            elif '-' in args.year:
                start, end = args.year.split('-')
                conditions.extend([
                    RuleCondition(field="year", operator=ComparisonOperator.GREATER_EQUAL, value=start),
                    RuleCondition(field="year", operator=ComparisonOperator.LESS_EQUAL, value=end)
                ])
            else:
                conditions.append(RuleCondition(
                    field="year",
                    operator=ComparisonOperator.EQUALS,
                    value=args.year
                ))

        if args.album:
            conditions.append(RuleCondition(
                field="album",
                operator=ComparisonOperator.MATCHES,
                value=args.album,
                case_sensitive=False
            ))

        # Create the rule
        rule = Rule(
            name=args.name,
            description=args.description or "",
            conditions=conditions,
            pattern=args.pattern,
            priority=args.priority
        )

        # Load existing rules or create new engine
        rules_file = Path(args.rules_file) if args.rules_file else None
        if rules_file and rules_file.exists():
            engine = RegexRuleEngine(rules_file)
        else:
            engine = RegexRuleEngine()

        # Add the rule
        engine.add_rule(rule)

        # Save if file specified
        if rules_file:
            engine.export_rules(rules_file)
            console.success(f"Added rule '{args.name}' to {rules_file}")
        else:
            console.success(f"Added rule '{args.name}' (in-memory only)")

        return 0

    except Exception as e:
        console.error(f"Error adding rule: {e}")
        return 1


async def cmd_rules_remove(args) -> int:
    """Handle the rules remove command."""
    try:
        rules_file = Path(args.rules_file) if args.rules_file else None
        if not rules_file or not rules_file.exists():
            console.error("No rules file specified")
            return 1

        engine = RegexRuleEngine(rules_file)
        removed = engine.remove_rule(args.name)

        if removed:
            engine.export_rules(rules_file)
            console.success(f"Removed rule '{args.name}'")
        else:
            console.warning(f"Rule '{args.name}' not found")

        return 0

    except Exception as e:
        console.error(f"Error removing rule: {e}")
        return 1


async def cmd_rules_enable_disable(args, enable: bool) -> int:
    """Handle enable/disable rule commands."""
    try:
        rules_file = Path(args.rules_file) if args.rules_file else None
        if not rules_file or not rules_file.exists():
            console.error("No rules file specified")
            return 1

        engine = RegexRuleEngine(rules_file)
        if enable:
            success = engine.enable_rule(args.name)
            action = "enabled"
        else:
            success = engine.disable_rule(args.name)
            action = "disabled"

        if success:
            engine.export_rules(rules_file)
            console.success(f"Rule '{args.name}' {action}")
        else:
            console.warning(f"Rule '{args.name}' not found")

        return 0

    except Exception as e:
        console.error(f"Error {action} rule: {e}")
        return 1


async def cmd_rules_preview(args) -> int:
    """Handle the rules preview command."""
    try:
        # Create the plugin with rules
        config = {
            'rules_file': args.rules_file,
            'debug_mode': True
        }
        plugin = RegexRulesPlugin(config)

        # Scan for files
        scanner = AsyncFileScanner()
        files = await scanner.scan_directory(Path(args.source), recursive=True)
        metadata_handler = MetadataHandler()

        console.header(f"Previewing organization with regex rules")
        console.print(f"Source: {args.source}")
        console.print(f"Target: {args.target}")
        console.print(f"Files to preview: {min(len(files), args.limit)}")
        console.print()

        # Process sample files
        base_dir = Path(args.target)
        processed = 0

        for file_path in files[:args.limit]:
            try:
                audio_file = await metadata_handler.extract_metadata(file_path)
                target_path = await plugin.generate_target_path(audio_file, base_dir)
                filename = await plugin.generate_filename(audio_file)

                if target_path:
                    console.print(f"Source: {file_path.relative_to(Path(args.source))}")
                    console.print(f"Target: {target_path.relative_to(base_dir) / filename}")

                    # Show which rule matched
                    matching_rule = plugin.rule_engine.find_matching_rule(audio_file)
                    if matching_rule:
                        console.print(f"Rule: {matching_rule.name}")
                    console.print()
                    processed += 1
                elif args.show_unmatched:
                    console.print(f"Source: {file_path.relative_to(Path(args.source))}")
                    console.print("Target: (no rule matched)")
                    console.print()

            except Exception as e:
                console.warning(f"Error processing {file_path.name}: {e}")

        # Show statistics
        stats = plugin.get_statistics()
        console.header("Preview Statistics")
        console.print(f"Files processed: {processed}")
        console.print(f"Rules matched: {stats['rules_matched']}")
        console.print(f"Fallback used: {stats['fallback_used']}")

        return 0

    except Exception as e:
        console.error(f"Error previewing organization: {e}")
        return 1


async def cmd_rules_fields(args) -> int:
    """Handle the rules fields command."""
    try:
        fields = get_field_examples()

        console.header("Available Fields for Rule Conditions")
        for category, field_list in fields.items():
            console.print(f"\n{category}:")
            for field in field_list:
                console.print(f"  • {field}")
                if args.examples:
                    # Add some example values
                    examples = {
                        'title': '"Bohemian Rhapsody"',
                        'artist': '"Queen"',
                        'album': '"A Night at the Opera"',
                        'year': '1975',
                        'genre': '"Rock"',
                        'track_number': '11',
                        'duration': '354',
                        'first_letter': '"Q"',
                        'decade': '"1970s"',
                        'file_extension': '"mp3"'
                    }
                    if field in examples:
                        console.print(f"    Example: {examples[field]}")

        return 0

    except Exception as e:
        console.error(f"Error showing fields: {e}")
        return 1


async def cmd_rules_operators(args) -> int:
    """Handle the rules operators command."""
    try:
        operators = get_operator_examples()

        console.header("Available Comparison Operators")
        for op_name, op_info in operators.items():
            console.print(f"\n{op_name}:")
            console.print(f"  Description: {op_info['description']}")
            console.print(f"  Example: {op_info['example']}")

        return 0

    except Exception as e:
        console.error(f"Error showing operators: {e}")
        return 1


async def handle_rules_command(args) -> int:
    """Main handler for rules commands."""
    if args.rules_command == 'list':
        return await cmd_rules_list(args)
    elif args.rules_command == 'test':
        return await cmd_rules_test(args)
    elif args.rules_command == 'validate':
        return await cmd_rules_validate(args)
    elif args.rules_command == 'init':
        return await cmd_rules_init(args)
    elif args.rules_command == 'add':
        return await cmd_rules_add(args)
    elif args.rules_command == 'remove':
        return await cmd_rules_remove(args)
    elif args.rules_command == 'enable':
        return await cmd_rules_enable_disable(args, enable=True)
    elif args.rules_command == 'disable':
        return await cmd_rules_enable_disable(args, enable=False)
    elif args.rules_command == 'preview':
        return await cmd_rules_preview(args)
    elif args.rules_command == 'fields':
        return await cmd_rules_fields(args)
    elif args.rules_command == 'operators':
        return await cmd_rules_operators(args)
    else:
        console.error("Unknown rules command")
        return 1