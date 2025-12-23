"""Tests for rule schema validation."""

import json
import pytest
from pathlib import Path

from music_organizer.core.rule_schema import (
    validate_rule_json, validate_rule_file, create_example_rules_file,
    get_field_examples, get_operator_examples, create_rule_template,
    RULE_SCHEMA, EXTENDED_RULE_SCHEMA
)


class TestRuleSchemaValidation:
    """Test rule schema validation."""

    def test_valid_basic_rule(self):
        """Test validation of a valid basic rule."""
        rule_data = {
            "rules": [
                {
                    "name": "Test Rule",
                    "description": "A test rule",
                    "conditions": [
                        {
                            "field": "genre",
                            "operator": "eq",
                            "value": "Rock"
                        }
                    ],
                    "pattern": "Rock/{artist}/{album}",
                    "priority": 100,
                    "enabled": True,
                    "tags": ["rock"]
                }
            ]
        }

        errors = validate_rule_json(rule_data)
        assert len(errors) == 0

    def test_valid_extended_rule(self):
        """Test validation of a valid extended rule."""
        rule_data = create_example_rules_file()

        errors = validate_rule_json(rule_data, extended=True)
        assert len(errors) == 0

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        # Missing name
        rule_data = {
            "rules": [
                {
                    "description": "A rule without name",
                    "pattern": "{artist}/{album}"
                }
            ]
        }

        errors = validate_rule_json(rule_data)
        assert len(errors) > 0
        assert any("name" in error for error in errors)

        # Missing pattern
        rule_data = {
            "rules": [
                {
                    "name": "Test Rule",
                    "description": "A rule without pattern"
                }
            ]
        }

        errors = validate_rule_json(rule_data)
        assert len(errors) > 0
        assert any("pattern" in error for error in errors)

    def test_invalid_operator(self):
        """Test validation with invalid operator."""
        rule_data = {
            "rules": [
                {
                    "name": "Test Rule",
                    "conditions": [
                        {
                            "field": "genre",
                            "operator": "invalid_operator",
                            "value": "Rock"
                        }
                    ],
                    "pattern": "{artist}/{album}"
                }
            ]
        }

        errors = validate_rule_json(rule_data)
        assert len(errors) > 0

    def test_is_empty_without_value(self):
        """Test that is_empty operator doesn't require a value."""
        rule_data = {
            "rules": [
                {
                    "name": "Test Rule",
                    "conditions": [
                        {
                            "field": "composer",
                            "operator": "is_empty"
                        }
                    ],
                    "pattern": "{artist}/{album}"
                }
            ]
        }

        errors = validate_rule_json(rule_data)
        assert len(errors) == 0

    def test_is_not_empty_without_value(self):
        """Test that is_not_empty operator doesn't require a value."""
        rule_data = {
            "rules": [
                {
                    "name": "Test Rule",
                    "conditions": [
                        {
                            "field": "artist",
                            "operator": "is_not_empty"
                        }
                    ],
                    "pattern": "{artist}/{album}"
                }
            ]
        }

        errors = validate_rule_json(rule_data)
        assert len(errors) == 0

    def test_other_operators_require_value(self):
        """Test that other operators require a value."""
        rule_data = {
            "rules": [
                {
                    "name": "Test Rule",
                    "conditions": [
                        {
                            "field": "genre",
                            "operator": "eq"
                            # Missing value
                        }
                    ],
                    "pattern": "{artist}/{album}"
                }
            ]
        }

        errors = validate_rule_json(rule_data)
        assert len(errors) > 0

    def test_invalid_priority_range(self):
        """Test validation of priority range."""
        rule_data = {
            "rules": [
                {
                    "name": "Test Rule",
                    "conditions": [
                        {
                            "field": "genre",
                            "operator": "eq",
                            "value": "Rock"
                        }
                    ],
                    "pattern": "{artist}/{album}",
                    "priority": -1  # Invalid (must be >= 0)
                }
            ]
        }

        errors = validate_rule_json(rule_data)
        assert len(errors) > 0

    def test_duplicate_rule_names(self):
        """Test validation of duplicate rule names."""
        rule_data = {
            "rules": [
                {
                    "name": "Same Name",
                    "pattern": "{artist}/{album}"
                },
                {
                    "name": "Same Name",
                    "pattern": "{genre}/{artist}"
                }
            ]
        }

        # Note: Schema validation won't catch this, but the engine should
        errors = validate_rule_json(rule_data)
        # Schema itself doesn't prevent duplicates, but engine validation should
        # This is more of an integration test
        assert len(errors) == 0  # Schema validates fine

    def test_empty_rules_array(self):
        """Test validation with empty rules array."""
        rule_data = {
            "rules": []
        }

        errors = validate_rule_json(rule_data)
        assert len(errors) == 0

    def test_no_rules_field(self):
        """Test validation without rules field."""
        rule_data = {
            "metadata": {
                "version": "1.0.0"
            }
        }

        errors = validate_rule_json(rule_data)
        assert len(errors) > 0

    def test_multiple_conditions(self):
        """Test validation with multiple conditions."""
        rule_data = {
            "rules": [
                {
                    "name": "Multi-condition Rule",
                    "conditions": [
                        {
                            "field": "genre",
                            "operator": "eq",
                            "value": "Rock"
                        },
                        {
                            "field": "year",
                            "operator": "gt",
                            "value": "2000"
                        }
                    ],
                    "condition_operator": "and",
                    "pattern": "Rock/{artist}/{album}"
                }
            ]
        }

        errors = validate_rule_json(rule_data)
        assert len(errors) == 0

    def test_invalid_condition_operator(self):
        """Test validation with invalid condition operator."""
        rule_data = {
            "rules": [
                {
                    "name": "Test Rule",
                    "conditions": [
                        {
                            "field": "genre",
                            "operator": "eq",
                            "value": "Rock"
                        }
                    ],
                    "condition_operator": "invalid",  # Should be 'and', 'or', or 'not'
                    "pattern": "{artist}/{album}"
                }
            ]
        }

        errors = validate_rule_json(rule_data)
        assert len(errors) > 0


class TestRuleFileValidation:
    """Test validation of rule files."""

    def test_valid_json_file(self):
        """Test validation of a valid JSON file."""
        rule_data = {
            "rules": [
                {
                    "name": "Test Rule",
                    "conditions": [
                        {
                            "field": "genre",
                            "operator": "eq",
                            "value": "Rock"
                        }
                    ],
                    "pattern": "Rock/{artist}/{album}"
                }
            ]
        }

        with open('/tmp/test_rules.json', 'w') as f:
            json.dump(rule_data, f)

        errors = validate_rule_file(Path('/tmp/test_rules.json'))
        assert len(errors) == 0

        Path('/tmp/test_rules.json').unlink()

    def test_invalid_json_file(self):
        """Test validation of an invalid JSON file."""
        with open('/tmp/invalid_rules.json', 'w') as f:
            f.write('{"rules": [{ "name": "Test"')  # Invalid JSON

        errors = validate_rule_file(Path('/tmp/invalid_rules.json'))
        assert len(errors) > 0
        assert any("JSON parsing error" in error for error in errors)

        Path('/tmp/invalid_rules.json').unlink()

    def test_nonexistent_file(self):
        """Test validation of a nonexistent file."""
        errors = validate_rule_file(Path('/tmp/nonexistent.json'))
        assert len(errors) > 0
        assert any("Error reading file" in error for error in errors)


class TestRuleHelpers:
    """Test helper functions for rules."""

    def test_get_field_examples(self):
        """Test getting field examples."""
        fields = get_field_examples()
        assert isinstance(fields, dict)
        assert len(fields) > 0
        assert "Direct AudioFile fields" in fields
        assert "title" in fields["Direct AudioFile fields"]
        assert "artist" in fields["Direct AudioFile fields"]

    def test_get_operator_examples(self):
        """Test getting operator examples."""
        operators = get_operator_examples()
        assert isinstance(operators, dict)
        assert len(operators) > 0
        assert "eq" in operators
        assert "description" in operators["eq"]
        assert "example" in operators["eq"]

    def test_create_rule_template(self):
        """Test creating a rule template."""
        template = create_rule_template(
            name="Test Template",
            description="A test template",
            pattern="{artist}/{album}",
            conditions=[
                {
                    "field": "genre",
                    "operator": "eq",
                    "value": "Rock"
                }
            ],
            priority=50,
            tags=["test"]
        )

        assert template["name"] == "Test Template"
        assert template["description"] == "A test template"
        assert template["pattern"] == "{artist}/{album}"
        assert len(template["conditions"]) == 1
        assert template["priority"] == 50
        assert "test" in template["tags"]
        assert template["condition_operator"] == "and"
        assert template["enabled"] is True

    def test_create_example_rules_file(self):
        """Test creating example rules file."""
        example = create_example_rules_file()

        assert "metadata" in example
        assert "rules" in example
        assert "templates" in example
        assert len(example["rules"]) > 0
        assert len(example["templates"]) > 0

        # Check structure
        assert "version" in example["metadata"]
        assert "description" in example["metadata"]
        assert all("name" in rule for rule in example["rules"])
        assert all("pattern" in rule for rule in example["rules"])
        assert all("name" in tmpl for tmpl in example["templates"])
        assert all("pattern" in tmpl for tmpl in example["templates"])


class TestSchemaConstants:
    """Test schema constants."""

    def test_rule_schema_structure(self):
        """Test the basic rule schema structure."""
        assert "type" in RULE_SCHEMA
        assert "properties" in RULE_SCHEMA
        assert "rules" in RULE_SCHEMA["properties"]
        assert RULE_SCHEMA["properties"]["rules"]["type"] == "array"

    def test_extended_schema_structure(self):
        """Test the extended rule schema structure."""
        assert "type" in EXTENDED_RULE_SCHEMA
        assert "properties" in EXTENDED_RULE_SCHEMA
        assert "metadata" in EXTENDED_RULE_SCHEMA["properties"]
        assert "templates" in EXTENDED_RULE_SCHEMA["properties"]
        assert "allOf" in EXTENDED_RULE_SCHEMA

    def test_operator_enum_values(self):
        """Test that all expected operators are in the enum."""
        # EXTENDED_RULE_SCHEMA uses allOf to include RULE_SCHEMA, access via RULE_SCHEMA directly
        operators_path = RULE_SCHEMA["properties"]["rules"]["items"]["properties"]["conditions"]["items"]["properties"]["operator"]["enum"]
        expected_operators = [
            "eq", "ne", "contains", "not_contains",
            "matches", "not_matches", "starts_with", "ends_with",
            "gt", "lt", "ge", "le", "in", "not_in",
            "is_empty", "is_not_empty"
        ]
        for op in expected_operators:
            assert op in operators_path