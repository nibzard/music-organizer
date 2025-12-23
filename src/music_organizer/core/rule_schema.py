"""JSON schema and validation for rule definitions."""

import json
import jsonschema
from pathlib import Path
from typing import Dict, Any, List, Optional

# JSON Schema for rule definitions
RULE_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "pattern"],
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Unique name for the rule"
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of what the rule does"
                    },
                    "conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["field", "operator"],
                            "properties": {
                                "field": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": "The AudioFile field to check"
                                },
                                "operator": {
                                    "type": "string",
                                    "enum": [
                                        "eq", "ne", "contains", "not_contains",
                                        "matches", "not_matches", "starts_with", "ends_with",
                                        "gt", "lt", "ge", "le", "in", "not_in",
                                        "is_empty", "is_not_empty"
                                    ],
                                    "description": "Comparison operator"
                                },
                                "value": {
                                    "description": "Value to compare against (not required for is_empty/is_not_empty)"
                                },
                                "case_sensitive": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Whether the comparison is case sensitive"
                                }
                            },
                            "allOf": [
                                {
                                    "if": {
                                        "properties": {"operator": {"const": "is_empty"}},
                                        "required": ["operator"]
                                    },
                                    "then": {
                                        "not": {"required": ["value"]}
                                    }
                                },
                                {
                                    "if": {
                                        "properties": {"operator": {"const": "is_not_empty"}},
                                        "required": ["operator"]
                                    },
                                    "then": {
                                        "not": {"required": ["value"]}
                                    }
                                },
                                {
                                    "if": {
                                        "properties": {"operator": {"not": {"enum": ["is_empty", "is_not_empty"]}}},
                                        "required": ["operator"]
                                    },
                                    "then": {
                                        "required": ["value"]
                                    }
                                }
                            ]
                        }
                    },
                    "condition_operator": {
                        "type": "string",
                        "enum": ["and", "or", "not"],
                        "default": "and",
                        "description": "Logical operator for combining conditions"
                    },
                    "pattern": {
                        "type": "string",
                        "minLength": 1,
                        "description": "The path/filename pattern to apply when rule matches"
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1000,
                        "default": 0,
                        "description": "Priority of the rule (higher values checked first)"
                    },
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether the rule is enabled"
                    },
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Tags for categorizing rules"
                    }
                }
            }
        }
    },
    "required": ["rules"]
}

# Extended schema with additional metadata
EXTENDED_RULE_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "version": {
                    "type": "string",
                    "pattern": r"^\d+\.\d+\.\d+$",
                    "description": "Schema version"
                },
                "description": {
                    "type": "string",
                    "description": "Description of this rule set"
                },
                "author": {
                    "type": "string",
                    "description": "Author of the rule set"
                },
                "created_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "When the rule set was created"
                },
                "updated_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "When the rule set was last updated"
                }
            }
        },
        "templates": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "description", "pattern"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Template name"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of when to use this template"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Path pattern template"
                    },
                    "example": {
                        "type": "string",
                        "description": "Example of the pattern in action"
                    }
                }
            }
        }
    },
    "allOf": [
        {
            "required": ["rules"]
        },
        {
            "properties": {
                "rules": RULE_SCHEMA["properties"]["rules"]
            }
        }
    ]
}


def validate_rule_json(rule_data: Dict[str, Any], extended: bool = False) -> List[str]:
    """Validate a rule definition JSON object.

    Args:
        rule_data: The rule data to validate
        extended: Whether to use the extended schema with metadata

    Returns:
        List of validation error messages
    """
    schema = EXTENDED_RULE_SCHEMA if extended else RULE_SCHEMA

    try:
        # Use Draft7 for compatibility
        jsonschema.validate(rule_data, schema, format_checker=jsonschema.draft7_format_checker)
        return []
    except jsonschema.ValidationError as e:
        # Get the full error path
        path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
        return [f"Validation error at {path}: {e.message}"]
    except jsonschema.SchemaError as e:
        return [f"Schema error: {e.message}"]
    except Exception as e:
        return [f"Unexpected validation error: {str(e)}"]


def validate_rule_file(file_path: Path, extended: bool = False) -> List[str]:
    """Validate a rule definition JSON file.

    Args:
        file_path: Path to the rule file
        extended: Whether to use the extended schema with metadata

    Returns:
        List of validation error messages
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            rule_data = json.load(f)
        return validate_rule_json(rule_data, extended)
    except json.JSONDecodeError as e:
        return [f"JSON parsing error: {e.msg} at line {e.lineno}, column {e.colno}"]
    except Exception as e:
        return [f"Error reading file: {str(e)}"]


def get_field_examples() -> Dict[str, List[str]]:
    """Get examples of available fields for rule conditions.

    Returns:
        Dictionary mapping field names to example values
    """
    return {
        "Direct AudioFile fields": [
            "title", "artist", "artists", "primary_artist", "album",
            "year", "track_number", "genre", "date", "location", "content_type",
            "file_type", "has_cover_art"
        ],
        "Metadata fields": [
            "albumartist", "composer", "lyricist", "producer",
            "bpm", "key", "comment", "copyright", "duration", "bitrate"
        ],
        "Computed fields": [
            "first_letter", "decade", "file_extension", "duration_minutes",
            "disc_number", "total_discs", "total_tracks", "format"
        ]
    }


def get_operator_examples() -> Dict[str, Dict[str, Any]]:
    """Get examples of available operators for rule conditions.

    Returns:
        Dictionary mapping operator names to descriptions and examples
    """
    return {
        "eq": {
            "description": "Equals (exact match)",
            "example": {"field": "genre", "operator": "eq", "value": "Rock"}
        },
        "ne": {
            "description": "Not equals",
            "example": {"field": "artist", "operator": "ne", "value": "Various Artists"}
        },
        "contains": {
            "description": "Contains substring",
            "example": {"field": "title", "operator": "contains", "value": "Remix"}
        },
        "not_contains": {
            "description": "Does not contain substring",
            "example": {"field": "album", "operator": "not_contains", "value": "Greatest Hits"}
        },
        "matches": {
            "description": "Matches regex pattern",
            "example": {"field": "genre", "operator": "matches", "value": "rock|pop|indie"}
        },
        "not_matches": {
            "description": "Does not match regex pattern",
            "example": {"field": "title", "operator": "not_matches", "value": r"^\d+\."}
        },
        "starts_with": {
            "description": "Starts with",
            "example": {"field": "album", "operator": "starts_with", "value": "Live"}
        },
        "ends_with": {
            "description": "Ends with",
            "example": {"field": "title", "operator": "ends_with", "value": "(Remix)"}
        },
        "gt": {
            "description": "Greater than (numeric)",
            "example": {"field": "year", "operator": "gt", "value": "2000"}
        },
        "lt": {
            "description": "Less than (numeric)",
            "example": {"field": "track_number", "operator": "lt", "value": "5"}
        },
        "ge": {
            "description": "Greater than or equal",
            "example": {"field": "duration", "operator": "ge", "value": "180"}
        },
        "le": {
            "description": "Less than or equal",
            "example": {"field": "bpm", "operator": "le", "value": "120"}
        },
        "in": {
            "description": "Value is in list",
            "example": {"field": "genre", "operator": "in", "value": ["Rock", "Pop", "Alternative"]}
        },
        "not_in": {
            "description": "Value is not in list",
            "example": {"field": "format", "operator": "not_in", "value": ["MP3", "WMA"]}
        },
        "is_empty": {
            "description": "Field is empty or null",
            "example": {"field": "composer", "operator": "is_empty"}
        },
        "is_not_empty": {
            "description": "Field is not empty",
            "example": {"field": "albumartist", "operator": "is_not_empty"}
        }
    }


def create_rule_template(name: str, description: str, pattern: str,
                        conditions: List[Dict[str, Any]],
                        priority: int = 0,
                        tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a rule template with proper structure.

    Args:
        name: Rule name
        description: Rule description
        pattern: Path pattern
        conditions: List of condition dictionaries
        priority: Rule priority (default: 0)
        tags: Optional tags list

    Returns:
        Rule template dictionary
    """
    rule_template = {
        "name": name,
        "description": description,
        "conditions": conditions,
        "condition_operator": "and",
        "pattern": pattern,
        "priority": priority,
        "enabled": True
    }

    if tags:
        rule_template["tags"] = tags

    return rule_template


def create_example_rules_file() -> Dict[str, Any]:
    """Create an example rules file with common patterns.

    Returns:
        Example rules data structure
    """
    return {
        "metadata": {
            "version": "1.0.0",
            "description": "Example music organization rules",
            "author": "Music Organizer",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        },
        "templates": [
            {
                "name": "Basic Artist/Album",
                "description": "Standard organization by artist and album",
                "pattern": "{artist}/{album} ({year})",
                "example": "The Beatles/Abbey Road (1969)"
            },
            {
                "name": "Genre-based",
                "description": "Organize by genre first, then artist",
                "pattern": "{genre}/{artist}/{album}",
                "example": "Rock/Queen/A Night at the Opera"
            },
            {
                "name": "With Track Number",
                "description": "Include track numbers in filenames",
                "pattern": "{artist}/{album}/{track_number:02} {title}",
                "example": "Pink Floyd/The Dark Side of the Moon/01 Speak to Me"
            }
        ],
        "rules": [
            {
                "name": "Soundtracks",
                "description": "Organize soundtrack and score albums separately",
                "conditions": [
                    {
                        "field": "genre",
                        "operator": "matches",
                        "value": "soundtrack|score|film|motion picture",
                        "case_sensitive": False
                    }
                ],
                "pattern": "Soundtracks/{album} ({year})",
                "priority": 100,
                "tags": ["genre", "special"]
            },
            {
                "name": "Classical Music",
                "description": "Organize classical music by composer",
                "conditions": [
                    {
                        "field": "genre",
                        "operator": "matches",
                        "value": "classical|orchestra|symphony",
                        "case_sensitive": False
                    }
                ],
                "pattern": "Classical/{composer}/{album}/{track_number:02} - {title}",
                "priority": 90,
                "tags": ["genre"]
            },
            {
                "name": "Multi-disc Albums",
                "description": "Handle albums with multiple discs",
                "conditions": [
                    {
                        "field": "disc_number",
                        "operator": "gt",
                        "value": 1
                    }
                ],
                "pattern": "{artist}/{album} (Disc {disc_number})/{track_number:02} {title}",
                "priority": 110,
                "tags": ["special"]
            },
            {
                "name": "Compilations",
                "description": "Various Artists compilations",
                "conditions": [
                    {
                        "field": "albumartist",
                        "operator": "eq",
                        "value": "Various Artists"
                    }
                ],
                "pattern": "Compilations/{album} ({year})",
                "priority": 80,
                "tags": ["special"]
            }
        ]
    }