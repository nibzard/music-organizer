"""Configuration model for music organizer."""

from pathlib import Path
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, field


@dataclass
class DirectoryConfig:
    """Configuration for directory names."""
    albums: str = "Albums"
    live: str = "Live"
    collaborations: str = "Collaborations"
    compilations: str = "Compilations"
    rarities: str = "Rarities"


@dataclass
class NamingConfig:
    """Configuration for file and directory naming patterns."""
    album_format: str = "{artist}/{album} ({year})"
    live_format: str = "{artist}/{date} - {location}"
    collab_format: str = "{album} ({year}) - {artists}"
    compilation_format: str = "{artist}/{album} ({year})"
    rarity_format: str = "{artist}/{album} ({edition})"


@dataclass
class MetadataConfig:
    """Configuration for metadata handling."""
    enhance: bool = True
    musicbrainz: bool = True
    fix_capitalization: bool = True
    standardize_genres: bool = True


@dataclass
class FileOperationsConfig:
    """Configuration for file operations."""
    strategy: str = "move"  # "copy" or "move"
    backup: bool = True
    handle_duplicates: str = "number"  # "number", "skip", or "overwrite"


@dataclass
class Config:
    """Main configuration model."""
    source_directory: Path
    target_directory: Path
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    naming: NamingConfig = field(default_factory=NamingConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    file_operations: FileOperationsConfig = field(default_factory=FileOperationsConfig)

    @classmethod
    def default(cls) -> "Config":
        """Create a default configuration with placeholder paths."""
        return cls(
            source_directory=Path("."),
            target_directory=Path(".")
        )


def _dataclass_to_dict(obj):
    """Convert dataclass to dict recursively."""
    from dataclasses import is_dataclass, asdict
    if is_dataclass(obj):
        result = {}
        for key, value in asdict(obj).items():
            result[key] = _dataclass_to_dict(value)
        return result
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: _dataclass_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    else:
        return obj


def _dict_to_dataclass(data, dataclass_type):
    """Convert dict to dataclass recursively."""
    from dataclasses import is_dataclass, fields
    if not is_dataclass(dataclass_type):
        return data

    # Get field types
    field_types = {f.name: f.type for f in fields(dataclass_type)}

    kwargs = {}
    for field_name, field_type in field_types.items():
        if field_name in data:
            if hasattr(field_type, '__dataclass_fields__'):
                # It's a dataclass
                kwargs[field_name] = _dict_to_dataclass(data[field_name], field_type)
            elif field_type is Path:
                kwargs[field_name] = Path(data[field_name])
            else:
                kwargs[field_name] = data[field_name]

    return dataclass_type(**kwargs)


def load_config(config_path: Path) -> Config:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    return _dict_to_dataclass(config_data, Config)


def save_config(config: Config, config_path: Path) -> None:
    """Save configuration to JSON file."""
    config_dict = _dataclass_to_dict(config)

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def create_default_config(config_path: Path) -> None:
    """Create a default configuration file."""
    default_config = Config(
        source_directory=Path("/path/to/source"),
        target_directory=Path("/path/to/target")
    )
    save_config(default_config, config_path)