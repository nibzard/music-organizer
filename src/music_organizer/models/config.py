"""Configuration model for music organizer."""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from pydantic import BaseModel, Field


class DirectoryConfig(BaseModel):
    """Configuration for directory names."""
    albums: str = Field(default="Albums", description="Albums directory name")
    live: str = Field(default="Live", description="Live recordings directory")
    collaborations: str = Field(default="Collaborations", description="Collaborations directory")
    compilations: str = Field(default="Compilations", description="Compilations directory")
    rarities: str = Field(default="Rarities", description="Rarities directory")


class NamingConfig(BaseModel):
    """Configuration for file and directory naming patterns."""
    album_format: str = Field(default="{artist}/{album} ({year})")
    live_format: str = Field(default="{artist}/{date} - {location}")
    collab_format: str = Field(default="{album} ({year}) - {artists}")
    compilation_format: str = Field(default="{artist}/{album} ({year})")
    rarity_format: str = Field(default="{artist}/{album} ({edition})")


class MetadataConfig(BaseModel):
    """Configuration for metadata handling."""
    enhance: bool = Field(default=True, description="Enable metadata enhancement")
    musicbrainz: bool = Field(default=True, description="Use MusicBrainz for metadata lookup")
    fix_capitalization: bool = Field(default=True, description="Fix capitalization in tags")
    standardize_genres: bool = Field(default=True, description="Standardize genre names")


class FileOperationsConfig(BaseModel):
    """Configuration for file operations."""
    strategy: str = Field(default="move", pattern="^(copy|move)$")
    backup: bool = Field(default=True, description="Create backup before changes")
    handle_duplicates: str = Field(default="number", pattern="^(number|skip|overwrite)$")


class Config(BaseModel):
    """Main configuration model."""
    source_directory: Path
    target_directory: Path
    directories: DirectoryConfig = Field(default_factory=DirectoryConfig)
    naming: NamingConfig = Field(default_factory=NamingConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    file_operations: FileOperationsConfig = Field(default_factory=FileOperationsConfig)


def load_config(config_path: Path) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Convert path strings to Path objects
    if 'source_directory' in config_data:
        config_data['source_directory'] = Path(config_data['source_directory'])
    if 'target_directory' in config_data:
        config_data['target_directory'] = Path(config_data['target_directory'])

    return Config(**config_data)


def save_config(config: Config, config_path: Path) -> None:
    """Save configuration to YAML file."""
    # Convert to dict and Path objects to strings
    config_dict = config.model_dump()
    config_dict['source_directory'] = str(config.source_directory)
    config_dict['target_directory'] = str(config.target_directory)

    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def create_default_config(config_path: Path) -> None:
    """Create a default configuration file."""
    default_config = Config(
        source_directory=Path("/path/to/source"),
        target_directory=Path("/path/to/target")
    )
    save_config(default_config, config_path)