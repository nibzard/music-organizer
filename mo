#!/bin/bash
# Music Organizer wrapper script

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run music-organizer with uv
exec uv run python "$SCRIPT_DIR/music-organizer" "$@"