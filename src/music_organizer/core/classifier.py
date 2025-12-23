"""Content classification for music files."""

import re
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from ..models.audio_file import AudioFile, ContentType
from ..exceptions import ClassificationError


class ContentClassifier:
    """Classify audio content type based on metadata and patterns."""

    # Patterns for identifying live recordings
    LIVE_PATTERNS = [
        r'\blive\b',
        r'live at',
        r'live in',
        r'bootleg',
        r'recording',
        r'concert',
        r'performance',
        r'show',
        r'festival',
    ]

    # Date patterns for live recordings
    DATE_PATTERNS = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{1,2}\.\d{1,2}\.\d{4}',  # D.M.YYYY
    ]

    # Location patterns
    LOCATION_PATTERNS = [
        r'\b\d{4}\s+-\s+.+\b',  # YEAR - LOCATION
        r',\s*[A-Z]{2}$',  # STATE initials
        r',\s*[A-Za-z\s]+$',  # City, Country
    ]

    # Compilation indicators
    COMPILATION_PATTERNS = [
        r'\bgreatest hits\b',
        r'\bbest of\b',
        r'\bthe best\b',
        r'\bessential\b',
        r'\bcollection\b',
        r'\banthology\b',
        r'\bgold\b',
        r'\bplatinum\b',
        r'\bultimate\b',
        r'\bhits\b',
        r'\bsingles\b',
        r'\bthe very best\b',
    ]

    # Rarity/bootleg indicators
    RARITY_PATTERNS = [
        r'\bdemo\b',
        r'\brare\b',
        r'\bunreleased\b',
        r'\bbonus\b',
        r'\bextra\b',
        r'\bspecial edition\b',
        r'\blimited edition\b',
        r'\banniversary\b',
        r'\bdeluxe\b',
        r'\bremastered\b',
        r'\bexpanded\b',
        r'\bbootleg\b',
    ]

    # Artist separators for collaborations
    COLLAB_SEPARATORS = [
        r'\s+feat\.\s+',
        r'\s+featuring\s+',
        r'\s+with\s+',
        r'\s+&\s+',
        r'\s+x\s+',
        r'\s+and\s+',
    ]

    @classmethod
    def classify(cls, audio_file: AudioFile) -> Tuple[ContentType, float]:
        """
        Classify audio file content type.

        Returns:
            Tuple of (content_type, confidence_score)
        """
        # Check for collaborations first
        collab_type, collab_score = cls._classify_collaboration(audio_file)
        if collab_type == ContentType.COLLABORATION and collab_score > 0.6:
            return collab_type, collab_score

        # Check for live recordings
        live_type, live_score = cls._classify_live(audio_file)
        if live_type == ContentType.LIVE and live_score > 0.5:
            return live_type, live_score

        # Check for compilations
        comp_type, comp_score = cls._classify_compilation(audio_file)
        if comp_type == ContentType.COMPILATION and comp_score > 0.6:
            return comp_type, comp_score

        # Check for rarities
        rarity_type, rarity_score = cls._classify_rarity(audio_file)
        if rarity_type == ContentType.RARITY and rarity_score > 0.6:
            return rarity_type, rarity_score

        # Default to studio album
        return ContentType.STUDIO, 0.5

    @classmethod
    def _classify_live(cls, audio_file: AudioFile) -> Tuple[ContentType, float]:
        """Classify if content is a live recording."""
        score = 0.0
        reasons = []

        # Check album/title for live indicators
        text_to_check = []
        if audio_file.album:
            text_to_check.append(audio_file.album.lower())
        if audio_file.title:
            text_to_check.append(audio_file.title.lower())
        if audio_file.path:
            text_to_check.append(audio_file.path.name.lower())

        for text in text_to_check:
            # Live patterns
            for pattern in cls.LIVE_PATTERNS:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    score += 0.3
                    reasons.append(f"Live pattern matched: {pattern}")

            # Date patterns
            for pattern in cls.DATE_PATTERNS:
                if re.search(pattern, text):
                    score += 0.2
                    # Extract date
                    match = re.search(pattern, text)
                    if match and not audio_file.date:
                        audio_file.date = match.group()
                        reasons.append(f"Found date: {audio_file.date}")

        # Check for explicit location in metadata
        if audio_file.location:
            score += 0.4
            reasons.append("Location found in metadata")

        # Check file path for location patterns
        if audio_file.path:
            path_parts = audio_file.path.parts
            for part in path_parts:
                for pattern in cls.LOCATION_PATTERNS:
                    if re.search(pattern, part):
                        score += 0.2
                        if not audio_file.location:
                            audio_file.location = part
                        reasons.append(f"Location pattern in path: {part}")
                        break

        # Check genre for live-related terms
        if audio_file.genre:
            genre_lower = audio_file.genre.lower()
            if 'live' in genre_lower:
                score += 0.1

        # Cap score
        score = min(score, 1.0)

        if score > 0.5:
            return ContentType.LIVE, score

        return ContentType.STUDIO, score

    @classmethod
    def _classify_collaboration(cls, audio_file: AudioFile) -> Tuple[ContentType, float]:
        """Classify if content is a collaboration."""
        score = 0.0

        # Check if primary artist field contains multiple artists separated by commas
        if audio_file.primary_artist:
            if ',' in audio_file.primary_artist and len([a for a in audio_file.primary_artist.split(',') if a.strip()]) > 1:
                # Split the primary artist field
                primary_artists = [a.strip() for a in audio_file.primary_artist.split(',') if a.strip()]

                # If we have 2-3 artists and it doesn't look like session musicians, it's likely a collaboration
                if len(primary_artists) <= 3:
                    # Check for session musician patterns (jazz albums with many musicians)
                    # Jazz albums often list many musicians but have clear primary artist
                    if len(primary_artists) == 2:
                        # Two artists could be a collaboration or could be primary + featured
                        # Check if it's a known collaboration pattern
                        artist_str = ' & '.join(primary_artists).lower()
                        if any(pattern in audio_file.album.lower() for pattern in [' with ', ' and ', ' & ', ' featuring ']):
                            score += 0.5
                        elif audio_file.path and any(artist.lower() in str(audio_file.path).lower() for artist in primary_artists):
                            # If both artists are in the folder name, it's likely a collaboration
                            score += 0.6
                    elif len(primary_artists) == 3:
                        # Three artists might be a trio or session musicians
                        # Check if the album title or folder suggests a trio
                        if 'trio' in audio_file.album.lower() or 'quartet' in audio_file.album.lower() or 'quintet' in audio_file.album.lower():
                            # Likely a jazz group with primary artist as leader
                            score += 0.0  # Don't mark as collaboration
                        else:
                            # Could be a collaboration
                            score += 0.4

        # Check number of artists in the artists list (not primary)
        if len(audio_file.artists) > 1:
            # Check if this looks like session musicians vs actual collaborators
            # Jazz albums often have many artists (5-10) but it's still one artist's album
            if len(audio_file.artists) > 5:
                # Likely session musicians - don't treat as collaboration
                score -= 0.2
            elif len(audio_file.artists) == 2:
                # Two artists is strong collaboration indicator
                score += 0.7
            elif len(audio_file.artists) <= 3:
                # Small number could be actual collaborators
                score += 0.5

        # Check album/artist names for collaboration indicators
        text_to_check = []
        if audio_file.album:
            text_to_check.append(audio_file.album)

        # Look for explicit collaboration terms
        collab_terms = [' with ', ' and ', ' & ', ' featuring ', ' feat ', 'featuring ', 'presents']

        for text in text_to_check:
            if not text:
                continue
            for term in collab_terms:
                if term in text.lower():
                    score += 0.4
                    break

        # Check for "vs" battles or duets
        if audio_file.album:
            if ' vs ' in audio_file.album.lower():
                score += 0.5
            if ' duets' in audio_file.album.lower():
                score += 0.3
            if ' duo' in audio_file.album.lower() or ' trio' in audio_file.album.lower():
                score += 0.3

        # Check file path for collaboration patterns
        if audio_file.path:
            path_str = str(audio_file.path).lower()

            # Check for pattern like "Artist1 & Artist2 - Album"
            if ' & ' in path_str or ' and ' in path_str:
                # Extract folder name
                folder = Path(path_str).name
                if ' & ' in folder or ' and ' in folder:
                    score += 0.4

        # Special check for specific known collaborations
        if audio_file.primary_artist:
            primary_lower = audio_file.primary_artist.lower()

            # Santana/McLaughlin is a known collaboration
            if 'santana' in primary_lower and 'mclaughlin' in primary_lower:
                score += 0.6

            # Joint albums where both names appear
            if ' love devotion surrender' in audio_file.album.lower():
                score += 0.6

        # Check jazz albums specifically - they often list many musicians but aren't collaborations
        if audio_file.genre:
            genre_lower = audio_file.genre.lower()
            if 'jazz' in genre_lower:
                # Jazz albums are less likely to be collaborations even with many artists
                score -= 0.2

        # Cap score
        score = max(0, min(score, 1.0))

        if score > 0.6:
            return ContentType.COLLABORATION, score

        return ContentType.STUDIO, score

    @classmethod
    def _classify_compilation(cls, audio_file: AudioFile) -> Tuple[ContentType, float]:
        """Classify if content is a compilation."""
        score = 0.0

        # Check album/title for compilation indicators
        text_to_check = []
        if audio_file.album:
            text_to_check.append(audio_file.album.lower())
        if audio_file.title:
            text_to_check.append(audio_file.title.lower())

        for text in text_to_check:
            for pattern in cls.COMPILATION_PATTERNS:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    score += 0.4

        # Check if it's a various artists album
        if audio_file.primary_artist and audio_file.primary_artist.lower() in ['various artists', 'va', 'various']:
            score += 0.6

        # Check if there are many different artists but album is the same
        # This would need context from multiple files, so we can't determine from single file

        # Check for DJ mixes
        if audio_file.primary_artist:
            if 'dj' in audio_file.primary_artist.lower() and 'mix' in audio_file.album.lower():
                score += 0.3

        # Cap score
        score = min(score, 1.0)

        if score > 0.6:
            return ContentType.COMPILATION, score

        return ContentType.STUDIO, score

    @classmethod
    def _classify_rarity(cls, audio_file: AudioFile) -> Tuple[ContentType, float]:
        """Classify if content is a rarity or special edition."""
        score = 0.0

        # Check album/title for rarity indicators
        text_to_check = []
        if audio_file.album:
            text_to_check.append(audio_file.album.lower())
        if audio_file.title:
            text_to_check.append(audio_file.title.lower())

        for text in text_to_check:
            for pattern in cls.RARITY_PATTERNS:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    score += 0.3

        # Check for years in parentheses or brackets (anniversary editions)
        if audio_file.album:
            # Pattern: Album Name (25th Anniversary)
            anniversary_match = re.search(r'\(\d+(?:st|nd|rd|th)\s+anniversary\)', audio_file.album.lower())
            if anniversary_match:
                score += 0.5

            # Pattern: [Remastered] or similar
            if re.search(r'\[(?:remastered|deluxe|expanded|bonus)\]', audio_file.album, flags=re.IGNORECASE):
                score += 0.4

        # Check file path
        if audio_file.path:
            path_str = str(audio_file.path).lower()
            for pattern in cls.RARITY_PATTERNS:
                if re.search(pattern, path_str):
                    score += 0.2

        # Cap score
        score = min(score, 1.0)

        if score > 0.6:
            return ContentType.RARITY, score

        return ContentType.STUDIO, score

    @classmethod
    def extract_date_from_string(cls, text: str) -> Optional[str]:
        """Extract date string from text using various patterns."""
        # YYYY-MM-DD
        match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
        if match:
            return match.group(1)

        # MM/DD/YYYY or DD/MM/YYYY
        match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', text)
        if match:
            return match.group(1)

        # D.M.YYYY
        match = re.search(r'(\d{1,2}\.\d{1,2}\.\d{4})', text)
        if match:
            return match.group(1)

        # Just the year
        match = re.search(r'\b((19|20)\d{2})\b', text)
        if match:
            return match.group(1)

        return None

    @classmethod
    def extract_location_from_string(cls, text: str) -> Optional[str]:
        """Extract location information from text."""
        # Pattern: YYYY-MM-DD - LOCATION (full date stamp before location)
        match = re.search(r'\d{4}-\d{2}-\d{2}\s*-\s*(.+)', text)
        if match:
            return match.group(1).strip()

        # Pattern: Live at LOCATION
        match = re.search(r'live at\s+(.+)', text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern: YYYY - LOCATION (year before location)
        match = re.search(r'\d{4}\s*-\s*(.+)', text)
        if match:
            return match.group(1).strip()

        # Pattern: City, State/Country
        match = re.search(r'([A-Za-z\s]+,\s*[A-Za-z]{2,})', text)
        if match:
            return match.group(1).strip()

        return None

    @classmethod
    def is_ambiguous(cls, audio_file: AudioFile) -> bool:
        """Check if classification is ambiguous and requires user input."""
        content_type, score = cls.classify(audio_file)

        # Studio albums with default score and no indicators are not ambiguous
        if content_type == ContentType.STUDIO and score == 0.5:
            # Check if there are any conflicting indicators
            indicators = [
                cls._has_live_indicators(audio_file),
                cls._has_compilation_indicators(audio_file),
                cls._has_rarity_indicators(audio_file),
            ]
            # Only consider ambiguous if there are multiple indicators
            if sum(indicators) <= 1:
                return False

        # Low confidence (other than default studio) indicates ambiguity
        if score < 0.6 and score != 0.5:
            return True

        # Multiple indicators suggest ambiguity
        indicators = [
            cls._has_live_indicators(audio_file),
            cls._has_compilation_indicators(audio_file),
            cls._has_rarity_indicators(audio_file),
            len(audio_file.artists) > 1,
        ]

        if sum(indicators) > 1:
            return True

        # Special cases
        if audio_file.album and 'greatest hits' in audio_file.album.lower():
            # Could be compilation or studio album depending on context
            return True

        return False

    @classmethod
    def _has_live_indicators(cls, audio_file: AudioFile) -> bool:
        """Check if file has live recording indicators."""
        text = f"{audio_file.album or ''} {audio_file.title or ''}".lower()
        return any(re.search(p, text, flags=re.IGNORECASE) for p in cls.LIVE_PATTERNS)

    @classmethod
    def _has_compilation_indicators(cls, audio_file: AudioFile) -> bool:
        """Check if file has compilation indicators."""
        text = f"{audio_file.album or ''} {audio_file.title or ''}".lower()
        return any(re.search(p, text, flags=re.IGNORECASE) for p in cls.COMPILATION_PATTERNS)

    @classmethod
    def _has_rarity_indicators(cls, audio_file: AudioFile) -> bool:
        """Check if file has rarity indicators."""
        text = f"{audio_file.album or ''} {audio_file.title or ''}".lower()
        return any(re.search(p, text, flags=re.IGNORECASE) for p in cls.RARITY_PATTERNS)