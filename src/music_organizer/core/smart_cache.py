"""Smart caching strategy with intelligent TTL and modification time integration.

This module implements a sophisticated caching system that adapts to file modification
patterns, reduces unnecessary filesystem calls, and provides optimal cache performance
for music libraries of all sizes.
"""

from __future__ import annotations

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

from ..models.audio_file import AudioFile, ContentType
from ..domain.value_objects import ArtistName, TrackNumber

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class CacheEntry:
    """Represents a cache entry with smart metadata."""
    file_path: str
    file_mtime: float
    file_size: int
    cached_at: datetime
    expires_at: datetime
    access_count: int
    last_accessed: datetime
    file_hash: Optional[str]
    directory_mtime: float
    access_frequency: float  # accesses per day
    stability_score: float   # 0.0-1.0, higher = more stable


@dataclass(slots=True, frozen=True)
class DirectoryStats:
    """Statistics for directory-level change detection."""
    directory_path: str
    directory_mtime: float
    file_count: int
    last_change: datetime
    change_frequency: float  # changes per day
    stability_score: float


class SmartCacheManager:
    """Intelligent cache manager with adaptive TTL and optimization strategies."""

    _instance: Optional[SmartCacheManager] = None
    _lock = threading.Lock()

    def __new__(cls, cache_dir: Optional[Path] = None) -> SmartCacheManager:
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize the smart cache manager."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Default cache directory
            if cache_dir is None:
                cache_dir = Path.home() / ".cache" / "music-organizer"

            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # SQLite database path
            self.db_path = self.cache_dir / "smart_cache.db"

            # Configuration
            self.default_ttl = timedelta(days=30)
            self.min_ttl = timedelta(days=1)
            self.max_ttl = timedelta(days=365)
            self.stability_threshold = 0.8  # Files with stability > 0.8 get longer TTL
            self.frequency_bonus_multiplier = 2.0  # Frequently accessed files get TTL bonus

            # In-memory caches for performance
            self._directory_cache: Dict[str, DirectoryStats] = {}
            self._access_stats: Dict[str, List[datetime]] = defaultdict(list)

            # Initialize database
            self._init_db()

            self._initialized = True

    def _init_db(self) -> None:
        """Initialize SQLite database with schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Smart cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS smart_cache (
                    file_path TEXT PRIMARY KEY,
                    file_mtime REAL NOT NULL,
                    file_size INTEGER NOT NULL,
                    cached_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    file_hash TEXT,
                    directory_path TEXT NOT NULL,
                    directory_mtime REAL NOT NULL,
                    access_frequency REAL DEFAULT 0.0,
                    stability_score REAL DEFAULT 0.0,
                    file_type TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    artists TEXT,
                    primary_artist TEXT,
                    album TEXT,
                    title TEXT,
                    year INTEGER,
                    date TEXT,
                    location TEXT,
                    track_number INTEGER,
                    genre TEXT,
                    has_cover_art BOOLEAN,
                    metadata TEXT
                )
            """)

            # Directory statistics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS directory_stats (
                    directory_path TEXT PRIMARY KEY,
                    directory_mtime REAL NOT NULL,
                    file_count INTEGER DEFAULT 0,
                    last_change TIMESTAMP NOT NULL,
                    change_frequency REAL DEFAULT 0.0,
                    stability_score REAL DEFAULT 0.0,
                    last_updated TIMESTAMP NOT NULL
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_smart_expires_at ON smart_cache(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_smart_directory ON smart_cache(directory_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_smart_frequency ON smart_cache(access_frequency)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_smart_stability ON smart_cache(stability_score)")

            conn.commit()

    def get(self, file_path: Path) -> Optional[AudioFile]:
        """Get cached metadata with intelligent validation."""
        try:
            stat = file_path.stat()
            file_mtime = stat.st_mtime
            file_size = stat.st_size
            directory_path = str(file_path.parent)
            directory_mtime = file_path.parent.stat().st_mtime
        except (OSError, FileNotFoundError, AttributeError):
            return None

        # Check directory-level cache first
        if not self._should_check_directory(directory_path, directory_mtime):
            # Directory unchanged, can trust individual file cache more
            return self._get_from_cache_fast(file_path, file_mtime, file_size)

        # Directory may have changed, do full check
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM smart_cache
                WHERE file_path = ? AND expires_at > datetime('now')
                """,
                (str(file_path),)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Validate file hasn't changed
            if row['file_mtime'] != file_mtime or row['file_size'] != file_size:
                self.invalidate(file_path)
                return None

            # Update access statistics
            self._update_access_stats(file_path)

            # Update database record
            new_expires_at = datetime.now() + self._calculate_adaptive_ttl(file_path, row)
            conn.execute(
                """
                UPDATE smart_cache
                SET access_count = access_count + 1,
                    last_accessed = datetime('now'),
                    access_frequency = ?,
                    expires_at = ?
                WHERE file_path = ?
                """,
                (
                    self._calculate_access_frequency(file_path),
                    new_expires_at,
                    str(file_path)
                )
            )
            conn.commit()

            return self._row_to_audiofile(row, file_path)

    def put(self, audio_file: AudioFile) -> None:
        """Cache metadata with intelligent TTL calculation."""
        try:
            stat = audio_file.path.stat()
            file_mtime = stat.st_mtime
            file_size = stat.st_size
            directory_path = str(audio_file.path.parent)
            directory_mtime = audio_file.path.parent.stat().st_mtime
        except (OSError, FileNotFoundError, AttributeError):
            file_mtime = 0
            file_size = 0
            directory_mtime = 0
            directory_path = str(audio_file.path.parent)

        # Calculate file hash for integrity checking
        file_hash = self._calculate_file_hash(audio_file.path) if audio_file.path.exists() else None

        # Get access statistics
        access_count = len(self._access_stats.get(str(audio_file.path), []))
        access_frequency = self._calculate_access_frequency(audio_file.path)
        stability_score = self._calculate_stability_score(audio_file.path, file_mtime)

        # Calculate adaptive TTL
        expires_at = datetime.now() + self._calculate_adaptive_ttl_from_scores(
            access_frequency, stability_score
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO smart_cache (
                    file_path, file_mtime, file_size, cached_at, expires_at,
                    access_count, last_accessed, file_hash, directory_path, directory_mtime,
                    access_frequency, stability_score, file_type, content_type,
                    artists, primary_artist, album, title, year, date, location,
                    track_number, genre, has_cover_art, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(audio_file.path),
                    file_mtime,
                    file_size,
                    datetime.now(),
                    expires_at,
                    access_count,
                    datetime.now(),
                    file_hash,
                    directory_path,
                    directory_mtime,
                    access_frequency,
                    stability_score,
                    audio_file.file_type,
                    audio_file.content_type.value,
                    json.dumps([str(artist) for artist in audio_file.artists]) if audio_file.artists else None,
                    audio_file.primary_artist,
                    audio_file.album,
                    audio_file.title,
                    audio_file.year,
                    audio_file.date,
                    audio_file.location,
                    int(audio_file.track_number) if audio_file.track_number else None,
                    audio_file.genre,
                    audio_file.has_cover_art,
                    json.dumps(audio_file.metadata) if audio_file.metadata else None
                )
            )
            conn.commit()

        # Update directory statistics
        self._update_directory_stats(directory_path, directory_mtime)

    def _should_check_directory(self, directory_path: str, directory_mtime: float) -> bool:
        """Check if we should validate files in this directory."""
        # Check in-memory cache first
        if directory_path in self._directory_cache:
            stats = self._directory_cache[directory_path]
            if stats.directory_mtime == directory_mtime:
                return False

        # Check database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT directory_mtime FROM directory_stats
                WHERE directory_path = ?
                """,
                (directory_path,)
            )
            row = cursor.fetchone()

            if row and row[0] == directory_mtime:
                # Update memory cache
                self._directory_cache[directory_path] = DirectoryStats(
                    directory_path=directory_path,
                    directory_mtime=directory_mtime,
                    file_count=0,  # Not needed for this check
                    last_change=datetime.now(),
                    change_frequency=0.0,
                    stability_score=1.0
                )
                return False

        return True

    def _get_from_cache_fast(self, file_path: Path, file_mtime: float, file_size: float) -> Optional[AudioFile]:
        """Fast cache lookup when directory is unchanged."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM smart_cache
                WHERE file_path = ? AND file_mtime = ? AND file_size = ?
                    AND expires_at > datetime('now')
                """,
                (str(file_path), file_mtime, file_size)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Quick access count update (can be deferred)
            self._access_stats[str(file_path)].append(datetime.now())

            return self._row_to_audiofile(row, file_path)

    def _update_access_stats(self, file_path: Path) -> None:
        """Update access statistics for a file."""
        file_path_str = str(file_path)
        now = datetime.now()

        # Add to access history
        self._access_stats[file_path_str].append(now)

        # Clean old access records (keep last 90 days)
        cutoff = now - timedelta(days=90)
        self._access_stats[file_path_str] = [
            access_time for access_time in self._access_stats[file_path_str]
            if access_time > cutoff
        ]

    def _calculate_access_frequency(self, file_path: Path) -> float:
        """Calculate access frequency as accesses per day."""
        file_path_str = str(file_path)
        accesses = self._access_stats.get(file_path_str, [])

        if not accesses:
            return 0.0

        # Calculate accesses per day over the last 30 days
        cutoff = datetime.now() - timedelta(days=30)
        recent_accesses = [a for a in accesses if a > cutoff]

        if not recent_accesses:
            return 0.0

        # Calculate days spanned
        days_span = max(1.0, (max(recent_accesses) - min(recent_accesses)).total_seconds() / 86400)

        return len(recent_accesses) / days_span

    def _calculate_stability_score(self, file_path: Path, current_mtime: float) -> float:
        """Calculate stability score based on file modification history."""
        file_path_str = str(file_path)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT file_mtime, cached_at FROM smart_cache
                WHERE file_path = ?
                ORDER BY cached_at DESC
                LIMIT 10
                """,
                (file_path_str,)
            )
            rows = cursor.fetchall()

        if not rows:
            # No history, assume moderate stability
            return 0.5

        # Check if file has been stable
        oldest_mtime = rows[-1][0] if rows else current_mtime

        # If file hasn't changed in a long time, high stability
        days_since_change = (datetime.now().timestamp() - oldest_mtime) / 86400
        stability = min(1.0, days_since_change / 90)  # Normalize to 0-1 over 90 days

        # Reduce stability if file changes frequently
        if len(rows) > 1:
            recent_changes = sum(1 for i in range(1, len(rows)) if rows[i][0] != rows[i-1][0])
            if recent_changes > len(rows) * 0.5:  # Changed in >50% of accesses
                stability *= 0.5

        return stability

    def _calculate_adaptive_ttl(self, file_path: Path, row: sqlite3.Row) -> timedelta:
        """Calculate adaptive TTL based on access patterns and stability."""
        access_frequency = row['access_frequency']
        stability_score = row['stability_score']

        return self._calculate_adaptive_ttl_from_scores(access_frequency, stability_score)

    def _calculate_adaptive_ttl_from_scores(self, access_frequency: float, stability_score: float) -> timedelta:
        """Calculate adaptive TTL from access frequency and stability scores."""
        # Base TTL
        base_ttl = self.default_ttl

        # Stability bonus - more stable files get longer TTL
        if stability_score > self.stability_threshold:
            stability_bonus = base_ttl * (stability_score - self.stability_threshold)
            base_ttl = base_ttl + stability_bonus

        # Frequency bonus - frequently accessed files get longer TTL
        if access_frequency > 0.1:  # More than 0.1 accesses per day
            frequency_bonus = min(
                base_ttl * (access_frequency * self.frequency_bonus_multiplier),
                base_ttl * 2  # Cap at 2x bonus
            )
            base_ttl = base_ttl + frequency_bonus

        # Clamp to min/max
        return max(self.min_ttl, min(self.max_ttl, base_ttl))

    def _calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate a quick file hash for integrity checking."""
        if not file_path.exists() or file_path.stat().st_size > 100 * 1024 * 1024:  # Skip >100MB files
            return None

        try:
            # Use first 8KB for quick hash
            with open(file_path, 'rb') as f:
                data = f.read(8192)
                return hashlib.sha256(data).hexdigest()[:16]
        except (OSError, IOError):
            return None

    def _update_directory_stats(self, directory_path: str, directory_mtime: float) -> None:
        """Update directory-level statistics."""
        now = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            # Check if directory exists
            cursor = conn.execute(
                """
                SELECT file_count, last_change, change_frequency FROM directory_stats
                WHERE directory_path = ?
                """,
                (directory_path,)
            )
            row = cursor.fetchone()

            if row:
                file_count, last_change_str, change_frequency = row

                # Parse last_change from string if it's stored as text
                if isinstance(last_change_str, str):
                    # SQLite returns timestamps as strings
                    last_change = datetime.fromisoformat(last_change_str.replace('Z', '+00:00'))
                else:
                    last_change = last_change_str

                # Update change frequency
                if directory_mtime != last_change.timestamp():
                    days_since_last = (now - last_change).total_seconds() / 86400
                    if days_since_last > 0:
                        change_frequency = (change_frequency + 1.0 / days_since_last) / 2
                    last_change = now
            else:
                # New directory
                file_count = 0
                change_frequency = 0.0
                last_change = now

            # Calculate stability score (inverse of change frequency)
            stability_score = max(0.0, min(1.0, 1.0 - (change_frequency * 30)))

            # Update database
            conn.execute(
                """
                INSERT OR REPLACE INTO directory_stats
                (directory_path, directory_mtime, file_count, last_change,
                 change_frequency, stability_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    directory_path,
                    directory_mtime,
                    file_count,
                    last_change,
                    change_frequency,
                    stability_score,
                    now
                )
            )
            conn.commit()

        # Update memory cache
        self._directory_cache[directory_path] = DirectoryStats(
            directory_path=directory_path,
            directory_mtime=directory_mtime,
            file_count=file_count,
            last_change=last_change,
            change_frequency=change_frequency,
            stability_score=stability_score
        )

    def _row_to_audiofile(self, row: sqlite3.Row, file_path: Path) -> AudioFile:
        """Convert database row to AudioFile instance."""
        artists_data = json.loads(row['artists']) if row['artists'] else []
        artists = [ArtistName(name) for name in artists_data] if artists_data else []
        metadata = json.loads(row['metadata']) if row['metadata'] else {}

        return AudioFile(
            path=file_path,
            file_type=row['file_type'],
            metadata=metadata,
            content_type=ContentType(row['content_type']),
            artists=artists,
            primary_artist=row['primary_artist'],
            album=row['album'],
            title=row['title'],
            year=row['year'],
            date=row['date'],
            location=row['location'],
            track_number=TrackNumber(row['track_number']) if row['track_number'] else None,
            genre=row['genre'],
            has_cover_art=bool(row['has_cover_art'])
        )

    def invalidate(self, file_path: Path) -> None:
        """Invalidate cache entry for a file."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM smart_cache WHERE file_path = ?",
                (str(file_path),)
            )
            conn.commit()

        # Clear access stats
        self._access_stats.pop(str(file_path), None)

    def cleanup_expired(self) -> int:
        """Remove expired entries and update statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Remove expired entries
            cursor = conn.execute(
                "DELETE FROM smart_cache WHERE expires_at <= datetime('now')"
            )
            expired_count = cursor.rowcount

            # Update directory file counts
            conn.execute("""
                UPDATE directory_stats
                SET file_count = (
                    SELECT COUNT(*) FROM smart_cache
                    WHERE smart_cache.directory_path = directory_stats.directory_path
                ),
                last_updated = datetime('now')
            """)

            conn.commit()

        return expired_count

    def get_smart_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Basic stats
            total = conn.execute("SELECT COUNT(*) FROM smart_cache").fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM smart_cache WHERE expires_at <= datetime('now')"
            ).fetchone()[0]
            valid = total - expired

            # Access frequency stats
            freq_cursor = conn.execute("""
                SELECT
                    AVG(access_frequency) as avg_frequency,
                    MAX(access_frequency) as max_frequency,
                    COUNT(CASE WHEN access_frequency > 0.1 THEN 1 END) as frequently_accessed
                FROM smart_cache
            """)
            freq_stats = freq_cursor.fetchone()

            # Stability stats
            stability_cursor = conn.execute("""
                SELECT
                    AVG(stability_score) as avg_stability,
                    COUNT(CASE WHEN stability_score > 0.8 THEN 1 END) as stable_files
                FROM smart_cache
            """)
            stability_stats = stability_cursor.fetchone()

            # Directory stats
            dir_cursor = conn.execute("""
                SELECT COUNT(*) as total_directories,
                       AVG(change_frequency) as avg_dir_change_freq,
                       AVG(stability_score) as avg_dir_stability
                FROM directory_stats
            """)
            dir_stats = dir_cursor.fetchone()

            # Memory usage
            cursor = conn.execute(
                "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
            )
            size_row = cursor.fetchone()
            size_bytes = size_row[0] if size_row else 0

            return {
                'total_entries': total,
                'valid_entries': valid,
                'expired_entries': expired,
                'size_bytes': size_bytes,
                'size_mb': size_bytes / (1024 * 1024),
                'avg_access_frequency': freq_stats[0] or 0.0,
                'max_access_frequency': freq_stats[1] or 0.0,
                'frequently_accessed_files': freq_stats[2] or 0,
                'avg_stability_score': stability_stats[0] or 0.0,
                'stable_files': stability_stats[1] or 0,
                'total_directories': dir_stats[0] or 0,
                'avg_directory_change_frequency': dir_stats[1] or 0.0,
                'avg_directory_stability': dir_stats[2] or 0.0,
                'cached_directories': len(self._directory_cache),
                'tracked_files': sum(len(accesses) for accesses in self._access_stats.values())
            }

    def warm_cache(self, directory: Path, recursive: bool = True, max_files: int = 1000) -> int:
        """Warm cache with likely-to-be-accessed files."""
        from concurrent.futures import ThreadPoolExecutor
        import os

        logger.info(f"Warming cache for {directory}")

        # Find audio files
        audio_extensions = {'.flac', '.mp3', '.wav', '.m4a', '.mp4', '.aac', '.ogg', '.opus', '.wma'}
        files_to_process = []

        if recursive:
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    if Path(filename).suffix.lower() in audio_extensions:
                        files_to_process.append(Path(root) / filename)
        else:
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    files_to_process.append(file_path)

        # Prioritize by access frequency and stability
        prioritized_files = self._prioritize_files_for_warming(files_to_process[:max_files])

        # Warm cache in parallel
        warmed = 0
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for file_path in prioritized_files:
                if self.get(file_path) is None:
                    futures.append(executor.submit(self._warm_file, file_path))

            for future in futures:
                try:
                    future.result()
                    warmed += 1
                except Exception as e:
                    logger.warning(f"Failed to warm cache: {e}")

        logger.info(f"Warmed cache with {warmed} files")
        return warmed

    def _prioritize_files_for_warming(self, files: List[Path]) -> List[Path]:
        """Prioritize files for cache warming based on access patterns."""
        prioritized = []

        with sqlite3.connect(self.db_path) as conn:
            for file_path in files:
                cursor = conn.execute(
                    """
                    SELECT access_frequency, stability_score FROM smart_cache
                    WHERE file_path = ?
                    """,
                    (str(file_path),)
                )
                row = cursor.fetchone()

                if row:
                    frequency, stability = row
                    # Priority = frequency * 10 + stability * 5
                    priority = frequency * 10 + stability * 5
                else:
                    # New files get moderate priority
                    priority = 1.0

                prioritized.append((file_path, priority))

        # Sort by priority (descending)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        return [file_path for file_path, _ in prioritized]

    def _warm_file(self, file_path: Path) -> None:
        """Warm cache for a single file."""
        try:
            from .metadata import MetadataHandler
            audio_file = MetadataHandler.extract_metadata(file_path)
            self.put(audio_file)
        except Exception as e:
            logger.warning(f"Failed to warm cache for {file_path}: {e}")

    def clear(self) -> None:
        """Clear all cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM smart_cache")
            conn.execute("DELETE FROM directory_stats")
            conn.commit()

        # Clear memory caches
        self._directory_cache.clear()
        self._access_stats.clear()

    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache performance by cleaning and reorganizing."""
        stats_before = self.get_smart_stats()

        # Clean expired entries
        expired_removed = self.cleanup_expired()

        # Vacuum database to reclaim space
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")
            conn.commit()

        # Clear old access statistics (older than 90 days)
        cutoff = datetime.now() - timedelta(days=90)
        for file_path, accesses in list(self._access_stats.items()):
            self._access_stats[file_path] = [a for a in accesses if a > cutoff]
            if not self._access_stats[file_path]:
                del self._access_stats[file_path]

        stats_after = self.get_smart_stats()

        return {
            'expired_entries_removed': expired_removed,
            'size_before_mb': stats_before['size_mb'],
            'size_after_mb': stats_after['size_mb'],
            'space_saved_mb': stats_before['size_mb'] - stats_after['size_mb'],
            'entries_before': stats_before['total_entries'],
            'entries_after': stats_after['total_entries']
        }


# Global instance for easy access
_default_smart_cache: Optional[SmartCacheManager] = None


def get_smart_cache_manager(cache_dir: Optional[Path] = None) -> SmartCacheManager:
    """Get the default smart cache manager.

    If cache_dir is provided and differs from the current instance's directory,
    the global instance is replaced with a new one using that directory.
    """
    global _default_smart_cache
    if _default_smart_cache is None:
        _default_smart_cache = SmartCacheManager(cache_dir)
    elif cache_dir is not None and _default_smart_cache.cache_dir != cache_dir:
        # Reset the global instance with the new cache_dir
        _default_smart_cache = SmartCacheManager(cache_dir)
    return _default_smart_cache