"""Base classes for CQRS query pattern."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import uuid4

# Type variables for generic query handling
Q = TypeVar("Q", bound="Query")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class Query:
    """Base query class with metadata."""

    query_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    include_metadata: bool = False
    cache_key: Optional[str] = None
    cache_ttl_seconds: int = 300  # 5 minutes default cache

    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "query_type": self.__class__.__name__,
            "timestamp": self.timestamp.isoformat(),
            "include_metadata": self.include_metadata,
            "cache_key": self.cache_key,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            **{
                field.name: getattr(self, field.name)
                for field in self.__dataclass_fields__.values()
                if field.name not in {"query_id", "timestamp", "include_metadata", "cache_key", "cache_ttl_seconds"}
            }
        }


class QueryHandler(ABC, Generic[Q, R]):
    """Abstract base class for query handlers."""

    @abstractmethod
    async def handle(self, query: Q) -> R:
        """Handle the query and return results."""
        pass

    @abstractmethod
    def can_handle(self, query_type: type) -> bool:
        """Check if this handler can handle the given query type."""
        pass

    def get_cache_key(self, query: Q) -> Optional[str]:
        """Generate a cache key for the query. Override for custom caching."""
        if query.cache_key:
            return query.cache_key
        return None


@dataclass(frozen=True, slots=True)
class QueryResult(Generic[R]):
    """Result wrapper for query responses."""

    data: Optional[R] = None
    success: bool = True
    query_id: str = ""
    message: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    from_cache: bool = False
    execution_time_ms: Optional[float] = None
    cached_at: Optional[datetime] = None
    total_count: Optional[int] = None  # For paginated results


class QueryBus:
    """Mediates queries to appropriate handlers with caching support."""

    def __init__(self):
        self._handlers: Dict[type, QueryHandler] = {}
        self._middleware: List[Callable] = []
        self._cache: Optional[QueryCache] = None

    def register(self, query_type: type, handler: QueryHandler) -> None:
        """Register a handler for a query type."""
        self._handlers[query_type] = handler

    def register_middleware(self, middleware: Callable) -> None:
        """Register middleware for query processing pipeline."""
        self._middleware.append(middleware)

    def set_cache(self, cache: "QueryCache") -> None:
        """Set the query cache implementation."""
        self._cache = cache

    async def dispatch(self, query: Query) -> QueryResult:
        """Dispatch a query to its registered handler."""
        query_type = type(query)
        start_time = datetime.utcnow()

        if query_type not in self._handlers:
            return QueryResult(
                success=False,
                query_id=query.query_id,
                errors=[f"No handler registered for query type: {query_type.__name__}"]
            )

        handler = self._handlers[query_type]

        # Check cache first
        cache_key = None
        if self._cache:
            cache_key = handler.get_cache_key(query)
            if cache_key:
                cached_result = await self._cache.get(cache_key)
                if cached_result:
                    execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    return QueryResult(
                        data=cached_result.data,
                        query_id=query.query_id,
                        from_cache=True,
                        cached_at=cached_result.timestamp,
                        execution_time_ms=execution_time
                    )

        try:
            # Apply middleware pipeline
            current_handler = handler.handle
            for middleware in reversed(self._middleware):
                current_handler = middleware(current_handler)

            # Execute query
            result_data = await current_handler(query)

            # Cache the result
            if self._cache and cache_key:
                await self._cache.set(
                    cache_key,
                    result_data,
                    ttl_seconds=query.cache_ttl_seconds
                )

            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return QueryResult(
                data=result_data,
                success=True,
                query_id=query.query_id,
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return QueryResult(
                success=False,
                query_id=query.query_id,
                errors=[str(e)],
                execution_time_ms=execution_time
            )

    def get_registered_queries(self) -> List[type]:
        """Get list of registered query types."""
        return list(self._handlers.keys())


class QueryCache:
    """Simple in-memory query cache with TTL support."""

    def __init__(self):
        self._cache: Dict[str, "CacheEntry"] = {}

    async def get(self, key: str) -> Optional["CacheEntry"]:
        """Get cached entry if not expired."""
        entry = self._cache.get(key)
        if entry and not entry.is_expired():
            return entry
        elif entry:
            # Remove expired entry
            del self._cache[key]
        return None

    async def set(self, key: str, data: Any, ttl_seconds: int = 300) -> None:
        """Cache data with TTL."""
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        self._cache[key] = CacheEntry(
            data=data,
            timestamp=datetime.utcnow(),
            expires_at=expires_at
        )

    async def invalidate(self, pattern: str = None) -> None:
        """Invalidate cache entries. If pattern is provided, only matching keys are removed."""
        if pattern:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                self._cache.pop(key, None)
        else:
            self._cache.clear()


@dataclass
class CacheEntry:
    """Cache entry with expiration support."""

    data: Any
    timestamp: datetime
    expires_at: datetime

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return datetime.utcnow() > self.expires_at