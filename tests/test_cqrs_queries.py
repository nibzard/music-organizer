"""Comprehensive tests for CQRS query modules."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

from music_organizer.application.queries import (
    Query,
    QueryHandler,
    QueryBus,
    QueryResult,
    QueryCache,
    CacheEntry,
)


def async_test(func):
    """Decorator to run async tests."""
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


class TestQuery:
    """Test Query base class."""

    def test_query_creation_with_defaults(self):
        """Test query creation with default values."""
        query = Query()

        assert query.query_id is not None
        assert isinstance(query.query_id, str)
        assert query.timestamp is not None
        assert isinstance(query.timestamp, datetime)
        assert query.include_metadata is False
        assert query.cache_key is None
        assert query.cache_ttl_seconds == 300

    def test_query_creation_with_values(self):
        """Test query creation with provided values."""
        cache_key = "custom-cache-key"
        ttl = 600

        query = Query(
            include_metadata=True,
            cache_key=cache_key,
            cache_ttl_seconds=ttl
        )

        assert query.include_metadata is True
        assert query.cache_key == cache_key
        assert query.cache_ttl_seconds == ttl

    def test_query_to_dict(self):
        """Test query serialization to dictionary."""
        cache_key = "test-cache-key"

        query = Query(
            include_metadata=True,
            cache_key=cache_key,
            cache_ttl_seconds=600
        )

        result = query.to_dict()

        assert result["query_id"] == query.query_id
        assert result["query_type"] == "Query"
        assert "timestamp" in result
        assert result["include_metadata"] is True
        assert result["cache_key"] == cache_key
        assert result["cache_ttl_seconds"] == 600

    def test_query_is_frozen(self):
        """Test that query instances are immutable."""
        from dataclasses import FrozenInstanceError
        query = Query()

        with pytest.raises(FrozenInstanceError):
            query.query_id = "new-id"

    def test_query_with_custom_fields(self):
        """Test query subclass with custom fields."""
        @dataclass(frozen=True, slots=True, kw_only=True)
        class CustomQuery(Query):
            search_term: str
            limit: int

        query = CustomQuery(search_term="test", limit=10)

        result = query.to_dict()
        assert result["search_term"] == "test"
        assert result["limit"] == 10


class TestQueryResult:
    """Test QueryResult class."""

    def test_result_creation_success(self):
        """Test successful result creation."""
        result = QueryResult(
            data={"key": "value"},
            success=True,
            query_id="query-123"
        )

        assert result.data == {"key": "value"}
        assert result.success is True
        assert result.query_id == "query-123"
        assert result.message is None
        assert result.errors == []
        assert result.from_cache is False
        assert result.execution_time_ms is None
        assert result.cached_at is None

    def test_result_creation_failure(self):
        """Test failed result creation."""
        result = QueryResult(
            success=False,
            query_id="query-456",
            errors=["Error 1", "Error 2"],
            execution_time_ms=50.5
        )

        assert result.success is False
        assert result.query_id == "query-456"
        assert result.errors == ["Error 1", "Error 2"]
        assert result.execution_time_ms == 50.5

    def test_result_from_cache(self):
        """Test result created from cache."""
        cached_at = datetime.utcnow()
        result = QueryResult(
            data={"cached": "data"},
            query_id="query-789",
            from_cache=True,
            cached_at=cached_at
        )

        assert result.from_cache is True
        assert result.cached_at == cached_at

    def test_result_with_total_count(self):
        """Test result with total count for pagination."""
        result = QueryResult(
            data=[{"id": 1}, {"id": 2}],
            query_id="query-101",
            total_count=100
        )

        assert result.total_count == 100

    def test_result_generic_type(self):
        """Test QueryResult generic type support."""
        result = QueryResult[str](
            data="test string",
            query_id="query-202"
        )

        assert result.data == "test string"


class TestQueryHandler:
    """Test QueryHandler abstract base class."""

    def test_handler_requires_handle_implementation(self):
        """Test that handle method must be implemented."""
        with pytest.raises(TypeError):
            class IncompleteHandler(QueryHandler):
                pass

            IncompleteHandler()

    def test_handler_requires_can_handle_implementation(self):
        """Test that can_handle method must be implemented."""
        with pytest.raises(TypeError):
            class IncompleteHandler(QueryHandler):
                async def handle(self, query):
                    return QueryResult(data=None, query_id="123")

            IncompleteHandler()

    def test_concrete_handler_implementation(self):
        """Test concrete handler implementation."""
        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestQuery(Query):
            search: str

        class TestHandler(QueryHandler):
            async def handle(self, query: TestQuery) -> str:
                return f"Result for: {query.search}"

            def can_handle(self, query_type: type) -> bool:
                return query_type == TestQuery

        handler = TestHandler()
        query = TestQuery(search="test-value")

        assert handler.can_handle(TestQuery) is True
        assert handler.can_handle(Query) is False

    def test_get_cache_key_default(self):
        """Test default cache key generation."""
        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestQuery(Query):
            search: str

        class TestHandler(QueryHandler):
            async def handle(self, query: TestQuery) -> str:
                return "result"

            def can_handle(self, query_type: type) -> bool:
                return query_type == TestQuery

        handler = TestHandler()

        # No cache key on query
        query_no_key = TestQuery(search="test")
        assert handler.get_cache_key(query_no_key) is None

        # With cache key
        query_with_key = TestQuery(search="test", cache_key="custom-key")
        assert handler.get_cache_key(query_with_key) == "custom-key"

    def test_get_cache_key_custom(self):
        """Test custom cache key generation."""
        @dataclass(frozen=True, slots=True, kw_only=True)
        class SearchQuery(Query):
            term: str
            filters: dict

        class SearchHandler(QueryHandler):
            async def handle(self, query: SearchQuery) -> list:
                return []

            def can_handle(self, query_type: type) -> bool:
                return query_type == SearchQuery

            def get_cache_key(self, query: SearchQuery) -> str:
                import json
                return f"search:{query.term}:{json.dumps(query.filters, sort_keys=True)}"

        handler = SearchHandler()
        query = SearchQuery(term="jazz", filters={"genre": "jazz", "year": 2020})

        cache_key = handler.get_cache_key(query)
        assert "search:jazz" in cache_key
        assert "genre" in cache_key


class TestQueryCache:
    """Test QueryCache class."""

    @async_test
    async def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = QueryCache()

        await cache.set("key1", "value1")
        result = await cache.get("key1")

        assert result is not None
        assert result.data == "value1"

    @async_test
    async def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = QueryCache()

        # Set entry with 1 second TTL
        await cache.set("key1", "value1", ttl_seconds=1)

        # Immediately get - should exist
        result = await cache.get("key1")
        assert result is not None
        assert result.data == "value1"

        # Wait for expiration
        import asyncio
        await asyncio.sleep(1.1)

        # Should be expired
        result = await cache.get("key1")
        assert result is None

    @async_test
    async def test_cache_invalidates_expired_entries(self):
        """Test that expired entries are removed from cache."""
        cache = QueryCache()

        await cache.set("key1", "value1", ttl_seconds=1)
        await cache.set("key2", "value2", ttl_seconds=10)

        # Wait for key1 to expire
        import asyncio
        await asyncio.sleep(1.1)

        # Try to get expired entry - should return None and remove it
        result1 = await cache.get("key1")
        assert result1 is None

        # key2 should still exist
        result2 = await cache.get("key2")
        assert result2 is not None
        assert result2.data == "value2"

    @async_test
    async def test_cache_invalidate_all(self):
        """Test invalidating all cache entries."""
        cache = QueryCache()

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        await cache.invalidate()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    @async_test
    async def test_cache_invalidate_with_pattern(self):
        """Test invalidating cache entries with pattern."""
        cache = QueryCache()

        await cache.set("user:1:data", "value1")
        await cache.set("user:2:data", "value2")
        await cache.set("product:1:data", "value3")

        await cache.invalidate(pattern="user:")

        assert await cache.get("user:1:data") is None
        assert await cache.get("user:2:data") is None
        assert await cache.get("product:1:data") is not None

    @async_test
    async def test_cache_nonexistent_key(self):
        """Test getting nonexistent cache key."""
        cache = QueryCache()

        result = await cache.get("nonexistent")
        assert result is None


class TestCacheEntry:
    """Test CacheEntry class."""

    def test_entry_creation(self):
        """Test cache entry creation."""
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=300)

        entry = CacheEntry(
            data="test data",
            timestamp=now,
            expires_at=expires_at
        )

        assert entry.data == "test data"
        assert entry.timestamp == now
        assert entry.expires_at == expires_at

    def test_entry_not_expired(self):
        """Test entry expiration check for valid entry."""
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=300)

        entry = CacheEntry(
            data="test data",
            timestamp=now,
            expires_at=expires_at
        )

        assert entry.is_expired() is False

    def test_entry_is_expired(self):
        """Test entry expiration check for expired entry."""
        now = datetime.utcnow()
        expires_at = now - timedelta(seconds=10)  # Expired 10 seconds ago

        entry = CacheEntry(
            data="test data",
            timestamp=now,
            expires_at=expires_at
        )

        assert entry.is_expired() is True


class TestQueryBus:
    """Test QueryBus class."""

    def test_bus_initialization(self):
        """Test bus initialization."""
        bus = QueryBus()

        assert bus.get_registered_queries() == []
        assert bus._handlers == {}
        assert bus._middleware == []
        assert bus._cache is None

    @async_test
    async def test_register_handler(self):
        """Test registering a query handler."""
        bus = QueryBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestQuery(Query):
            search: str

        class TestHandler(QueryHandler):
            async def handle(self, query: TestQuery) -> str:
                return f"Result: {query.search}"

            def can_handle(self, query_type: type) -> bool:
                return query_type == TestQuery

        handler = TestHandler()
        bus.register(TestQuery, handler)

        assert TestQuery in bus.get_registered_queries()
        assert bus._handlers[TestQuery] == handler

    @async_test
    async def test_dispatch_success(self):
        """Test successful query dispatch."""
        bus = QueryBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestQuery(Query):
            search: str

        class TestHandler(QueryHandler):
            async def handle(self, query: TestQuery) -> str:
                return f"Result: {query.search}"

            def can_handle(self, query_type: type) -> bool:
                return query_type == TestQuery

        handler = TestHandler()
        bus.register(TestQuery, handler)

        query = TestQuery(search="test-data")
        result = await bus.dispatch(query)

        assert result.success is True
        assert result.query_id == query.query_id
        assert result.data == "Result: test-data"
        assert result.from_cache is False
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0

    @async_test
    async def test_dispatch_no_handler(self):
        """Test dispatch when no handler is registered."""
        bus = QueryBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class UnhandledQuery(Query):
            search: str

        query = UnhandledQuery(search="test")
        result = await bus.dispatch(query)

        assert result.success is False
        assert result.query_id == query.query_id
        assert len(result.errors) == 1
        assert "No handler registered" in result.errors[0]

    @async_test
    async def test_dispatch_handler_exception(self):
        """Test dispatch when handler raises exception."""
        bus = QueryBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class FailingQuery(Query):
            search: str

        class FailingHandler(QueryHandler):
            async def handle(self, query: FailingQuery) -> str:
                raise ValueError("Handler error")

            def can_handle(self, query_type: type) -> bool:
                return query_type == FailingQuery

        handler = FailingHandler()
        bus.register(FailingQuery, handler)

        query = FailingQuery(search="test")
        result = await bus.dispatch(query)

        assert result.success is False
        assert result.query_id == query.query_id
        assert len(result.errors) == 1
        assert "Handler error" in result.errors[0]
        assert result.execution_time_ms is not None

    @async_test
    async def test_middleware_execution(self):
        """Test middleware pipeline execution."""
        bus = QueryBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestQuery(Query):
            search: str

        class TestHandler(QueryHandler):
            async def handle(self, query: TestQuery) -> str:
                return f"Result: {query.search}"

            def can_handle(self, query_type: type) -> bool:
                return query_type == TestQuery

        middleware_calls = []

        def logging_middleware(next_handler):
            async def wrapper(query):
                middleware_calls.append("logging")
                return await next_handler(query)
            return wrapper

        def validation_middleware(next_handler):
            async def wrapper(query):
                middleware_calls.append("validation")
                return await next_handler(query)
            return wrapper

        handler = TestHandler()
        bus.register(TestQuery, handler)
        bus.register_middleware(logging_middleware)
        bus.register_middleware(validation_middleware)

        query = TestQuery(search="test-data")
        result = await bus.dispatch(query)

        assert result.success is True
        assert len(middleware_calls) == 2
        assert middleware_calls == ["logging", "validation"]

    @async_test
    async def test_cache_hit(self):
        """Test cache hit scenario."""
        bus = QueryBus()
        cache = QueryCache()
        bus.set_cache(cache)

        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestQuery(Query):
            search: str

        class TestHandler(QueryHandler):
            async def handle(self, query: TestQuery) -> str:
                return f"Result: {query.search}"

            def can_handle(self, query_type: type) -> bool:
                return query_type == TestQuery

            def get_cache_key(self, query: TestQuery) -> str:
                return f"search:{query.search}"

        handler = TestHandler()
        bus.register(TestQuery, handler)

        # First call - cache miss
        query1 = TestQuery(search="test", cache_key="search:test")
        result1 = await bus.dispatch(query1)

        assert result1.success is True
        assert result1.data == "Result: test"
        assert result1.from_cache is False

        # Second call - cache hit
        query2 = TestQuery(search="test", cache_key="search:test")
        result2 = await bus.dispatch(query2)

        assert result2.success is True
        assert result2.data == "Result: test"
        assert result2.from_cache is True
        assert result2.cached_at is not None

    @async_test
    async def test_cache_miss(self):
        """Test cache miss scenario."""
        bus = QueryBus()
        cache = QueryCache()
        bus.set_cache(cache)

        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestQuery(Query):
            search: str

        class TestHandler(QueryHandler):
            async def handle(self, query: TestQuery) -> str:
                return f"Result: {query.search}"

            def can_handle(self, query_type: type) -> bool:
                return query_type == TestQuery

        handler = TestHandler()
        bus.register(TestQuery, handler)

        query = TestQuery(search="test")  # No cache key
        result = await bus.dispatch(query)

        assert result.success is True
        assert result.data == "Result: test"
        assert result.from_cache is False

    @async_test
    async def test_cache_set_after_query(self):
        """Test that results are cached after query."""
        bus = QueryBus()
        cache = QueryCache()
        bus.set_cache(cache)

        @dataclass(frozen=True, slots=True, kw_only=True)
        class TestQuery(Query):
            search: str

        class TestHandler(QueryHandler):
            async def handle(self, query: TestQuery) -> str:
                return f"Result: {query.search}"

            def can_handle(self, query_type: type) -> bool:
                return query_type == TestQuery

            def get_cache_key(self, query: TestQuery) -> str:
                return f"search:{query.search}"

        handler = TestHandler()
        bus.register(TestQuery, handler)

        query = TestQuery(search="test", cache_key="search:test")
        await bus.dispatch(query)

        # Check that it was cached
        cached_entry = await cache.get("search:test")
        assert cached_entry is not None
        assert cached_entry.data == "Result: test"

    @async_test
    async def test_get_registered_queries(self):
        """Test getting list of registered queries."""
        bus = QueryBus()

        @dataclass(frozen=True, slots=True, kw_only=True)
        class Query1(Query):
            pass

        @dataclass(frozen=True, slots=True, kw_only=True)
        class Query2(Query):
            pass

        class Handler1(QueryHandler):
            async def handle(self, query):
                return "result1"
            def can_handle(self, query_type):
                return query_type == Query1

        class Handler2(QueryHandler):
            async def handle(self, query):
                return "result2"
            def can_handle(self, query_type):
                return query_type == Query2

        bus.register(Query1, Handler1())
        bus.register(Query2, Handler2())

        registered = bus.get_registered_queries()
        assert Query1 in registered
        assert Query2 in registered
        assert len(registered) == 2


class TestQueryIntegration:
    """Integration tests for query pattern."""

    @async_test
    async def test_complex_query_with_caching(self):
        """Test complex query with caching and pagination."""
        bus = QueryBus()
        cache = QueryCache()
        bus.set_cache(cache)

        @dataclass(frozen=True, slots=True, kw_only=True)
        class SearchRecordingsQuery(Query):
            search_term: str
            page: int
            page_size: int

        class SearchRecordingsHandler(QueryHandler):
            async def handle(self, query: SearchRecordingsQuery) -> dict:
                # Simulate search results
                results = [
                    {"id": i, "title": f"Recording {i}"}
                    for i in range(query.page_size)
                ]
                return {
                    "results": results,
                    "page": query.page,
                    "page_size": query.page_size,
                    "total_count": 100
                }

            def can_handle(self, query_type: type) -> bool:
                return query_type == SearchRecordingsQuery

            def get_cache_key(self, query: SearchRecordingsQuery) -> str:
                return f"search:{query.search_term}:page{query.page}"

        handler = SearchRecordingsHandler()
        bus.register(SearchRecordingsQuery, handler)

        query = SearchRecordingsQuery(
            search_term="jazz",
            page=1,
            page_size=10,
            cache_key="search:jazz:page1"
        )

        result = await bus.dispatch(query)

        assert result.success is True
        assert len(result.data["results"]) == 10
        assert result.data["total_count"] == 100

    @async_test
    async def test_query_cache_invalidation_flow(self):
        """Test query with cache invalidation."""
        bus = QueryBus()
        cache = QueryCache()
        bus.set_cache(cache)

        @dataclass(frozen=True, slots=True, kw_only=True)
        class GetRecordingQuery(Query):
            recording_id: str

        class GetRecordingHandler(QueryHandler):
            def __init__(self):
                self.version = 1

            async def handle(self, query: GetRecordingQuery) -> dict:
                return {
                    "id": query.recording_id,
                    "title": f"Recording {query.recording_id}",
                    "version": self.version
                }

            def can_handle(self, query_type: type) -> bool:
                return query_type == GetRecordingQuery

            def get_cache_key(self, query: GetRecordingQuery) -> str:
                return f"recording:{query.recording_id}"

        handler = GetRecordingHandler()
        bus.register(GetRecordingQuery, handler)

        # First query
        query1 = GetRecordingQuery(recording_id="rec-1", cache_key="recording:rec-1")
        result1 = await bus.dispatch(query1)

        assert result1.data["version"] == 1
        assert result1.from_cache is False

        # Invalidate cache
        await cache.invalidate(pattern="recording:")

        # Update data
        handler.version = 2

        # Second query after invalidation
        query2 = GetRecordingQuery(recording_id="rec-1", cache_key="recording:rec-1")
        result2 = await bus.dispatch(query2)

        assert result2.data["version"] == 2
        assert result2.from_cache is False


# Import dataclass for custom query tests
from dataclasses import dataclass
