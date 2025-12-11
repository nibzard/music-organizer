"""Query side of CQRS pattern."""

from .base import Query, QueryHandler, QueryBus, QueryResult

__all__ = ["Query", "QueryHandler", "QueryBus", "QueryResult"]