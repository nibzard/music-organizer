"""Result pattern implementation for error handling.

This module implements the Result pattern, providing a type-safe way to handle
operations that can succeed or fail without relying on exceptions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, Union, cast, overload
from functools import wraps
import traceback

T = TypeVar('T')  # Success type
E = TypeVar('E', bound=Exception)  # Error type


class Result(ABC, Generic[T, E]):
    """Abstract base class for Result pattern.

    A Result represents either a successful operation with a value,
    or a failed operation with an error.
    """

    @abstractmethod
    def is_success(self) -> bool:
        """Check if the result is a success."""
        ...

    @abstractmethod
    def is_failure(self) -> bool:
        """Check if the result is a failure."""
        ...

    @abstractmethod
    def value(self) -> T:
        """Get the success value.

        Raises:
            ValueError: If the result is a failure.
        """
        ...

    @abstractmethod
    def error(self) -> E:
        """Get the error.

        Raises:
            ValueError: If the result is a success.
        """
        ...

    @abstractmethod
    def map(self, fn: Callable[[T], Any]) -> Result[Any, E]:
        """Map the success value through a function."""
        ...

    @abstractmethod
    def flat_map(self, fn: Callable[[T], Result[Any, E]]) -> Result[Any, E]:
        """Flat map the success value through a function that returns a Result."""
        ...

    @abstractmethod
    def map_error(self, fn: Callable[[E], Any]) -> Result[T, Any]:
        """Map the error through a function."""
        ...

    def or_else(self, default: T) -> T:
        """Get the success value or return a default."""
        return self.value() if self.is_success() else default

    def or_else_get(self, fn: Callable[[], T]) -> T:
        """Get the success value or compute a default."""
        return self.value() if self.is_success() else fn()

    def or_else_raise(self) -> T:
        """Get the success value or raise the error."""
        if self.is_failure():
            raise self.error()
        return self.value()

    @overload
    def match(self, *, success: Callable[[T], Any]) -> Any: ...

    @overload
    def match(self, *, failure: Callable[[E], Any]) -> Any: ...

    @overload
    def match(self, *, success: Callable[[T], Any], failure: Callable[[E], Any]) -> Any: ...

    def match(self, *, success: Callable[[T], Any] | None = None,
              failure: Callable[[E], Any] | None = None) -> Any:
        """Pattern match on the result."""
        if self.is_success() and success:
            return success(self.value())
        elif self.is_failure() and failure:
            return failure(self.error())
        return None


@dataclass(frozen=True, slots=True)
class Success(Result[T, E]):
    """Represents a successful operation with a value."""
    _value: T

    def __repr__(self) -> str:
        return f"Success({self._value!r})"

    def is_success(self) -> bool:
        return True

    def is_failure(self) -> bool:
        return False

    def value(self) -> T:
        return self._value

    def error(self) -> E:
        raise ValueError("Cannot get error from Success result")

    def map(self, fn: Callable[[T], Any]) -> Result[Any, E]:
        try:
            return Success(fn(self._value))
        except Exception as e:
            return Failure(cast(E, e))

    def flat_map(self, fn: Callable[[T], Result[Any, E]]) -> Result[Any, E]:
        try:
            return fn(self._value)
        except Exception as e:
            return Failure(cast(E, e))

    def map_error(self, fn: Callable[[E], Any]) -> Result[T, Any]:
        return self


@dataclass(frozen=True, slots=True)
class Failure(Result[T, E]):
    """Represents a failed operation with an error."""
    _error: E

    def __repr__(self) -> str:
        return f"Failure({self._error!r})"

    def is_success(self) -> bool:
        return False

    def is_failure(self) -> bool:
        return True

    def value(self) -> T:
        raise ValueError(f"Cannot get value from Failure result: {self._error}")

    def error(self) -> E:
        return self._error

    def map(self, fn: Callable[[T], Any]) -> Result[Any, E]:
        return self

    def flat_map(self, fn: Callable[[T], Result[Any, E]]) -> Result[Any, E]:
        return self

    def map_error(self, fn: Callable[[E], Any]) -> Result[T, Any]:
        try:
            return Failure(cast(Any, fn(self._error)))
        except Exception as e:
            return Failure(cast(E, e))


# Helper functions for creating Results
def success(value: T) -> Result[T, Any]:
    """Create a Success result."""
    return Success(value)


def failure(error: E) -> Result[Any, E]:
    """Create a Failure result."""
    return Failure(error)


def as_result(fn: Callable[..., T]) -> Callable[..., Result[T, Exception]]:
    """Decorator to convert function that raises exceptions to return Result.

    Example:
        @as_result
        def divide(a: int, b: int) -> float:
            return a / b

        result = divide(10, 0)  # Returns Failure(ZeroDivisionError(...))
    """
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Result[T, Exception]:
        try:
            return Success(fn(*args, **kwargs))
        except Exception as e:
            return Failure(e)
    return wrapper


async def as_result_async(fn: Callable[..., T]) -> Callable[..., Result[T, Exception]]:
    """Decorator to convert async function that raises exceptions to return Result."""
    @wraps(fn)
    async def wrapper(*args, **kwargs) -> Result[T, Exception]:
        try:
            result = await fn(*args, **kwargs)
            return Success(result)
        except Exception as e:
            return Failure(e)
    return wrapper


def collect(results: list[Result[T, E]]) -> Result[list[T], list[E]]:
    """Collect a list of Results into a single Result.

    If all results are Success, returns Success with a list of all values.
    If any results are Failure, returns Failure with a list of all errors.
    """
    values = []
    errors = []

    for result in results:
        if result.is_success():
            values.append(result.value())
        else:
            errors.append(result.error())

    return Success(values) if not errors else Failure(errors)


def partition(results: list[Result[T, E]]) -> tuple[list[T], list[E]]:
    """Partition a list of Results into successes and failures.

    Returns:
        A tuple of (success_values, failure_errors)
    """
    successes = []
    failures = []

    for result in results:
        if result.is_success():
            successes.append(result.value())
        else:
            failures.append(result.error())

    return successes, failures


def try_catch(fn: Callable[[], T], error_class: type[E] | tuple[type[E], ...] = Exception) -> Result[T, E]:
    """Try to execute a function and catch exceptions of specific type(s).

    Args:
        fn: Function to execute
        error_class: Exception class(es) to catch

    Returns:
        Success(value) if no exception, Failure(exception) if caught
    """
    try:
        return Success(fn())
    except error_class as e:
        return Failure(cast(E, e))


class ResultBuilder:
    """Builder pattern for chaining operations that might fail."""

    def __init__(self, initial_value: T | None = None):
        self._result = Success(initial_value) if initial_value is not None else None

    def bind(self, fn: Callable[[T], Result[Any, E]]) -> ResultBuilder:
        """Bind a function that returns a Result."""
        if self._result is None:
            self._result = Success(cast(T, None))

        if self._result.is_success():
            self._result = self._result.flat_map(fn)
        return self

    def map(self, fn: Callable[[T], Any]) -> ResultBuilder:
        """Map a function over the current value."""
        if self._result is None:
            self._result = Success(cast(T, None))

        if self._result.is_success():
            self._result = self._result.map(fn)
        return self

    def build(self) -> Result[T, E]:
        """Build the final Result."""
        if self._result is None:
            return Failure(cast(E, ValueError("No operations were performed")))
        return self._result


# Domain-specific errors for the music organizer
class DomainError(Exception):
    """Base class for domain-specific errors."""
    pass


class ValidationError(DomainError):
    """Raised when validation fails."""
    pass


class NotFoundError(DomainError):
    """Raised when a resource is not found."""
    pass


class DuplicateError(DomainError):
    """Raised when a duplicate is detected."""
    pass


class OrganizationError(DomainError):
    """Raised when file organization fails."""
    pass


class MetadataError(DomainError):
    """Raised when metadata operations fail."""
    pass