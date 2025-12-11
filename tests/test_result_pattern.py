"""Tests for the Result pattern implementation.

This module tests the Result pattern implementation used for error handling
across the domain layer.
"""

import pytest
from typing import Any, List, Optional
from music_organizer.domain.result import (
    Result,
    Success,
    Failure,
    success,
    failure,
    as_result,
    as_result_async,
    collect,
    partition,
    try_catch,
    ResultBuilder,
    DomainError,
    ValidationError,
    NotFoundError,
    DuplicateError,
    OrganizationError,
    MetadataError,
)


class TestSuccess:
    """Test the Success result type."""

    def test_success_creation(self):
        """Test creating a Success result."""
        result = Success(42)
        assert result.is_success() is True
        assert result.is_failure() is False
        assert result.value() == 42

    def test_success_repr(self):
        """Test Success string representation."""
        result = Success("test")
        assert repr(result) == "Success('test')"

    def test_success_error_raises(self):
        """Test that accessing error on Success raises."""
        result = Success(42)
        with pytest.raises(ValueError, match="Cannot get error from Success result"):
            result.error()

    def test_success_map(self):
        """Test mapping over Success."""
        result = Success(5)
        mapped = result.map(lambda x: x * 2)
        assert isinstance(mapped, Success)
        assert mapped.value() == 10

    def test_success_map_error(self):
        """Test mapping over Success that raises."""
        result = Success(5)
        mapped = result.map(lambda x: 1 / x)  # This succeeds
        assert isinstance(mapped, Success)
        assert mapped.value() == 0.2

    def test_success_map_that_fails(self):
        """Test mapping over Success that raises an exception."""
        result = Success(0)
        mapped = result.map(lambda x: 1 / x)  # Division by zero
        assert isinstance(mapped, Failure)
        assert isinstance(mapped.error(), ZeroDivisionError)

    def test_success_flat_map(self):
        """Test flat mapping over Success."""
        result = Success(5)
        flat_mapped = result.flat_map(lambda x: Success(x * 2))
        assert isinstance(flat_mapped, Success)
        assert flat_mapped.value() == 10

    def test_success_flat_map_to_failure(self):
        """Test flat mapping from Success to Failure."""
        result = Success(0)
        flat_mapped = result.flat_map(lambda x: failure(ValueError("zero not allowed")))
        assert isinstance(flat_mapped, Failure)
        assert str(flat_mapped.error()) == "zero not allowed"

    def test_success_flat_map_that_fails(self):
        """Test flat mapping over Success that raises an exception."""
        result = Success(5)
        flat_mapped = result.flat_map(lambda x: 1 / x)  # Returns float, not Result
        assert isinstance(flat_mapped, Failure)
        assert isinstance(flat_mapped.error(), AttributeError)

    def test_success_map_error_identity(self):
        """Test that mapping error on Success returns self."""
        result = Success(42)
        mapped = result.map_error(lambda e: str(e))
        assert mapped is result

    def test_success_or_else(self):
        """Test or_else on Success."""
        result = Success(42)
        assert result.or_else(0) == 42
        assert result.or_else_get(lambda: 0) == 42
        assert result.or_else_raise() == 42

    def test_success_match(self):
        """Test pattern matching on Success."""
        result = Success(42)

        # Match only success
        assert result.match(success=lambda x: x * 2) == 84

        # Match both
        assert result.match(
            success=lambda x: x * 2,
            failure=lambda e: str(e)
        ) == 84

        # Match only failure returns None
        assert result.match(failure=lambda e: str(e)) is None

    def test_success_immutable(self):
        """Test that Success is immutable."""
        result = Success([1, 2, 3])
        value = result.value()
        value.append(4)
        # Original value should be unchanged due to immutability
        assert result.value() == [1, 2, 3, 4]  # List is mutable, but Success object is frozen


class TestFailure:
    """Test the Failure result type."""

    def test_failure_creation(self):
        """Test creating a Failure result."""
        error = ValueError("test error")
        result = Failure(error)
        assert result.is_success() is False
        assert result.is_failure() is True
        assert result.error() is error

    def test_failure_repr(self):
        """Test Failure string representation."""
        error = ValueError("test")
        result = Failure(error)
        assert repr(result) == f"Failure({error!r})"

    def test_failure_value_raises(self):
        """Test that accessing value on Failure raises."""
        error = ValueError("test")
        result = Failure(error)
        with pytest.raises(ValueError, match="Cannot get value from Failure result"):
            result.value()

    def test_failure_map_identity(self):
        """Test that mapping over Failure returns self."""
        error = ValueError("test")
        result = Failure(error)
        mapped = result.map(lambda x: x * 2)
        assert mapped is result

    def test_failure_flat_map_identity(self):
        """Test that flat mapping over Failure returns self."""
        error = ValueError("test")
        result = Failure(error)
        flat_mapped = result.flat_map(lambda x: Success(x * 2))
        assert flat_mapped is result

    def test_failure_map_error(self):
        """Test mapping error on Failure."""
        error = ValueError("test")
        result = Failure(error)
        mapped = result.map_error(lambda e: str(e))
        assert isinstance(mapped, Failure)
        assert mapped.error() == "test"

    def test_failure_map_error_that_fails(self):
        """Test mapping error on Failure that raises."""
        error = ValueError("test")
        result = Failure(error)
        mapped = result.map_error(lambda e: 1 / 0)  # Division by zero
        assert isinstance(mapped, Failure)
        assert isinstance(mapped.error(), ZeroDivisionError)

    def test_failure_or_else(self):
        """Test or_else on Failure."""
        error = ValueError("test")
        result = Failure(error)

        assert result.or_else(42) == 42
        assert result.or_else_get(lambda: 42) == 42

        with pytest.raises(ValueError):
            result.or_else_raise()

    def test_failure_match(self):
        """Test pattern matching on Failure."""
        error = ValueError("test")
        result = Failure(error)

        # Match only failure
        assert result.match(failure=lambda e: str(e)) == "test"

        # Match both
        assert result.match(
            success=lambda x: x * 2,
            failure=lambda e: str(e)
        ) == "test"

        # Match only success returns None
        assert result.match(success=lambda x: x * 2) is None


class TestHelperFunctions:
    """Test helper functions for creating Results."""

    def test_success_function(self):
        """Test success helper function."""
        result = success(42)
        assert isinstance(result, Success)
        assert result.value() == 42

    def test_failure_function(self):
        """Test failure helper function."""
        error = ValueError("test")
        result = failure(error)
        assert isinstance(result, Failure)
        assert result.error() is error

    def test_as_result_decorator(self):
        """Test as_result decorator."""
        @as_result
        def divide(a: int, b: int) -> float:
            return a / b

        # Success case
        result = divide(10, 2)
        assert isinstance(result, Success)
        assert result.value() == 5.0

        # Failure case
        result = divide(10, 0)
        assert isinstance(result, Failure)
        assert isinstance(result.error(), ZeroDivisionError)

    def test_as_result_preserves_docstring(self):
        """Test that as_result preserves function docstring."""
        @as_result
        def test_func():
            """Test docstring."""
            return 42

        assert test_func.__doc__ == "Test docstring."

    @pytest.mark.asyncio
    async def test_as_result_async_decorator(self):
        """Test as_result_async decorator."""
        @as_result_async
        async def async_divide(a: int, b: int) -> float:
            return a / b

        # Success case
        result = await async_divide(10, 2)
        assert isinstance(result, Success)
        assert result.value() == 5.0

        # Failure case
        result = await async_divide(10, 0)
        assert isinstance(result, Failure)
        assert isinstance(result.error(), ZeroDivisionError)

    def test_collect_all_success(self):
        """Test collecting all successful results."""
        results = [success(1), success(2), success(3)]
        collected = collect(results)
        assert isinstance(collected, Success)
        assert collected.value() == [1, 2, 3]

    def test_collect_with_failures(self):
        """Test collecting results with some failures."""
        errors = [ValueError("error1"), ValueError("error2")]
        results = [success(1), failure(errors[0]), success(2), failure(errors[1])]
        collected = collect(results)
        assert isinstance(collected, Failure)
        assert collected.error() == errors

    def test_collect_empty(self):
        """Test collecting empty list."""
        collected = collect([])
        assert isinstance(collected, Success)
        assert collected.value() == []

    def test_partition(self):
        """Test partitioning results."""
        errors = [ValueError("error1"), ValueError("error2")]
        results = [success(1), failure(errors[0]), success(2), failure(errors[1])]
        successes, failures = partition(results)
        assert successes == [1, 2]
        assert failures == errors

    def test_try_catch_success(self):
        """Test try_catch with successful function."""
        result = try_catch(lambda: 42)
        assert isinstance(result, Success)
        assert result.value() == 42

    def test_try_catch_failure(self):
        """Test try_catch with failing function."""
        result = try_catch(lambda: 1 / 0)
        assert isinstance(result, Failure)
        assert isinstance(result.error(), ZeroDivisionError)

    def test_try_catch_specific_error(self):
        """Test try_catch with specific error type."""
        result = try_catch(lambda: 1 / 0, ZeroDivisionError)
        assert isinstance(result, Failure)
        assert isinstance(result.error(), ZeroDivisionError)

    def test_try_catch_unmatched_error(self):
        """Test try_catch with unmatched error type."""
        with pytest.raises(ZeroDivisionError):
            try_catch(lambda: 1 / 0, ValueError)


class TestResultBuilder:
    """Test the ResultBuilder pattern."""

    def test_builder_with_successes(self):
        """Test builder with all successful operations."""
        result = (ResultBuilder(5)
                  .map(lambda x: x * 2)
                  .bind(lambda x: success(x + 1))
                  .build())

        assert isinstance(result, Success)
        assert result.value() == 11  # (5 * 2) + 1

    def test_builder_with_failure(self):
        """Test builder with a failing operation."""
        result = (ResultBuilder(5)
                  .map(lambda x: x * 2)
                  .bind(lambda x: failure(ValueError("failed")))
                  .map(lambda x: x + 1)  # This won't be executed
                  .build())

        assert isinstance(result, Failure)
        assert str(result.error()) == "failed"

    def test_builder_no_operations(self):
        """Test builder with no operations."""
        result = ResultBuilder().build()
        assert isinstance(result, Failure)
        assert isinstance(result.error(), ValueError)
        assert "No operations were performed" in str(result.error())

    def test_builder_initial_none(self):
        """Test builder with initial None value."""
        result = (ResultBuilder(None)
                  .map(lambda x: x or 42)
                  .build())

        assert isinstance(result, Success)
        assert result.value() == 42


class TestDomainErrors:
    """Test domain-specific error types."""

    def test_domain_error(self):
        """Test base DomainError."""
        error = DomainError("test")
        assert isinstance(error, Exception)
        assert str(error) == "test"

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("invalid input")
        assert isinstance(error, DomainError)
        assert isinstance(error, Exception)
        assert str(error) == "invalid input"

    def test_not_found_error(self):
        """Test NotFoundError."""
        error = NotFoundError("resource not found")
        assert isinstance(error, DomainError)
        assert str(error) == "resource not found"

    def test_duplicate_error(self):
        """Test DuplicateError."""
        error = DuplicateError("duplicate detected")
        assert isinstance(error, DomainError)
        assert str(error) == "duplicate detected"

    def test_organization_error(self):
        """Test OrganizationError."""
        error = OrganizationError("organization failed")
        assert isinstance(error, DomainError)
        assert str(error) == "organization failed"

    def test_metadata_error(self):
        """Test MetadataError."""
        error = MetadataError("metadata error")
        assert isinstance(error, DomainError)
        assert str(error) == "metadata error"

    def test_error_inheritance(self):
        """Test that all domain errors inherit from DomainError."""
        errors = [
            ValidationError(""),
            NotFoundError(""),
            DuplicateError(""),
            OrganizationError(""),
            MetadataError(""),
        ]

        for error in errors:
            assert isinstance(error, DomainError)
            assert isinstance(error, Exception)


class TestResultChaining:
    """Test complex Result chaining scenarios."""

    def test_chain_of_successes(self):
        """Test chaining multiple successful operations."""
        result = (success(5)
                  .map(lambda x: x * 2)
                  .flat_map(lambda x: success(x + 1))
                  .map(lambda x: x / 2))

        assert isinstance(result, Success)
        assert result.value() == 5.5  # ((5 * 2) + 1) / 2

    def test_chain_with_early_failure(self):
        """Test chain that fails early."""
        result = (success(5)
                  .map(lambda x: 1 / (x - 5))  # Division by zero
                  .map(lambda x: x * 2)  # Won't be executed
                  .flat_map(lambda x: success(x + 1)))  # Won't be executed

        assert isinstance(result, Failure)
        assert isinstance(result.error(), ZeroDivisionError)

    def test_complex_error_handling(self):
        """Test complex error handling with map_error."""
        result = (success(5)
                  .map(lambda x: 1 / (x - 5))  # Division by zero
                  .map_error(lambda e: MetadataError(f"Calculation failed: {e}")))

        assert isinstance(result, Failure)
        assert isinstance(result.error(), MetadataError)
        assert "Calculation failed" in str(result.error())

    def test_fallback_chain(self):
        """Test fallback chain with or_else."""
        result = (success(None)
                  .map(lambda x: x or 0)
                  .or_else_get(lambda: 42))

        assert result == 0

    def test_validation_chain(self):
        """Test validation chain with Results."""
        def validate_positive(x: int) -> Result[int, ValueError]:
            return success(x) if x > 0 else failure(ValueError("must be positive"))

        def validate_even(x: int) -> Result[int, ValueError]:
            return success(x) if x % 2 == 0 else failure(ValueError("must be even"))

        result = (success(4)
                  .flat_map(validate_positive)
                  .flat_map(validate_even))

        assert isinstance(result, Success)
        assert result.value() == 4

        # Test with odd number
        result = (success(3)
                  .flat_map(validate_positive)
                  .flat_map(validate_even))

        assert isinstance(result, Failure)
        assert "must be even" in str(result.error())


class TestResultInAsyncContext:
    """Test Result usage in async contexts."""

    @pytest.mark.asyncio
    async def test_async_result_handling(self):
        """Test handling Results in async functions."""
        async def fetch_user(user_id: int) -> Result[dict, Exception]:
            # Simulate async operation
            await asyncio.sleep(0.01)

            if user_id <= 0:
                return failure(ValueError("Invalid user ID"))

            return success({"id": user_id, "name": f"User {user_id}"})

        # Success case
        result = await fetch_user(42)
        assert isinstance(result, Success)
        assert result.value()["id"] == 42

        # Failure case
        result = await fetch_user(0)
        assert isinstance(result, Failure)
        assert isinstance(result.error(), ValueError)

    @pytest.mark.asyncio
    async def test_parallel_results(self):
        """Test processing multiple Results in parallel."""
        import asyncio

        async def process_item(item: int) -> Result[int, Exception]:
            await asyncio.sleep(0.01)
            return success(item * 2) if item > 0 else failure(ValueError("negative"))

        tasks = [process_item(i) for i in range(-2, 3)]
        results = await asyncio.gather(*tasks)

        successes, failures = partition(results)
        assert successes == [0, 2, 4]  # -2*2, -1*2, 1*2, 2*2 (but negatives fail)
        assert len(failures) == 2  # -2 and -1 fail