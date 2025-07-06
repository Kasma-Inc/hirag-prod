"""
Tests for utility functions in hirag_prod._utils
"""

import asyncio
import time

import pytest

from hirag_prod._utils import _limited_gather_with_factory


class TestLimitedGatherWithFactory:
    """Test suite for _limited_gather_with_factory function"""

    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test basic concurrent execution of coroutine factories"""

        # Setup: Create simple async functions that return their input values
        async def async_double(x):
            await asyncio.sleep(0.1)  # Simulate some work
            return x * 2

        # Create factories for values 1-5
        factories = [lambda x=i: async_double(x) for i in range(1, 6)]

        # Execute with high concurrency limit
        results = await _limited_gather_with_factory(factories, limit=10)

        # Verify results are correct and in order
        assert results == [2, 4, 6, 8, 10]
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency limit is respected"""
        concurrent_count = 0
        max_concurrent = 0

        async def track_concurrency():
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.2)
            concurrent_count -= 1
            return "done"

        # Create 10 factories but limit concurrency to 3
        factories = [lambda: track_concurrency() for _ in range(10)]
        results = await _limited_gather_with_factory(factories, limit=3)

        # Verify concurrency was limited and all tasks completed
        assert max_concurrent <= 3
        assert len(results) == 10
        assert all(r == "done" for r in results)

    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test retry mechanism with temporary failures"""
        attempt_counts = {}

        async def flaky_function(task_id):
            attempt_counts[task_id] = attempt_counts.get(task_id, 0) + 1

            # Fail on first two attempts, succeed on third
            if attempt_counts[task_id] < 3:
                raise ValueError(f"Temporary failure for task {task_id}")
            return f"success_{task_id}"

        # Create factories for 3 tasks
        factories = [lambda i=i: flaky_function(i) for i in range(3)]

        results = await _limited_gather_with_factory(
            factories, limit=2, max_retries=3, retry_delay=0.01
        )

        # Verify all tasks eventually succeeded after retries
        assert results == ["success_0", "success_1", "success_2"]
        assert all(attempt_counts[i] == 3 for i in range(3))

    @pytest.mark.asyncio
    async def test_permanent_failure_handling(self):
        """Test handling of permanent failures after max retries"""

        async def always_fails():
            raise RuntimeError("This always fails")

        async def always_succeeds():
            return "success"

        # Mix of failing and succeeding factories
        factories = [
            lambda: always_fails(),
            lambda: always_succeeds(),
            lambda: always_fails(),
        ]

        results = await _limited_gather_with_factory(
            factories, limit=2, max_retries=2, retry_delay=0.01
        )

        # Verify failed tasks return None, successful tasks return their values
        assert results == [None, "success", None]

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff delay between retries"""
        retry_times = []

        async def failing_function():
            retry_times.append(time.time())
            raise ValueError("Always fails")

        factories = [lambda: failing_function()]

        await _limited_gather_with_factory(
            factories, limit=1, max_retries=3, retry_delay=0.1
        )

        # Verify exponential backoff: delays should be ~0.1, ~0.2, ~0.4 seconds
        assert len(retry_times) == 3
        if len(retry_times) >= 2:
            delay1 = retry_times[1] - retry_times[0]
            assert 0.08 <= delay1 <= 0.15  # ~0.1s with some tolerance
        if len(retry_times) >= 3:
            delay2 = retry_times[2] - retry_times[1]
            assert 0.15 <= delay2 <= 0.25  # ~0.2s with some tolerance

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test handling of empty factory list"""
        results = await _limited_gather_with_factory([], limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_single_factory(self):
        """Test with single factory function"""

        async def single_task():
            return "single_result"

        factories = [lambda: single_task()]
        results = await _limited_gather_with_factory(factories, limit=1)

        assert results == ["single_result"]

    @pytest.mark.asyncio
    async def test_mixed_execution_times(self):
        """Test factories with different execution times"""

        async def fast_task():
            await asyncio.sleep(0.05)
            return "fast"

        async def slow_task():
            await asyncio.sleep(0.2)
            return "slow"

        # Mix fast and slow tasks
        factories = [
            lambda: fast_task(),
            lambda: slow_task(),
            lambda: fast_task(),
            lambda: slow_task(),
        ]

        start_time = time.time()
        results = await _limited_gather_with_factory(factories, limit=2)
        execution_time = time.time() - start_time

        # Should complete in roughly the time of the slower tasks
        assert results.count("fast") == 2
        assert results.count("slow") == 2
        assert 0.15 <= execution_time <= 0.35  # Account for concurrency

    @pytest.mark.asyncio
    async def test_coroutine_factory_isolation(self):
        """Test that each factory creates independent coroutines"""
        shared_state = {"counter": 0}

        async def increment_counter(increment):
            # Each coroutine should get its own increment value
            await asyncio.sleep(0.01)
            shared_state["counter"] += increment
            return shared_state["counter"]

        # Create factories with different increment values
        factories = [lambda inc=i: increment_counter(inc) for i in [1, 2, 3]]

        results = await _limited_gather_with_factory(factories, limit=3)

        # Verify each factory used its own increment value
        assert shared_state["counter"] == 6  # 1 + 2 + 3
        assert len(set(results)) == 3  # All results should be different


if __name__ == "__main__":
    pytest.main([__file__])
