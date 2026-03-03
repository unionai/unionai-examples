"""
DynamicBatcher — maximize resource utilization by batching work from many
concurrent producers through a single async processing function.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Awaitable,
    Callable,
    Generic,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

RecordT = TypeVar("RecordT")
ResultT = TypeVar("ResultT")


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class CostEstimator(Protocol):
    def estimate_cost(self) -> int: ...


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ProcessFn = Callable[[list[RecordT]], Awaitable[list[ResultT]]]
CostEstimatorFn = Callable[[RecordT], int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_future() -> asyncio.Future:
    return asyncio.get_running_loop().create_future()


@dataclass
class _Envelope(Generic[RecordT, ResultT]):
    record: RecordT
    estimated_cost: int
    future: asyncio.Future[ResultT] = field(default_factory=_make_future)


# ---------------------------------------------------------------------------
# BatchStats
# ---------------------------------------------------------------------------


@dataclass
class BatchStats:
    total_submitted: int = 0
    total_completed: int = 0
    total_batches: int = 0
    total_batch_cost: int = 0
    avg_batch_size: float = 0.0
    avg_batch_cost: float = 0.0
    busy_time_s: float = 0.0
    idle_time_s: float = 0.0

    @property
    def utilization(self) -> float:
        total = self.busy_time_s + self.idle_time_s
        return self.busy_time_s / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# DynamicBatcher
# ---------------------------------------------------------------------------


class DynamicBatcher(Generic[RecordT, ResultT]):
    """Batches records from many concurrent producers and runs them through
    a single async processing function, maximizing resource utilization.

    The batcher runs two internal loops:

    1. **Aggregation loop** — drains the submission queue and assembles
       cost-budgeted batches, respecting ``target_batch_cost``,
       ``max_batch_size``, and ``batch_timeout_s``.
    2. **Processing loop** — pulls assembled batches and calls
       ``process_fn``, resolving each record's :class:`asyncio.Future`.

    Args:
        process_fn:
            ``async def f(batch: list[RecordT]) -> list[ResultT]``
            Must return results in the **same order** as the input batch.
        cost_estimator:
            Optional ``(RecordT) -> int`` function. Falls back to
            ``record.estimate_cost()`` if the record implements
            :class:`CostEstimator`, then to ``default_cost``.
        target_batch_cost:
            Cost budget per batch. The aggregator fills batches up to
            this limit before dispatching.
        max_batch_size:
            Hard cap on records per batch regardless of cost budget.
        min_batch_size:
            Minimum records before dispatching. Ignored when the timeout
            fires or shutdown is in progress.
        batch_timeout_s:
            Maximum seconds to wait for a full batch.
        max_queue_size:
            Bounded queue size. When full, :meth:`submit` awaits
            (backpressure).
        prefetch_batches:
            Number of pre-assembled batches to buffer between the
            aggregation and processing loops.
        default_cost:
            Fallback cost when no estimator is available.
    """

    def __init__(
        self,
        process_fn: ProcessFn[RecordT, ResultT],
        *,
        cost_estimator: CostEstimatorFn[RecordT] | None = None,
        target_batch_cost: int = 32_000,
        max_batch_size: int = 256,
        min_batch_size: int = 1,
        batch_timeout_s: float = 0.05,
        max_queue_size: int = 5_000,
        prefetch_batches: int = 2,
        default_cost: int = 1,
    ):
        self._process_fn = process_fn
        self._cost_estimator = cost_estimator
        self._target_batch_cost = target_batch_cost
        self._max_batch_size = max_batch_size
        self._min_batch_size = min_batch_size
        self._batch_timeout_s = batch_timeout_s
        self._prefetch_batches = prefetch_batches
        self._default_cost = default_cost

        self._queue: asyncio.Queue[_Envelope[RecordT, ResultT] | None] = asyncio.Queue(
            maxsize=max_queue_size,
        )
        self._batch_queue: asyncio.Queue[list[_Envelope[RecordT, ResultT]] | None] = asyncio.Queue(
            maxsize=prefetch_batches,
        )
        self._stats = BatchStats()
        self._running = False
        self._aggregator_task: asyncio.Task | None = None
        self._consumer_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    @property
    def stats(self) -> BatchStats:
        return self._stats

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            raise RuntimeError(f"{type(self).__name__} is already running")
        self._running = True
        self._shutdown_event.clear()
        self._aggregator_task = asyncio.create_task(
            self._aggregation_loop(), name="dynamic-batcher-aggregator"
        )
        self._consumer_task = asyncio.create_task(
            self._processing_loop(), name="dynamic-batcher-consumer"
        )
        logger.info("%s started", type(self).__name__)

    async def stop(self) -> None:
        if not self._running:
            return
        self._shutdown_event.set()
        await self._queue.put(None)
        if self._aggregator_task:
            await self._aggregator_task
        if self._consumer_task:
            await self._consumer_task
        self._running = False
        logger.info(
            "%s stopped — %d records in %d batches, utilization %.1f%%",
            type(self).__name__,
            self._stats.total_completed,
            self._stats.total_batches,
            self._stats.utilization * 100,
        )

    async def submit(
        self,
        record: RecordT,
        *,
        estimated_cost: int | None = None,
    ) -> asyncio.Future[ResultT]:
        if not self._running:
            raise RuntimeError(f"{type(self).__name__} is not running. Call start() or use 'async with'.")
        cost = self._estimate_cost(record, estimated_cost)
        envelope: _Envelope[RecordT, ResultT] = _Envelope(record=record, estimated_cost=cost)
        await self._queue.put(envelope)
        self._stats.total_submitted += 1
        return envelope.future

    async def submit_batch(
        self,
        records: Sequence[RecordT],
        *,
        estimated_cost: Sequence[int] | None = None,
    ) -> list[asyncio.Future[ResultT]]:
        futures = []
        for i, record in enumerate(records):
            cost = estimated_cost[i] if estimated_cost is not None else None
            f = await self.submit(record, estimated_cost=cost)
            futures.append(f)
        return futures

    async def __aenter__(self) -> DynamicBatcher[RecordT, ResultT]:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    def _estimate_cost(self, record: RecordT, override: int | None) -> int:
        if override is not None:
            return override
        if self._cost_estimator is not None:
            return self._cost_estimator(record)
        if isinstance(record, CostEstimator):
            return record.estimate_cost()
        return self._default_cost

    async def _aggregation_loop(self) -> None:
        while True:
            batch: list[_Envelope[RecordT, ResultT]] = []
            cost_count = 0

            envelope = await self._queue.get()
            if envelope is None:
                if batch:
                    await self._batch_queue.put(batch)
                await self._batch_queue.put(None)
                return

            batch.append(envelope)
            cost_count += envelope.estimated_cost

            deadline = time.monotonic() + self._batch_timeout_s
            while cost_count < self._target_batch_cost and len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    envelope = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if envelope is None:
                    if batch:
                        await self._batch_queue.put(batch)
                    await self._batch_queue.put(None)
                    return
                batch.append(envelope)
                cost_count += envelope.estimated_cost

            if len(batch) >= self._min_batch_size or self._shutdown_event.is_set():
                await self._batch_queue.put(batch)
            else:
                for env in batch:
                    await self._queue.put(env)
                await asyncio.sleep(self._batch_timeout_s)

    async def _processing_loop(self) -> None:
        idle_start = time.monotonic()
        while True:
            batch = await self._batch_queue.get()
            self._stats.idle_time_s += time.monotonic() - idle_start

            if batch is None:
                return

            records = [env.record for env in batch]
            batch_cost = sum(env.estimated_cost for env in batch)
            busy_start = time.monotonic()
            try:
                results = await self._process_fn(records)
                if len(results) != len(batch):
                    raise ValueError(
                        f"process_fn returned {len(results)} results for batch of {len(batch)}"
                    )
                for envelope, result in zip(batch, results):
                    if not envelope.future.done():
                        envelope.future.set_result(result)
            except Exception as exc:
                logger.error("%s batch failed: %s", type(self).__name__, exc)
                for envelope in batch:
                    if not envelope.future.done():
                        envelope.future.set_exception(exc)

            self._stats.busy_time_s += time.monotonic() - busy_start
            self._stats.total_batches += 1
            self._stats.total_completed += len(batch)
            self._stats.total_batch_cost += batch_cost
            self._stats.avg_batch_size = self._stats.total_completed / self._stats.total_batches
            self._stats.avg_batch_cost = self._stats.total_batch_cost / self._stats.total_batches
            idle_start = time.monotonic()


__all__ = [
    "BatchStats",
    "CostEstimator",
    "CostEstimatorFn",
    "DynamicBatcher",
    "ProcessFn",
]