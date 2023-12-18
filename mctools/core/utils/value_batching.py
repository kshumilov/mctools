from __future__ import annotations

import heapq

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    'Batch',
    'batch_values',
    'split_batch',
    'split_batches',
    'combine_batches',
    'merge_batches',

    'construct_from_arrays',
    'print_batches',
]


@dataclass
class Batch:
    n: int  # Number of values in the batch
    lb: float  # Lower bound of the batch, such that ∀v, v >= lb
    ub: float  # Upper bound of the batch, such that ∀v, v <  ub
    gap: float  # Distance between highest state of this batch and lowest state of next batch

    @property
    def width(self) -> float:
        return self.ub - self.lb

    def is_adjacent(self, other: 'Batch') -> bool:
        return np.allclose(self.ub, other.lb) or np.allclose(other.lb, self.ub)

    def get_n(self, energy: npt.NDArray) -> int:
        return self.get_idx(energy).sum()

    def get_idx(self, energy: npt.NDArray) -> npt.NDArray:
        return (self.lb <= energy) & (energy < self.ub)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n:>5d}, lb={self.lb:>11.5f}, width={self.width:>11.5f}, gap={self.gap:>11.5f})"


LB = np.s_[:-1, np.newaxis]
UB = np.s_[1:, np.newaxis]


def batch_values(values: npt.ArrayLike, min_diff, /, n_batch_start: int = 0, max_batch_size: int = 96,
                 max_iter: int = 1):
    if n_batch_start < 0:
        raise ValueError(f'Number of start batches must be non-negative: n_batch_start = {n_batch_start} < 0')

    values = np.asarray(values)

    dv0 = (values.max() - values.min()) / values.shape[0] * 4.0
    lb0 = values.min() - dv0  # -np.inf
    ub0 = values.max() + dv0  # +np.inf

    dv = np.diff(values)
    dv_idx, *_ = np.where(dv > min_diff)

    upper_bounds = (values[dv_idx + 1] + values[dv_idx]) / 2
    upper_gaps = (values[dv_idx + 1] - values[dv_idx]) / 2

    idx = np.argsort(upper_gaps)[::-1]
    upper_gaps = upper_gaps[idx]
    upper_bounds = upper_bounds[idx]
    max_batches = len(upper_gaps)

    iter = 0
    max_n = len(values)
    n_batches = n_batch_start
    while max_n > max_batch_size and iter < max_iter and n_batches - 1 < max_batches:
        # Select 'n_batches' biggest gaps between states
        selected = np.s_[:n_batches - 1 if n_batches > 0 else max_batches]
        gaps = upper_gaps[selected]
        bounds = upper_bounds[selected]

        # Sort bounds in ascending order
        idx = np.argsort(bounds)
        bounds = bounds[idx]
        gaps = gaps[idx]

        # Pad bounds with edge bounds
        bounds = np.pad(bounds, 1, constant_values=(lb0, ub0))
        gaps = np.pad(gaps, (0, 1), constant_values=np.inf)

        # Assign each value to a batch
        batch_assign = (bounds[LB] <= values) & (values < bounds[UB])
        n = batch_assign.sum(axis=1)  # Calculate number of values per batch

        max_n = n.max()
        n_batches += 1
        iter += 1

    return n, bounds, gaps


def split_batch(batch: Batch, values: npt.ArrayLike, idx: int | None = None) -> tuple[Batch, Batch]:
    v = values[batch.get_idx(values)]

    if idx is None:
        idx = np.diff(v).argmax()

    mb = (v[idx + 1] + v[idx]) / 2
    gap = (v[idx + 1] - v[idx]) / 2
    n_l = ((batch.lb <= v) & (v < mb)).sum()
    b_l = batch._replace(n=n_l, ub=mb, gap=gap)

    n_u = ((mb <= v) & (v < batch.ub)).sum()
    b_u = batch._replace(n=n_u, lb=mb)

    return b_l, b_u


def split_batches(batches: list[Batch], values: npt.ArrayLike,
                  max_batch_size: int = 96, min_gap: float = 0.0005) -> list[Batch]:
    excluded = []

    values = np.asarray(values)

    heap = [(-b.n, b) for b in batches]
    heapq.heapify(heap)
    while -heap[0][0] > max_batch_size:
        _, batch = heapq.heappop(heap)

        v = values[batch.get_idx(values)]
        dv = np.diff(v)
        i = dv.argmax()
        if dv[i] < min_gap:
            excluded.append(batch)
            continue

        b_l, b_u = split_batch(batch, v, idx=i)
        heapq.heappush(heap, (-b_l.n, b_l))
        heapq.heappush(heap, (-b_u.n, b_u))

    new_batches = list((b for _, b in heap))
    new_batches.extend(excluded)

    return sorted(new_batches, key=lambda b: b.lb)


def combine_batches(batches: list[Batch], max_batch_size: int = 96, min_gap: float = 0.0005) -> list[Batch]:
    ready = []
    while batches:
        current = batches.pop()
        if current.n > 96:
            ready.append(current)
            continue

        while batches:
            # print(f'Found batch: {current}', end=' - ')

            other = batches.pop()
            if (current.n + other.n) <= max_batch_size and other.gap < min_gap:
                # print('merged')
                current = merge_batches(other, current)
            else:
                # print('ready')
                ready.append(current)
                batches.append(other)
                break
    else:
        ready.append(current)

    return ready[::-1]


def construct_from_arrays(n, bounds, gaps, fmt='dict') -> dict[int, Batch] | list[Batch]:
    if fmt == 'dict':
        batches: dict[int, Batch] = {}
        for i, (n, lb, ub, g) in enumerate(zip(n, bounds[:-1], bounds[1:], gaps)):
            batches[i] = Batch(n, lb, ub, g)
    else:
        batches: list[Batch] = []
        for i, (n, lb, ub, g) in enumerate(zip(n, bounds[:-1], bounds[1:], gaps)):
            batches.append(Batch(n, lb, ub, g))

    return batches


def print_batches(batches: dict[int, Batch] | list[Batch]) -> None:
    if isinstance(batches, dict):
        n = sum(b.n for b in batches.values())
        iter = batches.items()
    else:
        n = sum(b.n for b in batches)
        iter = enumerate(batches)

    print(f'Total of {n} values in {len(batches)} batches'.center(4 + 65 + 3))
    print('=' * (4 + 65 + 3))
    print('ID'.center(4), 'Batch'.center(65), sep=' | ')
    print('-' * (4 + 65 + 3))

    for i, b in iter:
        print(f'{i:4d}', b, sep=' | ')


def merge_batches(b1: Batch, b2: Batch) -> Batch:
    if not b1.is_adjacent(b2):
        raise ValueError(f'Cannot merge non-adjacent batches: {b1!r} and {b2!r}')

    lb = min(b1.lb, b2.lb)
    ub = max(b1.ub, b2.ub)
    n = b1.n + b2.n

    if ub == b1.ub:
        gap = b1.gap
    else:
        gap = b2.gap

    return Batch(n, lb, ub, gap)
