"""DSSP two-tier SLO budget decomposition (Che et al., arXiv:2308.11985).

Upper tier (query -> task): a query fans out to kq IADs and finishes with its
slowest task, so a query tail percentile pq maps to a per-task percentile pt
via order statistics (Eq. 1-2). Every task shares the query's latency budget.

Lower tier (task -> subtask): a task fans out to edge nodes and finishes with
its slowest subtask. The unloaded task response time distribution is the
product of the per-node unloaded subtask CDFs (Eq. 3); its pt-quantile (Eq. 4)
is what the task costs without queuing, so the rest of the budget is the
queuing budget tQ (Eq. 6) and each subtask gets deadline t0 + tQ (Eq. 5).

Pure Python, no dependencies.
"""
import bisect
from collections import deque


def query_percentile(pt, kq):
    """Eq. 1: query percentile reached when each of kq tasks meets percentile pt."""
    return 100.0 * (pt / 100.0) ** kq


def task_percentile(pq, kq):
    """Eq. 2: per-task percentile needed so kq parallel tasks meet query percentile pq."""
    if kq < 1:
        raise ValueError("kq must be >= 1")
    return 100.0 * (pq / 100.0) ** (1.0 / kq)


class UnloadedProfile:
    """Sliding-window empirical CDF of one edge node's unloaded subtask response time.

    "Unloaded" = network + execution, excluding time spent waiting in the queue.
    The prototype in the paper keeps a histogram over a moving window; we keep
    the raw samples sorted, which is the same CDF without binning error.
    """

    def __init__(self, window=500):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._fifo = deque()
        self._sorted = []

    def __len__(self):
        return len(self._sorted)

    def add(self, x):
        self._fifo.append(x)
        bisect.insort(self._sorted, x)
        if len(self._fifo) > self.window:
            old = self._fifo.popleft()
            del self._sorted[bisect.bisect_left(self._sorted, old)]

    def cdf(self, t):
        n = len(self._sorted)
        # No samples yet: assume the node costs nothing, i.e. never constrains the task.
        return bisect.bisect_right(self._sorted, t) / n if n else 1.0

    def max(self):
        return self._sorted[-1] if self._sorted else 0.0


def unloaded_task_tail(profiles, pt, iters=40):
    """Eq. 3-4: x^u_pt = Gu^-1(pt/100) with Gu(t) = prod_i F^u_i(t), by bisection on t."""
    target = pt / 100.0
    lo, hi = 0.0, max((p.max() for p in profiles), default=0.0)
    if hi <= 0.0:
        return 0.0
    for _ in range(iters):  # invariant: Gu(hi) >= target
        mid = 0.5 * (lo + hi)
        g = 1.0
        for p in profiles:
            g *= p.cdf(mid)
            if g < target:
                break
        if g >= target:
            hi = mid
        else:
            lo = mid
    return hi


def queuing_budget(task_budget, unloaded_tail):
    """Eq. 6: tQ = x_pt - x^u_pt, clamped at 0 (a negative budget means infeasible)."""
    return max(0.0, task_budget - unloaded_tail)
