"""Discrete-event simulator of DSSP query processing (pure Python).

Flow of one query:
  CSP receives query -> dispatches one task per selected IAD -> each IAD splits
  the task into subtasks on its edge nodes -> subtask travels to the node
  (network), waits in the node queue, executes on a pod, travels back.
  A task completes with its slowest subtask, a query with its slowest task.

Queuing policies at each edge node:
  fifo - first come, first served (budget-unaware baseline)
  spr  - strict priority by class SLO, tighter SLO first (budget-unaware baseline)
  edfq - DSSP: earliest subtask queuing deadline first, deadlines from budget.py
"""
import heapq
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List

from .budget import UnloadedProfile, queuing_budget, task_percentile, unloaded_task_tail

POLICIES = ("fifo", "spr", "edfq")

_ARRIVAL, _ENQUEUE, _DONE, _RETURN = range(4)


def percentile(xs, p):
    """Nearest-rank p-th percentile of xs (p in [0, 100])."""
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[max(0, math.ceil(p / 100.0 * len(xs)) - 1)]


class EdgeNode:
    __slots__ = ("iad", "index", "speed", "pods", "busy", "queue", "profile", "busy_ms")

    def __init__(self, iad, index, speed, pods, window):
        self.iad, self.index, self.speed, self.pods = iad, index, speed, pods
        self.busy = 0
        self.queue = []  # heap of (key, seq, subtask)
        self.profile = UnloadedProfile(window)
        self.busy_ms = 0.0


class _Query:
    __slots__ = ("cls", "arrival", "pending", "measured")


class _Task:
    __slots__ = ("query", "pending")


class _Subtask:
    __slots__ = ("task", "node", "tq", "deadline", "net_out", "service", "net_back")


@dataclass
class ClassStats:
    name: str
    count: int
    slo_ms: float
    percentile: float
    tail_ms: float
    mean_ms: float

    @property
    def met(self):
        return self.count > 0 and self.tail_ms <= self.slo_ms


@dataclass
class SimResult:
    scenario: str
    policy: str
    rate: float
    classes: Dict[str, ClassStats]
    utilization: float
    deadline_miss_rate: float  # edfq only: subtasks dequeued after their deadline
    infeasible_rate: float     # edfq only: tasks whose unloaded tail already exceeds budget
    sim_time_ms: float
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def meets_slo(self):
        return all(c.met for c in self.classes.values())


class Simulator:
    def __init__(self, scenario, policy="edfq", rate=10.0, seed=0,
                 profile_window=500, profile_samples=500, warmup_fraction=0.1):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if rate <= 0:
            raise ValueError("rate must be > 0 (queries per second)")
        self.scenario, self.policy, self.rate = scenario, policy, rate
        self.warmup_fraction = warmup_fraction
        self.rng = random.Random(seed)

        layout = random.Random(scenario.layout_seed)
        self.iads: List[List[EdgeNode]] = []
        for i in range(scenario.n_iads):
            nodes = []
            for j in range(scenario.nodes_per_iad):
                speed = scenario.slow_factor if layout.random() < scenario.slow_fraction else 1.0
                nodes.append(EdgeNode(i, j, speed, scenario.pods_per_node, profile_window))
            self.iads.append(nodes)

        # Offline profiling: seed each node's unloaded profile before serving queries.
        for nodes in self.iads:
            for node in nodes:
                for _ in range(profile_samples):
                    node.profile.add(self._net() + self._service(node) + self._net())

        # SPR: tighter SLO -> higher priority (lower rank).
        slos = sorted({c.slo_ms for c in scenario.classes})
        self._spr_rank = {c.name: slos.index(c.slo_ms) for c in scenario.classes}
        total = sum(c.share for c in scenario.classes)
        self._cum_shares = list(itertools.accumulate(c.share / total for c in scenario.classes))

        self._events = []
        self._seq = itertools.count()
        self._latencies = {c.name: [] for c in scenario.classes}
        self._subtasks = 0
        self._misses = 0
        self._tasks = 0
        self._infeasible = 0

    # -- sampling ---------------------------------------------------------
    def _service(self, node):
        s = self.scenario
        return node.speed * (s.service_base_ms + self.rng.expovariate(1.0 / s.service_exp_ms))

    def _net(self):
        half = self.scenario.net_mean_ms / 2.0
        return self.rng.expovariate(1.0 / half) if half > 0 else 0.0

    def _fanout(self, spec, available):
        mean, std = spec
        k = round(self.rng.gauss(mean, std)) if std > 0 else round(mean)
        return min(available, max(1, k))

    def _pick(self, n, k):
        if self.scenario.placement == "middle":
            # Contiguous block whose start is centred on the middle of the
            # ordered list: middle entries are hit most, creating hotspots.
            start = round(self.rng.gauss((n - k) / 2.0, n / 4.0))
            start = min(n - k, max(0, start))
            return range(start, start + k)
        return self.rng.sample(range(n), k)

    def _choose_class(self):
        r = self.rng.random()
        for cls, c in zip(self.scenario.classes, self._cum_shares):
            if r <= c:
                return cls
        return self.scenario.classes[-1]

    # -- event handling ---------------------------------------------------
    def _push(self, t, kind, obj):
        heapq.heappush(self._events, (t, next(self._seq), kind, obj))

    def _arrive(self, now, measured):
        cls = self._choose_class()
        q = _Query()
        q.cls, q.arrival, q.measured = cls, now, measured
        iads = cls.pinned_iads or self._pick(self.scenario.n_iads,
                                             self._fanout(cls.iad_fanout, self.scenario.n_iads))
        q.pending = len(iads)
        pt = task_percentile(cls.percentile, len(iads))
        for i in iads:
            pool = self.iads[i]
            nodes = [pool[j] for j in self._pick(len(pool), self._fanout(cls.node_fanout, len(pool)))]
            task = _Task()
            task.query, task.pending = q, len(nodes)
            tq = None
            if self.policy == "edfq":
                # All tasks share the query budget (x_pt = x_pq); the IAD subtracts
                # its unloaded task tail to get the queuing budget.
                x_u = unloaded_task_tail([n.profile for n in nodes], pt)
                tq = queuing_budget(cls.slo_ms, x_u)
                self._tasks += 1
                self._infeasible += cls.slo_ms < x_u
            for node in nodes:
                st = _Subtask()
                st.task, st.node, st.tq, st.deadline = task, node, tq, None
                st.net_out, st.service, st.net_back = self._net(), self._service(node), self._net()
                self._push(now + st.net_out, _ENQUEUE, st)

    def _enqueue(self, now, st):
        node = st.node
        if self.policy == "edfq":
            st.deadline = now + st.tq  # Eq. 5: tD = t0 + tQ
            key = st.deadline
        elif self.policy == "spr":
            key = (self._spr_rank[st.task.query.cls.name], now)
        else:
            key = now
        heapq.heappush(node.queue, (key, next(self._seq), st))
        self._start(now, node)

    def _start(self, now, node):
        while node.busy < node.pods and node.queue:
            _, _, st = heapq.heappop(node.queue)
            node.busy += 1
            self._subtasks += 1
            if st.deadline is not None and now > st.deadline:
                self._misses += 1
            self._push(now + st.service, _DONE, st)

    def _done(self, now, st):
        node = st.node
        node.busy -= 1
        node.busy_ms += st.service
        node.profile.add(st.net_out + st.service + st.net_back)
        self._push(now + st.net_back, _RETURN, st)
        self._start(now, node)

    def _return(self, now, st):
        task = st.task
        task.pending -= 1
        if task.pending:
            return
        q = task.query
        q.pending -= 1
        if q.pending == 0 and q.measured:
            self._latencies[q.cls.name].append(now - q.arrival)

    # -- main loop --------------------------------------------------------
    def run(self, n_queries=None):
        n_queries = n_queries or self.scenario.default_queries
        warmup = int(self.warmup_fraction * n_queries)
        mean_gap = 1000.0 / self.rate
        arrived = 0
        self._push(self.rng.expovariate(1.0 / mean_gap), _ARRIVAL, None)
        now = 0.0
        while self._events:
            now, _, kind, obj = heapq.heappop(self._events)
            if kind == _ARRIVAL:
                self._arrive(now, arrived >= warmup)
                arrived += 1
                if arrived < n_queries:
                    self._push(now + self.rng.expovariate(1.0 / mean_gap), _ARRIVAL, None)
            elif kind == _ENQUEUE:
                self._enqueue(now, obj)
            elif kind == _DONE:
                self._done(now, obj)
            else:
                self._return(now, obj)
        return self._result(now)

    def _result(self, end):
        s = self.scenario
        classes = {}
        for c in s.classes:
            lats = self._latencies[c.name]
            classes[c.name] = ClassStats(c.name, len(lats), c.slo_ms, c.percentile,
                                         percentile(lats, c.percentile),
                                         sum(lats) / len(lats) if lats else float("nan"))
        n_pods = s.n_iads * s.nodes_per_iad * s.pods_per_node
        busy = sum(n.busy_ms for nodes in self.iads for n in nodes)
        return SimResult(
            scenario=s.name,
            policy=self.policy,
            rate=self.rate,
            classes=classes,
            utilization=busy / (n_pods * end) if end > 0 else 0.0,
            deadline_miss_rate=self._misses / self._subtasks if self._subtasks else 0.0,
            infeasible_rate=self._infeasible / self._tasks if self._tasks else 0.0,
            sim_time_ms=end,
        )


def simulate(scenario, policy="edfq", rate=10.0, n_queries=None, seed=0, **kwargs):
    return Simulator(scenario, policy, rate, seed, **kwargs).run(n_queries)
