# Phase 0 — Learn the codebase

A self-study guide to ReEnlightment before starting
[Phase 1](../docs/PHASE1_TODO.md). Budget about **2 hours**: ~45 min reading,
the rest on the exercises. Every exercise is safe to try. Undo any experiment
with `git checkout -- <file>`.

**Contents**
1. [The big picture](#1-the-big-picture)
2. [The DSSP vocabulary](#2-the-dssp-vocabulary)
3. [Discrete-event simulation in 5 minutes](#3-discrete-event-simulation-in-5-minutes)
4. [Life of one query](#4-life-of-one-query-simpy)
5. [The budget math](#5-the-budget-math-budgetpy)
6. [Queuing policies: FIFO vs SPR vs EDFQ](#6-queuing-policies-fifo-vs-spr-vs-edfq)
7. [Scenarios, experiments, CLI](#7-scenarios-experiments-cli)
8. [The RL half](#8-the-rl-half-older-separate)
9. [Tests](#9-tests)
10. [Exercises](#10-exercises)
11. [Self-check questions](#11-self-check-questions)
12. [Glossary](#12-glossary)

---

## 1. The big picture

The repository has **two halves that don't import each other yet**:

```
reenlightment/
├── dssp/                    ← Phase 1: DSSP simulator (focus now)
│   ├── scenarios.py         WHAT to simulate    topology + workload presets
│   ├── budget.py            the paper's MATH    Eq. 1–6: SLO → subtask deadlines
│   ├── sim.py               the ENGINE          discrete-event simulator + policies
│   ├── experiments.py       what to MEASURE     sweeps, max sustainable rate, CSV
│   └── __main__.py          COMMAND LINE        python -m reenlightment.dssp ...
├── agent.py                 RL: tail-latency (p99) agent
├── q_agent.py               RL: tabular Q-learning agent
├── env.py                   RL toy env: pick a network path
├── research_env.py          RL toy envs: research DAG, DSSPSensorEnv, DSSPBudgetEnv
└── train.py                 RL entry point: python -m reenlightment.train ...
tests/                       one file per module; `make test` runs them all
docs/PHASE1_TODO.md          Phase 1 checklist
learn/                       you are here
Makefile                     shortcuts: make test / run / sweep / maxrate
```

How the DSSP modules depend on each other (arrows mean "imports"):

```
__main__.py ──► experiments.py ──► sim.py ──► budget.py
     │                               ▲
     └──────────► scenarios.py ──────┘ (Scenario objects are passed in)
```

`budget.py` has no dependencies. It's pure math, so it's the easiest file
to start reading.

**Phase 2** will connect the halves by wrapping `Simulator` in an RL
environment, so an agent can learn to tune DSSP.

---

## 2. The DSSP vocabulary

DSSP comes from *DSSP: An SLO-aware Multi-Domain-based Sensing-as-a-Service
Platform*, by Che, Sun, Wang and Jiang (PerCom 2026 Workshops; preprint
[arXiv:2308.11985](https://arxiv.org/abs/2308.11985)).

**The setting:** many independent organisations own sensors. A cloud
provider sells "sensing as a service". A user asks a question such as
"temperature over this area for the last hour", and the answer has to come
back fast: *99 % of queries in under 500 ms*.

| Term | Meaning | In the code |
|---|---|---|
| **CSP** | Cloud service provider. It receives the user's query and dispatches it | Implicit: the dispatch logic in `Simulator._arrive` |
| **IAD** | Independent sensing administrative domain: one organisation's pool of edge nodes | `sim.iads[i]`, a list of `EdgeNode` |
| **Edge node** | A device (Raspberry Pi) with sensors, one queue and one or more pods | `EdgeNode` |
| **Pod** | A worker on a node that executes one subtask at a time | `EdgeNode.pods`, `EdgeNode.busy` |
| **Query** | One user request; belongs to a class | `_Query` |
| **Task** | The part of a query sent to one IAD | `_Task` |
| **Subtask** | The part of a task sent to one edge node | `_Subtask` |
| **kq / kt** | Query fanout (number of IADs) / task fanout (nodes per IAD) | `QueryClass.iad_fanout`, `.node_fanout` |
| **SLO** | Service-level objective: "p99 latency ≤ X ms" per class | `QueryClass.slo_ms`, `.percentile` |

```
                   ┌──► IAD 1 ──► node a, node b         (task 1: kt = 2)
query ──► CSP ─────┼──► IAD 2 ──► node c                 (task 2: kt = 1)
       (kq = 3)    └──► IAD 3 ──► node d, node e, node f (task 3: kt = 3)
```

> **The one idea to remember:** a query finishes when its **slowest** subtask
> finishes. With 32 subtasks, one slow node ruins the whole query, so DSSP is
> about controlling **tail** latency, not average latency.

---

## 3. Discrete-event simulation in 5 minutes

A naive simulator advances a clock 1 ms at a time and checks everything. A
**discrete-event** simulator instead jumps straight from one interesting
moment to the next:

```python
events = []                                   # a min-heap ordered by time
heapq.heappush(events, (t, seq, kind, obj))   # schedule something in the future
while events:
    now, _, kind, obj = heapq.heappop(events) # jump to the earliest event
    handle(kind, obj)                         # handling may push new events
```

Why `seq`? Two events can have the same time. `seq` is an ever-increasing
counter that breaks ties in insertion order, so Python never has to compare
two `obj`s, which would crash. Look for `self._seq = itertools.count()` in
`sim.py`.

Nothing actually waits. "A subtask spends 120 ms executing" just means
"push a `_DONE` event at `now + 120`".

---

## 4. Life of one query ([sim.py](../reenlightment/dssp/sim.py))

There are four event kinds, each with one handler:

```
_ARRIVAL ──► _arrive()    pick class, IADs, nodes; [EDFQ] compute budget;
   │                      draw each subtask's times; schedule next arrival
   │  + network delay out
_ENQUEUE ──► _enqueue()   put subtask in node's queue with the policy's sort key
   │                      then _start(): if a pod is free, begin the best subtask
   │  + service time
_DONE    ──► _done()      free the pod; add (net_out+service+net_back) to the
   │                      node's profile; _start() the next queued subtask
   │  + network delay back
_RETURN  ──► _return()    task.pending -= 1; when 0 → query.pending -= 1;
                          when 0 → record latency = now − query.arrival
```

Read the methods in this order, which is the order events happen:

1. **`run(n_queries)`**: the main loop. Arrivals are a Poisson process: the
   gap between queries is `expovariate(rate)`, with a mean of `1000/rate` ms.
   Only one future arrival is in the heap at any time; each arrival schedules
   the next.
2. **`_arrive(now, measured)`**:
   - `_choose_class()` picks a class with probability equal to its `share`.
   - `_fanout()` draws `kq` (and later `kt`) from a normal distribution and
     clamps it to `[1, available]`.
   - `_pick()` chooses which IADs and nodes: random, or a "middle" block that
     creates hotspots.
   - For EDFQ only, it computes the queuing budget `tq` once per task (§5).
   - Each subtask's `net_out`, `service` and `net_back` are drawn **up front**.
     That makes runs reproducible and keeps the handlers simple.
3. **`_enqueue(now, st)`**: the only place the policies differ (§6).
4. **`_start(now, node)`**: while there's a free pod and a waiting subtask,
   pop the lowest key. For EDFQ, a subtask popped after its deadline counts
   as a *deadline miss*.
5. **`_done(now, st)`**: the profile update here is what makes the budgets
   **adaptive**: nodes that get slower produce bigger unloaded tails.
6. **`_return(now, st)`**: countdown counters. `pending` starts at the
   number of children, and the parent completes when it reaches 0.

The **first 10 % of queries are warm-up** (`measured=False`): they run but
aren't recorded, because the queues start empty and unrealistically fast.

At the end, `_result()` builds a `SimResult`:

```
SimResult
├── classes: {name → ClassStats(count, tail_ms (p99), mean_ms, slo_ms, met)}
├── meets_slo          True if every class met its SLO
├── utilization        busy pod-time / total pod-time
├── deadline_miss_rate EDFQ only
└── infeasible_rate    EDFQ only: tasks whose budget was negative before any queuing
```

### Two kinds of randomness

| What | Controlled by | Why |
|---|---|---|
| Which nodes are slow (the "hardware") | `Scenario.layout_seed` | Fixed, so every run sees the same testbed |
| Traffic, service times, fanouts | `seed` argument | Changes per run; Phase 1 task 1 averages over many seeds |

---

## 5. The budget math ([budget.py](../reenlightment/dssp/budget.py))

This is DSSP's main contribution. It turns *"p99 query latency ≤ SLO"* into
*"this subtask must start by time X"*, which a single node can act on.

### Step 1: query percentile → task percentile (Eq. 1–2), `task_percentile`

A query waits for all `kq` tasks. If each task independently meets
percentile `pt`, all of them do with probability `(pt/100)^kq`. To make that
equal `pq/100`:

```
pt = 100 · (pq/100)^(1/kq)        p99, kq=4  →  p99.749     p99, kq=100 → p99.99
```

Every task gets the query's whole SLO as its budget, but must hit a
stricter percentile.

### Step 2: unloaded task tail (Eq. 3–4), `unloaded_task_tail`

How long would this task take **with no queuing at all**, at percentile `pt`?

- Each node keeps an `UnloadedProfile`, a sliding window (default 500) of
  its recent `net_out + service + net_back` times. That's an empirical CDF
  `Fᵢ(t)`. The window is pre-filled by "offline profiling" in
  `Simulator.__init__`.
- A task finishes when its slowest subtask does, so its unloaded CDF is the
  product: `G(t) = F₁(t) · F₂(t) · … · F_kt(t)`.
- `unloaded_task_tail` **bisects** on `t` until `G(t) = pt/100`. That point is
  `x_u`.

### Step 3: queuing budget and deadline (Eq. 5–6), `queuing_budget`

```
tQ = SLO − x_u           (clamped at 0; negative means "infeasible")
deadline = enqueue_time + tQ
```

### Worked example (real numbers, `testbed`, IAD4)

| Class | kq | kt | pt | x_u (unloaded tail) | SLO | tQ (queuing budget) |
|---|---|---|---|---|---|---|
| 1.2 | 1 | 1 | 99.000 | 210 ms | 500 | **290 ms** |
| 2   | 4 | 1 | 99.749 | 242 ms | 800 | **558 ms** |
| 3   | 4 | 8 | 99.749 | 443 ms | 1200 | **757 ms** |

How to read it: class 3 needs a higher percentile (step 1) **and** waits for
the slowest of 8 nodes (step 2), so its unloaded tail is double class 1.2's.
Its SLO is still much looser, so it ends up with more slack to wait in the
queue. You'll reproduce this table in Exercise 3.

---

## 6. Queuing policies: FIFO vs SPR vs EDFQ

All three use the same heap per node and differ only in the **sort key**
built in `_enqueue`:

| Policy | Key | Behaviour |
|---|---|---|
| `fifo` | `now` | First come, first served. Ignores SLOs entirely. |
| `spr` | `(class rank, now)` | Strict priority: the tightest SLO always goes first. |
| `edfq` | `now + tq` (deadline) | DSSP: the most urgent subtask goes first, judged by its remaining budget. |

Why EDFQ wins (look at `make run RATE=20`):
- **FIFO** makes the tight 500 ms classes wait behind loose 1200 ms work, so
  class 1.x misses its SLO first.
- **SPR** fixes that but **starves** class 3. Under load, class 3 can wait
  forever, because higher classes always cut in.
- **EDFQ** lets class 1.2 cut in only while it is actually more urgent. As a
  class-3 subtask waits, its deadline gets closer and it eventually wins.

Reference result (`make maxrate`, seed 0): FIFO 13.0, SPR 14.75, **EDFQ
17.25** queries/s. The paper reports 13.23 / 15.73 / 17.98. This is a
single seed; Phase 1 task 1 checks whether the gap holds up across seeds.

---

## 7. Scenarios, experiments, CLI

### [scenarios.py](../reenlightment/dssp/scenarios.py): only data

- `QueryClass`: name, share, SLO, percentile, fanouts, and optionally
  `pinned_iads` (class 1.2 always goes to IAD4, which makes it a hotspot).
- `Scenario`: topology (`n_iads`, `nodes_per_iad`), service model
  (`speed × (service_base_ms + Exp(service_exp_ms))`), network, slow nodes,
  placement, and a default run length.
- Presets: `testbed` (paper Table II), `sim-small` and `sim-medium` (paper
  Table V, built by `table_v_scenario`).

### [experiments.py](../reenlightment/dssp/experiments.py): loops over `simulate()`

- `sweep`: every (policy, rate) pair.
- `max_rate`: **double** the rate until the SLO breaks, then **bisect**
  between the last pass and the first failure. It reuses the same seed for
  every probe ("common random numbers"), so the policies face identical
  traffic.
- `write_csv`: one row per run, with a column per class.

### [\_\_main\_\_.py](../reenlightment/dssp/__main__.py): argparse only

No logic: it parses arguments → calls the functions above → prints.
Subcommands: `scenarios`, `run`, `sweep`, `maxrate`. The [Makefile](../Makefile)
wraps them as `make scenarios / run / sweep / maxrate`.

---

## 8. The RL half (older, separate)

| File | What | Run |
|---|---|---|
| [agent.py](../reenlightment/agent.py) | `TailLatencyAgent`: learns each action's p99 by **quantile regression** (nudges the estimate up by `τ` when a sample is above it, down by `1−τ` when below) and picks the lowest | `python3 -m reenlightment.train` |
| [env.py](../reenlightment/env.py) | `TransportEnv`: 3 paths; the fastest-on-average one has the worst tail | (same) |
| [q_agent.py](../reenlightment/q_agent.py) | `QLearningAgent`: textbook tabular Q-learning, ε-greedy | — |
| [research_env.py](../reenlightment/research_env.py) | `ResearchNetworkEnv` (task DAG), and `DSSPSensorEnv` / `DSSPBudgetEnv`: **toy** DSSP models with made-up latency formulas | `python3 -m reenlightment.train research / dssp / dssp-pipeline` |

`DSSPSensorEnv` doesn't use the real simulator. In Phase 2 it would be
replaced by an environment whose `step()` advances `Simulator` and whose
reward comes from real SLO misses.

---

## 9. Tests

| File | What it guarantees |
|---|---|
| [test_dssp_budget.py](../tests/test_dssp_budget.py) | Eq. 2 matches the paper's numbers; the profile window evicts correctly; the unloaded tail matches the known math (max of n uniforms) |
| [test_dssp_sim.py](../tests/test_dssp_sim.py) | Same seed gives the same result; every query completes; light load meets the SLO and overload violates it; SPR favours tight classes |
| [test_dssp_experiments.py](../tests/test_dssp_experiments.py) | `max_rate` returns a rate that really passes; CSV has the right columns; CLI runs |
| [test_agent.py](../tests/test_agent.py) | The quantile estimate converges; the RL agent beats the baselines on p99 |

Run `make test` (all), `make test-dssp`, or a single file:
`python3 -m unittest tests.test_dssp_sim -v`.

---

## 10. Exercises

Do them in order. Each one takes 5–15 minutes.

**Ex 1: First runs.**
```
make scenarios
make run RATE=10
make run RATE=20
```
Which policy fails at 20 q/s, and which class causes it? Compare class 3's
p99 under `spr` and `edfq`.

**Ex 2: Follow one subtask.** In `Simulator._enqueue`, temporarily add this
just before the `heapq.heappush(...)` line (after the deadline is set):
```python
if self.policy == "edfq" and self._subtasks < 10:
    print(f"t={now:8.1f} class={st.task.query.cls.name:4s} node={node.iad}/{node.index} "
          f"tq={st.tq:6.1f} deadline={st.deadline:8.1f}")
```
Run `python3 -m reenlightment.dssp run --queries 50`. Do the `tq` values
match the table in §5? Why are they not *exactly* equal? (Hint: which nodes
were picked.) Revert afterwards.

**Ex 3: Reproduce the budget table.** In `python3`:
```python
from reenlightment.dssp.scenarios import get_scenario
from reenlightment.dssp.sim import Simulator
from reenlightment.dssp.budget import task_percentile, unloaded_task_tail, queuing_budget
sim = Simulator(get_scenario("testbed"), "edfq", 15, seed=0)
iad4 = sim.iads[3]
print([n.speed for n in iad4])
pt = task_percentile(99, 4)
xu = unloaded_task_tail([n.profile for n in iad4], pt)
print(pt, xu, queuing_budget(1200, xu))
```
Then compute `x_u` for a **single slow node** versus a **single fast node**.
How much budget does a slow node cost a subtask?

**Ex 4: Break a policy on purpose.** In `_enqueue`, make `spr` rank the
*loosest* SLO first (reverse the rank). Run `make run RATE=15` and explain
what happens to classes 1.1 and 1.2. Revert.

**Ex 5: Your own scenario.** In `scenarios.py`, copy `TESTBED` to a
`TESTBED_NO_HOTSPOT` where class 1.2 is not pinned, register it in
`SCENARIOS`, and run `python3 -m reenlightment.dssp maxrate --scenario <name>`.
Does EDFQ's advantage grow or shrink without the hotspot? Why?

**Ex 6: Write a test.** Add a test to `tests/test_dssp_sim.py` checking that
at very light load (rate 1), `deadline_miss_rate == 0` for EDFQ. Run
`make test-dssp`. Commit it. This one is worth keeping.

---

## 11. Self-check questions

<details><summary>1. Why does a query with kq = 4 need each task at p99.749 rather than p99?</summary>

The query is late if *any* of its 4 tasks is late. Assuming independence,
P(all 4 on time) = pt⁴. Solving pt⁴ = 0.99 gives pt = 0.99^(1/4) ≈ 0.99749.
</details>

<details><summary>2. Why does the simulator draw a subtask's service time at dispatch instead of when it starts?</summary>

Reproducibility, and it keeps the handlers simple. The draws happen in the
same order regardless of queue state, so a given seed gives identical
results. It also lets `_done` know `net_back` without another random call.
</details>

<details><summary>3. What would happen to EDFQ budgets if `UnloadedProfile` included queuing time?</summary>

Under load, the "unloaded" tail would grow with the queues, so `tQ` would
shrink, so every subtask would look urgent and deadlines would stop
discriminating between classes. EDFQ would collapse towards FIFO. That's
why `_done` records `net_out + service + net_back` only.
</details>

<details><summary>4. Why does max_rate use the same seed for every probe?</summary>

Common random numbers. Every rate and every policy sees the same random
stream (as far as possible), so a pass/fail difference comes from the
policy, not from luck. It reduces variance in comparisons.
</details>

<details><summary>5. Why discard the first 10 % of queries?</summary>

The simulation starts with empty queues, which is unrealistically fast.
Counting those queries would make p99 look better than the steady state.
Phase 1 task 2 checks whether 10 % is enough.
</details>

<details><summary>6. What does `infeasible_rate > 0` mean?</summary>

For some tasks, the unloaded tail alone already exceeds the SLO (`x_u >
SLO`). Even with zero queuing, that task would miss its percentile. The
budget is clamped to 0, so the subtask is maximally urgent, but no
scheduler can save it. It signals that the hardware or the SLO is wrong, not
the policy.
</details>

<details><summary>7. Where would you plug in a learned (RL) scheduler?</summary>

In `Simulator._enqueue`: it decides the sort key. For example, an agent
could output a per-IAD multiplier on `tq`, or a whole new key. The rest of
the simulator stays unchanged.
</details>

---

## 12. Glossary

| Term | Meaning |
|---|---|
| p99 / tail latency | The latency that 99 % of queries beat; the slowest 1 % sits above it |
| SLO | Service-level objective: a target such as "p99 ≤ 500 ms" |
| Fanout | How many parallel children a request splits into |
| Unloaded time | Response time without queuing (network + execution) |
| Budget decomposition | Splitting an end-to-end SLO into per-component targets |
| EDF / EDFQ | Earliest deadline first (queuing) |
| SPR | Strict priority |
| Max sustainable rate | Highest arrival rate at which every class still meets its SLO |
| Common random numbers | Reusing seeds across configurations to make comparisons fairer |
| Poisson arrivals | Exponential gaps between queries; a memoryless "random traffic" model |
| Warm-up | Initial period excluded from measurement |

Next: [docs/PHASE1_TODO.md](../docs/PHASE1_TODO.md).
