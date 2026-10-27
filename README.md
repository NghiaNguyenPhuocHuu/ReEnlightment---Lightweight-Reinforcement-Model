# ReEnlightment
Lightweight Reinforcement AI

Pure-Python (no dependencies, Python >= 3.9) experiments on SLO-aware data transport:

1. **DSSP simulation** (`reenlightment/dssp/`) is a discrete-event recreation of
   *DSSP: An SLO-aware Multi-Domain-based Sensing-as-a-Service Platform*
   (Che, Sun, Wang, Jiang; PerCom 2026 Workshops; preprint
   [arXiv:2308.11985](https://arxiv.org/abs/2308.11985)).
2. **RL agents** (`reenlightment/agent.py`, `q_agent.py`) include a tail-latency
   (p99) path-selection agent and a tabular Q-learning agent, plus toy DSSP RL
   environments.

## Quick start

    git clone <repo> && cd ReEnlightment
    python3 -m unittest discover -s tests        # ~5 s, all should pass
    python3 -m reenlightment.dssp scenarios      # list DSSP presets
    python3 -m reenlightment.dssp run --rate 15  # one simulation run
    python3 -m reenlightment.dssp maxrate        # FIFO vs SPR vs EDFQ (~40 s)

Nothing needs to be installed. Run commands from the repository root.

### One-command shortcuts (Makefile)

    make test                  # every unit test (verbose)
    make test-dssp             # only tests/test_dssp*.py
    make test-rl               # only the RL agent tests
    make run RATE=20           # one DSSP run, all policies
    make sweep SCENARIO=sim-small   # writes results/sweep-sim-small.csv
    make maxrate               # FIFO vs SPR vs EDFQ
    make demo                  # tests, then scenarios, run, maxrate
    make help                  # list targets

`SCENARIO` (default `testbed`), `RATE` (default `15`) and `PYTHON` (default
`python3`) can be overridden on any target. New tests are picked up
automatically if they live in `tests/` and are named `test_*.py`.

---

## Phase 1: DSSP data-transport simulation

### What is modelled

```
query (class, p99 SLO) ──► CSP ──fan-out kq──► IAD 1..kq ──fan-out kt──► edge nodes
                                                                │
                         latency = slowest task = slowest subtask    queue (FIFO | SPR | EDFQ)
                                                                │      └► pod executes
query done ◄── all tasks done ◄── all subtasks returned ◄────────┘
```

* **IAD** (independent sensing administrative domain) is a pool of edge nodes.
  Nodes are heterogeneous: a fixed `slow_fraction` of them run `slow_factor`
  times slower (in the testbed, Raspberry Pi 3 vs Pi 4).
* **Subtask time** is network out, then queueing, then execution, then network
  back. Execution is `speed * (base + Exp(mean))` ms and the network is
  exponential.
* **Budget decomposition** ([budget.py](reenlightment/dssp/budget.py)) follows
  the paper's two tiers:
  * Eq. 1–2: per-task percentile `pt = 100·(pq/100)^(1/kq)`. Every task gets
    the query SLO as its budget.
  * Eq. 3–4: unloaded task tail `x^u_pt = Gu⁻¹(pt/100)`, where `Gu(t) = ∏ F^u_i(t)`
    over the task's edge nodes. `F^u_i` is a sliding-window empirical CDF of
    node *i*'s queue-free response time. It is seeded by offline profiling and
    then updated online.
  * Eq. 5–6: queuing budget `tQ = x_pt − x^u_pt`, with subtask deadline
    `tD = t_enqueue + tQ`.
* **Queuing policies** ([sim.py](reenlightment/dssp/sim.py)):
  * `fifo` serves subtasks in arrival order.
  * `spr` uses strict priority by class, so the tightest SLO goes first.
  * `edfq` is DSSP's policy: the earliest subtask deadline goes first.
* **Metric:** per-class p99 query latency against the SLO, and the **max
  sustainable arrival rate** at which *all* classes meet their SLO. This is the
  paper's headline metric (Tables IV and VI).

### Scenarios ([scenarios.py](reenlightment/dssp/scenarios.py))

| preset | topology | workload | default queries |
|---|---|---|---|
| `testbed` | 4 IADs × 8 nodes | Paper Table II: classes 1.1/1.2/2/3, IAD4 hotspot | 20000 |
| `sim-small` | 4 IADs × 25 nodes | Paper Table V classes, hotspots in the middle of the list | 5000 |
| `sim-medium` | 20 IADs × 50 nodes | same as `sim-small` | 2000 (slow) |

### Commands

```
python3 -m reenlightment.dssp scenarios
python3 -m reenlightment.dssp run     [--scenario S] [--policy edfq|fifo|spr|all] [--rate Q] [--queries N] [--seed K]
python3 -m reenlightment.dssp sweep   [--scenario S] [--policy all] [--rates 5,10,15] [--csv out.csv]
python3 -m reenlightment.dssp maxrate [--scenario S] [--policy all] [--queries N] [--verbose]
```

* `run` prints per-class count, mean, p99, and SLO, plus utilisation. For
  EDFQ it also prints the subtask deadline-miss rate and the share of tasks
  whose budget was infeasible.
* `sweep` prints a policy × rate table and can write a CSV for plotting.
* `maxrate` doubles and then bisects the arrival rate for each policy, using
  common random numbers, and reports EDFQ's gain over each baseline.

### Reference results (seed 0, default settings)

`python3 -m reenlightment.dssp maxrate --scenario testbed`:

| policy | this sim (q/s) | paper Table IV (q/s) |
|---|---|---|
| FIFO | 13.00 | 13.23 |
| SPR  | 14.75 | 15.73 |
| EDFQ | 17.25 | 17.98 |
| EDFQ gain vs FIFO / SPR | +32.7% / +16.9% | +35.9% / +14.3% |

`python3 -m reenlightment.dssp maxrate --scenario sim-small`: FIFO 16.50,
SPR 13.00, EDFQ 17.25 (+4.5% / +32.7%).

### Assumptions and limitations (read before citing numbers)

* Absolute rates are **not calibrated** to the paper's hardware. Service time
  (60 ms + Exp(40 ms), about 100 ms unloaded as in the paper), slow-node share,
  and network delay are our choices. Only the relative ordering of the
  policies is meaningful.
* We map the Table V fanout parameters (μ = 100–700) to fractions of the
  IAD/node pool (μ/1000). The extracted parameters were ambiguous, so this
  mapping is our assumption.
* There is one pod per edge node and no autoscaling (the paper leaves
  autoscaling to future work). Discovery and registration (DRS) are not
  modelled, because they are off the data path.
* EDFQ under overload degrades like any EDF scheduler. Expect a domino effect
  above the max rate.

### Customising

* **New scenario:** add a `Scenario(...)` to `SCENARIOS` in
  [scenarios.py](reenlightment/dssp/scenarios.py), or build one with
  `table_v_scenario(n_iads, nodes_per_iad)`.
* **From Python:**
  ```python
  from reenlightment.dssp.scenarios import get_scenario
  from reenlightment.dssp.sim import simulate
  r = simulate(get_scenario("testbed"), "edfq", rate=15, n_queries=20000, seed=0)
  print(r.meets_slo, {c: s.tail_ms for c, s in r.classes.items()})
  ```
* **New queuing policy:** add its name to `POLICIES` and its queue key in
  `Simulator._enqueue` in [sim.py](reenlightment/dssp/sim.py). The key is
  where a learned (RL) scheduler can plug in later.

---

## RL agents and toy environments

    python3 -m reenlightment.train                # tail-latency agent vs baselines
    python3 -m reenlightment.train research       # toy task-oriented "research network"
    python3 -m reenlightment.train dssp           # DSSP sensors-only adaptive timeout + anomaly alerts
    python3 -m reenlightment.train dssp-pipeline  # older multi-stage DSSP toy

The tail-latency agent keeps an online tau-quantile estimate of latency per
(state, action) and acts epsilon-greedily on it.

## Project layout

```
reenlightment/
  agent.py, q_agent.py      RL agents
  env.py, research_env.py   toy RL environments (incl. DSSPSensorEnv, DSSPBudgetEnv)
  train.py                  RL training entry point
  dssp/
    budget.py               DSSP budget decomposition (Eq. 1-6)
    scenarios.py            topology/workload presets
    sim.py                  discrete-event simulator + FIFO/SPR/EDFQ
    experiments.py          sweeps, max-rate search, CSV
    __main__.py             CLI
tests/                      unittest suite
```

## Development workflow

* Run `make test` (or `python3 -m unittest discover -s tests`) before every commit.
* Commit one milestone or feature at a time, with a conventional prefix
  (`feat(dssp): ...`, `docs: ...`).
