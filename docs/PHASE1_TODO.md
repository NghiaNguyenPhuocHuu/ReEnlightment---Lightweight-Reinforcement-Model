# Phase 1 — firm up the DSSP simulation

Goal: make the FIFO / SPR / EDFQ comparison **statistically trustworthy** and
check that the simulator reproduces the paper's *trends*, not just one table.

Work on a branch: `git switch -c phase1-firm-up`. Tick boxes as you go, run
`make test` before each commit, one commit per task.

Timings measured on this machine (single run, default queries):
testbed 0.8 s (fifo) / 2.3 s (edfq) · sim-small 0.7 / 1.9 s · sim-medium 3.1 / 5.8 s.
A `maxrate` search is ~10–12 runs per policy.

---

## 1. Multi-seed max rate with confidence intervals  ⭐ start here

Today `maxrate` uses one seed, so we don't know if +16.9 % over SPR is real or noise.

- [ ] In [experiments.py](../reenlightment/dssp/experiments.py) add
      `max_rate_over_seeds(scenario, policy, seeds, n_queries=None)` → list of rates.
- [ ] Add a small stats helper: mean, sample std, 95 % CI.
      With n ≤ 30 seeds use the Student-t critical value, not 1.96
      (hint: hard-code t₀.₉₇₅ for df = 1..30; no scipy needed).
- [ ] Report **paired gains**: for each seed compute `edfq / fifo − 1` on the
      *same seed*, then CI over those ratios. Pairing removes shared noise and
      gives much tighter intervals than comparing two means.
- [ ] CLI: `maxrate --seeds 0-9` (or `--n-seeds 10`), print mean ± CI per policy
      and the paired-gain CIs.
- [ ] Test: with `--seeds 0` the result equals the current single-seed value;
      the CI half-width shrinks as seeds increase (use small `n_queries`).
- [ ] `make maxrate-ci` target.

**Done when:** README reference table shows `mean ± 95 % CI` over ≥ 10 seeds,
and you can say whether EDFQ > SPR is significant on `testbed`.
Commit: `feat(dssp): multi-seed max rate with confidence intervals`

## 2. Check the measurement itself (resolution & convergence)

- [ ] **Bisection resolution:** `max_rate` stops at `rel_tol=0.02` and starts from
      `1.0`, which is why results look quantised (13.00, 14.75, 17.25). Expose
      `--tol` and confirm the ranking doesn't change at 0.5 %.
- [ ] **Run length:** class 1.1 only has ~1 800 measured queries at 20 000 total,
      so its p99 rests on ~18 samples above the tail. Re-run `maxrate` at
      `--queries 10000 / 20000 / 40000` and record how much the max rate moves.
      Pick the smallest length where it stabilises (± 2 %).
- [ ] **Warm-up:** `warmup_fraction=0.1` is a guess. Plot (or print) p99 over
      successive windows of a run near the max rate; confirm 10 % is past the
      transient.

**Done when:** you've written the chosen tol / queries / warm-up into the README
with one sentence of justification each.
Commit: `chore(dssp): justify run length, warm-up and bisection tolerance`

## 3. Speed up EDFQ budget computation (needed for task 4)

Profile (`python3 -m cProfile -s cumtime -m reenlightment.dssp run --scenario sim-small`)
shows ~60 % of EDFQ time in `unloaded_task_tail` → 3.5 M calls to `cdf()`.

Ideas, simplest first — measure after each:
- [ ] For `kt == 1` the tail is just the node's own pt-quantile: index the sorted
      window directly instead of bisecting.
- [ ] Bisect over the *candidate sample values* instead of 40 continuous
      iterations (the answer is always one of the samples).
- [ ] Cache each node's quantiles and invalidate only every N new samples
      (profiles change slowly; the paper's prototype refreshes a histogram).
- [ ] Optional: `pypy3` runs the unchanged code several times faster.

- [ ] Guard rail test: the optimised `unloaded_task_tail` matches the old one
      within 1 % on random profiles (keep the old version in the test as reference).

**Done when:** `sim-medium` edfq run is ≥ 2× faster and `make test` passes.
Commit: `perf(dssp): faster unloaded task tail`

## 4. Scale trend — does EDFQ's advantage grow with size? (paper Table VI)

Paper: EDFQ gain over FIFO grows 31 % → 53 % → 80 % → 110 % → 145 % as the
system grows from 100 to 18 000 nodes. This is the strongest check that the
simulator captures DSSP's behaviour.

- [ ] Add a scale study, e.g. `experiments.scale_study(sizes, seeds)` using
      `table_v_scenario(n_iads, nodes_per_iad)` for sizes like
      (4×25), (10×25), (20×50), (40×50) — go as far as runtime allows.
- [ ] CLI `scale` subcommand → CSV `results/scale.csv`
      (size, policy, mean rate, CI, paired gain vs FIFO and vs SPR).
- [ ] Compare the *trend* to Table VI. If gain does **not** grow, write down
      why you think so (hotspot placement? fanout mapping μ/1000? service
      distribution?) — that's a finding, not a failure.

**Done when:** a table/plot of gain vs. total nodes is in the README.
Commit: `feat(dssp): scale study vs paper Table VI`

## 5. Sensitivity / robustness of the ranking

All of these are *our* assumptions (see README "Assumptions"). Show the ranking
survives them.

- [ ] CLI `--set key=value` that overrides `Scenario` fields
      (hint: `dataclasses.replace(scenario, **overrides)`, cast to the field's type).
- [ ] Sweep one knob at a time on `testbed`, 5+ seeds each:
  - [ ] `slow_fraction` ∈ {0, 0.25, 0.5}
  - [ ] `service_exp_ms` ∈ {20, 40, 80} (lighter ↔ heavier tail)
  - [ ] `net_mean_ms` ∈ {0, 3, 20} (paper assumes network is negligible)
  - [ ] `profile_window` ∈ {50, 500, 5000} (needs a Simulator kwarg on the CLI)
- [ ] Record where (if anywhere) EDFQ stops winning.

Commit: `feat(dssp): scenario overrides and sensitivity results`

## 6. Plots

- [ ] `scripts/plot.py` reading the CSVs: p99-vs-rate per class & policy
      (with the SLO as a horizontal line), and gain-vs-scale.
      matplotlib is fine as a **dev-only** dependency — keep the simulator itself
      dependency-free (`pip install matplotlib`, import it only in the script).
- [ ] Save PNGs to `results/` (git-ignored) and copy the final ones you want to
      keep into `docs/figures/`.
- [ ] `make plots` target.

Commit: `feat: plotting script for sweep and scale results`

## 7. Write-up & wrap

- [ ] README "Reference results": replace single-seed numbers with CI tables,
      add figures, scale trend, sensitivity summary.
- [ ] Update "Assumptions and limitations" with what tasks 2 & 5 taught you.
- [ ] `make test` green, merge `phase1-firm-up` into `main`, tag `phase1`.

Commit: `docs: phase 1 results`

---

### Suggested order
1 → 2 → 3 → 4 → 5 → 6 → 7. Tasks 1–2 are quick and change how you read every
later number; 3 unblocks 4. 5 and 6 can be done in parallel.

### Where things live
| concern | file |
|---|---|
| Eq. 1–6 budget math | `reenlightment/dssp/budget.py` |
| event loop, policies | `reenlightment/dssp/sim.py` |
| presets / `Scenario` fields | `reenlightment/dssp/scenarios.py` |
| sweeps, max-rate search | `reenlightment/dssp/experiments.py` |
| CLI | `reenlightment/dssp/__main__.py` |
| tests | `tests/test_dssp_*.py` |
