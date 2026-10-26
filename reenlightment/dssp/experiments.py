"""Experiments on top of the simulator: rate sweeps and max sustainable rate.

The paper's headline metric is the maximum query arrival rate at which every
query class still meets its tail-latency SLO (Tables IV and VI).
"""
import csv

from .sim import simulate


def sweep(scenario, policies, rates, n_queries=None, seed=0):
    """Run every (policy, rate) pair; returns a list of SimResult."""
    return [simulate(scenario, p, r, n_queries, seed) for p in policies for r in rates]


def max_rate(scenario, policy, n_queries=None, seed=0, start=1.0, rel_tol=0.02, verbose=False):
    """Largest rate (queries/s) meeting all class SLOs, by doubling then bisection.

    Every probe uses the same seed (common random numbers), so the comparison
    between policies is not dominated by sampling noise.
    """
    def ok(rate):
        res = simulate(scenario, policy, rate, n_queries, seed)
        if verbose:
            print(f"  {policy:5s} rate={rate:7.2f}  {'meets' if res.meets_slo else 'violates'} SLO")
        return res.meets_slo

    lo, hi = 0.0, start
    while ok(hi):
        lo, hi = hi, hi * 2.0
    while hi - lo > rel_tol * hi:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def compare_max_rates(scenario, policies, n_queries=None, seed=0, verbose=False):
    """{policy: max sustainable rate in queries/s}."""
    rates = {p: max_rate(scenario, p, n_queries, seed, verbose=verbose) for p in policies}
    return rates


def write_csv(path, results):
    classes = list(results[0].classes) if results else []
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "policy", "rate_qps", "meets_slo", "utilization",
                    "deadline_miss_rate", "infeasible_rate"]
                   + [f"class_{c}_tail_ms" for c in classes]
                   + [f"class_{c}_slo_ms" for c in classes])
        for r in results:
            w.writerow([r.scenario, r.policy, r.rate, int(r.meets_slo), f"{r.utilization:.4f}",
                        f"{r.deadline_miss_rate:.4f}", f"{r.infeasible_rate:.4f}"]
                       + [f"{r.classes[c].tail_ms:.1f}" for c in classes]
                       + [r.classes[c].slo_ms for c in classes])
