"""Command line for the DSSP simulation: python -m reenlightment.dssp --help"""
import argparse

from .experiments import compare_max_rates, sweep, write_csv
from .scenarios import SCENARIOS, get_scenario
from .sim import POLICIES, simulate


def _policies(arg):
    return list(POLICIES) if arg == "all" else arg.split(",")


def print_result(r):
    print(f"[{r.scenario}] policy={r.policy} rate={r.rate:g} q/s  "
          f"utilization={r.utilization:.2f}  SLO {'MET' if r.meets_slo else 'VIOLATED'}")
    if r.policy == "edfq":
        print(f"  subtask deadline misses={r.deadline_miss_rate:.2%}  "
              f"infeasible task budgets={r.infeasible_rate:.2%}")
    print(f"  {'class':6s} {'queries':>8s} {'mean ms':>9s} {'tail ms':>9s} {'SLO ms':>8s}  ok")
    for c in r.classes.values():
        print(f"  {c.name:6s} {c.count:8d} {c.mean_ms:9.1f} {c.tail_ms:9.1f} "
              f"{c.slo_ms:8.0f}  {'yes' if c.met else 'NO'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m reenlightment.dssp",
                                 description="DSSP SLO-aware Sensing-as-a-Service simulation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scenarios", help="list scenario presets")

    def common(p, policy_default):
        p.add_argument("--scenario", default="testbed", choices=sorted(SCENARIOS))
        p.add_argument("--policy", default=policy_default,
                       help=f"one of {', '.join(POLICIES)}, comma list, or 'all'")
        p.add_argument("--queries", type=int, default=None,
                       help="queries per run (default: scenario preset)")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("run", help="simulate one arrival rate")
    common(p, "edfq")
    p.add_argument("--rate", type=float, default=10.0, help="query arrival rate (queries/s)")

    p = sub.add_parser("sweep", help="simulate a list of arrival rates")
    common(p, "all")
    p.add_argument("--rates", default="5,10,15,20,25", help="comma-separated queries/s")
    p.add_argument("--csv", help="write results to this CSV file")

    p = sub.add_parser("maxrate", help="max sustainable rate per policy (paper Table IV/VI)")
    common(p, "all")
    p.add_argument("--verbose", action="store_true", help="print every probe")

    a = ap.parse_args(argv)
    if a.cmd == "scenarios":
        for s in SCENARIOS.values():
            print(f"{s.name:11s} {s.description}  (default {s.default_queries} queries)")
            for c in s.classes:
                print(f"    class {c.name:4s} share={c.share:.2f}  p{c.percentile:g} SLO={c.slo_ms:g} ms  "
                      f"kq~N{c.iad_fanout}  kt~N{c.node_fanout}"
                      + (f"  pinned IADs={c.pinned_iads}" if c.pinned_iads else ""))
            for n in s.notes:
                print(f"    note: {n}")
        return

    scenario = get_scenario(a.scenario)
    policies = _policies(a.policy)
    if a.cmd == "run":
        for pol in policies:
            print_result(simulate(scenario, pol, a.rate, a.queries, a.seed))
    elif a.cmd == "sweep":
        rates = [float(x) for x in a.rates.split(",")]
        results = sweep(scenario, policies, rates, a.queries, a.seed)
        print(f"{'policy':6s} {'rate':>6s} {'util':>5s}  SLO  "
              + " ".join(f"{'p-tail ' + c.name:>10s}" for c in scenario.classes))
        for r in results:
            print(f"{r.policy:6s} {r.rate:6g} {r.utilization:5.2f}  {'ok ' if r.meets_slo else 'NO '}  "
                  + " ".join(f"{c.tail_ms:10.0f}" for c in r.classes.values()))
        if a.csv:
            write_csv(a.csv, results)
            print(f"wrote {a.csv}")
    else:
        rates = compare_max_rates(scenario, policies, a.queries, a.seed, a.verbose)
        print(f"[{scenario.name}] max sustainable query rate (all class SLOs met)")
        for pol, r in rates.items():
            print(f"  {pol:5s} {r:7.2f} q/s")
        if "edfq" in rates:
            for pol, r in rates.items():
                if pol != "edfq" and r > 0:
                    print(f"  EDFQ gain over {pol.upper()}: {100 * (rates['edfq'] / r - 1):+.1f}%")


if __name__ == "__main__":
    main()
