"""Workload/topology presets for the DSSP simulator. All times are in ms.

A query of a class fans out to kq IADs (query fanout); inside each IAD the task
fans out to kt edge nodes (task fanout). Fanouts are (mean, std) pairs drawn
from a normal distribution truncated to [1, available]; std 0 means fixed.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class QueryClass:
    name: str
    share: float                      # fraction of arriving queries
    slo_ms: float                     # tail-latency SLO
    iad_fanout: Tuple[float, float]   # kq ~ N(mean, std)
    node_fanout: Tuple[float, float]  # kt ~ N(mean, std)
    percentile: float = 99.0          # the SLO is on this percentile
    pinned_iads: Optional[Tuple[int, ...]] = None  # always go to these IADs (hotspot)


@dataclass
class Scenario:
    name: str
    description: str
    n_iads: int
    nodes_per_iad: int
    classes: List[QueryClass]
    service_base_ms: float = 60.0     # execution = speed * (base + Exp(service_exp_ms))
    service_exp_ms: float = 40.0
    net_mean_ms: float = 3.0          # round-trip non-queuing delay, Exp, split out/back
    slow_fraction: float = 0.25       # share of slower edge nodes (e.g. Pi 3 vs Pi 4)
    slow_factor: float = 1.5
    pods_per_node: int = 1
    placement: str = "random"         # "random" or "middle" (paper's hotspot fan-out)
    layout_seed: int = 42             # fixes which nodes are slow, independent of run seed
    default_queries: int = 20000
    notes: List[str] = field(default_factory=list)


# Paper Table II: 4 IADs x 8 Raspberry Pi edge nodes, IAD4 is a hotspot.
TESTBED = Scenario(
    name="testbed",
    description="Paper Table II campus testbed: 4 IADs x 8 edge nodes, IAD4 hotspot",
    n_iads=4,
    nodes_per_iad=8,
    classes=[
        QueryClass("1.1", 0.10, 500.0, (1, 0), (1, 0)),
        QueryClass("1.2", 0.40, 500.0, (1, 0), (1, 0), pinned_iads=(3,)),
        QueryClass("2", 0.40, 800.0, (4, 0), (1, 0)),
        QueryClass("3", 0.10, 1200.0, (4, 0), (8, 0)),
    ],
    notes=["Unloaded subtask time ~100 ms as measured in the paper; absolute "
           "rates are not calibrated to the testbed hardware."],
)

# Paper Table V classes on the 100-node row of Table VI (4 IADs x 25 nodes).
# Table V fanout means (100..700) are mapped to fractions of the pool (/1000):
# this mapping is our assumption, the relative class ordering is preserved.
_TABLE_V = (("1", 0.10, 1200.0, 0.1, 0.1, 0.1, 0.05),
            ("2", 0.40, 1800.0, 0.3, 0.05, 0.3, 0.1),
            ("3", 0.30, 2400.0, 0.5, 0.2, 0.5, 0.2),
            ("4", 0.20, 3000.0, 0.7, 0.2, 0.7, 0.3))


def table_v_scenario(n_iads, nodes_per_iad, name=None, default_queries=5000):
    classes = [QueryClass(n, share, slo,
                          (fi * n_iads, si * n_iads),
                          (fn * nodes_per_iad, sn * nodes_per_iad))
               for n, share, slo, fi, si, fn, sn in _TABLE_V]
    return Scenario(
        name=name or f"sim-{n_iads}x{nodes_per_iad}",
        description=f"Paper Table V classes, {n_iads} IADs x {nodes_per_iad} nodes, middle hotspots",
        n_iads=n_iads,
        nodes_per_iad=nodes_per_iad,
        classes=classes,
        placement="middle",
        default_queries=default_queries,
    )


SCENARIOS: Dict[str, Scenario] = {
    "testbed": TESTBED,
    "sim-small": table_v_scenario(4, 25, name="sim-small"),
    "sim-medium": table_v_scenario(20, 50, name="sim-medium", default_queries=2000),
}


def get_scenario(name):
    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(f"unknown scenario {name!r}; choose from {', '.join(SCENARIOS)}")
