import csv
import os
import tempfile
import unittest

from reenlightment.dssp.__main__ import main as cli
from reenlightment.dssp.experiments import max_rate, sweep, write_csv
from reenlightment.dssp.scenarios import get_scenario
from reenlightment.dssp.sim import simulate


class ExperimentsTest(unittest.TestCase):
    def test_max_rate_is_boundary(self):
        s = get_scenario("testbed")
        r = max_rate(s, "edfq", n_queries=2000, seed=0, rel_tol=0.05)
        self.assertGreater(r, 0)
        self.assertTrue(simulate(s, "edfq", r, 2000, seed=0).meets_slo)

    def test_sweep_csv(self):
        results = sweep(get_scenario("testbed"), ["fifo", "edfq"], [5, 10], n_queries=500)
        self.assertEqual(len(results), 4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.csv")
            write_csv(path, results)
            with open(path) as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertIn("class_3_tail_ms", rows[0])

    def test_cli_run(self):
        cli(["run", "--rate", "5", "--queries", "500", "--policy", "all"])


if __name__ == "__main__":
    unittest.main()
