import unittest

from reenlightment.dssp.scenarios import SCENARIOS, get_scenario
from reenlightment.dssp.sim import POLICIES, percentile, simulate


class SimulatorTest(unittest.TestCase):
    def test_percentile_nearest_rank(self):
        xs = list(range(1, 101))
        self.assertEqual(percentile(xs, 99), 99)
        self.assertEqual(percentile(xs, 100), 100)
        self.assertEqual(percentile([5], 99), 5)

    def test_deterministic_for_seed(self):
        s = get_scenario("testbed")
        a = simulate(s, "edfq", 10, n_queries=2000, seed=3)
        b = simulate(s, "edfq", 10, n_queries=2000, seed=3)
        self.assertEqual(a.classes["3"].tail_ms, b.classes["3"].tail_ms)

    def test_all_queries_complete_and_light_load_meets_slo(self):
        s = get_scenario("testbed")
        for policy in POLICIES:
            r = simulate(s, policy, 5, n_queries=3000, seed=0)
            self.assertEqual(sum(c.count for c in r.classes.values()), 2700)  # minus 10% warm-up
            self.assertTrue(r.meets_slo, policy)
            self.assertLess(r.utilization, 0.2)

    def test_overload_violates_slo(self):
        r = simulate(get_scenario("testbed"), "edfq", 80, n_queries=3000, seed=0)
        self.assertFalse(r.meets_slo)

    def test_spr_favours_tight_slo_class(self):
        s = get_scenario("testbed")
        fifo = simulate(s, "fifo", 20, n_queries=5000, seed=0)
        spr = simulate(s, "spr", 20, n_queries=5000, seed=0)
        self.assertLess(spr.classes["1.2"].tail_ms, fifo.classes["1.2"].tail_ms)
        self.assertGreater(spr.classes["3"].tail_ms, fifo.classes["3"].tail_ms)

    def test_presets_and_errors(self):
        self.assertIn("testbed", SCENARIOS)
        with self.assertRaises(ValueError):
            get_scenario("nope")
        with self.assertRaises(ValueError):
            simulate(get_scenario("testbed"), "lifo", 10)


if __name__ == "__main__":
    unittest.main()
