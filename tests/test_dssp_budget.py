import math
import random
import unittest

from reenlightment.dssp.budget import (UnloadedProfile, query_percentile, queuing_budget,
                                       task_percentile, unloaded_task_tail)


class BudgetDecompositionTest(unittest.TestCase):
    def test_task_percentile_matches_paper(self):
        self.assertAlmostEqual(task_percentile(99, 1), 99.0)
        self.assertAlmostEqual(task_percentile(99, 100), 99.99, places=2)

    def test_percentile_round_trip(self):
        for kq in (1, 4, 37):
            self.assertAlmostEqual(query_percentile(task_percentile(99, kq), kq), 99.0)

    def test_profile_window_evicts_oldest(self):
        p = UnloadedProfile(window=3)
        for x in (10, 1, 2, 3):
            p.add(x)
        self.assertEqual(len(p), 3)
        self.assertEqual(p.max(), 3)
        self.assertAlmostEqual(p.cdf(2), 2 / 3)

    def test_unloaded_tail_single_node_is_quantile(self):
        p = UnloadedProfile(window=10000)
        r = random.Random(0)
        for _ in range(10000):
            p.add(r.uniform(0, 100))
        self.assertAlmostEqual(unloaded_task_tail([p], 90), 90, delta=2)

    def test_unloaded_tail_grows_with_fanout(self):
        # Max of n iid U(0,1) has CDF t^n, so the 90th percentile is 0.9^(1/n).
        r = random.Random(1)
        profiles = []
        for _ in range(4):
            p = UnloadedProfile(window=20000)
            for _ in range(20000):
                p.add(r.random())
            profiles.append(p)
        self.assertAlmostEqual(unloaded_task_tail(profiles, 90), 0.9 ** 0.25, delta=0.01)
        self.assertLess(unloaded_task_tail(profiles[:1], 90), unloaded_task_tail(profiles, 90))

    def test_queuing_budget_clamped(self):
        self.assertEqual(queuing_budget(500, 120), 380)
        self.assertEqual(queuing_budget(100, 120), 0.0)
        self.assertFalse(math.isnan(unloaded_task_tail([UnloadedProfile()], 99)))


if __name__ == "__main__":
    unittest.main()
