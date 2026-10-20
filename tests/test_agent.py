import unittest

from reenlightment.agent import TailLatencyAgent
from reenlightment.train import main


class TailLatencyTest(unittest.TestCase):
    def test_quantile_estimate(self):
        a = TailLatencyAgent(1, tau=0.9, seed=1)
        import random
        r = random.Random(2)
        for _ in range(20000):
            a.update(0, 0, r.uniform(0, 100))
        self.assertAlmostEqual(a.predict(0, 0), 90, delta=8)

    def test_invalid_tau(self):
        with self.assertRaises(ValueError):
            TailLatencyAgent(2, tau=1.5)

    def test_agent_beats_baselines_on_tail(self):
        res = main(steps=20000)
        self.assertLess(res["RL agent"][1], res["random"][1])
        self.assertLess(res["RL agent"][1], res["lowest-mean-path"][1])


if __name__ == "__main__":
    unittest.main()
