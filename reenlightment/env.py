"""Simulated data-transport flow: pick a path per packet; latency is heavy-tailed."""
import random


class TransportEnv:
    """Each path has a base latency and a probability of a large spike.

    The state is a congestion level (0..n_states-1) that drifts randomly and
    inflates spike probability on the "risky" paths. A low-mean path can have a
    terrible p99, which is exactly what the agent must learn to avoid.
    """

    # (base latency ms, jitter ms, spike probability, spike size ms)
    PATHS = [(10.0, 2.0, 0.05, 200.0),   # fast on average, bad tail
             (14.0, 2.0, 0.00, 0.0),     # steady
             (25.0, 5.0, 0.01, 40.0)]    # slow, mild tail

    def __init__(self, n_states=3, seed=None):
        self.n_states = n_states
        self.n_actions = len(self.PATHS)
        self.rng = random.Random(seed)
        self.state = 0

    def reset(self):
        self.state = self.rng.randrange(self.n_states)
        return self.state

    def step(self, action):
        base, jitter, p_spike, spike = self.PATHS[action]
        boost = 1.0 + self.state * (1.0 if action == 0 else 0.0)
        latency = base + abs(self.rng.gauss(0.0, jitter))
        if self.rng.random() < min(1.0, p_spike * boost):
            latency += spike * self.rng.uniform(0.5, 1.5)
        self.state = self.rng.randrange(self.n_states)
        return self.state, latency
