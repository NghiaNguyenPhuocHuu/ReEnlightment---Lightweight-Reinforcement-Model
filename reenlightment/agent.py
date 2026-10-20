"""Lightweight RL agent that learns to minimise tail latency.

Pure Python, no dependencies. Each (state, action) pair keeps an online
estimate of the tau-quantile (e.g. p99) of observed latency, updated with a
stochastic-approximation rule (quantile regression). The policy is
epsilon-greedy over the lowest estimated tail latency, so it optimises the
tail rather than the mean.
"""
import random


class TailLatencyAgent:
    def __init__(self, n_actions, n_states=1, tau=0.99, lr=0.05,
                 epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.999, seed=None):
        if n_actions < 1 or n_states < 1:
            raise ValueError("n_actions and n_states must be >= 1")
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be in (0, 1)")
        self.n_actions, self.n_states, self.tau, self.lr = n_actions, n_states, tau, lr
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.rng = random.Random(seed)
        self.q = [[0.0] * n_actions for _ in range(n_states)]
        self.n = [[0] * n_actions for _ in range(n_states)]

    def predict(self, state, action):
        """Estimated tau-quantile latency for taking `action` in `state`."""
        return self.q[state][action]

    def act(self, state, explore=True):
        # Try every untried action first, then exploit the lowest tail estimate.
        untried = [a for a in range(self.n_actions) if self.n[state][a] == 0]
        if explore and untried:
            return self.rng.choice(untried)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        row = self.q[state]
        return min(range(self.n_actions), key=row.__getitem__)

    def update(self, state, action, latency):
        n = self.n[state][action] = self.n[state][action] + 1
        q = self.q[state][action]
        if n == 1:
            q = latency
        else:
            step = max(self.lr, 1.0 / n)
            scale = max(abs(q), abs(latency), 1e-9)  # keeps updates unit-free
            q += step * scale * (self.tau - (1.0 if latency <= q else 0.0))
        self.q[state][action] = q
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
