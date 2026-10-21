"""Tiny tabular Q-learning agent (pure Python, no deps).

Designed to stay small enough for edge experiments:
- discrete actions
- discrete state (int)
- lazily allocated Q-table (dict) to keep memory bounded
"""

import random
from typing import Dict, List, Optional


class QLearningAgent:
    def __init__(
        self,
        n_actions: int,
        *,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        if n_actions < 1:
            raise ValueError("n_actions must be >= 1")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")

        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)

        # state(int) -> list[float] length n_actions
        self.q: Dict[int, List[float]] = {}

    def _row(self, state: int) -> List[float]:
        row = self.q.get(state)
        if row is None:
            row = [0.0] * self.n_actions
            self.q[state] = row
        return row

    def act(self, state: int, *, explore: bool = True) -> int:
        row = self._row(state)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)

        best_value = max(row)
        best_actions = [a for a, v in enumerate(row) if v == best_value]
        return self.rng.choice(best_actions)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        row = self._row(state)
        q_sa = row[action]

        if done:
            target = reward
        else:
            next_row = self._row(next_state)
            target = reward + self.gamma * max(next_row)

        row[action] = q_sa + self.alpha * (target - q_sa)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
