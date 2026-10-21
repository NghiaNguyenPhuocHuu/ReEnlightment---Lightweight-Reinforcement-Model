"""A tiny simulated 'research network' environment.

This is a toy task-oriented world meant for practicing RL algorithms.

Mental model:
- There is a DAG of research tasks (nodes).
- Each task may depend on prerequisites.
- Each step you choose one task to attempt.
- If prerequisites are met and the task isn't already complete, it may succeed
  stochastically and mark that task as done.

State representation:
- An integer bitmask of completed tasks.
  Example: mask bit i = 1 means task i is done.

Action representation:
- An integer task index (0..n_tasks-1).
"""

import random
from typing import Optional, Tuple


class ResearchNetworkEnv:
    # (name, prereq_mask, reward_if_complete, success_prob)
    TASKS = [
        ("literature_survey", 0b0000_0000, 3.0, 0.95),
        ("pick_problem", 0b0000_0001, 2.0, 0.90),
        ("build_simulator", 0b0000_0011, 4.0, 0.85),
        ("baseline_agent", 0b0000_0111, 4.0, 0.80),
        ("rl_agent", 0b0000_0111, 6.0, 0.75),
        ("run_experiments", 0b0001_1000, 7.0, 0.70),  # needs baseline + rl_agent
        ("analyze_results", 0b0011_1000, 5.0, 0.85),
        ("write_report", 0b0111_1000, 6.0, 0.90),
    ]

    def __init__(self, *, step_cost: float = 0.05, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.step_cost = step_cost

        self.n_actions = len(self.TASKS)
        self.goal_mask = (1 << self.n_actions) - 1
        self.state = 0

    @property
    def n_states(self) -> int:
        return 1 << self.n_actions

    def reset(self) -> int:
        # Always start with no progress; the stochasticity is in task success.
        self.state = 0
        return self.state

    def is_done(self, state: Optional[int] = None) -> bool:
        if state is None:
            state = self.state
        return state == self.goal_mask

    def step(self, action: int) -> Tuple[int, float, bool]:
        if not 0 <= action < self.n_actions:
            raise ValueError("action out of bounds")

        name, prereq_mask, reward, p_success = self.TASKS[action]
        _ = name  # currently unused, kept for debugging/printing

        cost = -float(self.step_cost)

        already_done = (self.state >> action) & 1
        if already_done:
            # Wasting a step repeating finished work.
            return self.state, cost - 0.25, self.is_done()

        prereqs_met = (self.state & prereq_mask) == prereq_mask
        if not prereqs_met:
            # Bumping into missing prerequisites.
            return self.state, cost - 0.40, self.is_done()

        # Attempt the task.
        if self.rng.random() < p_success:
            self.state |= 1 << action
            return self.state, cost + reward, self.is_done()

        # Failure: you learned something but didn't finish.
        return self.state, cost - 0.05, self.is_done()
