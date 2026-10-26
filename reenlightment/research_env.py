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
        # "Machine" initialization: RNG + internal state.
        self.rng = random.Random(seed)
        self.step_cost = step_cost

        self.n_actions = len(self.TASKS)
        self.goal_mask = (1 << self.n_actions) - 1
        self.state = 0

    def seed(self, seed: Optional[int]) -> None:
        """Re-seed the environment RNG (useful for reproducible runs)."""
        self.rng.seed(seed)

    def task_name(self, action: int) -> str:
        return self.TASKS[action][0]

    def is_task_done(self, state: int, action: int) -> bool:
        return ((state >> action) & 1) == 1

    def prereqs_met(self, state: int, action: int) -> bool:
        prereq_mask = self.TASKS[action][1]
        return (state & prereq_mask) == prereq_mask

    def available_actions(self, state: Optional[int] = None):
        """Actions that can make progress now (not done, prereqs met)."""
        if state is None:
            state = self.state
        return [
            a
            for a in range(self.n_actions)
            if (not self.is_task_done(state, a)) and self.prereqs_met(state, a)
        ]

    @property
    def n_states(self) -> int:
        return 1 << self.n_actions

    def reset(self) -> int:
        """Reset simulator to the start state (no tasks completed)."""
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


class DSSPBudgetEnv:
    """Toy DSSP simulator for adaptive, dynamic, but stable budget decomposition.

    You can treat each env.step() as handling one user request with an SLO.

    The simulator includes:
    - A hidden network "regime" (normal/jittery/lossy/storm) that can change.
    - Measured jitter + packet loss observations (discretized into bins).
    - A small discrete action space: choose a budget template + whether to alert.
    - A stability penalty to discourage thrashing budgets between requests.

    This stays edge-friendly:
    - discrete state -> int
    - discrete actions -> int
    - no deps beyond stdlib
    """

    # SLO deadline (ms) for the whole request.
    SLO_MS = 200.0

    # Minimal pipeline stages ("silos"). Each stage has some processing latency.
    # These are base means; randomness + network effects are added in step().
    STAGES = (
        "csp_frontend",
        "backbone_transport",
        "iad_gateway",
        "sensor_fetch",
    )

    # Budget templates in milliseconds per stage.
    # A "more adaptive" agent chooses among these based on observed jitter/loss.
    BUDGET_TEMPLATES = (
        (25.0, 85.0, 30.0, 60.0),   # balanced
        (25.0, 110.0, 25.0, 40.0),  # network-heavy (assume transport is volatile)
        (20.0, 70.0, 35.0, 75.0),   # edge-heavy (assume sensor side is slow)
        (30.0, 95.0, 35.0, 40.0),   # conservative (more slack earlier)
        (15.0, 70.0, 25.0, 50.0),   # aggressive (tighter budgets)
    )

    # Network regimes: (mean extra transport latency ms, jitter std ms, loss prob).
    REGIMES = (
        (10.0, 2.0, 0.002),  # normal
        (25.0, 10.0, 0.005),  # jittery
        (15.0, 4.0, 0.030),  # lossy
        (50.0, 20.0, 0.080),  # storm
    )

    # Markov transition for the hidden regime (kept simple and stable).
    # Rows sum to ~1.
    REGIME_TRANSITION = (
        (0.92, 0.05, 0.02, 0.01),
        (0.15, 0.75, 0.05, 0.05),
        (0.10, 0.10, 0.75, 0.05),
        (0.20, 0.10, 0.10, 0.60),
    )

    # Observation bins (edge-friendly discretization).
    JITTER_BINS_MS = (3.0, 8.0, 15.0)  # -> 4 bins
    LOSS_BINS = (0.005, 0.02, 0.06)  # -> 4 bins

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        stability_cost: float = 0.10,
        alert_reward: float = 0.60,
    ):
        self.rng = random.Random(seed)
        self.stability_cost = stability_cost
        self.alert_reward = alert_reward

        self.n_templates = len(self.BUDGET_TEMPLATES)

        # action = template_id * 2 + alert_flag
        self.n_actions = self.n_templates * 2

        self.regime = 0
        self.last_template = 0
        self.last_obs_jitter_bin = 0
        self.last_obs_loss_bin = 0

    def seed(self, seed: Optional[int]) -> None:
        self.rng.seed(seed)

    def reset(self) -> int:
        # Start stable: normal regime and default template.
        self.regime = 0
        self.last_template = 0
        self.last_obs_jitter_bin = 0
        self.last_obs_loss_bin = 0
        return self._encode_state(self.last_obs_jitter_bin, self.last_obs_loss_bin, self.last_template)

    def _sample_next_regime(self) -> None:
        row = self.REGIME_TRANSITION[self.regime]
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(row):
            acc += p
            if r <= acc:
                self.regime = i
                return
        self.regime = len(row) - 1

    def _bin(self, x: float, thresholds) -> int:
        for i, t in enumerate(thresholds):
            if x <= t:
                return i
        return len(thresholds)

    def _encode_state(self, jitter_bin: int, loss_bin: int, last_template: int) -> int:
        # 2 bits jitter + 2 bits loss + 3 bits template (<= 8 templates).
        return (last_template << 4) | (loss_bin << 2) | jitter_bin

    def _decode_action(self, action: int) -> Tuple[int, int]:
        if not 0 <= action < self.n_actions:
            raise ValueError("action out of bounds")
        template = action // 2
        alert = action % 2
        return template, alert

    def is_anomaly(self, jitter_bin: int, loss_bin: int) -> bool:
        # Simple anomaly definition: "high" jitter or "high" loss.
        return jitter_bin >= 2 or loss_bin >= 2

    def step(self, action: int) -> Tuple[int, float, bool]:
        template, alert = self._decode_action(action)

        # Hidden regime evolves independently of agent choices.
        self._sample_next_regime()
        mean_extra, jitter_std, loss_p = self.REGIMES[self.regime]

        # Observations (what your lightweight AI would estimate online).
        obs_jitter = abs(self.rng.gauss(0.0, jitter_std))
        obs_loss = loss_p
        jitter_bin = self._bin(obs_jitter, self.JITTER_BINS_MS)
        loss_bin = self._bin(obs_loss, self.LOSS_BINS)

        # Simulate request execution.
        budgets = self.BUDGET_TEMPLATES[template]
        proc = [
            18.0 + abs(self.rng.gauss(0.0, 3.0)),  # csp_frontend
            55.0 + mean_extra + abs(self.rng.gauss(0.0, 6.0)),  # backbone_transport
            20.0 + abs(self.rng.gauss(0.0, 4.0)),  # iad_gateway
            45.0 + abs(self.rng.gauss(0.0, 10.0)),  # sensor_fetch
        ]

        # Packet loss -> retry penalty. (Simple model: either ok or one retry.)
        lost = self.rng.random() < loss_p
        if lost:
            proc[1] += 35.0 + abs(self.rng.gauss(0.0, 10.0))

        total_ms = sum(proc)
        violates_slo = total_ms > self.SLO_MS

        # "Budget decomposition breaks" is modeled as stage budget violations.
        stage_violations = sum(1 for i, ms in enumerate(proc) if ms > budgets[i])

        # Reward shaping:
        # - SLO success is the core objective
        # - stage_violations represent decomposition mismatch
        # - stability_cost discourages budget thrashing
        # - alert is rewarded only when there is an anomaly
        reward = 0.0
        reward += 1.0 if not violates_slo else -2.0
        reward += -0.35 * stage_violations
        reward += -5.0 if lost else 0.0

        if template != self.last_template:
            reward -= self.stability_cost

        anomaly = self.is_anomaly(jitter_bin, loss_bin)
        if alert and anomaly:
            reward += self.alert_reward
        elif alert and not anomaly:
            reward -= self.alert_reward
        elif (not alert) and anomaly:
            reward -= 0.30

        # Update internal (for state encoding).
        self.last_template = template
        self.last_obs_jitter_bin = jitter_bin
        self.last_obs_loss_bin = loss_bin

        # This environment is continuing (not episodic) by default.
        done = False
        next_state = self._encode_state(jitter_bin, loss_bin, template)
        return next_state, reward, done


class DSSPSensorEnv:
    """Sensors-only DSSP simulator (single silo focus).

    Goal:
    - Choose a per-request sensor timeout/budget (adaptive & stable).
    - Detect anomalies (jitter/loss spikes) and decide whether to alert.

    Each step simulates one user request for one sensor's data.

    State (discrete int) encodes:
    - which sensor is requested
    - observed jitter bin
    - observed loss bin
    - last chosen timeout template (to penalize thrashing)

    Action (discrete int) encodes:
    - timeout template id
    - alert flag
    """

    # End-to-end SLO for the request.
    SLO_MS = 200.0

    # Timeout/budget templates for the sensor-fetch portion (ms).
    # (tighter) -> (looser). The agent learns when to relax.
    TIMEOUT_TEMPLATES_MS = (55.0, 75.0, 100.0, 130.0)

    # A few example sensors with different tail behavior.
    # (base_ms, jitter_std_ms, spike_prob, spike_ms, loss_prob)
    SENSORS = (
        (35.0, 6.0, 0.02, 60.0, 0.005),
        (45.0, 8.0, 0.03, 90.0, 0.010),
        (30.0, 5.0, 0.01, 40.0, 0.002),
        (55.0, 10.0, 0.05, 120.0, 0.020),
    )

    # Hidden network regimes affect measured jitter/loss, independent of sensor.
    # (transport_mean_ms, jitter_std_ms, loss_prob)
    REGIMES = (
        (18.0, 2.0, 0.002),  # normal
        (30.0, 8.0, 0.006),  # jittery
        (22.0, 3.0, 0.030),  # lossy
        (60.0, 18.0, 0.080),  # storm
    )

    REGIME_TRANSITION = (
        (0.92, 0.05, 0.02, 0.01),
        (0.15, 0.75, 0.05, 0.05),
        (0.10, 0.10, 0.75, 0.05),
        (0.20, 0.10, 0.10, 0.60),
    )

    # Discretization bins to keep the state small (edge-friendly).
    JITTER_BINS_MS = (3.0, 8.0, 15.0)  # -> 4 bins
    LOSS_BINS = (0.005, 0.02, 0.06)  # -> 4 bins

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        stability_cost: float = 0.10,
        alert_reward: float = 0.60,
        step_cost: float = 0.01,
    ):
        self.rng = random.Random(seed)
        self.stability_cost = stability_cost
        self.alert_reward = alert_reward
        self.step_cost = step_cost

        self.n_sensors = len(self.SENSORS)
        self.n_templates = len(self.TIMEOUT_TEMPLATES_MS)

        # action = template_id * 2 + alert_flag
        self.n_actions = self.n_templates * 2

        self.regime = 0
        self.last_template = 0

        self.sensor_id = 0
        self.jitter_bin = 0
        self.loss_bin = 0

        # Telemetry from the most recent step() for evaluation/debugging.
        self.last_total_ms = 0.0
        self.last_sensor_ms = 0.0
        self.last_transport_ms = 0.0
        self.last_violates_slo = False
        self.last_violates_sensor_budget = False
        self.last_lost = False
        self.last_anomaly = False
        self.last_alert = 0
        self.last_timeout_ms = float(self.TIMEOUT_TEMPLATES_MS[0])

    def seed(self, seed: Optional[int]) -> None:
        self.rng.seed(seed)

    def _sample_next_regime(self) -> None:
        row = self.REGIME_TRANSITION[self.regime]
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(row):
            acc += p
            if r <= acc:
                self.regime = i
                return
        self.regime = len(row) - 1

    def _bin(self, x: float, thresholds) -> int:
        for i, t in enumerate(thresholds):
            if x <= t:
                return i
        return len(thresholds)

    def _encode_state(self, sensor_id: int, jitter_bin: int, loss_bin: int, last_template: int) -> int:
        # 2 bits jitter + 2 bits loss + 2 bits template + 3 bits sensor (<= 8).
        return (sensor_id << 6) | (last_template << 4) | (loss_bin << 2) | jitter_bin

    def _decode_action(self, action: int) -> Tuple[int, int]:
        if not 0 <= action < self.n_actions:
            raise ValueError("action out of bounds")
        return action // 2, action % 2

    def is_anomaly(self, jitter_bin: int, loss_bin: int) -> bool:
        return jitter_bin >= 2 or loss_bin >= 2

    def reset(self) -> int:
        # Start stable.
        self.regime = 0
        self.last_template = 0
        self.sensor_id = self.rng.randrange(self.n_sensors)
        self.jitter_bin = 0
        self.loss_bin = 0
        self.last_total_ms = 0.0
        self.last_sensor_ms = 0.0
        self.last_transport_ms = 0.0
        self.last_violates_slo = False
        self.last_violates_sensor_budget = False
        self.last_lost = False
        self.last_anomaly = False
        self.last_alert = 0
        self.last_timeout_ms = float(self.TIMEOUT_TEMPLATES_MS[self.last_template])
        return self._encode_state(self.sensor_id, self.jitter_bin, self.loss_bin, self.last_template)

    def last_info(self):
        """Return a dict of the most recent request's debug metrics."""
        return {
            "sensor_id": self.sensor_id,
            "jitter_bin": self.jitter_bin,
            "loss_bin": self.loss_bin,
            "template": self.last_template,
            "timeout_ms": self.last_timeout_ms,
            "transport_ms": self.last_transport_ms,
            "sensor_ms": self.last_sensor_ms,
            "total_ms": self.last_total_ms,
            "violates_slo": self.last_violates_slo,
            "violates_sensor_budget": self.last_violates_sensor_budget,
            "lost": self.last_lost,
            "anomaly": self.last_anomaly,
            "alert": self.last_alert,
        }

    def step(self, action: int) -> Tuple[int, float, bool]:
        template, alert = self._decode_action(action)
        timeout_ms = self.TIMEOUT_TEMPLATES_MS[template]
        self.last_alert = alert
        self.last_timeout_ms = float(timeout_ms)

        # New request -> sensor can change.
        self.sensor_id = self.rng.randrange(self.n_sensors)

        # Hidden regime evolves.
        self._sample_next_regime()
        transport_mean, jitter_std, net_loss_p = self.REGIMES[self.regime]

        # Observation: jitter/loss signals that break naive budget decomposition.
        obs_jitter = abs(self.rng.gauss(0.0, jitter_std))
        obs_loss = net_loss_p
        self.jitter_bin = self._bin(obs_jitter, self.JITTER_BINS_MS)
        self.loss_bin = self._bin(obs_loss, self.LOSS_BINS)

        base, s_jitter, spike_p, spike_ms, sensor_loss_p = self.SENSORS[self.sensor_id]

        # Transport + sensor latency.
        transport = transport_mean + abs(self.rng.gauss(0.0, 4.0))
        sensor = base + abs(self.rng.gauss(0.0, s_jitter))
        if self.rng.random() < spike_p:
            sensor += spike_ms * self.rng.uniform(0.6, 1.4)

        # Loss -> retry penalty (either net loss or sensor-side loss).
        lost = (self.rng.random() < net_loss_p) or (self.rng.random() < sensor_loss_p)
        if lost:
            transport += 35.0 + abs(self.rng.gauss(0.0, 10.0))

        total_ms = transport + sensor
        violates_slo = total_ms > self.SLO_MS

        # Timeout is the "budget" we decompose to the sensor silo.
        violates_sensor_budget = sensor > timeout_ms

        anomaly = self.is_anomaly(self.jitter_bin, self.loss_bin)

        # Record telemetry (for p99 + miss-rate evaluation).
        self.last_total_ms = float(total_ms)
        self.last_sensor_ms = float(sensor)
        self.last_transport_ms = float(transport)
        self.last_violates_slo = bool(violates_slo)
        self.last_violates_sensor_budget = bool(violates_sensor_budget)
        self.last_lost = bool(lost)
        self.last_anomaly = bool(anomaly)

        # Reward balances: deadline misses + tail reduction + stability + alert correctness.
        reward = -float(self.step_cost)
        reward += 1.0 if not violates_slo else -2.0
        reward += -1.0 if violates_sensor_budget else 0.2
        reward += -5.0 if lost else 0.0

        # Penalize long tail events even if SLO holds (proxy for p99 focus).
        if total_ms > 0.90 * self.SLO_MS:
            reward -= 0.5

        if template != self.last_template:
            reward -= self.stability_cost

        if alert and anomaly:
            reward += self.alert_reward
        elif alert and not anomaly:
            reward -= self.alert_reward
        elif (not alert) and anomaly:
            reward -= 0.30

        self.last_template = template

        done = False
        next_state = self._encode_state(self.sensor_id, self.jitter_bin, self.loss_bin, self.last_template)
        return next_state, reward, done
