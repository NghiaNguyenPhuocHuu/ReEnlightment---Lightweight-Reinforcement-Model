# ReEnlightment
Lightweight Reinforcement AI

Pure-Python (no dependencies) RL agent that learns to pick transport paths minimising
tail latency (e.g. p99). It keeps an online tau-quantile estimate of latency per
(state, action) and acts epsilon-greedily on it.

    python -m reenlightment.train      # train + compare against baselines
    python -m reenlightment.train research  # toy task-oriented "research network"
    python -m reenlightment.train dssp  # DSSP sensors-only adaptive timeout + anomaly alerts
    python -m reenlightment.train dssp-pipeline  # older multi-stage DSSP toy
    python -m unittest discover -s tests
