# ReEnlightment
Lightweight Reinforcement AI

Pure-Python (no dependencies) RL agent that learns to pick transport paths minimising
tail latency (e.g. p99). It keeps an online tau-quantile estimate of latency per
(state, action) and acts epsilon-greedily on it.

    python -m reenlightment.train      # train + compare against baselines
    python -m unittest discover -s tests
