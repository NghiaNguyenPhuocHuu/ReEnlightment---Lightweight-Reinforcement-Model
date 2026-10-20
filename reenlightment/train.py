"""Train and compare the agent against baselines: python -m reenlightment.train"""
import random

from .agent import TailLatencyAgent
from .env import TransportEnv


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def run(policy, env, steps):
    state, lats = env.reset(), []
    for _ in range(steps):
        state, lat = env.step(policy(state))
        lats.append(lat)
    return lats


def main(steps=20000, tau=0.99, seed=0):
    env = TransportEnv(seed=seed)
    agent = TailLatencyAgent(env.n_actions, env.n_states, tau=tau, seed=seed)
    state = env.reset()
    for _ in range(steps):
        a = agent.act(state)
        nxt, lat = env.step(a)
        agent.update(state, a, lat)
        state = nxt
    rng = random.Random(seed)
    policies = {
        "random": lambda s: rng.randrange(env.n_actions),
        "lowest-mean-path": lambda s: 0,
        "RL agent": lambda s: agent.act(s, explore=False),
    }
    results = {}
    for name, pol in policies.items():
        lats = run(pol, TransportEnv(seed=seed + 1), steps)
        results[name] = (sum(lats) / len(lats), percentile(lats, tau))
        print(f"{name:18s} mean={results[name][0]:7.2f}ms  p{int(tau*100)}={results[name][1]:7.2f}ms")
    return results


if __name__ == "__main__":
    main()
