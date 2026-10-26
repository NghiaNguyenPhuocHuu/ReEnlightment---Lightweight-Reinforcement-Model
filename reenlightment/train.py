"""Train and compare the agent against baselines: python -m reenlightment.train"""
import random

from .agent import TailLatencyAgent
from .env import TransportEnv
from .q_agent import QLearningAgent
from .research_env import DSSPBudgetEnv, DSSPSensorEnv, ResearchNetworkEnv


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


def main_research(episodes=500, max_steps=40, seed=0):
    """Train a small Q-learning agent on a toy task-oriented 'research network'."""
    env = ResearchNetworkEnv(seed=seed)
    agent = QLearningAgent(env.n_actions, seed=seed)

    def run_episode(explore):
        state = env.reset()
        total = 0.0
        for _ in range(max_steps):
            action = agent.act(state, explore=explore)
            next_state, reward, done = env.step(action)
            if explore:
                agent.update(state, action, reward, next_state, done)
            total += reward
            state = next_state
            if done:
                break
        return total, env.is_done(state)

    # Train.
    solved = 0
    for ep in range(1, episodes + 1):
        _, done = run_episode(explore=True)
        solved += 1 if done else 0
        if ep % 50 == 0:
            rate = solved / 50.0
            solved = 0
            print(f"episode={ep:4d}  epsilon={agent.epsilon:5.2f}  solved@50={rate:4.2f}")

    # Evaluate.
    eval_episodes = 200
    rl_solved = sum(1 for _ in range(eval_episodes) if run_episode(explore=False)[1]) / eval_episodes
    rng = random.Random(seed)

    def random_policy(_state):
        return rng.randrange(env.n_actions)

    def greedy_prereqs(state):
        # Deterministic baseline: always pick the lowest-index available task.
        for a, (_name, prereq, _r, _p) in enumerate(env.TASKS):
            if (state >> a) & 1:
                continue
            if (state & prereq) == prereq:
                return a
        return 0

    def eval_policy(policy):
        wins = 0
        for _ in range(eval_episodes):
            state = env.reset()
            for _ in range(max_steps):
                action = policy(state)
                state, _reward, done = env.step(action)
                if done:
                    break
            wins += 1 if done else 0
        return wins / eval_episodes

    results = {
        "random": eval_policy(random_policy),
        "greedy-prereqs": eval_policy(greedy_prereqs),
        "RL agent": rl_solved,
    }
    for name, winrate in results.items():
        print(f"{name:14s} solved_rate={winrate:5.2f}")
    return results


if __name__ == "__main__":
    import sys

    # Keep the existing default so current tests/docs keep working.
    if len(sys.argv) > 1 and sys.argv[1] == "research":
        main_research()
    elif len(sys.argv) > 1 and sys.argv[1] == "dssp":
        # Continuous (non-episodic) training loop over requests.
        # Sensors-only DSSP focus for now.
        env = DSSPSensorEnv(seed=0)
        agent = QLearningAgent(env.n_actions, seed=0)

        state = env.reset()
        steps = 20000
        for t in range(1, steps + 1):
            action = agent.act(state, explore=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if t % 2000 == 0:
                print(f"step={t:5d}  epsilon={agent.epsilon:5.2f}")
    elif len(sys.argv) > 1 and sys.argv[1] == "dssp-pipeline":
        # Older multi-stage DSSP toy (kept for comparison).
        env = DSSPBudgetEnv(seed=0)
        agent = QLearningAgent(env.n_actions, seed=0)

        state = env.reset()
        steps = 20000
        for t in range(1, steps + 1):
            action = agent.act(state, explore=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if t % 2000 == 0:
                print(f"step={t:5d}  epsilon={agent.epsilon:5.2f}")
    else:
        main()
