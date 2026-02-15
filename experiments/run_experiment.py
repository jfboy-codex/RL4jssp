from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl4jssp.baselines import BASELINES, rollout, spt_policy, mwkr_policy, atc_policy
from rl4jssp.env import CERJSSPEnv
from rl4jssp.g4dqn import G4DQNAgent, Transition
from rl4jssp.instance import generate_instance
from rl4jssp.plotting import save_bar_svg, save_line_svg


def generate_train_config(seed):
    random.seed(seed)
    return dict(
        num_jobs=random.choice([6, 10, 15, 20]),
        num_machines=random.choice([6, 10, 15]),
        reentry_prob=random.uniform(0.2, 0.6),
        hotspot_intensity=random.uniform(0.3, 0.7),
        setup_level=random.choice([2, 4, 6, 8]),
        breakdown_freq=random.choice([0.02, 0.05, 0.1]),
        due_tightness=random.choice([0.3, 0.5, 0.7]),
        seed=seed,
    )


def collect_imitation(env: CERJSSPEnv):
    trajs = []
    for policy in [spt_policy, mwkr_policy, atc_policy]:
        env.reset()
        traj = []
        while True:
            s = env.get_state()
            a = policy(env)
            if a is None:
                break
            ns, r, d, _ = env.step(a)
            traj.append(Transition(s=s, a=a, r=r, ns=ns, done=d))
            if d:
                break
        trajs.append(traj)
    return trajs


def train_agent(episodes: int, seed: int = 42):
    random.seed(seed)
    agent = G4DQNAgent(seed=seed)
    history = []
    for ep in range(episodes):
        cfg = generate_train_config(seed + ep)
        inst = generate_instance(**cfg)
        env = CERJSSPEnv(inst)
        if ep == 0:
            agent.imitation_warm_start(collect_imitation(env))
        s = env.reset()
        total_r = 0.0
        eps = max(0.05, 0.4 - ep / max(1, episodes) * 0.35)
        while True:
            a = agent.act(s, eps=eps)
            if a is None:
                break
            ns, r, d, _ = env.step(a)
            agent.push(Transition(s=s, a=a, r=r, ns=ns, done=d))
            agent.train_step()
            s = ns
            total_r += r
            if d:
                break
        history.append(total_r)
    return agent, history


def evaluate(agent: G4DQNAgent, n_instances=12, seed=100):
    metrics = defaultdict(list)
    for i in range(n_instances):
        cfg = generate_train_config(seed + i)
        inst = generate_instance(**cfg)
        env = CERJSSPEnv(inst)
        for name, policy in BASELINES.items():
            metrics[name].append(rollout(env, policy))

        env.reset()
        while True:
            s = env.get_state()
            a = agent.act(s, eps=0.0)
            if a is None:
                break
            _, _, done, info = env.step(a)
            if done:
                metrics["G4DQN"].append({"makespan": info.makespan, "tardiness": info.tardiness, "energy": info.energy, "wip": info.wip})
                break
    return metrics


def summarize(metrics):
    out = {}
    for name, vals in metrics.items():
        out[name] = {k: float(statistics.mean([x[k] for x in vals])) for k in ["makespan", "tardiness", "energy", "wip"]}
    return out


def plot_curves(train_rewards, summary, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_line_svg(train_rewards, "G4DQN Training Reward Curve", out_dir / "training_curve.svg")
    methods = list(summary.keys())
    for metric in ["makespan", "tardiness", "energy", "wip"]:
        save_bar_svg({m: summary[m][metric] for m in methods}, metric, out_dir / f"compare_{metric}.svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", type=str, default="outputs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    agent, train_rewards = train_agent(args.episodes)
    metrics = evaluate(agent)
    summary = summarize(metrics)
    plot_curves(train_rewards, summary, out_dir)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
