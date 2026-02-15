from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_experiment import train_agent, evaluate, summarize
from rl4jssp.g4dqn import G4DQNAgent
from rl4jssp.plotting import save_bar_svg


def train_variant(name: str, episodes: int):
    if name == "full":
        return train_agent(episodes)

    # lightweight ablation by disabling specific modules
    if name == "w/o candidate":
        agent = G4DQNAgent(use_candidate=False)
    elif name == "MLP encoder":
        agent = G4DQNAgent(use_graph=False)
    elif name == "w/o imitation":
        agent, _ = train_agent(0)
        return agent, []
    else:
        agent = G4DQNAgent()

    # simple reuse of train loop from run_experiment with manual update
    from experiments.run_experiment import generate_train_config
    from rl4jssp.instance import generate_instance
    from rl4jssp.env import CERJSSPEnv
    from rl4jssp.g4dqn import Transition

    rewards = []
    for ep in range(max(1, episodes)):
        cfg = generate_train_config(200 + ep)
        env = CERJSSPEnv(generate_instance(**cfg))
        s = env.reset()
        total = 0
        while True:
            a = agent.act(s, eps=0.15)
            if a is None:
                break
            ns, r, d, _ = env.step(a)
            agent.push(Transition(s=s, a=a, r=r, ns=ns, done=d))
            agent.train_step()
            total += r
            s = ns
            if d:
                break
        rewards.append(total)
    return agent, rewards


def main():
    out = Path("outputs/ablation")
    out.mkdir(parents=True, exist_ok=True)
    variants = ["full", "w/o candidate", "MLP encoder"]

    results = {}
    for v in variants:
        agent, _ = train_variant(v, episodes=8)
        results[v] = summarize(evaluate(agent, n_instances=6))

    makespan = {v: results[v]["G4DQN"]["makespan"] for v in variants if "G4DQN" in results[v]}
    save_bar_svg(makespan, "ablation_makespan", out / "ablation_makespan.svg")
    (out / "ablation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
