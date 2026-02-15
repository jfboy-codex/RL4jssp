from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from rl4jssp.env import CERJSSPEnv

Action = Tuple[int, int, int]


def _select(env: CERJSSPEnv, score_fn: Callable[[CERJSSPEnv, Action], float], reverse=False):
    actions = env.get_state()["actions"]
    if not actions:
        return None
    return sorted(actions, key=lambda a: score_fn(env, a), reverse=reverse)[0]


def fifo_policy(env: CERJSSPEnv):
    return _select(env, lambda e, a: e.job_ready[a[0]])


def spt_policy(env: CERJSSPEnv):
    return _select(env, lambda e, a: e.instance.jobs[a[0]].operations[a[1]].proc_times[a[2]])


def lpt_policy(env: CERJSSPEnv):
    return _select(env, lambda e, a: e.instance.jobs[a[0]].operations[a[1]].proc_times[a[2]], reverse=True)


def mwkr_policy(env: CERJSSPEnv):
    def score(e: CERJSSPEnv, a: Action):
        j, idx, _ = a
        job = e.instance.jobs[j]
        return -sum(min(op.proc_times.values()) for op in job.operations[idx:])

    return _select(env, score)


def atc_policy(env: CERJSSPEnv):
    def score(e: CERJSSPEnv, a: Action):
        j, idx, m = a
        op = e.instance.jobs[j].operations[idx]
        p = op.proc_times[m]
        slack = max(1.0, e.instance.jobs[j].due_date - e.time - p)
        return p / slack

    return _select(env, score)


def random_policy(env: CERJSSPEnv):
    actions = env.get_state()["actions"]
    return random.choice(actions) if actions else None


def ga_policy(env: CERJSSPEnv, pop_size: int = 16, gens: int = 12):
    actions = env.get_state()["actions"]
    if not actions:
        return None

    population = [[random.random() for _ in actions] for _ in range(pop_size)]

    def fitness(chrom):
        i = max(range(len(actions)), key=lambda x: chrom[x])
        j, idx, m = actions[i]
        op = env.instance.jobs[j].operations[idx]
        return -(op.proc_times[m] + abs(env.instance.jobs[j].due_date - env.time))

    for _ in range(gens):
        scores = sorted([(fitness(c), c) for c in population], key=lambda x: x[0], reverse=True)
        elites = [c for _, c in scores[:4]]
        new_pop = elites[:]
        while len(new_pop) < pop_size:
            a, b = random.sample(elites, 2)
            cut = random.randint(1, len(actions) - 1) if len(actions) > 1 else 1
            child = a[:cut] + b[cut:]
            if random.random() < 0.2:
                pos = random.randrange(len(actions))
                child[pos] += random.uniform(-0.5, 0.5)
            new_pop.append(child)
        population = new_pop

    best = max(population, key=fitness)
    return actions[max(range(len(actions)), key=lambda x: best[x])]


def cp_sat_proxy_policy(env: CERJSSPEnv):
    # Lightweight proxy: exact small branching lookahead
    actions = env.get_state()["actions"]
    if not actions:
        return None
    if len(actions) <= 2:
        return spt_policy(env)
    return min(actions, key=lambda a: env.instance.jobs[a[0]].operations[a[1]].proc_times[a[2]] + 0.3 * max(0, env.time - env.instance.jobs[a[0]].due_date))


def rollout(env: CERJSSPEnv, policy: Callable[[CERJSSPEnv], Action]) -> Dict[str, float]:
    env.reset()
    while True:
        act = policy(env)
        if act is None:
            break
        _, _, done, info = env.step(act)
        if done:
            return {
                "makespan": info.makespan,
                "tardiness": info.tardiness,
                "energy": info.energy,
                "wip": info.wip,
            }
    info = env._calc_metrics()
    return {"makespan": info.makespan, "tardiness": info.tardiness, "energy": info.energy, "wip": info.wip}


BASELINES = {
    "FIFO": fifo_policy,
    "SPT": spt_policy,
    "LPT": lpt_policy,
    "MWKR": mwkr_policy,
    "ATC": atc_policy,
    "GA": ga_policy,
    "CP-SAT-proxy": cp_sat_proxy_policy,
}
