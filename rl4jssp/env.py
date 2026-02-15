from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Tuple

from rl4jssp.instance import CEInstance


@dataclass
class StepInfo:
    makespan: float
    tardiness: float
    energy: float
    wip: float


class CERJSSPEnv:
    def __init__(self, instance: CEInstance, alpha=0.6, beta=0.2, gamma=0.2):
        self.instance = instance
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.time = 0.0
        self.machine_ready = [0.0] * self.instance.num_machines
        self.machine_last_family = [-1] * self.instance.num_machines
        self.job_next_op = [0] * len(self.instance.jobs)
        self.job_ready = [0.0] * len(self.instance.jobs)
        self.job_completion = [0.0] * len(self.instance.jobs)
        self.total_energy = 0.0
        self.area_wip = 0.0
        self.last_time = 0.0
        self.done = False
        return self.get_state()

    def _ready_ops(self):
        r = []
        for j, job in enumerate(self.instance.jobs):
            idx = self.job_next_op[j]
            if idx < len(job.operations):
                op = job.operations[idx]
                for m in op.proc_times:
                    if self._feasible(op, m):
                        r.append((j, idx, m))
        return r

    def _maintenance_adjust(self, m: int, start: float, dur: float):
        t = start
        for ws, we in self.instance.maintenance_windows.get(m, []):
            if t < we and (t + dur) > ws:
                t = we
        return t

    def _feasible(self, op, machine):
        if machine in self.instance.batch_machines:
            same_family = 0
            for j, job in enumerate(self.instance.jobs):
                idx = self.job_next_op[j]
                if idx < len(job.operations) and job.operations[idx].family == op.family and machine in job.operations[idx].proc_times:
                    same_family += 1
            if same_family < self.instance.batch_min_size and random.random() < 0.5:
                return False
        return True

    def _calc_metrics(self):
        makespan = max(self.job_completion)
        tardiness = sum(max(0.0, self.job_completion[j] - self.instance.jobs[j].due_date) for j in range(len(self.instance.jobs)))
        wip = self.area_wip / max(1.0, self.time)
        return StepInfo(makespan, tardiness, self.total_energy, wip)

    def _mean(self, seq):
        return sum(seq) / max(1, len(seq))

    def get_state(self):
        op_features = []
        machine_features = []
        for j, job in enumerate(self.instance.jobs):
            idx = self.job_next_op[j]
            rem = sum(min(job.operations[k].proc_times.values()) for k in range(idx, len(job.operations)))
            slack = job.due_date - self.time - rem
            op_ratio = idx / max(1, len(job.operations))
            est_wait = max(0.0, self.job_ready[j] - self.time)
            if idx < len(job.operations):
                op = job.operations[idx]
                avail = min(self.machine_ready[m] for m in op.proc_times)
                op_features.append([rem, slack, op_ratio, avail, est_wait, op.energy_rate])
            else:
                op_features.append([0, 0, 1, 0, 0, 0])
        for m in range(self.instance.num_machines):
            load = max(0.0, self.machine_ready[m] - self.time)
            queue_len = sum(1 for j, job in enumerate(self.instance.jobs) if self.job_next_op[j] < len(job.operations) and m in job.operations[self.job_next_op[j]].proc_times)
            machine_features.append([load, self.machine_ready[m], queue_len, random.uniform(1.0, 3.0)])
        return {"op_features": op_features, "machine_features": machine_features, "actions": self._ready_ops()}

    def step(self, action: Tuple[int, int, int]):
        if self.done:
            raise RuntimeError("Episode done")
        prev = self._calc_metrics()

        j, idx, m = action
        op = self.instance.jobs[j].operations[idx]

        start = max(self.time, self.job_ready[j], self.machine_ready[m])
        if self.machine_last_family[m] >= 0:
            start += self.instance.setup_time[m][self.machine_last_family[m]][op.family]
        proc = op.proc_times[m]
        start = self._maintenance_adjust(m, start, proc)
        dur = proc + (random.randint(1, 10) if random.random() < self.instance.breakdown_prob else 0)
        end = start + dur

        active_wip = sum(1 for x in self.job_next_op if x > 0) - sum(1 for x, jb in zip(self.job_next_op, self.instance.jobs) if x >= len(jb.operations))
        self.area_wip += active_wip * max(0.0, start - self.last_time)
        self.last_time = start

        self.machine_ready[m] = end
        self.machine_last_family[m] = op.family
        self.job_ready[j] = end
        self.job_next_op[j] += 1
        if self.job_next_op[j] >= len(self.instance.jobs[j].operations):
            self.job_completion[j] = end

        self.total_energy += dur * op.energy_rate
        self.time = min(self.machine_ready[m], self.job_ready[j])

        self.done = all(self.job_next_op[j_idx] >= len(job.operations) for j_idx, job in enumerate(self.instance.jobs))
        cur = self._calc_metrics()
        reward = -((cur.makespan - prev.makespan) + self.alpha * (cur.tardiness - prev.tardiness) + self.beta * (cur.energy - prev.energy) + self.gamma * (cur.wip - prev.wip))
        return self.get_state(), reward, self.done, cur
