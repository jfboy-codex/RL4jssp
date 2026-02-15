from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Tuple, Optional


@dataclass
class Operation:
    job_id: int
    op_id: int
    family: int
    proc_times: dict[int, int]
    energy_rate: float
    due_date: int


@dataclass
class Job:
    job_id: int
    operations: List[Operation]
    due_date: int


@dataclass
class CEInstance:
    jobs: List[Job]
    num_machines: int
    setup_time: List[List[List[int]]]  # machine x family x family
    maintenance_windows: dict[int, List[Tuple[int, int]]]
    batch_machines: set[int]
    batch_min_size: int
    breakdown_prob: float


def _sample_due_date(base_work: int, tightness: float) -> int:
    return max(10, int(base_work * (1.1 + (1.0 - tightness) * random.uniform(0.8, 1.4))))


def generate_instance(
    num_jobs: int,
    num_machines: int,
    reentry_prob: float,
    hotspot_intensity: float,
    setup_level: int,
    breakdown_freq: float,
    due_tightness: float,
    seed: Optional[int] = None,
) -> CEInstance:
    if seed is not None:
        random.seed(seed)

    families = max(3, num_machines // 2)
    hotspot = max(1, int(num_machines * hotspot_intensity))
    hotspot_machines = set(random.sample(range(num_machines), k=hotspot))

    jobs: List[Job] = []
    for j in range(num_jobs):
        num_ops = random.randint(max(4, num_machines // 2), max(6, num_machines + 2))
        ops: List[Operation] = []
        last_family = random.randint(0, families - 1)
        expected_work = 0
        for l in range(num_ops):
            if random.random() > reentry_prob:
                last_family = random.randint(0, families - 1)
            cand_size = random.randint(1, min(4, num_machines))
            base_pool = list(hotspot_machines) if random.random() < hotspot_intensity else list(range(num_machines))
            cand = random.sample(base_pool, k=min(cand_size, len(base_pool)))
            proc_times = {m: random.randint(2, 20) for m in cand}
            expected_work += min(proc_times.values())
            ops.append(
                Operation(
                    job_id=j,
                    op_id=l,
                    family=last_family,
                    proc_times=proc_times,
                    energy_rate=round(random.uniform(1.0, 3.5), 2),
                    due_date=0,
                )
            )
        due = _sample_due_date(expected_work, due_tightness)
        for op in ops:
            op.due_date = due
        jobs.append(Job(job_id=j, operations=ops, due_date=due))

    setup_time = []
    for _ in range(num_machines):
        mtx = []
        for f1 in range(families):
            row = []
            for f2 in range(families):
                if f1 == f2:
                    row.append(0)
                else:
                    row.append(random.randint(0, setup_level))
            mtx.append(row)
        setup_time.append(mtx)

    maintenance_windows = {}
    for m in range(num_machines):
        windows = []
        if random.random() < 0.4:
            s = random.randint(30, 80)
            e = s + random.randint(5, 15)
            windows.append((s, e))
        maintenance_windows[m] = windows

    batch_machines = set(random.sample(range(num_machines), k=max(1, num_machines // 5)))
    return CEInstance(
        jobs=jobs,
        num_machines=num_machines,
        setup_time=setup_time,
        maintenance_windows=maintenance_windows,
        batch_machines=batch_machines,
        batch_min_size=2,
        breakdown_prob=breakdown_freq,
    )
