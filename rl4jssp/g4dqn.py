from __future__ import annotations

from collections import deque
import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Transition:
    s: dict
    a: Tuple[int, int, int]
    r: float
    ns: dict
    done: bool


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]


def relu(v):
    return [x if x > 0 else 0.0 for x in v]


def matvec(v, w):
    return [sum(v[i] * w[i][j] for i in range(len(v))) for j in range(len(w[0]))]


def mean_vec(rows):
    if not rows:
        return []
    n = len(rows)
    return [sum(row[j] for row in rows) / n for j in range(len(rows[0]))]


class GraphEncoder:
    def __init__(self, op_dim: int, machine_dim: int, hidden: int = 16, seed: int = 42):
        rng = random.Random(seed)

        def rmat(i, j):
            return [[rng.uniform(-0.2, 0.2) for _ in range(j)] for _ in range(i)]

        self.w_op = rmat(op_dim, hidden)
        self.w_m = rmat(machine_dim, hidden)
        self.w_msg_o = rmat(hidden, hidden)
        self.w_msg_m = rmat(hidden, hidden)

    def encode(self, state: dict):
        op_h = [relu(matvec(x, self.w_op)) for x in state["op_features"]]
        m_h = [relu(matvec(x, self.w_m)) for x in state["machine_features"]]
        op_ctx = mean_vec(op_h)
        m_ctx = mean_vec(m_h)
        op_h = [relu(matvec(vec_add(h, m_ctx), self.w_msg_o)) for h in op_h]
        m_h = [relu(matvec(vec_add(h, op_ctx), self.w_msg_m)) for h in m_h]
        g = mean_vec(op_h) + mean_vec(m_h)
        return op_h, m_h, g


class G4DQNAgent:
    def __init__(self, op_dim=6, machine_dim=4, lr=1e-3, gamma=0.98, seed=42, use_graph=True, use_candidate=True):
        self.encoder = GraphEncoder(op_dim, machine_dim, seed=seed)
        self.t_encoder = GraphEncoder(op_dim, machine_dim, seed=seed)
        self.hidden = 16
        feat_dim = self.hidden * 4
        rng = random.Random(seed)
        self.w_v = [rng.uniform(-0.1, 0.1) for _ in range(feat_dim)]
        self.w_a = [rng.uniform(-0.1, 0.1) for _ in range(feat_dim)]
        self.tw_v = self.w_v[:]
        self.tw_a = self.w_a[:]
        self.lr = lr
        self.gamma = gamma
        self.use_graph = use_graph
        self.use_candidate = use_candidate
        self.buffer = deque(maxlen=20000)
        self.steps = 0

    def _action_feature(self, state, action, encoder):
        j, _, m = action
        if self.use_graph:
            op_h, m_h, g = encoder.encode(state)
            return op_h[j] + m_h[m] + g
        op = state["op_features"][j]
        mac = state["machine_features"][m]
        raw = op + mac
        while len(raw) < self.hidden * 4:
            raw += raw[: max(1, self.hidden * 4 - len(raw))]
        return raw[: self.hidden * 4]

    def _q_values(self, state, encoder, w_v, w_a):
        actions = state["actions"]
        if not actions:
            return actions, [], []
        feats = [self._action_feature(state, a, encoder) for a in actions]
        vals = [dot(f, w_v) for f in feats]
        adv = [dot(f, w_a) for f in feats]
        mean_adv = sum(adv) / len(adv)
        q = [v + a - mean_adv for v, a in zip(vals, adv)]
        return actions, q, feats

    def act(self, state, eps=0.1, candidate_k=8):
        actions = state["actions"]
        if not actions:
            return None
        scored = actions
        if self.use_candidate:
            scored = sorted(actions, key=lambda a: (state["op_features"][a[0]][0], state["machine_features"][a[2]][0]))[:candidate_k]
        if random.random() < eps:
            return random.choice(scored)
        sub = dict(state)
        sub["actions"] = scored
        acts, q, _ = self._q_values(sub, self.encoder, self.w_v, self.w_a)
        return acts[max(range(len(q)), key=lambda i: q[i])]

    def push(self, tr: Transition, priority=1.0):
        self.buffer.append((priority, tr))

    def imitation_warm_start(self, trajectories: List[List[Transition]]):
        for traj in trajectories:
            for tr in traj:
                self.push(tr, priority=2.0)

    def train_step(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return None
        total_p = sum(p for p, _ in self.buffer)
        samples = []
        for _ in range(batch_size):
            r = random.random() * total_p
            acc = 0.0
            for i, (p, tr) in enumerate(self.buffer):
                acc += p
                if acc >= r:
                    samples.append((i, tr))
                    break

        loss = 0.0
        for idx, tr in samples:
            acts, q, feats = self._q_values(tr.s, self.encoder, self.w_v, self.w_a)
            if tr.a not in acts:
                continue
            ai = acts.index(tr.a)
            q_sa = q[ai]

            if tr.done:
                target = tr.r
            else:
                nacts, nq, _ = self._q_values(tr.ns, self.encoder, self.w_v, self.w_a)
                if not nacts:
                    target = tr.r
                else:
                    b = max(range(len(nq)), key=lambda i: nq[i])
                    tacts, tq, _ = self._q_values(tr.ns, self.t_encoder, self.tw_v, self.tw_a)
                    t_idx = tacts.index(nacts[b]) if nacts[b] in tacts else b
                    target = tr.r + self.gamma * tq[t_idx]

            err = q_sa - target
            loss += err * err
            grad = [2 * err * x for x in feats[ai]]
            self.w_v = [w - self.lr * g for w, g in zip(self.w_v, grad)]
            self.w_a = [w - self.lr * g for w, g in zip(self.w_a, grad)]
            self.buffer[idx] = (abs(err) + 1e-3, tr)

        self.steps += 1
        if self.steps % 100 == 0:
            self.tw_v, self.tw_a = self.w_v[:], self.w_a[:]
            self.t_encoder = self.encoder
        return loss / batch_size
