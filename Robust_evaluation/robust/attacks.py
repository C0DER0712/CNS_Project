"""Attacks over the realizable edit set (problem space) + a feature-space PGD reference.

Three search strategies, all restricted to pcap_ops edits and the ThreatModel budget:

  RandomSearch    no model knowledge: random edit chains, keep the first that works.
  GreedySearch    score-based black box: evaluate every neighbour, take the best.
  GradientSearch  white box: rank neighbours by the first-order change predicted by
                  the input gradient, then verify the top-k with real queries.

FeaturePGD is NOT realizable (it edits MFR bytes directly).  It is kept only to
show the feature-space vs problem-space gap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from robust.model_io import Objective
from robust.pcap_ops import Edit, Flow, apply_edit, cost, enumerate_edits
from robust.pipeline import flow_to_mfr
from robust.threat_model import ThreatModel


@dataclass
class AttackResult:
    success: bool
    edits: List[Edit]
    flow: Flow
    mfr: np.ndarray
    extra_bytes: int
    extra_packets: int
    queries: int
    final_score: float
    history: List[Tuple[int, float]] = field(default_factory=list)  # (queries, best score)


# Edit results are deterministic functions of (flow state, edit), so they are cached across
# budgets, attacks and seeds.  suite.run_suite() calls clear_cache() between flows.
_CACHE: Dict[Tuple[str, Edit], Optional[Tuple[int, int, np.ndarray, Flow]]] = {}


def clear_cache() -> None:
    _CACHE.clear()


def _cached_apply(cur: Flow, e: Edit):
    """(wire_bytes, n_packets, mfr, flow) of apply_edit(cur, e), or None if inapplicable."""
    k = (cur.key, e)
    if k not in _CACHE:
        g = apply_edit(cur, e)
        _CACHE[k] = None if g is None else (g.wire_bytes, g.n_packets, flow_to_mfr(g), g)
    return _CACHE[k]


def _neighbors(cur: Flow, orig: Flow, cur_mfr: np.ndarray, tm: ThreatModel,
               rng: random.Random):
    """Budget-feasible, MFR-changing neighbours.  Returns [(edit, mfr, flow_getter)]."""
    seen = {cur_mfr.tobytes()}
    out = []
    for e in enumerate_edits(cur, tm.allowed, rng):
        r = _cached_apply(cur, e)
        if r is None:
            continue
        wb, npk, m, _g = r
        if wb - orig.wire_bytes > tm.max_extra_bytes or npk - orig.n_packets > tm.max_extra_packets:
            continue
        key = m.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((e, cur, m))
    return out


def _materialise(cur: Flow, e: Edit) -> Flow:
    r = _cached_apply(cur, e)
    assert r is not None
    return r[3]


class _Base:
    name = "base"

    def run(self, flow: Flow, obj: Objective, tm: ThreatModel, seed: int = 0) -> AttackResult:
        raise NotImplementedError

    @staticmethod
    def _finish(orig, cur, cur_mfr, edits, queries, score, success, hist):
        b, n = cost(orig, cur)
        return AttackResult(bool(success), edits, cur, cur_mfr, b, n, queries, float(score), hist)


class RandomSearch(_Base):
    """Random edit chains (no model knowledge).  At most `max_chains` chains are tried, each of
    random length 1..max_steps, every step a uniformly random applicable edit that respects the
    budget.  Candidate flows come from the shared edit cache, so repeated prefixes are free."""
    name = "random"

    def __init__(self, batch: int = 32, max_chains: int = 100):
        self.batch, self.max_chains = batch, max_chains

    def run(self, flow, obj, tm, seed=0):
        rng = random.Random(seed)
        m0 = flow_to_mfr(flow)
        best = (float(obj.scores(m0[None])[0]), flow, m0, [])
        queries, hist = 1, [(1, best[0])]
        if obj.success(m0[None])[0]:
            return self._finish(flow, flow, m0, [], queries, best[0], True, hist)
        n_chains = max(0, min(self.max_chains, tm.max_queries - 1))
        done = 0
        while done < n_chains:
            cand = []
            for _ in range(min(self.batch, n_chains - done)):
                done += 1
                cur, cur_mfr, edits = flow, m0, []
                for _s in range(rng.randint(1, tm.max_steps)):
                    options = enumerate_edits(cur, tm.allowed, rng)
                    rng.shuffle(options)
                    for e in options:
                        r = _cached_apply(cur, e)
                        if r is None:
                            continue
                        if r[0] - flow.wire_bytes > tm.max_extra_bytes or r[1] - flow.n_packets > tm.max_extra_packets:
                            continue
                        cur, cur_mfr, edits = r[3], r[2], edits + [e]
                        break
                if edits:
                    cand.append((edits, cur, cur_mfr))
            if not cand:
                break
            mf = np.stack([c[2] for c in cand])
            sc, ok = obj.scores(mf), obj.success(mf)
            queries += len(cand)
            i = int(np.argmax(sc))
            if sc[i] > best[0]:
                best = (float(sc[i]), cand[i][1], cand[i][2], cand[i][0])
            hist.append((queries, best[0]))
            if ok.any():
                j = int(np.argmax(np.where(ok, sc, -np.inf)))
                return self._finish(flow, cand[j][1], cand[j][2], cand[j][0], queries, sc[j], True, hist)
        return self._finish(flow, best[1], best[2], best[3], queries, best[0], False, hist)


class GreedySearch(_Base):
    name = "greedy"

    def run(self, flow, obj, tm, seed=0):
        rng = random.Random(seed)
        cur, cur_mfr, edits = flow, flow_to_mfr(flow), []
        cur_score = float(obj.scores(cur_mfr[None])[0])
        queries, hist = 1, [(1, cur_score)]
        if obj.success(cur_mfr[None])[0]:
            return self._finish(flow, cur, cur_mfr, edits, queries, cur_score, True, hist)
        for _ in range(tm.max_steps):
            nb = _neighbors(cur, flow, cur_mfr, tm, rng)
            if not nb:
                break
            nb = nb[: max(0, tm.max_queries - queries)]
            if not nb:
                break
            mf = np.stack([x[2] for x in nb])
            sc, ok = obj.scores(mf), obj.success(mf)
            queries += len(nb)
            if ok.any():
                j = int(np.argmax(np.where(ok, sc, -np.inf)))
                hist.append((queries, float(sc[j])))
                return self._finish(flow, _materialise(nb[j][1], nb[j][0]), nb[j][2], edits + [nb[j][0]], queries, sc[j], True, hist)
            j = int(np.argmax(sc))
            if sc[j] <= cur_score:
                break
            cur, cur_mfr, cur_score = _materialise(nb[j][1], nb[j][0]), nb[j][2], float(sc[j])
            edits = edits + [nb[j][0]]
            hist.append((queries, cur_score))
        return self._finish(flow, cur, cur_mfr, edits, queries, cur_score, False, hist)


class GradientSearch(_Base):
    name = "gradient"

    def __init__(self, top_k: int = 8, n_explore: int = 2):
        self.top_k, self.n_explore = top_k, n_explore

    def run(self, flow, obj, tm, seed=0):
        if not obj.has_grad:
            raise ValueError("GradientSearch needs an objective with gradients")
        rng = random.Random(seed)
        cur, cur_mfr, edits = flow, flow_to_mfr(flow), []
        cur_score = float(obj.scores(cur_mfr[None])[0])
        queries, hist = 1, [(1, cur_score)]
        if obj.success(cur_mfr[None])[0]:
            return self._finish(flow, cur, cur_mfr, edits, queries, cur_score, True, hist)
        for _ in range(tm.max_steps):
            nb = _neighbors(cur, flow, cur_mfr, tm, rng)
            if not nb:
                break
            g = obj.grad(cur_mfr)
            est = np.array([float((g * (m.astype(np.float32) - cur_mfr.astype(np.float32))).sum())
                            for _, _, m in nb])
            order = list(np.argsort(-est))
            pick = order[: self.top_k]
            rest = order[self.top_k:]
            rng.shuffle(rest)
            pick += rest[: self.n_explore]
            pick = pick[: max(0, tm.max_queries - queries)]
            if not pick:
                break
            sub = [nb[i] for i in pick]
            mf = np.stack([x[2] for x in sub])
            sc, ok = obj.scores(mf), obj.success(mf)
            queries += len(sub)
            if ok.any():
                j = int(np.argmax(np.where(ok, sc, -np.inf)))
                hist.append((queries, float(sc[j])))
                return self._finish(flow, _materialise(sub[j][1], sub[j][0]), sub[j][2], edits + [sub[j][0]], queries, sc[j], True, hist)
            j = int(np.argmax(sc))
            if sc[j] <= cur_score:
                break
            cur, cur_mfr, cur_score = _materialise(sub[j][1], sub[j][0]), sub[j][2], float(sc[j])
            edits = edits + [sub[j][0]]
            hist.append((queries, cur_score))
        return self._finish(flow, cur, cur_mfr, edits, queries, cur_score, False, hist)


class RestartGreedy(_Base):
    """Adaptive attacker: several greedy runs with different candidate seeds, widened
    step count and query budget.  Used against defended models."""
    name = "greedy_restarts"

    def __init__(self, restarts: int = 4, step_mult: int = 2):
        self.restarts, self.step_mult = restarts, step_mult

    def run(self, flow, obj, tm, seed=0):
        d = tm.to_dict()
        d["max_steps"] = tm.max_steps * self.step_mult
        d["allowed"] = tuple(d["allowed"])
        wide = ThreatModel(**d)
        best = None
        total_q = 0
        for r in range(self.restarts):
            res = GreedySearch().run(flow, obj, wide, seed=seed * 1000 + r)
            total_q += res.queries
            if best is None or (res.success and not best.success) or \
                    (res.success == best.success and res.final_score > best.final_score):
                best = res
            if res.success:
                break
        best.queries = total_q
        return best


ATTACKS = {"random": RandomSearch, "greedy": GreedySearch, "gradient": GradientSearch,
           "greedy_restarts": RestartGreedy}


# ─────────────────────── feature-space reference ───────────────────────

def payload_mask() -> np.ndarray:
    m = np.zeros((40, 40), dtype=np.float32)
    for p in range(5):
        m[p * 8 + 2: p * 8 + 8, :] = 1.0
    return m


def header_mask() -> np.ndarray:
    return 1.0 - payload_mask()


def feature_pgd(obj: Objective, mfr: np.ndarray, eps: int, steps: int = 20,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
    """L-infinity PGD directly on MFR bytes (feature space; NOT wire-realizable)."""
    mask = np.ones((40, 40), np.float32) if mask is None else mask
    x0 = mfr.astype(np.float32)
    x = x0.copy()
    alpha = max(1.0, eps / 4.0)
    for _ in range(steps):
        g = obj.grad(np.clip(np.rint(x), 0, 255).astype(np.uint8))
        x = x + alpha * np.sign(g) * mask
        x = np.clip(x, x0 - eps, x0 + eps)
        x = np.clip(x, 0, 255)
        xi = np.clip(np.rint(x), 0, 255).astype(np.uint8)
        if obj.success(xi[None])[0]:
            return xi, True
    xi = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    return xi, bool(obj.success(xi[None])[0])


def sample_chain(flow: Flow, tm: ThreatModel, rng: random.Random,
                 length: Optional[int] = None) -> Tuple[Flow, List[Edit]]:
    """One random edit chain within the budget (used for augmentation and control baselines)."""
    length = length or rng.randint(1, tm.max_steps)
    cur, edits = flow, []
    for _ in range(length):
        options = enumerate_edits(cur, tm.allowed, rng)
        rng.shuffle(options)
        for e in options:
            g = apply_edit(cur, e)
            if g is None:
                continue
            b, n = cost(flow, g)
            if b <= tm.max_extra_bytes and n <= tm.max_extra_packets:
                cur, edits = g, edits + [e]
                break
    return cur, edits
