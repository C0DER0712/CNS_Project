import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust.attacks import GreedySearch, GradientSearch, RandomSearch, RestartGreedy
from robust.model_io import Objective
from robust.pcap_ops import apply_edits, cost
from robust.pipeline import flow_to_mfr, introduced_violations
from robust.tests.test_pcap_ops import synth_flow
from robust.threat_model import ThreatModel


class ToyObjective(Objective):
    """Score = how many bytes of packet 3's payload region are non-zero (needs padding)."""
    has_grad = True
    name = "toy"

    def scores(self, mfrs):
        return np.array([float((m[26:32, :] > 0).sum()) for m in mfrs])

    def success(self, mfrs):
        return self.scores(mfrs) >= 200

    def grad(self, mfr):
        g = np.zeros((40, 40), np.float32)
        g[26:32, :] = 1.0
        return g


def _check(res, f, tm):
    b, n = cost(f, res.flow)
    assert b <= tm.max_extra_bytes and n <= tm.max_extra_packets
    assert introduced_violations(f, res.flow) == []
    assert np.array_equal(flow_to_mfr(res.flow), res.mfr)


def test_attacks_respect_budget_and_validity():
    f = synth_flow(0)
    tm = ThreatModel(max_extra_bytes=2048, max_extra_packets=4, max_steps=3, max_queries=300)
    for A in (RandomSearch(), GreedySearch(), GradientSearch(), RestartGreedy(restarts=2)):
        res = A.run(f, ToyObjective(), tm, seed=0)
        _check(res, f, tm)


def test_zero_budget_only_uses_free_edits():
    f = synth_flow(0)
    tm = ThreatModel(max_extra_bytes=0, max_extra_packets=0, max_steps=3, max_queries=200)
    res = GreedySearch().run(f, ToyObjective(), tm)
    assert all(e.kind in ("ttl", "window", "ipid") for e in res.edits)
    assert (res.extra_bytes, res.extra_packets) == (0, 0)


def test_greedy_beats_random_on_toy():
    f = synth_flow(0)
    tm = ThreatModel(max_extra_bytes=4096, max_extra_packets=4, max_steps=3, max_queries=150)
    g = GreedySearch().run(f, ToyObjective(), tm)
    assert g.final_score >= RandomSearch().run(f, ToyObjective(), tm).final_score - 1e-9


def test_random_search_respects_chain_cap_and_budget():
    from robust.attacks import RandomSearch, clear_cache
    clear_cache()
    f = synth_flow(1)
    tm = ThreatModel(max_extra_bytes=512, max_extra_packets=2, max_steps=3, max_queries=500)

    class Never(ToyObjective):
        def success(self, mfrs):
            return np.zeros(len(mfrs), bool)

    res = RandomSearch(max_chains=40).run(f, Never(), tm, seed=3)
    assert res.queries <= 41
    _check(res, f, tm)
