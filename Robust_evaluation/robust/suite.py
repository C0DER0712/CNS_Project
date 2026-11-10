"""Run an attack suite over flows and return per-flow outcomes (shared by all scripts)."""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from robust.attacks import ATTACKS, AttackResult, clear_cache, feature_pgd, header_mask, payload_mask
from robust.data_io import Item
from robust.model_io import ClassifierObjective, Model
from robust.pcap_ops import cost
from robust.pipeline import flow_to_mfr, introduced_violations
from robust.stats import per_flow_success_ci, wilson_ci
from robust.threat_model import BUDGET_PRESETS, ThreatModel

DETERMINISTIC = {"greedy", "gradient", "greedy_restarts"}


def clean_predictions(model: Model, items: Sequence[Item]) -> np.ndarray:
    mf = np.stack([flow_to_mfr(i.flow) for i in items])
    return model.predict(mf)


def pick_target(label: int, n_classes: int, rng: random.Random) -> int:
    t = rng.randrange(n_classes - 1)
    return t + 1 if t >= label else t


def run_suite(model: Model, items: Sequence[Item], n_classes: int, attacks: Sequence[str],
              budgets: Dict[str, Tuple[int, int]], seeds: Sequence[int], tm_base: ThreatModel,
              goal: str = "evasion", check_validity: bool = True,
              all_seeds_all_attacks: bool = False, progress_every: int = 25,
              attack_kwargs: Optional[Dict] = None, deadline: Optional[float] = None) -> Dict:
    """Attack every item in `items` (callers pass only correctly classified flows).

    deadline: absolute time.time() after which no new flow is started (flows should be pre-shuffled
    so the processed prefix is a random subset).  Returns
    {"records": [...], "success": {(attack,budget): bool array (n_seeds, n_done)}, "n_done": int}.
    """
    attack_kwargs = attack_kwargs or {}
    records: List[Dict] = []
    success: Dict[Tuple[str, str], np.ndarray] = {}
    t0 = time.time()
    for an in attacks:
        for bn in budgets:
            n_seeds = len(seeds) if (an not in DETERMINISTIC or all_seeds_all_attacks) else 1
            success[(an, bn)] = np.zeros((n_seeds, len(items)), dtype=bool)
    # Flow-major loop so the per-flow edit cache (attacks._CACHE) is shared by every
    # attack / budget / seed that touches the same flow.
    n_done = 0
    for k, it in enumerate(items):
        if deadline is not None and time.time() > deadline:
            print(f"  time limit reached after {k}/{len(items)} flows", flush=True)
            break
        n_done = k + 1
        clear_cache()
        for an in attacks:
            seeds_for = list(seeds) if (an not in DETERMINISTIC or all_seeds_all_attacks) else [seeds[0]]
            for bn, (bb, bp) in budgets.items():
                tm = tm_base.with_budget(bb, bp)
                for si, seed in enumerate(seeds_for):
                    trng = random.Random(10_000 + seed * 100003 + k)
                    target = pick_target(it.label, n_classes, trng) if goal == "impersonation" else None
                    obj = ClassifierObjective(model, it.label, target)
                    atk = ATTACKS[an](**attack_kwargs.get(an, {}))
                    res: AttackResult = atk.run(it.flow, obj, tm, seed=seed)
                    valid = None
                    if res.success and check_validity:
                        valid = len(introduced_violations(it.flow, res.flow)) == 0
                    success[(an, bn)][si, k] = res.success and (valid is not False)
                    records.append(dict(
                        attack=an, budget=bn, seed=seed, idx=k, path=it.path, label=it.label,
                        capture=it.capture, target=-1 if target is None else target,
                        success=bool(res.success), valid=valid, extra_bytes=res.extra_bytes,
                        extra_packets=res.extra_packets, queries=res.queries,
                        edits=";".join(str(e) for e in res.edits)))
        if progress_every and (k + 1) % progress_every == 0:
            print(f"  {k + 1}/{len(items)} flows done ({time.time() - t0:.0f}s)", flush=True)
    for key in list(success):
        success[key] = success[key][:, :n_done]
    return {"records": records, "success": success, "n_done": n_done}


def run_feature_pgd(model: Model, items: Sequence[Item], n_classes: int,
                    eps_list: Sequence[int] = (4, 8, 16, 32, 64), goal: str = "evasion",
                    seed: int = 0) -> Dict[str, np.ndarray]:
    """Feature-space reference.  Returns {f"pgd_{mask}_eps{e}": bool array (1, n)}."""
    out = {}
    trng = random.Random(10_000 + seed)
    targets = [pick_target(i.label, n_classes, trng) if goal == "impersonation" else None for i in items]
    mfrs = [flow_to_mfr(i.flow) for i in items]
    for mname, mask in (("all", None), ("payload", payload_mask()), ("header", header_mask())):
        for e in eps_list:
            arr = np.zeros((1, len(items)), dtype=bool)
            for k, it in enumerate(items):
                obj = ClassifierObjective(model, it.label, targets[k])
                _, ok = feature_pgd(obj, mfrs[k], e, steps=20, mask=mask)
                arr[0, k] = ok
            out[f"pgd_{mname}_eps{e}"] = arr
    return out


def summarise(success: Dict[Tuple[str, str], np.ndarray], n_eval: int) -> List[Dict]:
    rows = []
    for (an, bn), arr in success.items():
        r = per_flow_success_ci(arr)
        r.update(attack=an, budget=bn)
        rows.append(r)
    return rows
