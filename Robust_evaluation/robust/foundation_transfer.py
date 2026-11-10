"""Foundation-level attack: craft against the PRE-TRAINED encoder only, test on fine-tuned models.

Objective: push the flow's pre-trained embedding away from its own clean embedding until cosine <=
tau_low (the 5th percentile of same-class similarity measured on the test set, i.e. the flow now looks
less like its own class than 95% of genuine same-class pairs).  The resulting flows are then fed to
every downstream classifier given in --ckpts.  A random-edit control with the same budget is reported.
No downstream weights or labels are used while crafting.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robust.attacks import ATTACKS, sample_chain
from robust.data_io import class_names, items_to_mfrs, load_items
from robust.model_io import EmbeddingObjective, Model
from robust.pipeline import flow_to_mfr
from robust.rq1_calibration import distributions
from robust.stats import wilson_ci
from robust.threat_model import BUDGET_PRESETS, ThreatModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows_root", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True, help="downstream fine-tuned models (any seed/task)")
    ap.add_argument("--nb_classes", type=int, default=0)
    ap.add_argument("--attack", default="gradient", choices=["gradient", "greedy"])
    ap.add_argument("--budget", default="L(2KB,4pkt)")
    ap.add_argument("--max_flows", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from attack.utils import load_finetuned_model, load_pretrained_encoder
    dev = a.device if torch.cuda.is_available() else "cpu"
    classes = class_names(a.flows_root, "train")
    n_cls = a.nb_classes or len(classes)
    items = load_items(a.flows_root, "test", classes, a.max_flows)
    mfr = items_to_mfrs(items)
    y = np.array([i.label for i in items])
    enc = Model(load_pretrained_encoder(a.pretrained, dev, n_cls), dev)
    emb = enc.forward(mfr)
    same, _ = distributions(emb, y)
    tau_low = float(np.quantile(same, 0.05))
    print(f"tau_low (5th pct same-class cosine) = {tau_low:.4f}")

    clfs = [Model(load_finetuned_model(c, n_cls, dev), dev) for c in a.ckpts]
    clean_ok = [m.predict(mfr) == y for m in clfs]
    tm = ThreatModel(max_queries=500).with_budget(*BUDGET_PRESETS[a.budget])
    rng = random.Random(a.seed)

    crafted, control, crafted_ok = [], [], []
    for k, it in enumerate(items):
        obj = EmbeddingObjective(enc, emb[k], tau_low, "away")
        r = ATTACKS[a.attack]().run(it.flow, obj, tm, seed=a.seed + k)
        crafted.append(r.mfr)
        crafted_ok.append(r.success)
        g, _ = sample_chain(it.flow, tm, rng, length=max(1, len(r.edits)))
        control.append(flow_to_mfr(g))
    crafted, control = np.stack(crafted), np.stack(control)
    crafted_ok = np.array(crafted_ok)
    rep = dict(tau_low=tau_low, attack=a.attack, budget=a.budget, n=len(items),
               encoder_level_success=float(crafted_ok.mean()), models=[])
    print(f"encoder-level success: {crafted_ok.mean():.3f}")
    for ci, m in enumerate(clfs):
        ok = clean_ok[ci]
        n = int(ok.sum())
        f_craft = int(((m.predict(crafted) != y) & ok).sum())
        f_ctrl = int(((m.predict(control) != y) & ok).sum())
        f_craft_succ = int(((m.predict(crafted) != y) & ok & crafted_ok).sum())
        row = dict(ckpt=a.ckpts[ci], n_clean_correct=n,
                   transfer_flip_rate=f_craft / max(1, n), transfer_ci=wilson_ci(f_craft, n),
                   control_flip_rate=f_ctrl / max(1, n), control_ci=wilson_ci(f_ctrl, n),
                   flip_rate_when_encoder_attack_succeeded=f_craft_succ / max(1, int((ok & crafted_ok).sum())))
        rep["models"].append(row)
        print(row)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "foundation_transfer.json", "w") as fh:
        json.dump(rep, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
