"""RQ1 done properly: calibrate the collision threshold, then test whether collisions flip predictions.

1. Same-class vs different-class cosine distributions on the TEST flows (pre-trained encoder
   and, optionally, the fine-tuned model's feature space).  Reports percentiles and AUC.
2. Threshold candidates derived from those distributions (NOT the arbitrary 0.85):
     tau_diff95 : 95th percentile of different-class similarity
     tau_same05 : 5th percentile of same-class similarity
3. Collision search with the pcap-level attacks (source flow -> embedding of a target-class flow).
   Pairs whose CLEAN similarity already exceeds tau are reported separately ("trivial"), never counted
   as attack successes.
4. Prediction-flip rate: does a collision change the fine-tuned classifier's decision?
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robust.attacks import ATTACKS
from robust.data_io import class_names, items_to_mfrs, load_items
from robust.model_io import EmbeddingObjective, Model
from robust.stats import wilson_ci
from robust.threat_model import BUDGET_PRESETS, ThreatModel


def cos_matrix(e: np.ndarray) -> np.ndarray:
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    return e @ e.T


def distributions(emb: np.ndarray, y: np.ndarray):
    S = cos_matrix(emb)
    iu = np.triu_indices(len(y), 1)
    sims, same = S[iu], (y[:, None] == y[None, :])[iu]
    return sims[same], sims[~same]


def hist_plot(same, diff, taus, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(same.min(), diff.min()), 1.0, 60)
    ax.hist(diff, bins=bins, alpha=0.6, density=True, label="different class")
    ax.hist(same, bins=bins, alpha=0.6, density=True, label="same class")
    for n, t in taus.items():
        ax.axvline(t, ls="--", lw=1, label=f"{n}={t:.3f}")
    ax.set_xlabel("cosine similarity")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows_root", required=True)
    ap.add_argument("--pretrained", required=True, help="pre-trained_TACO.pth")
    ap.add_argument("--ckpt", required=True, help="fine-tuned classifier checkpoint")
    ap.add_argument("--nb_classes", type=int, default=0)
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--budget", default="L(2KB,4pkt)")
    ap.add_argument("--attack", default="greedy", choices=["greedy", "gradient", "random"])
    ap.add_argument("--max_flows", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--center", action="store_true",
                    help="subtract the mean test embedding before taking cosines (removes the shared offset)")
    a = ap.parse_args()

    import torch
    from attack.utils import load_finetuned_model, load_pretrained_encoder
    dev = a.device if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = class_names(a.flows_root, "train")
    n_cls = a.nb_classes or len(classes)
    items = load_items(a.flows_root, "test", classes, a.max_flows)
    mfr = items_to_mfrs(items)
    y = np.array([i.label for i in items])

    enc = Model(load_pretrained_encoder(a.pretrained, dev, n_cls), dev)
    clf = Model(load_finetuned_model(a.ckpt, n_cls, dev), dev)
    emb = enc.forward(mfr)
    mu = emb.mean(0) if a.center else None
    if a.center:
        emb = emb - mu
        print("[rq1] using mean-centred embeddings")

    same, diff = distributions(emb, y)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(np.r_[np.ones(len(same)), np.zeros(len(diff))], np.r_[same, diff])
    taus = {"tau_diff95": float(np.quantile(diff, 0.95)), "tau_same05": float(np.quantile(same, 0.05)),
            "arbitrary_0.85": 0.85}
    pct = lambda v: {f"p{q}": float(np.quantile(v, q / 100)) for q in (5, 25, 50, 75, 95)}
    report = dict(n_flows=len(y), centered=bool(a.center), auc_same_vs_diff=float(auc), same=pct(same), diff=pct(diff),
                  frac_diff_pairs_above_0_85=float((diff > 0.85).mean()), taus=taus)
    print(json.dumps(report, indent=2))
    hist_plot(same, diff, taus, out / "cosine_distributions.png", "Pre-trained encoder: test-set cosine")

    # ---- collision search ----
    rng = random.Random(a.seed)
    by_cls = {c: [i for i, it in enumerate(items) if it.label == c] for c in range(n_cls)}
    bb, bp = BUDGET_PRESETS[a.budget]
    tm = ThreatModel(max_queries=400).with_budget(bb, bp)
    pairs = []
    while len(pairs) < a.n_pairs:
        s = rng.randrange(len(items))
        tc = rng.choice([c for c in range(n_cls) if c != items[s].label and by_cls[c]])
        pairs.append((s, rng.choice(by_cls[tc])))

    tau = taus["tau_diff95"]
    res = dict(n=0, trivial=0, attacked=0, collided=0, flipped_to_target=0, flipped_any=0,
               collided_and_flip_target=0, no_collision_but_flip=0)
    for (s, t) in pairs:
        res["n"] += 1
        e_t = emb[t]
        et_n = e_t / np.linalg.norm(e_t)
        clean_cos = float((emb[s] / np.linalg.norm(emb[s])) @ et_n)
        if clean_cos >= tau:
            res["trivial"] += 1
            continue
        res["attacked"] += 1
        obj = EmbeddingObjective(enc, e_t, tau, "toward", center=mu)
        r = ATTACKS[a.attack]().run(items[s].flow, obj, tm, seed=a.seed)
        p = int(clf.predict(r.mfr[None])[0])
        flip_target = p == items[t].label
        flip_any = p != items[s].label
        res["flipped_any"] += int(flip_any)
        res["flipped_to_target"] += int(flip_target)
        if r.success:
            res["collided"] += 1
            res["collided_and_flip_target"] += int(flip_target)
        else:
            res["no_collision_but_flip"] += int(flip_any)
    n = max(1, res["attacked"])
    res["collision_rate"] = res["collided"] / n
    res["collision_rate_ci"] = wilson_ci(res["collided"], n)
    res["flip_rate_given_collision"] = res["collided_and_flip_target"] / max(1, res["collided"])
    res["tau"], res["attack"], res["budget"] = tau, a.attack, a.budget
    report["collision_study"] = res
    print(json.dumps(res, indent=2, default=float))
    with open(out / "rq1_calibrated.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
