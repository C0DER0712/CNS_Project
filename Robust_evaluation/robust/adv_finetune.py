"""Defence + adaptive attack: obfuscation-aware augmentation and adversarial fine-tuning.

Training signal (all problem-space, all produced by pcap_ops):
  * clean flows
  * random edit chains ("obfuscation-aware" augmentation, no model needed)
  * per-round adversarial examples found by GreedySearch against the CURRENT model

Optional --holdout <edit kind> removes one edit family from training/augmentation so the adaptive
attacker (which may use every family) tests generalisation, not memorisation.

Evaluation of both the original and the defended model on the same test flows:
  * clean accuracy (the price of the defence)
  * static attacks (random, greedy) at each budget
  * ADAPTIVE attack: greedy with restarts, 2x steps, all edit families, defender weights known
  * transfer: adversarial flows crafted on the undefended model replayed on the defended one
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robust.attacks import ATTACKS, GreedySearch, sample_chain
from robust.data_io import class_names, items_to_mfrs, load_items
from robust.model_io import ClassifierObjective, Model, to_tensor
from robust.pipeline import flow_to_mfr
from robust.stats import wilson_ci
from robust.suite import clean_predictions, run_suite, summarise
from robust.threat_model import BUDGET_PRESETS, ThreatModel


def make_aug(items, tm: ThreatModel, per_flow: int, rng: random.Random):
    X, Y = [], []
    for it in items:
        for _ in range(per_flow):
            bb, bp = rng.choice(list(BUDGET_PRESETS.values()))
            g, _e = sample_chain(it.flow, tm.with_budget(bb, bp), rng)
            X.append(flow_to_mfr(g))
            Y.append(it.label)
    return (np.stack(X) if X else np.zeros((0, 40, 40), np.uint8)), np.array(Y, dtype=np.int64)


def attack_pool(model: Model, items, n_cls: int, tm: ThreatModel, n: int, rng: random.Random, seed: int):
    sel = rng.sample(range(len(items)), min(n, len(items)))
    X, Y = [], []
    for k in sel:
        it = items[k]
        if model.predict(flow_to_mfr(it.flow)[None])[0] != it.label:
            continue
        res = GreedySearch().run(it.flow, ClassifierObjective(model, it.label), tm, seed=seed + k)
        if res.success:
            X.append(res.mfr)
            Y.append(it.label)
    return (np.stack(X) if X else np.zeros((0, 40, 40), np.uint8)), np.array(Y, dtype=np.int64)


def train_epochs(model: Model, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float, bs: int = 64,
                 seed: int = 0):
    net, dev = model.net, model.device
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).numpy()
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            x = to_tensor(X[idx], dev)
            y = torch.from_numpy(Y[idx]).to(dev)
            loss = F.cross_entropy(net(x), y, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        print(f"    epoch {ep + 1}/{epochs} loss {tot / n:.4f}", flush=True)
    net.eval()


def evaluate(model: Model, test_items, n_cls, tm, budgets, seeds, label):
    pred = clean_predictions(model, test_items)
    y = np.array([i.label for i in test_items])
    k = int((pred == y).sum())
    lo, hi = wilson_ci(k, len(y))
    todo = [it for j, it in enumerate(test_items) if pred[j] == y[j]]
    out = dict(label=label, clean_acc=k / len(y), clean_acc_ci=[lo, hi], n_test=len(y), attacks=[])
    static = run_suite(model, todo, n_cls, ["random", "greedy"], budgets, seeds, tm)
    for r in summarise(static["success"], len(todo)):
        r["kind"] = "static"
        out["attacks"].append(r)
    adaptive = run_suite(model, todo, n_cls, ["greedy_restarts"], budgets, seeds, tm,
                         attack_kwargs={"greedy_restarts": dict(restarts=4, step_mult=2)})
    for r in summarise(adaptive["success"], len(todo)):
        r["kind"] = "adaptive"
        out["attacks"].append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows_root", required=True)
    ap.add_argument("--init_ckpt", required=True, help="fine-tuned (undefended) classifier")
    ap.add_argument("--nb_classes", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--epochs_per_round", type=int, default=5)
    ap.add_argument("--aug_per_flow", type=int, default=4)
    ap.add_argument("--attack_flows", type=int, default=200, help="train flows attacked per round")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train_budget", default="L(2KB,4pkt)")
    ap.add_argument("--holdout", default="", help="edit kind withheld from training, e.g. keepalive")
    ap.add_argument("--eval_budgets", nargs="+", default=["free(0B,0pkt)", "M(512B,2pkt)", "L(2KB,4pkt)"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max_test_flows", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from attack.utils import load_finetuned_model
    dev = a.device if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = class_names(a.flows_root, "train")
    n_cls = a.nb_classes or len(classes)
    train_items = load_items(a.flows_root, "train", classes)
    test_items = load_items(a.flows_root, "test", classes, a.max_test_flows)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    base = Model(load_finetuned_model(a.init_ckpt, n_cls, dev), dev)
    defended = Model(load_finetuned_model(a.init_ckpt, n_cls, dev), dev)

    tm_full = ThreatModel()
    tm_train = tm_full.without((a.holdout,)) if a.holdout else tm_full
    bb, bp = BUDGET_PRESETS[a.train_budget]
    tm_train = tm_train.with_budget(bb, bp)

    X_clean, Y_clean = items_to_mfrs(train_items), np.array([i.label for i in train_items])
    X_aug, Y_aug = make_aug(train_items, tm_train, a.aug_per_flow, rng)
    print(f"[defence] clean={len(X_clean)} augmented={len(X_aug)}")
    pool_X = np.zeros((0, 40, 40), np.uint8)
    pool_Y = np.zeros((0,), np.int64)
    for r in range(a.rounds):
        print(f"[round {r + 1}/{a.rounds}] attacking current model")
        ax, ay = attack_pool(defended, train_items, n_cls, tm_train, a.attack_flows, rng, a.seed * 100 + r)
        print(f"    collected {len(ax)} adversarial flows")
        pool_X, pool_Y = np.concatenate([pool_X, ax]), np.concatenate([pool_Y, ay])
        X = np.concatenate([X_clean, X_aug, pool_X])
        Y = np.concatenate([Y_clean, Y_aug, pool_Y])
        train_epochs(defended, X, Y, a.epochs_per_round, a.lr, seed=a.seed + r)
    torch.save({"model": defended.net.state_dict()}, out / "defended.pth")

    budgets = {k: BUDGET_PRESETS[k] for k in a.eval_budgets}
    tm_eval = ThreatModel()   # attacker may use EVERY edit family (incl. the held-out one)
    res = dict(args=vars(a), undefended=evaluate(base, test_items, n_cls, tm_eval, budgets, a.seeds, "undefended"),
               defended=evaluate(defended, test_items, n_cls, tm_eval, budgets, a.seeds, "defended"))

    # transfer: craft on undefended, replay on defended
    tm_t = tm_eval.with_budget(*BUDGET_PRESETS[a.eval_budgets[-1]])
    adv, labels = [], []
    pred_b = clean_predictions(base, test_items)
    pred_d = clean_predictions(defended, test_items)
    for j, it in enumerate(test_items):
        if pred_b[j] != it.label or pred_d[j] != it.label:
            continue
        r_ = GreedySearch().run(it.flow, ClassifierObjective(base, it.label), tm_t, seed=a.seed)
        if r_.success:
            adv.append(r_.mfr)
            labels.append(it.label)
    if adv:
        fooled = int((defended.predict(np.stack(adv)) != np.array(labels)).sum())
        res["transfer_undefended_to_defended"] = dict(
            n_adv=len(adv), fooled=fooled, rate=fooled / len(adv), ci=wilson_ci(fooled, len(adv)))
    with open(out / "defence_report.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    for key in ("undefended", "defended"):
        d = res[key]
        print(f"\n{key}: clean acc {d['clean_acc']:.4f} {d['clean_acc_ci']}")
        for r in d["attacks"]:
            print(f"   {r['kind']:<9s}{r['attack']:<18s}{r['budget']:<16s}{r['asr']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]")
    print("transfer:", res.get("transfer_undefended_to_defended"))


if __name__ == "__main__":
    main()
