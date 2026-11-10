"""Full-test-set, multi-seed, confidence-interval attack evaluation.

Example
-------
python -m robust.eval_attacks \
    --flows_root /kaggle/working/flows_by_class --mfr_root /kaggle/working/mfr_dataset \
    --ckpts ft_seed0/checkpoint-best.pth ft_seed1/checkpoint-best.pth ft_seed2/checkpoint-best.pth \
    --attacks random greedy gradient --seeds 0 1 2 --goal evasion --out results_v2/evasion
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robust.data_io import class_names, items_to_mfrs, load_items, parity_mask
from robust.model_io import Model
from robust.stats import bootstrap_ci, per_flow_success_ci, wilson_ci
from robust.suite import clean_predictions, run_feature_pgd, run_suite, summarise
from robust.threat_model import BUDGET_PRESETS, ThreatModel


def load_classifier(ckpt: str, n_classes: int, device: str) -> Model:
    from attack.utils import load_finetuned_model
    return Model(load_finetuned_model(ckpt, n_classes, device), device)


def plot_curves(rows, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    budgets = list(dict.fromkeys(r["budget"] for r in rows if not r["attack"].startswith("pgd")))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for an in dict.fromkeys(r["attack"] for r in rows if not r["attack"].startswith("pgd")):
        pts = {r["budget"]: r for r in rows if r["attack"] == an}
        xs = [i for i, b in enumerate(budgets) if b in pts]
        y = np.array([pts[budgets[i]]["asr"] for i in xs])
        lo = np.array([pts[budgets[i]]["lo"] for i in xs])
        hi = np.array([pts[budgets[i]]["hi"] for i in xs])
        ax.plot(xs, y, marker="o", label=an)
        ax.fill_between(xs, lo, hi, alpha=0.2)
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(budgets, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("attack success rate (correctly-classified flows)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows_root", required=True)
    ap.add_argument("--mfr_root", default="", help="PNG dataset the model was trained on (parity filter)")
    ap.add_argument("--keep_parity_mismatch", action="store_true",
                    help="do NOT drop flows whose in-memory MFR differs from the PNG")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpts", nargs="+", required=True, help="one fine-tuned checkpoint per training seed")
    ap.add_argument("--common_with", nargs="*", default=[],
                    help="extra checkpoints: attack only flows that ALL of them also classify correctly "
                         "(matched comparison between models)")
    ap.add_argument("--nb_classes", type=int, default=0, help="0 = infer from folders")
    ap.add_argument("--attacks", nargs="+", default=["random", "greedy", "gradient"])
    ap.add_argument("--budgets", nargs="+", default=list(BUDGET_PRESETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--goal", default="evasion", choices=["evasion", "impersonation"])
    ap.add_argument("--max_flows", type=int, default=0, help="0 = full split")
    ap.add_argument("--max_steps", type=int, default=4)
    ap.add_argument("--max_queries", type=int, default=600)
    ap.add_argument("--time_limit_s", type=float, default=0,
                    help="wall-clock limit for the attack phase of EACH model (0 = none); flows are "
                         "shuffled first so the processed part is a random subset")
    ap.add_argument("--pgd_max_flows", type=int, default=80)
    ap.add_argument("--all_seeds_all_attacks", action="store_true")
    ap.add_argument("--no_pgd", action="store_true")
    ap.add_argument("--no_validity_check", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th flow (parallel runs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import random
    import time
    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from robust import pipeline
    print(f"[mask] fields = {pipeline.MASK_FIELDS or 'none'}")
    classes = class_names(args.flows_root, "train")
    n_cls = args.nb_classes or len(classes)
    items = load_items(args.flows_root, args.split, classes, args.max_flows)
    n_loaded = len(items)
    if args.mfr_root:
        ok = parity_mask(items, args.mfr_root, args.split)
        print(f"[parity] in-memory MFR == PNG for {sum(ok)}/{len(ok)} flows ({sum(ok) / max(1, len(ok)):.3%})")
        if not args.keep_parity_mismatch:
            items = [it for it, k in zip(items, ok) if k]
            print(f"[parity] kept {len(items)}/{n_loaded} flows")
    tm = ThreatModel(goal=args.goal, max_steps=args.max_steps, max_queries=args.max_queries)
    budgets = {k: BUDGET_PRESETS[k] for k in args.budgets}
    si, sn = (int(x) for x in args.shard.split("/"))
    y = np.array([i.label for i in items])
    others = [load_classifier(c, n_cls, device) for c in args.common_with]
    other_ok = np.ones(len(items), dtype=bool)
    for m2 in others:
        other_ok &= clean_predictions(m2, items) == y

    summary = dict(args=vars(args), threat_model=tm.to_dict(), classes=classes, models=[],
                   mask_fields=list(pipeline.MASK_FIELDS))
    all_rows = []
    for ci, ck in enumerate(args.ckpts):
        model = load_classifier(ck, n_cls, device)
        pred = clean_predictions(model, items)
        correct = pred == y
        k = int(correct.sum())
        lo, hi = wilson_ci(k, len(items))
        print(f"[model {ci}] clean acc {k}/{len(items)} = {k / max(1, len(items)):.4f}  (95% Wilson {lo:.4f}-{hi:.4f})")
        todo = [it for j, it in enumerate(items) if correct[j] and other_ok[j]]
        random.Random(0).shuffle(todo)            # identical order in every shard / matched run
        todo = todo[si::sn]
        deadline = time.time() + args.time_limit_s if args.time_limit_s > 0 else None
        res = run_suite(model, todo, n_cls, args.attacks, budgets, args.seeds, tm, args.goal,
                        check_validity=not args.no_validity_check,
                        all_seeds_all_attacks=args.all_seeds_all_attacks, deadline=deadline)
        done = todo[:res["n_done"]]
        rows = summarise(res["success"], len(done))
        if not args.no_pgd and done:
            pg = run_feature_pgd(model, done[:args.pgd_max_flows], n_cls, goal=args.goal)
            for name, arr in pg.items():
                r = per_flow_success_ci(arr)
                r.update(attack=name, budget="feature-space")
                rows.append(r)
        for r in rows:
            r["model"] = ci
            all_rows.append(r)
        summary["models"].append(dict(ckpt=ck, clean_acc=k / max(1, len(items)), clean_acc_ci=[lo, hi],
                                      n_test=len(items), n_loaded=n_loaded, n_attacked=len(done),
                                      n_candidates=len(todo), matched_with=list(args.common_with)))
        if res["records"]:
            with open(out / f"records_model{ci}_shard{si}of{sn}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(res["records"][0].keys()))
                w.writeheader()
                w.writerows(res["records"])
        plot_curves(rows, out / f"asr_vs_budget_model{ci}_shard{si}.png",
                    f"{args.goal}: ASR vs budget (model {ci}, n={len(done)})")

    pooled = []
    keys = sorted({(r["attack"], r["budget"]) for r in all_rows})
    for a_, b_ in keys:
        v = [r["asr"] for r in all_rows if r["attack"] == a_ and r["budget"] == b_]
        pooled.append(dict(attack=a_, budget=b_, mean_asr=float(np.mean(v)),
                           std_across_models=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                           n_models=len(v)))
    summary["per_model_rows"] = all_rows
    summary["pooled_across_models"] = pooled
    with open(out / f"summary_shard{si}of{sn}.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("\n=== ASR (mean over flows, 95% bootstrap CI over flows) ===")
    for r in all_rows:
        print(f"model{r['model']} {r['attack']:<18s} {r['budget']:<16s} "
              f"{r['asr']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] n={r['n_flows']}")


if __name__ == "__main__":
    main()
