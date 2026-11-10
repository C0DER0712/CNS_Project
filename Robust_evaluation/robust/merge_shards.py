"""Merge sharded eval_attacks outputs into one table with bootstrap CIs.

python -m robust.merge_shards --dir res/evasion --out res/evasion/merged.json
Reads records_model{M}_shard{i}of{n}.csv and summary_shard*.json written by eval_attacks.
A flow counts as a success only if the attack succeeded and the result passed validation.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robust.stats import bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = Path(a.dir)
    frames = []
    for f in sorted(glob.glob(str(d / "records_model*_shard*of*.csv"))):
        m = re.search(r"records_model(\d+)_shard", f)
        df = pd.read_csv(f)
        df["model"] = int(m.group(1))
        frames.append(df)
    if not frames:
        raise SystemExit(f"no record files in {d}")
    df = pd.concat(frames, ignore_index=True)
    df["ok"] = df["success"] & (df["valid"].fillna(True).astype(bool))
    # clean accuracy / sizes from the shard summaries (identical across shards)
    info = {}
    for f in sorted(glob.glob(str(d / "summary_shard*.json"))):
        s = json.load(open(f))
        for mi, m in enumerate(s["models"]):
            info.setdefault(mi, dict(m))
    rows = []
    for (mi, an, bn), g in df.groupby(["model", "attack", "budget"]):
        per_flow = g.groupby("path")["ok"].mean()       # average over seeds inside each flow
        mean, lo, hi = bootstrap_ci(per_flow.values)
        row = dict(model=int(mi), attack=an, budget=bn, asr=mean, lo=lo, hi=hi, n_flows=int(len(per_flow)),
                   mean_extra_bytes_when_success=float(g[g.ok]["extra_bytes"].mean()) if g.ok.any() else None,
                   mean_extra_packets_when_success=float(g[g.ok]["extra_packets"].mean()) if g.ok.any() else None,
                   mean_queries=float(g["queries"].mean()),
                   validity_rate_of_successes=float(g[g.success]["valid"].fillna(True).astype(bool).mean())
                   if g.success.any() else None)
        if mi in info:
            n_test, acc = info[mi]["n_test"], info[mi]["clean_acc"]
            row["robust_acc_all_flows"] = float((acc * n_test - mean * len(per_flow)) / n_test)
        rows.append(row)
    # feature-space PGD reference rows (computed once, in shard 0)
    for f in sorted(glob.glob(str(d / "summary_shard*.json"))):
        for r in json.load(open(f)).get("per_model_rows", []):
            if str(r.get("attack", "")).startswith("pgd"):
                rows.append(dict(model=r["model"], attack=r["attack"], budget="feature-space (not realizable)",
                                 asr=r["asr"], lo=r["lo"], hi=r["hi"], n_flows=r["n_flows"]))
    # per-class ASR for the strongest-budget greedy run (where available)
    by_class = {}
    for (mi, an, bn), g in df.groupby(["model", "attack", "budget"]):
        if an in ("greedy", "gradient"):
            pf = g.groupby(["path", "label"])["ok"].mean().reset_index()
            by_class[f"{mi}|{an}|{bn}"] = {str(int(l)): dict(asr=float(x.ok.mean()), n=int(len(x)))
                                          for l, x in pf.groupby("label")}
    # paired greedy - random difference per budget (same flows, per-flow success averaged over seeds)
    gvr = []
    for (mi, bn), g in df.groupby(["model", "budget"]):
        pf = g.groupby(["attack", "path"])["ok"].mean().unstack(0)
        if "greedy" in pf.columns and "random" in pf.columns:
            pf = pf.dropna(subset=["greedy", "random"])
            d_, lo_, hi_ = bootstrap_ci((pf["greedy"] - pf["random"]).values)
            gvr.append(dict(model=int(mi), budget=bn, n_flows=int(len(pf)), greedy=float(pf["greedy"].mean()),
                            random=float(pf["random"].mean()), diff=d_, lo=lo_, hi=hi_,
                            significant=bool(lo_ > 0 or hi_ < 0)))
    if gvr:
        print("\ngreedy minus random (paired bootstrap):")
        print(pd.DataFrame(gvr).round(3).to_string(index=False))
    out = dict(rows=rows, models=info, by_class=by_class, greedy_vs_random=gvr)
    out_path = Path(a.out) if a.out else d / "merged.json"
    json.dump(out, open(out_path, "w"), indent=2, default=float)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t[["model", "attack", "budget", "asr", "lo", "hi", "n_flows", "robust_acc_all_flows"]].to_string(index=False))
    t.to_csv(out_path.with_suffix(".csv"), index=False)
    try:
        from robust.eval_attacks import plot_curves
        for mi in sorted(t.model.unique()):
            sub = [r for r in rows if r["model"] == int(mi) and not str(r["attack"]).startswith("pgd")]
            if sub:
                plot_curves(sub, d / f"asr_vs_budget_model{int(mi)}.png", f"ASR vs budget (model {int(mi)})")
    except Exception as e:  # plotting must never break the merge
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
