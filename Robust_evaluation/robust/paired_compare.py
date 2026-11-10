"""Paired comparison of two models attacked on the SAME flows (see eval_attacks --common_with).

python -m robust.paired_compare --a res/matched_pre --b res/matched_scratch --name_a pretrained --name_b scratch
Both directories hold records_model0_shard*.csv.  Flows are joined on pcap path; only flows
processed in both runs are used.  Reports ASR_b - ASR_a with a paired bootstrap 95% CI.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robust.stats import paired_diff_ci


def load(d):
    fs = sorted(glob.glob(str(Path(d) / "records_model0_shard*of*.csv")))
    if not fs:
        raise SystemExit(f"no records in {d}")
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["ok"] = df["success"] & df["valid"].fillna(True).astype(bool)
    return df.groupby(["attack", "budget", "path"])["ok"].mean().rename("ok").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    m = A.merge(B, on=["attack", "budget", "path"], suffixes=("_a", "_b"))
    rows = []
    for (an, bn), g in m.groupby(["attack", "budget"]):
        mean_a, mean_b = g.ok_a.mean(), g.ok_b.mean()
        d, lo, hi = paired_diff_ci(g.ok_b.values, g.ok_a.values)
        rows.append(dict(attack=an, budget=bn, n_flows=len(g), **{f"asr_{args.name_a}": mean_a,
                         f"asr_{args.name_b}": mean_b}, diff_b_minus_a=d, diff_lo=lo, diff_hi=hi,
                         significant=bool(lo > 0 or hi < 0)))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(t.to_string(index=False))
    out = Path(args.out or Path(args.a).parent / "paired_compare.json")
    out.write_text(json.dumps(rows, indent=2, default=float))
    t.to_csv(out.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
