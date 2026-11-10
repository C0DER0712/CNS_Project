"""Capture-disjoint split: all flows of a source capture go to the SAME side.

The existing split_flows_and_bucket.py splits flows at random, so flows from one pcap appear in both
train and test (same IPs, same ports, same host stack).  This script regroups the already-bucketed
flows by source capture (filename stem before "_flow<N>") and assigns whole captures to test until
~test_ratio of each class is reached.  Classes with a single capture cannot be split this way and
are reported (and by default excluded, unless --allow_single_capture, which falls back to a
time-ordered split inside that capture).

Usage:
  python -m robust.make_capture_split --flows_root flows_by_class --out flows_by_class_capdisjoint
  python -c "from data_process import MFR_generator; MFR_generator('flows_by_class_capdisjoint','mfr_capdisjoint')"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


def cap_of(p: Path) -> str:
    return re.sub(r"_flow\d+$", "", p.stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_single_capture", action="store_true")
    ap.add_argument("--symlink", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    root, out = Path(a.flows_root), Path(a.out)
    by_cls = defaultdict(lambda: defaultdict(list))
    for split in ("train", "test"):
        for p in sorted((root / split).glob("*/*.pcap")):
            by_cls[p.parent.name][cap_of(p)].append(p)
    report = {}
    for cls, caps in by_cls.items():
        names = sorted(caps)
        total = sum(len(v) for v in caps.values())
        if len(names) < 2:
            report[cls] = dict(status="single_capture", captures=names, flows=total)
            if not a.allow_single_capture:
                print(f"[skip] class {cls}: only one capture ({names[0]}); no capture-disjoint split possible")
                continue
            flows = sorted(caps[names[0]], key=lambda p: int(re.search(r"flow(\d+)$", p.stem).group(1)))
            k = max(1, int(len(flows) * a.test_ratio))
            test_f, train_f = flows[-k:], flows[:-k]
        else:
            order = names[:]
            rng.shuffle(order)
            test_caps, n_test = [], 0
            for c in order:
                if n_test >= a.test_ratio * total or len(test_caps) == len(order) - 1:
                    break
                test_caps.append(c)
                n_test += len(caps[c])
            test_f = [p for c in test_caps for p in caps[c]]
            train_f = [p for c in names if c not in test_caps for p in caps[c]]
            report[cls] = dict(status="ok", test_captures=test_caps,
                               train_captures=[c for c in names if c not in test_caps],
                               n_train=len(train_f), n_test=len(test_f))
        for split, fl in (("train", train_f), ("test", test_f)):
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for p in fl:
                (d / p.name).unlink(missing_ok=True)
                if a.symlink:
                    (d / p.name).symlink_to(p.resolve())
                else:
                    shutil.copy2(p, d / p.name)
        print(f"[{cls:14s}] train={len(train_f):4d} test={len(test_f):4d}  "
              f"{report[cls].get('test_captures', 'time-ordered split')}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "split_report.json", "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
