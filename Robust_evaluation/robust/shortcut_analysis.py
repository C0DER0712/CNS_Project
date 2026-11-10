"""Which header fields does the classifier lean on?  (shortcut / attackability audit)

MFR header = raw IP bytes (data_process.read_MFR_bytes keeps src/dst IP, ports and TCP seq/ack).
For every field of the standard IPv4+TCP layout we:
  * zero it in every packet                      -> accuracy drop
  * replace it with the value from a random flow -> accuracy drop (shuffle)
  * gradient x input attribution                 -> mean |g*x| mass per field
Field offsets assume IPv4 with a 20-byte header and TCP; UDP flows reuse the same offsets, so treat
UDP-heavy classes with care (listed in the output).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robust.data_io import class_names, items_to_mfrs, load_items
from robust.model_io import ClassifierObjective, Model
from robust.stats import wilson_ci

FIELDS = {  # name: (start, end) inside the 80-byte header
    "ip_ver_ihl_tos": (0, 2), "ip_total_len": (2, 4), "ip_id": (4, 6), "ip_flags_frag": (6, 8),
    "ip_ttl": (8, 9), "ip_proto": (9, 10), "ip_checksum": (10, 12), "ip_src": (12, 16),
    "ip_dst": (16, 20), "tcp_sport": (20, 22), "tcp_dport": (22, 24), "tcp_seq": (24, 28),
    "tcp_ack": (28, 32), "tcp_off_flags": (32, 34), "tcp_window": (34, 36),
    "tcp_checksum": (36, 38), "tcp_urg": (38, 40), "tcp_options_and_early_payload": (40, 80),
}


def field_mask(name: str, packets=range(5)) -> np.ndarray:
    s, e = FIELDS[name]
    m = np.zeros((40, 40), dtype=bool)
    for p in packets:
        for k in range(s, e):
            m[p * 8 + k // 40, k % 40] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nb_classes", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_flows", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from attack.utils import load_finetuned_model
    dev = a.device if torch.cuda.is_available() else "cpu"
    classes = class_names(a.flows_root, "train")
    n_cls = a.nb_classes or len(classes)
    items = load_items(a.flows_root, a.split, classes, a.max_flows)
    mfr = items_to_mfrs(items)
    y = np.array([i.label for i in items])
    model = Model(load_finetuned_model(a.ckpt, n_cls, dev), dev)
    base = model.predict(mfr)
    base_acc = float((base == y).mean())
    rng = np.random.default_rng(a.seed)
    rows = []
    for name in FIELDS:
        m = field_mask(name)
        z = mfr.copy()
        z[:, m] = 0
        acc_zero = float((model.predict(z) == y).mean())
        perm = rng.permutation(len(mfr))
        s = mfr.copy()
        s[:, m] = mfr[perm][:, m]
        acc_shuf = float((model.predict(s) == y).mean())
        # gradient x input attribution (loss = -log p_y approximated by margin gradient)
        mass = 0.0
        for k in range(len(mfr)):
            g = ClassifierObjective(model, int(y[k])).grad(mfr[k])
            mass += float(np.abs(g * (mfr[k].astype(np.float32) / 255.0))[m].sum())
        rows.append(dict(field=name, acc_zero=acc_zero, acc_shuffle=acc_shuf,
                         drop_zero=base_acc - acc_zero, drop_shuffle=base_acc - acc_shuf,
                         attribution=mass / len(mfr), n_bytes=int(m.sum())))
    tot = sum(r["attribution"] for r in rows) + 1e-12
    for r in rows:
        r["attribution_share"] = r["attribution"] / tot
    rows.sort(key=lambda r: -r["drop_shuffle"])
    lo, hi = wilson_ci(int(round(base_acc * len(y))), len(y))
    rep = dict(base_acc=base_acc, base_acc_ci=[lo, hi], n=len(y), fields=rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "shortcut_fields.json", "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"clean acc {base_acc:.4f}")
    print(f"{'field':<32s}{'zero':>8s}{'shuffle':>9s}{'attr%':>8s}")
    for r in rows:
        print(f"{r['field']:<32s}{r['acc_zero']:8.3f}{r['acc_shuffle']:9.3f}{100 * r['attribution_share']:8.1f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([r["field"] for r in rows][::-1], [r["drop_shuffle"] for r in rows][::-1])
    ax.set_xlabel("accuracy drop when field is replaced by another flow's value")
    fig.tight_layout()
    fig.savefig(Path(a.out) / "shortcut_fields.png", dpi=200)


if __name__ == "__main__":
    main()
