"""Is the pre-trained TaCo embedding class-informative?  (fixes the RQ1 near-chance AUC question)

Compares, on the PNG MFR dataset:
  pretrained   : released encoder, head=Identity
  random_init  : same architecture, random weights (control)
  finetuned    : fine-tuned classifier with its head removed (upper reference)
For each: linear probe (logistic regression on standardised embeddings, trained on the TRAIN split),
5-NN cosine accuracy, and AUC of "same class vs different class" using raw cosine, mean-centred
cosine and whitened (standardised) cosine, on the TEST split.  Centring removes the shared offset
that makes every raw cosine > 0.9.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robust.model_io import Model
from robust.stats import wilson_ci


def load_split(root, split):
    classes = sorted(p.name for p in (Path(root) / "train").iterdir() if p.is_dir())
    X, y = [], []
    for ci, c in enumerate(classes):
        for f in sorted((Path(root) / split / c).glob("*.png")):
            X.append(np.array(Image.open(f).convert("L"), dtype=np.uint8))
            y.append(ci)
    return np.stack(X), np.array(y), classes


def cos(e):
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    return e @ e.T


def pair_auc(emb, y):
    from sklearn.metrics import roc_auc_score
    S = cos(emb)
    iu = np.triu_indices(len(y), 1)
    same = (y[:, None] == y[None, :])[iu]
    return float(roc_auc_score(same.astype(int), S[iu]))


def evaluate_embeddings(Etr, ytr, Ete, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    mu = Etr.mean(0, keepdims=True)
    sc = StandardScaler().fit(Etr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Etr), ytr)
    pred = clf.predict(sc.transform(Ete))
    k = int((pred == yte).sum())
    lo, hi = wilson_ci(k, len(yte))
    # 5-NN with cosine on centred embeddings
    A = Etr - mu
    B = Ete - mu
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    nn_idx = np.argsort(-(B @ A.T), axis=1)[:, :5]
    votes = ytr[nn_idx]
    knn = np.array([np.bincount(v, minlength=int(ytr.max()) + 1).argmax() for v in votes])
    return dict(
        probe_acc=k / len(yte), probe_ci=[lo, hi], probe_macro_f1=float(f1_score(yte, pred, average="macro")),
        knn5_acc=float((knn == yte).mean()),
        auc_raw_cosine=pair_auc(Ete, yte),
        auc_centered_cosine=pair_auc(Ete - mu, yte),
        auc_whitened_cosine=pair_auc(sc.transform(Ete), yte),
        mean_raw_cosine=float(cos(Ete)[np.triu_indices(len(yte), 1)].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfr_root", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--ckpt", required=True, help="fine-tuned checkpoint")
    ap.add_argument("--nb_classes", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from attack.utils import _build_encoder, load_finetuned_model, load_pretrained_encoder
    dev = a.device if torch.cuda.is_available() else "cpu"
    Xtr, ytr, classes = load_split(a.mfr_root, "train")
    Xte, yte, _ = load_split(a.mfr_root, "test")
    n_cls = a.nb_classes or len(classes)
    print(f"train {len(Xtr)}  test {len(Xte)}  classes {classes}")

    torch.manual_seed(0)
    rand = _build_encoder(nb_classes=n_cls)
    rand.head = nn.Identity()
    ft = load_finetuned_model(a.ckpt, n_cls, dev)
    ft.head = nn.Identity()
    models = {"pretrained": load_pretrained_encoder(a.pretrained, dev, n_cls),
              "random_init": rand, "finetuned": ft}
    report = {}
    for name, net in models.items():
        m = Model(net, dev)
        report[name] = evaluate_embeddings(m.forward(Xtr), ytr, m.forward(Xte), yte)
        print(name, json.dumps(report[name], indent=1))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "linear_probe.json", "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
