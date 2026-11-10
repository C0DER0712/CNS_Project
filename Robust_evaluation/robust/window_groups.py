"""RQ3 corrected: analyse the REAL window grouping and attribute drift to it.

Facts established by tracing flow_window_attention (eval mode), verified numerically below:
  * Packet windows: 5 groups of 80 tokens; group p = packet p exactly.
  * "Flow" windows: 4 groups of 100 tokens each.  The released code does NOT group the same byte position
    across packets.  Group g takes a contiguous run of the packet-ordered, internally permuted token list:
    group0 = packet0 (80) + 20 tokens of packet1, group1 = 60 of packet1 + 40 of packet2, group2 = 40 of
    packet2 + 60 of packet3, group3 = 20 of packet3 + packet4 (80).  So each flow window is ~1 packet,
    with a partial neighbour.  (Training mode additionally draws a random permutation.)
Therefore "window boundary" conclusions must refer to these groups.  This script:
  1. prints/saves the token -> (packet, group) map,
  2. runs a mechanism probe (feature-space noise in ONE packet's header or payload region, a diagnostic,
     not an attack) and reports per-block hidden-state drift aggregated by packet window and by flow group.
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
from robust.model_io import Model, to_tensor

L, P, COUNT = 400, 80, 4


def group_map():
    ids = np.arange(P)
    shuf = ids.reshape(COUNT, -1).T.flatten()
    full = np.tile(shuf, 5) + np.repeat(np.arange(0, L, P), P)       # shuffled position -> token index
    group_of_token = np.zeros(L, dtype=int)
    for g in range(COUNT):
        group_of_token[full.reshape(COUNT, -1)[g]] = g
    packet_of_token = np.arange(L) // P
    comp = np.zeros((COUNT, 5), dtype=int)
    for t in range(L):
        comp[group_of_token[t], packet_of_token[t]] += 1
    return group_of_token, packet_of_token, comp


@torch.no_grad()
def traced_forward(net, x):
    """Per-block token states, shape (4, N, 400, D), in original token order."""
    s = net.patch_embed(x) + net.pos_embed[:, 1:, :]
    cls = net.cls_token + net.pos_embed[:, :1, :]
    states = []
    s, _ = net.packet_window_attention(s, cls, 0); states.append(s)
    s, _ = net.flow_window_attention(s, cls, 1); states.append(s)
    s, _ = net.packet_window_attention(s, cls, 2); states.append(s)
    s, _ = net.flow_window_attention(s, cls, 3); states.append(s)
    return torch.stack(states)


def region_mask(packet: int, region: str) -> np.ndarray:
    m = np.zeros((40, 40), dtype=bool)
    r0 = packet * 8
    if region == "header":
        m[r0:r0 + 2, :] = True
    else:
        m[r0 + 2:r0 + 8, :] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nb_classes", type=int, default=0)
    ap.add_argument("--max_flows", type=int, default=200)
    ap.add_argument("--noise", type=int, default=32, help="max byte change in the probed region")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from attack.utils import load_finetuned_model
    dev = a.device if torch.cuda.is_available() else "cpu"
    gmap, pmap, comp = group_map()
    print("flow-window composition (rows = group, cols = packet 0..4):\n", comp)
    classes = class_names(a.flows_root, "train")
    n_cls = a.nb_classes or len(classes)
    items = load_items(a.flows_root, "test", classes, a.max_flows)
    mfr = items_to_mfrs(items)
    net = load_finetuned_model(a.ckpt, n_cls, dev).eval()
    rng = np.random.default_rng(a.seed)
    base = traced_forward(net, to_tensor(mfr, dev)).cpu()

    out = {"composition": comp.tolist(), "probe": {}}
    for region in ("header", "payload"):
        for pk in range(5):
            m = region_mask(pk, region)
            x = mfr.astype(np.int32).copy()
            x[:, m] += rng.integers(-a.noise, a.noise + 1, size=(len(x), int(m.sum())))
            x = np.clip(x, 0, 255).astype(np.uint8)
            d = (traced_forward(net, to_tensor(x, dev)).cpu() - base).norm(dim=-1)   # (4, N, 400)
            by_packet = np.stack([d[:, :, pmap == p].mean((1, 2)).numpy() for p in range(5)], 1)  # (4 blocks, 5)
            by_group = np.stack([d[:, :, gmap == g].mean((1, 2)).numpy() for g in range(COUNT)], 1)  # (4, 4)
            out["probe"][f"{region}_pkt{pk}"] = dict(drift_by_packet=by_packet.tolist(),
                                                     drift_by_flow_group=by_group.tolist())
            print(f"{region:8s} pkt{pk}: block3 drift by packet {np.round(by_packet[3], 3)} | by flow-group {np.round(by_group[3], 3)}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "window_groups.json", "w") as fh:
        json.dump(out, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 5, figsize=(14, 5))
    for ri, region in enumerate(("header", "payload")):
        for pk in range(5):
            mat = np.array(out["probe"][f"{region}_pkt{pk}"]["drift_by_packet"])
            axs[ri, pk].imshow(mat, aspect="auto")
            axs[ri, pk].set_title(f"{region} of packet {pk}", fontsize=9)
            axs[ri, pk].set_xlabel("packet window")
            axs[ri, pk].set_ylabel("block")
    fig.tight_layout()
    fig.savefig(Path(a.out) / "window_drift_heatmaps.png", dpi=200)


if __name__ == "__main__":
    main()
