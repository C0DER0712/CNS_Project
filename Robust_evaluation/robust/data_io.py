"""Load per-flow pcaps (flows_by_class/{train,test}/{class}/*.pcap) as Flow objects."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from robust.pcap_ops import Flow
from robust.pipeline import flow_to_mfr


@dataclass
class Item:
    path: str
    label: int
    cls: str
    capture: str   # source capture id (filename stem before "_flow<N>")
    flow: Flow


def class_names(root: str, split: str = "train") -> List[str]:
    """Sorted class names -- identical to torchvision ImageFolder ordering."""
    base = Path(root) / split
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def capture_id(path: str) -> str:
    return re.sub(r"_flow\d+$", "", Path(path).stem)


def load_items(root: str, split: str = "test", classes: Optional[List[str]] = None,
               max_flows: int = 0, seed: int = 0, quiet: bool = False) -> List[Item]:
    classes = classes or class_names(root, "train")
    items: List[Item] = []
    for ci, c in enumerate(classes):
        for p in sorted((Path(root) / split / c).glob("*.pcap")):
            f = Flow.from_pcap(str(p))
            if f is None or f.n_packets < 1:
                continue
            items.append(Item(str(p), ci, c, capture_id(str(p)), f))
    if max_flows and len(items) > max_flows:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(len(items), max_flows, replace=False))
        items = [items[i] for i in keep]
    if not quiet:
        print(f"[data] {split}: {len(items)} flows, {len(classes)} classes, "
              f"{len(set(i.capture for i in items))} captures")
    return items


def items_to_mfrs(items: List[Item]) -> np.ndarray:
    return np.stack([flow_to_mfr(i.flow) for i in items]).astype(np.uint8)


def parity_check(items: List[Item], mfr_root: str, split: str = "test", n: int = 50) -> float:
    """Fraction of flows whose in-memory MFR equals the PNG written by MFR_generator."""
    ok = tot = 0
    for it in items[:n]:
        png = Path(mfr_root) / split / it.cls / (Path(it.path).stem + ".png")
        if not png.exists():
            continue
        ref = np.array(Image.open(png).convert("L"), dtype=np.uint8)
        ok += int(np.array_equal(ref, flow_to_mfr(it.flow)))
        tot += 1
    return ok / tot if tot else float("nan")


def parity_mask(items: List[Item], mfr_root: str, split: str = "test", verbose: bool = True) -> List[bool]:
    """Per-flow: True if the in-memory MFR equals the PNG that the classifier was trained/tested on.
    Flows that differ (extraction quirks) are reported and can be excluded from attacks."""
    ok = []
    shown = 0
    for it in items:
        png = Path(mfr_root) / split / it.cls / (Path(it.path).stem + ".png")
        if not png.exists():
            ok.append(False)
            continue
        ref = np.array(Image.open(png).convert("L"), dtype=np.uint8)
        mine = flow_to_mfr(it.flow)
        same = bool(np.array_equal(ref, mine))
        ok.append(same)
        if not same and verbose and shown < 3:
            shown += 1
            d = np.argwhere(ref != mine)
            print(f"[parity] mismatch {Path(it.path).name}: {len(d)} bytes differ, first at (row,col)={d[0].tolist()} "
                  f"png={int(ref[tuple(d[0])])} mem={int(mine[tuple(d[0])])}")
    return ok
