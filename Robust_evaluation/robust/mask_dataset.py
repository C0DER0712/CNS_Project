"""Copy an MFR PNG dataset (root/{train,test}/<class>/*.png) with identifier fields zeroed.

python -m robust.mask_dataset --src /kaggle/working/mfr_dataset --dst /kaggle/working/mfr_dataset_masked
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robust.masking import DEFAULT_FIELDS, apply_mask, parse_fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--fields", default="default")
    a = ap.parse_args()
    fields = parse_fields(a.fields) or DEFAULT_FIELDS
    n = 0
    for png in sorted(Path(a.src).glob("*/*/*.png")):
        rel = png.relative_to(a.src)
        out = Path(a.dst) / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = np.array(Image.open(png).convert("L"), dtype=np.uint8)
        Image.fromarray(apply_mask(arr, fields)).save(out)
        n += 1
    print(f"masked {n} images ({','.join(fields)}) -> {a.dst}")


if __name__ == "__main__":
    main()
