"""Field masking for MFR arrays (removes capture-specific identifiers).

read_MFR_bytes keeps the full IP header, so source/destination IPs, ports and the checksums that
depend on them reach the model.  Masking zeroes those bytes in every packet.  Offsets assume a
20-byte IPv4 header followed by TCP or UDP (the L4 checksum sits at 36-37 for TCP, 26-27 for UDP,
chosen per packet from the IP protocol byte).

Enable globally with the environment variable  ROBUST_MASK_FIELDS=ip_src,ip_dst,l4_sport,l4_dport,
ip_checksum,l4_checksum   (or "default").  Every robust/ script then masks identically.
"""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import numpy as np

FIELD_RANGES = {
    "ip_src": (12, 16),
    "ip_dst": (16, 20),
    "l4_sport": (20, 22),
    "l4_dport": (22, 24),
    "ip_checksum": (10, 12),
}
DEFAULT_FIELDS: Tuple[str, ...] = ("ip_src", "ip_dst", "l4_sport", "l4_dport", "ip_checksum", "l4_checksum")


def parse_fields(spec: str) -> Tuple[str, ...]:
    spec = (spec or "").strip()
    if not spec:
        return ()
    if spec == "default":
        return DEFAULT_FIELDS
    names = tuple(x.strip() for x in spec.split(",") if x.strip())
    for n in names:
        if n != "l4_checksum" and n not in FIELD_RANGES:
            raise ValueError(f"unknown mask field {n!r}; choose from {sorted(FIELD_RANGES) + ['l4_checksum']}")
    return names


def apply_mask(mfr: np.ndarray, fields: Iterable[str]) -> np.ndarray:
    """Return a masked copy of a (40, 40) uint8 MFR."""
    fields = tuple(fields)
    out = np.array(mfr, dtype=np.uint8, copy=True)
    if not fields:
        return out
    for p in range(5):
        r0 = p * 8
        proto = int(out[r0, 9])               # header byte 9 = IP protocol
        spans = [FIELD_RANGES[f] for f in fields if f in FIELD_RANGES]
        if "l4_checksum" in fields:
            spans.append((26, 28) if proto == 17 else (36, 38))
        for s, e in spans:
            for k in range(s, e):
                out[r0 + k // 40, k % 40] = 0
    return out


def apply_mask_batch(mfrs: np.ndarray, fields: Iterable[str]) -> np.ndarray:
    return np.stack([apply_mask(m, fields) for m in mfrs])


def env_fields() -> Tuple[str, ...]:
    return parse_fields(os.environ.get("ROBUST_MASK_FIELDS", ""))
