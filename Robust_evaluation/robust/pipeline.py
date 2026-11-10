"""Flow -> MFR extraction (identical to the repo's read_MFR_bytes) and validation."""

from __future__ import annotations

import binascii
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scapy.all as scapy
from scapy.all import IP, TCP, UDP

from robust.masking import apply_mask, env_fields
from robust.pcap_ops import (MOD32, TCP_ACK, TCP_FIN, TCP_RST, TCP_SYN, Flow,
                             _after, _is_tcp, _l4, _payload, _seq_end)

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "data"))


# ─────────────────────── extraction ───────────────────────

MASK_FIELDS = env_fields()      # set ROBUST_MASK_FIELDS=default to evaluate identifier-masked models


def set_mask_fields(fields) -> None:
    global MASK_FIELDS
    MASK_FIELDS = tuple(fields)

def packets_to_mfr(pkts: List[IP]) -> np.ndarray:
    """Same logic as data_process.read_MFR_bytes, but on in-memory packets.

    Packets are re-parsed from bytes first so they dissect exactly like packets
    read back from a pcap.  Verified against the repo function in tests.
    """
    data = []
    for p in pkts:
        packet = IP(bytes(p))
        header = binascii.hexlify(bytes(packet["IP"])).decode()
        try:
            payload = binascii.hexlify(bytes(packet["Raw"])).decode()
            header = header.replace(payload, "")
        except Exception:
            payload = ""
        header = header[:160] if len(header) > 160 else header + "0" * (160 - len(header))
        payload = payload[:480] if len(payload) > 480 else payload + "0" * (480 - len(payload))
        data.append((header, payload))
        if len(data) >= 5:
            break
    while len(data) < 5:
        data.append(("0" * 160, "0" * 480))
    s = "".join(h + p for h, p in data)
    arr = np.array([int(s[i:i + 2], 16) for i in range(0, len(s), 2)], dtype=np.uint8).reshape(40, 40)
    return apply_mask(arr, MASK_FIELDS) if MASK_FIELDS else arr


def flow_to_mfr(flow: Flow) -> np.ndarray:
    return packets_to_mfr(flow.pkts)


def mfr_via_repo_pipeline(flow: Flow) -> np.ndarray:
    """Round-trip through a real pcap and the repo's own read_MFR_bytes (for parity tests)."""
    from data_process import read_MFR_bytes
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "f.pcap")
        flow.to_pcap(path)
        content = read_MFR_bytes(path)
    arr = np.array([int(content[i:i + 2], 16) for i in range(0, len(content), 2)], dtype=np.uint8)
    return arr.reshape(40, 40)


# ─────────────────────── validation ───────────────────────

def _checksums_ok(p: IP) -> List[str]:
    errs = []
    raw = bytes(p)
    ref = IP(raw)
    a = IP(raw)
    del a.chksum
    if bytes(a)[10:12] != raw[10:12]:
        errs.append("ip_checksum")
    if ref.len != len(raw):
        errs.append("ip_length")
    l4 = _l4(p)
    if l4 is not None:
        b = IP(raw)
        layer = TCP if p.haslayer(TCP) else UDP
        del b[layer].chksum
        rb = bytes(b)
        if rb != raw and rb[:10] + raw[10:12] + rb[12:] != raw:
            errs.append("l4_checksum")
    return errs


def validate_flow(flow: Flow) -> List[str]:
    """Return a list of violations (empty = looks like a legal connection)."""
    errs: List[str] = []
    expected: Dict[bool, Optional[int]] = {True: None, False: None}
    last_id: Dict[bool, Optional[int]] = {True: None, False: None}
    peer_next: Dict[bool, Optional[int]] = {True: None, False: None}
    for i, p in enumerate(flow.pkts):
        for e in _checksums_ok(p):
            errs.append(f"pkt{i}:{e}")
        if p.version != 4 or p.ihl < 5:
            errs.append(f"pkt{i}:ip_header")
        if p.len > 1500:
            errs.append(f"pkt{i}:mtu")
        cl = flow.is_client(p)
        if p.id != 0 and last_id[cl] is not None and p.id == last_id[cl]:
            errs.append(f"pkt{i}:ip_id_repeat")
        if p.id != 0:
            last_id[cl] = p.id
        if _is_tcp(p):
            t = p[TCP]
            plen = len(_payload(p))
            nxt = _seq_end(p)
            exp = expected[cl]
            if exp is not None and not (int(t.flags) & TCP_SYN):
                keepalive_like = (t.seq == (exp - 1) % MOD32 and plen <= 1)
                if t.seq != exp and not keepalive_like:
                    errs.append(f"pkt{i}:seq_discontinuity")
            if exp is None or t.seq == exp or _after(nxt, exp):
                expected[cl] = nxt
            if (int(t.flags) & TCP_ACK) and peer_next[cl] is not None:
                if _after(t.ack, peer_next[cl]):
                    errs.append(f"pkt{i}:ack_beyond_sent")
            peer_next[not cl] = expected[cl]
    return errs


def introduced_violations(orig: Flow, adv: Flow) -> List[str]:
    """Violations in adv that the original flow did not already have (by type count)."""
    from collections import Counter
    kind = lambda e: e.split(":", 1)[1]
    c0 = Counter(kind(e) for e in validate_flow(orig))
    c1 = Counter(kind(e) for e in validate_flow(adv))
    return [f"{k}x{c1[k] - c0.get(k, 0)}" for k in c1 if c1[k] > c0.get(k, 0)]


def tshark_check(flow: Flow) -> Optional[Dict]:
    """Optional dissector check via tshark (checksum validation + expert errors).

    Returns None when tshark is not installed.
    """
    exe = shutil.which("tshark")
    if exe is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "f.pcap")
        flow.to_pcap(path)
        out = subprocess.run(
            [exe, "-r", path, "-o", "tcp.check_checksum:TRUE", "-o", "ip.check_checksum:TRUE",
             "-o", "udp.check_checksum:TRUE", "-Y", "_ws.expert.severity >= error",
             "-T", "fields", "-e", "frame.number", "-e", "_ws.expert.message"],
            capture_output=True, text=True, timeout=60)
    bad = [ln for ln in out.stdout.splitlines() if ln.strip()]
    return {"n_errors": len(bad), "errors": bad[:10], "stderr": out.stderr[:200]}
