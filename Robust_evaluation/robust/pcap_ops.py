"""Pcap-level, endpoint-realizable edits of a flow.

Rules every edit follows
------------------------
* The attacker controls only the CLIENT endpoint (the sender of the first packet).
* Lengths and IP/TCP/UDP checksums are always recomputed.
* TCP seq/ack numbers are shifted consistently (later client seq, later server ack).
* IP-ID stays monotone per direction when the original was sequential.
* Ciphertext bytes are never rewritten; padding appends fresh random-looking bytes
  (what a longer TLS record looks like on the wire).
* An edit that is not applicable to a flow returns None.

The result is again a list of IPv4 packets that goes back through the SAME
extraction code as the clean data (see pipeline.py).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import scapy.all as scapy
from scapy.all import IP, TCP, UDP, Raw

MOD32 = 1 << 32
TCP_FIN, TCP_SYN, TCP_RST, TCP_PSH, TCP_ACK = 0x01, 0x02, 0x04, 0x08, 0x10
MTU = 1500


# ───────────────────────────── data model ─────────────────────────────

@dataclass(frozen=True)
class Edit:
    kind: str
    a: int = 0
    b: int = 0
    seed: int = 0

    def __str__(self) -> str:
        return f"{self.kind}({self.a},{self.b})"


@dataclass
class Flow:
    pkts: List[IP]
    client_ip: str
    client_port: int
    _wire: Optional[int] = field(default=None, repr=False, compare=False)
    _key: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def n_packets(self) -> int:
        return len(self.pkts)

    @property
    def wire_bytes(self) -> int:
        # Packets are treated as immutable once inside a Flow, so this is cached.
        if self._wire is None:
            self._wire = sum(len(bytes(p)) for p in self.pkts)
        return self._wire

    @property
    def key(self) -> str:
        """Content hash of the flow state (used to cache edit results)."""
        if self._key is None:
            h = hashlib.blake2b(digest_size=12)
            for p in self.pkts:
                h.update(bytes(p))
                h.update(b"|")
            h.update(f"{self.client_ip}:{self.client_port}".encode())
            self._key = h.hexdigest()
        return self._key

    def is_client(self, p: IP) -> bool:
        if p.src != self.client_ip:
            return False
        l4 = _l4(p)
        return l4 is None or l4.sport == self.client_port

    def copy(self) -> "Flow":
        return Flow(list(self.pkts), self.client_ip, self.client_port)

    @staticmethod
    def from_pcap(path: str, max_packets: int = 10) -> Optional["Flow"]:
        pkts: List[IP] = []
        for p in scapy.rdpcap(path):
            if IP not in p:
                continue
            ip = _normalise(p[IP])
            ip.time = float(p.time)
            pkts.append(ip)
            if len(pkts) >= max_packets:
                break
        if not pkts:
            return None
        l4 = _l4(pkts[0])
        return Flow(pkts, pkts[0].src, l4.sport if l4 is not None else 0)

    def to_pcap(self, path: str) -> None:
        scapy.wrpcap(path, self.pkts)


# ───────────────────────────── helpers ─────────────────────────────

def _normalise(ip: IP) -> IP:
    """Re-parse and drop link-layer trailer padding (bytes beyond IP total length)."""
    raw = bytes(ip)
    try:
        total = IP(raw).len
    except Exception:
        total = None
    if total and 20 <= total <= len(raw):
        raw = raw[:total]
    return IP(raw)


def _l4(p: IP):
    if p.frag != 0 or (int(p.flags) & 0x1):
        return None
    if p.haslayer(TCP):
        return p[TCP]
    if p.haslayer(UDP):
        return p[UDP]
    return None


def _is_tcp(p: IP) -> bool:
    return _l4(p) is not None and p.haslayer(TCP)


def _payload(p: IP) -> bytes:
    raw = bytes(p)
    ihl = p.ihl * 4
    total = p.len if p.len and p.len <= len(raw) else len(raw)
    l4 = _l4(p)
    if l4 is None:
        return b""
    hl = l4.dataofs * 4 if p.haslayer(TCP) else 8
    return raw[ihl + hl: total]


def _ip_fields(p: IP) -> Dict:
    return dict(tos=p.tos, id=p.id, flags=int(p.flags), frag=p.frag, ttl=p.ttl,
                src=p.src, dst=p.dst, options=p.options)


def _l4_fields(p: IP) -> Dict:
    l4 = _l4(p)
    if p.haslayer(TCP):
        return dict(sport=l4.sport, dport=l4.dport, seq=l4.seq, ack=l4.ack,
                    reserved=l4.reserved, flags=int(l4.flags), window=l4.window,
                    urgptr=l4.urgptr, options=l4.options)
    return dict(sport=l4.sport, dport=l4.dport)


def _rebuild(p: IP, payload: Optional[bytes] = None, ip_over: Optional[Dict] = None,
             l4_over: Optional[Dict] = None, t: Optional[float] = None) -> IP:
    """Create a fresh packet with recomputed len/checksums."""
    ipf = _ip_fields(p)
    l4f = _l4_fields(p)
    if ip_over:
        ipf.update(ip_over)
    if l4_over:
        l4f.update(l4_over)
    pl = _payload(p) if payload is None else payload
    l4cls = TCP if p.haslayer(TCP) else UDP
    pkt = IP(**ipf) / l4cls(**l4f)
    if pl:
        pkt = pkt / Raw(pl)
    new = IP(bytes(pkt))
    new.time = float(p.time) if t is None else t
    return new


def _seq_end(p: IP) -> int:
    l4 = p[TCP]
    extra = 1 if (int(l4.flags) & (TCP_SYN | TCP_FIN)) else 0
    return (l4.seq + len(_payload(p)) + extra) % MOD32


def _after(a: int, b: int) -> bool:
    """True if a is strictly after b in 32-bit sequence space."""
    return a != b and ((a - b) % MOD32) < (1 << 31)


def _shift_ids(pkts: List[IP], start: int, flow: Flow, client: bool, delta: int) -> None:
    for j in range(start, len(pkts)):
        q = pkts[j]
        if flow.is_client(q) == client and q.id != 0:
            pkts[j] = _rebuild(q, ip_over={"id": (q.id + delta) & 0xFFFF})


def _data_packets(flow: Flow, limit: int = 5) -> List[int]:
    out = []
    for i, p in enumerate(flow.pkts[:limit]):
        if flow.is_client(p) and _l4(p) is not None and len(_payload(p)) > 0:
            out.append(i)
    return out


# ───────────────────────────── edits ─────────────────────────────

def pad(flow: Flow, idx: int, n: int, seed: int) -> Optional[Flow]:
    if idx >= len(flow.pkts) or n <= 0:
        return None
    p = flow.pkts[idx]
    if not flow.is_client(p) or _l4(p) is None:
        return None
    old = _payload(p)
    if not old or p.len + n > MTU:
        return None
    rng = random.Random(seed)
    new_pl = old + bytes(rng.getrandbits(8) for _ in range(n))
    pkts = list(flow.pkts)
    pkts[idx] = _rebuild(p, payload=new_pl)
    if _is_tcp(p):
        seq0 = p[TCP].seq
        end_old = (seq0 + len(old)) % MOD32
        for j in range(idx + 1, len(pkts)):
            q = pkts[j]
            if not _is_tcp(q):
                continue
            if flow.is_client(q):
                pkts[j] = _rebuild(q, l4_over={"seq": (q[TCP].seq + n) % MOD32})
            elif (int(q[TCP].flags) & TCP_ACK):
                a = q[TCP].ack
                if a == end_old or _after(a, end_old):
                    pkts[j] = _rebuild(q, l4_over={"ack": (a + n) % MOD32})
    return Flow(pkts, flow.client_ip, flow.client_port)


def split(flow: Flow, idx: int, at: int) -> Optional[Flow]:
    if idx >= len(flow.pkts) or at <= 0:
        return None
    p = flow.pkts[idx]
    if not flow.is_client(p) or not _is_tcp(p):
        return None
    pl = _payload(p)
    if len(pl) < 2 or at >= len(pl):
        return None
    l4 = p[TCP]
    fl = int(l4.flags)
    first = _rebuild(p, payload=pl[:at], l4_over={"flags": fl & ~(TCP_PSH | TCP_FIN)})
    nxt_t = float(flow.pkts[idx + 1].time) if idx + 1 < len(flow.pkts) else float(p.time) + 1e-3
    t2 = float(p.time) + 0.01 * (nxt_t - float(p.time)) if nxt_t > float(p.time) else float(p.time) + 1e-6
    new_id = p.id if p.id == 0 else (p.id + 1) & 0xFFFF
    second = _rebuild(p, payload=pl[at:], ip_over={"id": new_id},
                      l4_over={"seq": (l4.seq + at) % MOD32}, t=t2)
    pkts = list(flow.pkts[:idx]) + [first, second] + list(flow.pkts[idx + 1:])
    if p.id != 0:
        _shift_ids(pkts, idx + 2, flow, True, 1)
    return Flow(pkts, flow.client_ip, flow.client_port)


def mss(flow: Flow, max_payload: int, cap_packets: int = 64) -> Optional[Flow]:
    cur = flow
    changed = False
    i = 0
    while i < len(cur.pkts) and len(cur.pkts) < cap_packets:
        p = cur.pkts[i]
        if cur.is_client(p) and _is_tcp(p) and len(_payload(p)) > max_payload:
            nxt = split(cur, i, max_payload)
            if nxt is None:
                return None
            cur, changed = nxt, True
        i += 1
    return cur if changed else None


def keepalive(flow: Flow, after_idx: int) -> Optional[Flow]:
    if after_idx >= len(flow.pkts):
        return None
    pre = flow.pkts[: after_idx + 1]
    cl = [p for p in pre if flow.is_client(p) and _is_tcp(p)]
    sv = [p for p in pre if (not flow.is_client(p)) and _is_tcp(p)]
    if not cl or not sv:
        return None
    ref, sref = cl[-1], sv[-1]
    if int(ref[TCP].flags) & (TCP_SYN | TCP_RST | TCP_FIN):
        return None
    next_seq = _seq_end(ref)
    base_t = float(flow.pkts[after_idx].time)
    nxt_t = float(flow.pkts[after_idx + 1].time) if after_idx + 1 < len(flow.pkts) else base_t + 1e-3
    gap = max(nxt_t - base_t, 1e-6)
    ka_id = ref.id if ref.id == 0 else (ref.id + 1) & 0xFFFF
    probe = _rebuild(
        ref, payload=b"",
        ip_over={"id": ka_id},
        l4_over={"seq": (next_seq - 1) % MOD32, "flags": TCP_ACK},
        t=base_t + 0.3 * gap)
    rep_id = sref.id if sref.id == 0 else (sref.id + 1) & 0xFFFF
    reply = _rebuild(
        sref, payload=b"",
        ip_over={"id": rep_id},
        l4_over={"seq": _seq_end(sref), "ack": next_seq, "flags": TCP_ACK},
        t=base_t + 0.6 * gap)
    pkts = list(flow.pkts[: after_idx + 1]) + [probe, reply] + list(flow.pkts[after_idx + 1:])
    if ref.id != 0:
        _shift_ids(pkts, after_idx + 3, flow, True, 1)
    if sref.id != 0:
        _shift_ids(pkts, after_idx + 3, flow, False, 1)
    return Flow(pkts, flow.client_ip, flow.client_port)


def set_ttl(flow: Flow, shift: int) -> Optional[Flow]:
    pkts, changed = [], False
    for p in flow.pkts:
        if flow.is_client(p) and _l4(p) is not None:
            v = min(255, max(2, p.ttl + shift))
            if v != p.ttl:
                changed = True
            pkts.append(_rebuild(p, ip_over={"ttl": v}))
        else:
            pkts.append(p)
    return Flow(pkts, flow.client_ip, flow.client_port) if changed else None


def set_window(flow: Flow, milli: int) -> Optional[Flow]:
    pkts, changed = [], False
    for p in flow.pkts:
        if flow.is_client(p) and _is_tcp(p):
            v = int(min(65535, max(1, round(p[TCP].window * milli / 1000.0))))
            if v != p[TCP].window:
                changed = True
            pkts.append(_rebuild(p, l4_over={"window": v}))
        else:
            pkts.append(p)
    return Flow(pkts, flow.client_ip, flow.client_port) if changed else None


def set_ipid(flow: Flow, mode: int, seed: int) -> Optional[Flow]:
    """mode 0: DF set + IP-ID 0 (modern Linux).  mode 1: sequential from random start."""
    rng = random.Random(seed)
    start = rng.randrange(1, 0xFFF0)
    pkts, k, changed = [], 0, False
    for p in flow.pkts:
        if flow.is_client(p) and _l4(p) is not None:
            if mode == 0:
                fl = int(p.flags) | 0x2
                new = _rebuild(p, ip_over={"id": 0, "flags": fl})
            else:
                new = _rebuild(p, ip_over={"id": (start + k) & 0xFFFF})
                k += 1
            if bytes(new) != bytes(p):
                changed = True
            pkts.append(new)
        else:
            pkts.append(p)
    return Flow(pkts, flow.client_ip, flow.client_port) if changed else None


def apply_edit(flow: Flow, e: Edit) -> Optional[Flow]:
    if e.kind == "pad":
        return pad(flow, e.a, e.b, e.seed)
    if e.kind == "split":
        return split(flow, e.a, e.b)
    if e.kind == "mss":
        return mss(flow, e.a)
    if e.kind == "keepalive":
        return keepalive(flow, e.a)
    if e.kind == "ttl":
        return set_ttl(flow, e.a)
    if e.kind == "window":
        return set_window(flow, e.a)
    if e.kind == "ipid":
        return set_ipid(flow, e.a, e.seed)
    raise ValueError(f"unknown edit kind: {e.kind}")


def apply_edits(flow: Flow, edits: List[Edit]) -> Optional[Flow]:
    cur = flow
    for e in edits:
        cur = apply_edit(cur, e)
        if cur is None:
            return None
    return cur


# ───────────────────────────── candidate grid ─────────────────────────────

PAD_SIZES = (16, 64, 160, 400)
SPLIT_AT = (1, 16, 64, 128)
TTL_SHIFTS = (-48, -32, -16, 16, 32, 64)
WINDOW_MILLI = (250, 500, 2000, 4000)
MSS_VALUES = (64, 128, 256, 536)


def enumerate_edits(flow: Flow, allowed: Tuple[str, ...], rng: random.Random,
                    window_pkts: int = 5) -> List[Edit]:
    out: List[Edit] = []
    data_idx = _data_packets(flow, window_pkts)
    if "pad" in allowed:
        for i in data_idx:
            for n in PAD_SIZES:
                out.append(Edit("pad", i, n, i * 100003 + n))
    if "split" in allowed:
        for i in data_idx:
            if _is_tcp(flow.pkts[i]):
                L = len(_payload(flow.pkts[i]))
                for at in SPLIT_AT:
                    if at < L:
                        out.append(Edit("split", i, at))
    if "mss" in allowed:
        for m in MSS_VALUES:
            out.append(Edit("mss", m))
    if "keepalive" in allowed:
        for i in range(min(window_pkts, len(flow.pkts))):
            out.append(Edit("keepalive", i))
    if "ttl" in allowed:
        for s in TTL_SHIFTS:
            out.append(Edit("ttl", s))
    if "window" in allowed:
        for m in WINDOW_MILLI:
            out.append(Edit("window", m))
    if "ipid" in allowed:
        out.append(Edit("ipid", 0))
        out.append(Edit("ipid", 1, 0, 12345))
    return out


def cost(orig: Flow, adv: Flow) -> Tuple[int, int]:
    """(extra wire bytes, extra packets) of adv relative to orig."""
    return adv.wire_bytes - orig.wire_bytes, adv.n_packets - orig.n_packets
