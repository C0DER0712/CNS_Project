"""Tiny synthetic flows_by_class dataset for smoke tests (NOT for results)."""
import random
import sys
from pathlib import Path

import scapy.all as scapy
from scapy.all import IP, TCP, UDP, Raw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def make_flow(cls: int, rng: random.Random, capture: int):
    c = f"10.{capture}.0.{rng.randrange(2, 250)}"
    s = f"172.16.{cls}.{rng.randrange(2, 250)}"
    sp, dp = rng.randrange(20000, 60000), [443, 80, 6881][cls % 3]
    ttl_c, ttl_s = [64, 128, 64][cls % 3], [57, 117, 49][cls % 3]
    cs, ss = rng.randrange(1 << 30), rng.randrange(1 << 30)
    cid, sid = rng.randrange(1, 60000), rng.randrange(1, 60000)
    sizes = [(200, 900), (40, 120), (500, 1300)][cls % 3]
    pk, t = [], 1.0

    def add(p):
        nonlocal t
        p = IP(bytes(p)); p.time = t; t += 0.01; pk.append(p)

    add(IP(src=c, dst=s, id=cid, ttl=ttl_c, flags=2) / TCP(sport=sp, dport=dp, seq=cs, flags="S", window=29200)); cid += 1; cs += 1
    add(IP(src=s, dst=c, id=sid, ttl=ttl_s, flags=2) / TCP(sport=dp, dport=sp, seq=ss, ack=cs, flags="SA", window=65535)); sid += 1; ss += 1
    add(IP(src=c, dst=s, id=cid, ttl=ttl_c, flags=2) / TCP(sport=sp, dport=dp, seq=cs, ack=ss, flags="A", window=502)); cid += 1
    for k in range(7):
        if k % 2 == 0:
            n = rng.randint(*sizes)
            add(IP(src=c, dst=s, id=cid, ttl=ttl_c, flags=2) / TCP(sport=sp, dport=dp, seq=cs, ack=ss, flags="PA", window=502) / Raw(bytes(rng.getrandbits(8) for _ in range(n)))); cid += 1; cs += n
        else:
            n = rng.randint(*sizes)
            add(IP(src=s, dst=c, id=sid, ttl=ttl_s, flags=2) / TCP(sport=dp, dport=sp, seq=ss, ack=cs, flags="PA", window=700) / Raw(bytes(rng.getrandbits(8) for _ in range(n)))); sid += 1; ss += n
    return pk


def build(root: str, n_train=40, n_test=20, n_classes=3, seed=0):
    rng = random.Random(seed)
    names = ["chat", "email", "p2p", "voip", "streaming"][:n_classes]
    for split, n, cap_off in (("train", n_train, 0), ("test", n_test, 0)):
        for ci, cn in enumerate(names):
            d = Path(root) / split / cn
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                cap = i % 4
                scapy.wrpcap(str(d / f"vpn_{cn}{cap}_flow{i}.pcap"), make_flow(ci, rng, cap))
    return names
