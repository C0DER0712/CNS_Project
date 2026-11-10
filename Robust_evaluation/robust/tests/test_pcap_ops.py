import random
import sys
from pathlib import Path

import numpy as np
import pytest
import scapy.all as scapy
from scapy.all import IP, TCP, Raw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust.pcap_ops import (Edit, Flow, apply_edit, apply_edits, cost, enumerate_edits)
from robust.pipeline import (flow_to_mfr, introduced_violations, mfr_via_repo_pipeline, validate_flow)


def synth_flow(seed=0) -> Flow:
    rng = random.Random(seed)
    c, s = "10.0.0.2", "93.184.216.34"
    cs, ss = rng.randrange(1 << 31), rng.randrange(1 << 31)
    cid, sid = 1000, 5000
    pk = []
    t = 1.0

    def add(p):
        nonlocal t
        p = IP(bytes(p))
        p.time = t
        t += 0.01
        pk.append(p)

    add(IP(src=c, dst=s, id=cid, ttl=64, flags=2) / TCP(sport=40000, dport=443, seq=cs, flags="S", window=29200)); cid += 1
    add(IP(src=s, dst=c, id=sid, ttl=55, flags=2) / TCP(sport=443, dport=40000, seq=ss, ack=cs + 1, flags="SA", window=65535)); sid += 1
    cs += 1; ss += 1
    add(IP(src=c, dst=s, id=cid, ttl=64, flags=2) / TCP(sport=40000, dport=443, seq=cs, ack=ss, flags="A", window=502)); cid += 1
    pl = bytes(rng.getrandbits(8) for _ in range(517))
    add(IP(src=c, dst=s, id=cid, ttl=64, flags=2) / TCP(sport=40000, dport=443, seq=cs, ack=ss, flags="PA", window=502) / Raw(pl)); cid += 1
    cs += len(pl)
    add(IP(src=s, dst=c, id=sid, ttl=55, flags=2) / TCP(sport=443, dport=40000, seq=ss, ack=cs, flags="A", window=700)); sid += 1
    pl2 = bytes(rng.getrandbits(8) for _ in range(90))
    add(IP(src=s, dst=c, id=sid, ttl=55, flags=2) / TCP(sport=443, dport=40000, seq=ss, ack=cs, flags="PA", window=700) / Raw(pl2)); sid += 1
    ss += len(pl2)
    pl3 = bytes(rng.getrandbits(8) for _ in range(60))
    add(IP(src=c, dst=s, id=cid, ttl=64, flags=2) / TCP(sport=40000, dport=443, seq=cs, ack=ss, flags="PA", window=502) / Raw(pl3)); cid += 1
    cs += len(pl3)
    add(IP(src=s, dst=c, id=sid, ttl=55, flags=2) / TCP(sport=443, dport=40000, seq=ss, ack=cs, flags="A", window=700)); sid += 1
    return Flow(pk, c, 40000)


def test_clean_flow_is_valid():
    assert validate_flow(synth_flow()) == []


def test_parity_with_repo_pipeline():
    f = synth_flow()
    assert np.array_equal(flow_to_mfr(f), mfr_via_repo_pipeline(f))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_enumerated_edit_is_valid(seed):
    f = synth_flow(seed)
    rng = random.Random(seed)
    edits = enumerate_edits(f, ("pad", "split", "mss", "keepalive", "ttl", "window", "ipid"), rng)
    assert len(edits) > 20
    n_applied = 0
    for e in edits:
        g = apply_edit(f, e)
        if g is None:
            continue
        n_applied += 1
        assert introduced_violations(f, g) == [], (str(e), introduced_violations(f, g))
        assert np.array_equal(flow_to_mfr(g), mfr_via_repo_pipeline(g))
    assert n_applied > 15


def test_chained_edits_stay_valid():
    f = synth_flow()
    chain = [Edit("pad", 3, 64, 1), Edit("split", 3, 16), Edit("keepalive", 2), Edit("ttl", -16), Edit("window", 2000)]
    g = apply_edits(f, chain)
    assert g is not None
    assert introduced_violations(f, g) == []


def test_cost_accounting():
    f = synth_flow()
    g = apply_edit(f, Edit("pad", 3, 64, 1))
    assert cost(f, g) == (64, 0)
    g = apply_edit(f, Edit("keepalive", 2))
    b, n = cost(f, g)
    assert n == 2 and b > 0
    g = apply_edit(f, Edit("ttl", -16))
    assert cost(f, g) == (0, 0)


def test_server_packets_untouched_by_free_edits():
    f = synth_flow()
    g = apply_edit(f, Edit("ttl", 32))
    for p, q in zip(f.pkts, g.pkts):
        if not f.is_client(p):
            assert bytes(p) == bytes(q)


def test_pad_shifts_later_seq_and_ack():
    f = synth_flow()
    g = apply_edit(f, Edit("pad", 3, 64, 1))
    assert g.pkts[6][TCP].seq == (f.pkts[6][TCP].seq + 64) % (1 << 32)
    assert g.pkts[4][TCP].ack == (f.pkts[4][TCP].ack + 64) % (1 << 32)


def test_validator_catches_broken_edits():
    f = synth_flow()
    # 1. bytes changed without fixing checksum (what the old attack/ module did)
    bad = f.copy()
    raw = bytearray(bytes(bad.pkts[3]))
    raw[-5] ^= 0xFF
    bad.pkts[3] = IP(bytes(raw))
    assert any("l4_checksum" in e for e in validate_flow(bad))
    # 2. all-zero "fragment" header
    bad = f.copy()
    bad.pkts[3] = IP(b"\x00" * 40)
    assert validate_flow(bad) != []
    # 3. padding without shifting later seq numbers
    g = apply_edit(f, Edit("pad", 3, 64, 1))
    broken = g.copy()
    q = broken.pkts[6]
    q[TCP].seq = (q[TCP].seq - 64) % (1 << 32)
    broken.pkts[6] = IP(bytes(q))
    assert any("seq_discontinuity" in e for e in introduced_violations(f, broken)) or \
        any("seq_discontinuity" in e for e in validate_flow(broken))
