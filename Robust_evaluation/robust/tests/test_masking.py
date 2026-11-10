import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust.masking import apply_mask, parse_fields
from robust.pipeline import flow_to_mfr
from robust.tests.test_pcap_ops import synth_flow


def test_mask_zeroes_only_identifier_bytes():
    f = synth_flow(0)
    m = flow_to_mfr(f)
    fields = parse_fields("default")
    mm = apply_mask(m, fields)
    assert (mm != m).sum() > 0
    for p in range(5):
        hdr = m[p * 8:p * 8 + 2].reshape(-1)
        hm = mm[p * 8:p * 8 + 2].reshape(-1)
        if hdr[9] == 0:          # empty packet row
            continue
        keep = np.ones(80, bool)
        for a, b in ((10, 12), (12, 16), (16, 20), (20, 22), (22, 24), (36, 38)):
            keep[a:b] = False
        assert np.array_equal(hdr[keep], hm[keep])
        assert (hm[~keep] == 0).all()
    assert np.array_equal(mm[2:8], m[2:8])      # payload rows untouched


def test_mask_is_idempotent_and_empty_is_noop():
    m = flow_to_mfr(synth_flow(2))
    assert np.array_equal(apply_mask(m, ()), m)
    once = apply_mask(m, parse_fields("default"))
    assert np.array_equal(apply_mask(once, parse_fields("default")), once)
