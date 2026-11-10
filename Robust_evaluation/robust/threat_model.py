"""Explicit threat model + budget for the TaCo robustness study.

Every attack in this package runs under one ThreatModel instance, and every
reported number states which instance produced it.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

ALL_EDITS: Tuple[str, ...] = (
    "pad",        # app-/TLS-level padding: longer record, random-looking bytes
    "split",      # TCP segmentation at a chosen byte offset
    "mss",        # clamp MSS: re-segment all client data packets
    "keepalive",  # TCP keep-alive probe + the server's ACK reply
    "ttl",        # client OS default TTL (shifts observed TTL)
    "window",     # client TCP receive-window scaling
    "ipid",       # IP-ID policy (DF + id 0, or sequential from a random start)
)

# Edits whose cost is exactly zero extra bytes / zero extra packets.
FREE_EDITS: Tuple[str, ...] = ("ttl", "window", "ipid")


@dataclass
class ThreatModel:
    # Goal
    goal: str = "evasion"          # "evasion" | "impersonation"
    # Knowledge
    knowledge: str = "score"       # "none" (random) | "label" | "score" | "white_box"
    # Control: the attacker owns ONE endpoint (the client) and can only change
    # what that endpoint sends.  Server packets are never modified directly.
    control: str = "client_endpoint"
    # Budget (measured on the wire, between original and perturbed flow)
    max_extra_bytes: int = 1024
    max_extra_packets: int = 4
    max_steps: int = 4             # edits chained in one attack
    max_queries: int = 600         # model queries per flow
    allowed: Tuple[str, ...] = ALL_EDITS
    # Timing is NOT part of the budget: the MFR pipeline discards timestamps, so
    # latency cannot influence the model input.  Reported as a limitation.
    notes: str = (
        "Timing/IAT edits are excluded on purpose: read_MFR_bytes() keeps no "
        "timestamps, so they are unobservable to the model."
    )

    def to_dict(self) -> Dict:
        return asdict(self)

    def with_budget(self, extra_bytes: int, extra_packets: int) -> "ThreatModel":
        d = self.to_dict()
        d["max_extra_bytes"] = extra_bytes
        d["max_extra_packets"] = extra_packets
        d["allowed"] = tuple(d["allowed"])
        return ThreatModel(**d)

    def restrict(self, kinds: Tuple[str, ...]) -> "ThreatModel":
        d = self.to_dict()
        d["allowed"] = tuple(k for k in self.allowed if k in kinds)
        return ThreatModel(**d)

    def without(self, kinds: Tuple[str, ...]) -> "ThreatModel":
        d = self.to_dict()
        d["allowed"] = tuple(k for k in self.allowed if k not in kinds)
        return ThreatModel(**d)


# Budget presets swept in every ASR-vs-budget curve: name -> (bytes, packets)
BUDGET_PRESETS: Dict[str, Tuple[int, int]] = {
    "free(0B,0pkt)": (0, 0),
    "S(128B,1pkt)": (128, 1),
    "M(512B,2pkt)": (512, 2),
    "L(2KB,4pkt)": (2048, 4),
    "XL(8KB,8pkt)": (8192, 8),
}
