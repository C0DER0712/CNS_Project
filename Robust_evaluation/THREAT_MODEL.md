# Threat model (problem-space, endpoint attacker)

| Element | Definition used in every reported number |
|---|---|
| **Goal** | (a) *Evasion*: a flow of class *y* is classified as anything other than *y*. (b) *Impersonation*: it is classified as a chosen class *t ≠ y*. Both are reported separately (`--goal`). |
| **Knowledge** | `none` = random edit chains (baseline). `score` = black-box access to the logits (greedy search). `white_box` = weights and input gradients (gradient-guided search, adaptive attacker against defences). Embedding-collision studies state which encoder was queried. |
| **Control** | One endpoint (the client = sender of the first packet). The attacker may change what that endpoint *sends*: longer records (padding), TCP segmentation / MSS, keep-alive probes (and the server's genuine ACK reply), default TTL, receive-window scaling, IP-ID policy. The attacker cannot edit ciphertext in transit, cannot modify server packets directly, and cannot change timestamps in a way the model sees (the MFR pipeline discards time). |
| **Budget** | Extra wire bytes and extra packets, measured between original and perturbed flow (`pcap_ops.cost`), plus a cap on chained edits and on model queries. Presets: free (0 B, 0 pkt), S (128 B, 1), M (512 B, 2), L (2 KB, 4), XL (8 KB, 8). Latency is *not* in the budget because it cannot reach the model; this is stated as a limitation. |
| **Success** | Only flows the model classified correctly before the attack are attacked and counted in the ASR denominator. Robust accuracy over the full test set is reported alongside. |
| **Validity** | A perturbed flow counts as a success only if `validate_flow` finds no violation that the original did not already have (lengths, IP/TCP/UDP checksums, seq/ack continuity, IP-ID monotonicity, MTU). `pipeline.tshark_check` runs a protocol dissector when tshark is installed. |
| **Extraction** | Perturbed packets are re-parsed from bytes and pushed through the same `read_MFR_bytes` logic as the clean data. `tests/test_pcap_ops.py` asserts byte-for-byte equality with the repo function. |

## Feature space vs problem space

* **Problem space (the headline results):** `random`, `greedy`, `gradient`, `greedy_restarts` in `attacks.py`; all outputs are legal packet sequences.
* **Feature space (reference only):** `feature_pgd` edits MFR bytes directly. It is *not* realizable and is plotted only to show the gap between what the model can be fooled by and what a network attacker can actually do. Never call its success rate an attack on traffic.
* The old `attack/` matrix perturbations are feature-space operations. Their names (TLS padding, MTU fragmentation, keep-alive, IAT jitter) described network actions they did not implement.

## Dual-use statement

This code evaluates the robustness of an encrypted-traffic classifier so that defenders can harden it. The edits are ones endpoints already perform legitimately (segmentation, padding, keep-alives, OS TCP settings); the evaluation contains no exploit for any system and no payload manipulation. Evasion of traffic classification can also be used to bypass network policy or censorship measurement, so the repository ships the defence (adversarial fine-tuning, adaptive evaluation) with the attack and reports the accuracy cost of the defence. Releases should include seeds, exact data processing, and checkpoints so results are reproducible by defenders.
