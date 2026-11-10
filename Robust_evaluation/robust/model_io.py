"""Torch adapters: batched prediction, gradients, and attack objectives."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "fine-tune"))

BYTE_TO_INPUT = 2.0 / 255.0  # d(normalised input)/d(byte value)


def to_tensor(mfrs: np.ndarray, device: str, requires_grad: bool = False) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(mfrs, dtype=np.float32)).reshape(-1, 1, 40, 40).to(device)
    x = (x / 255.0 - 0.5) / 0.5
    if requires_grad:
        x.requires_grad_(True)
    return x


class Model:
    """Thin wrapper around a TaCo classifier or encoder (head=Identity)."""

    def __init__(self, net: nn.Module, device: str = "cpu", batch_size: int = 256):
        self.net = net.to(device).eval()
        self.device = device
        self.bs = batch_size

    @torch.no_grad()
    def forward(self, mfrs: np.ndarray) -> np.ndarray:
        outs = []
        for i in range(0, len(mfrs), self.bs):
            x = to_tensor(mfrs[i:i + self.bs], self.device)
            outs.append(self.net(x).float().cpu().numpy())
        return np.concatenate(outs, 0)

    def predict(self, mfrs: np.ndarray) -> np.ndarray:
        return self.forward(mfrs).argmax(1)

    def grad_of(self, mfr: np.ndarray, scalar_fn) -> np.ndarray:
        """d scalar_fn(net(x)) / d byte-value, shape (40, 40)."""
        self.net.eval()
        x = to_tensor(mfr[None], self.device, requires_grad=True)
        out = self.net(x)
        s = scalar_fn(out)
        g, = torch.autograd.grad(s, x)
        return (g[0, 0] * BYTE_TO_INPUT).detach().cpu().numpy()


# ───────────────────────── objectives ─────────────────────────
# Convention: HIGHER score is better for the attacker.

class Objective:
    name = "base"
    has_grad = False

    def scores(self, mfrs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def success(self, mfrs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad(self, mfr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ClassifierObjective(Objective):
    """Untargeted evasion (target=None) or targeted impersonation (target=class id).

    score = margin: untargeted  max_{k != y} z_k - z_y     (>0  <=> misclassified)
                    targeted    z_t - max_{k != t} z_k     (>0  <=> classified as t)
    """

    has_grad = True

    def __init__(self, model: Model, label: int, target: Optional[int] = None):
        self.m, self.y, self.t = model, int(label), None if target is None else int(target)
        self.name = "evasion" if target is None else "impersonation"

    def _margin_np(self, z: np.ndarray) -> np.ndarray:
        if self.t is None:
            zy = z[:, self.y]
            other = np.delete(z, self.y, axis=1).max(1)
            return other - zy
        zt = z[:, self.t]
        other = np.delete(z, self.t, axis=1).max(1)
        return zt - other

    def _margin_torch(self, z: torch.Tensor) -> torch.Tensor:
        k = self.y if self.t is None else self.t
        mask = torch.ones(z.shape[1], dtype=torch.bool, device=z.device)
        mask[k] = False
        other = z[:, mask].max(1).values
        return (other - z[:, k]) if self.t is None else (z[:, k] - other)

    def scores(self, mfrs):
        return self._margin_np(self.m.forward(mfrs))

    def success(self, mfrs):
        return self.scores(mfrs) > 0

    def grad(self, mfr):
        return self.m.grad_of(mfr, lambda z: self._margin_torch(z).sum())


class EmbeddingObjective(Objective):
    """Cosine objective on an encoder (head=Identity).

    mode="toward":  maximise cos(f(x), e_target);  success when cos >= tau
    mode="away":    minimise cos(f(x), e_ref);     success when cos <= tau
    """

    has_grad = True

    def __init__(self, encoder: Model, ref_embedding: np.ndarray, tau: float, mode: str = "toward",
                 center: Optional[np.ndarray] = None):
        """center: optional mean embedding subtracted from f(x) before the cosine (ref_embedding must
        then already be centred).  Removes the shared offset that makes raw cosines all > 0.9."""
        self.m = encoder
        self.mu = None if center is None else np.asarray(center, dtype=np.float32)
        e = np.asarray(ref_embedding, dtype=np.float32)
        self.e = e / (np.linalg.norm(e) + 1e-12)
        self.tau, self.mode = float(tau), mode
        self.name = f"embedding_{mode}"

    def _cos(self, z: np.ndarray) -> np.ndarray:
        if self.mu is not None:
            z = z - self.mu
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        return z @ self.e

    def scores(self, mfrs):
        c = self._cos(self.m.forward(mfrs))
        return c if self.mode == "toward" else -c

    def success(self, mfrs):
        c = self._cos(self.m.forward(mfrs))
        return c >= self.tau if self.mode == "toward" else c <= self.tau

    def grad(self, mfr):
        e = torch.from_numpy(self.e).to(self.m.device)
        mu = None if self.mu is None else torch.from_numpy(self.mu).to(self.m.device)
        sign = 1.0 if self.mode == "toward" else -1.0

        def fn(z):
            if mu is not None:
                z = z - mu[None]
            return sign * F.cosine_similarity(z, e[None], dim=1).sum()
        return self.m.grad_of(mfr, fn)
