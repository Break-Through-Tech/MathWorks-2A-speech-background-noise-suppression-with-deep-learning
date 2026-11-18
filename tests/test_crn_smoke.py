"""
Smoke tests for CRN-Lite. Run from the repo root:

    python tests/test_crn_smoke.py

No real data needed. All 4 checks should print PASS.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.crn import CRNDenoiser, count_params  # noqa: E402

torch.manual_seed(42)


def check_shapes():
    m = CRNDenoiser().eval()
    for t in (126, 100, 37):          # odd lengths too
        x = torch.randn(4, t, 257)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape, (t, y.shape)
    print("PASS shapes  [B,T,257] -> [B,T,257] for T=126,100,37")


def check_range():
    m = CRNDenoiser().eval()
    with torch.no_grad():
        y = m(torch.randn(2, 126, 257))
    assert y.min() > 0 and y.max() < 1
    assert torch.isfinite(y).all()
    print("PASS range   mask in (0,1), no NaN/Inf")


def check_params():
    n = count_params(CRNDenoiser())
    n128 = count_params(CRNDenoiser(hidden_size=128))
    print(f"PASS params  h=256: {n:,} | h=128: {n128:,}")


def check_overfit():
    """Loss on one fixed batch should fall sharply."""
    m = CRNDenoiser()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(8, 126, 257)
    irm = torch.rand(8, 126, 257)
    fmask = torch.ones(8, 126)
    fmask[:, 100:] = 0                # fake padding

    def loss_fn():
        err = (m(x) - irm) ** 2 * fmask[..., None]
        return err.sum() / (fmask.sum() * 257)

    first = loss_fn().item()
    for _ in range(150):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
    last = loss.item()
    assert last < 0.5 * first, (first, last)
    print(f"PASS overfit loss {first:.4f} -> {last:.4f}")


if __name__ == "__main__":
    check_shapes()
    check_range()
    check_params()
    check_overfit()
    print("\nAll smoke tests passed.")
