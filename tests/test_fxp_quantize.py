from mptorch.quant import fixed_point_quantize
from mptorch.number import FixedPoint
import torch
import pytest


def test_fixed_point_quantize_nearest_no_subset():
    a = torch.tensor([0.1, 0.9, -1.7, 3.3], dtype=torch.float32)
    fmt = FixedPoint(wl=8, fl=4, subset_fxp=None)

    q = fixed_point_quantize(a, wl=fmt.wl, fl=fmt.fl,
                             clamp=fmt.clamp, symmetric=fmt.symmetric,
                             rounding="nearest", subset_fxp=fmt.subset_fxp)

    scale = 2 ** fmt.fl
    expected = torch.round(a * scale) / scale
    assert torch.allclose(q, expected, atol=1e-6)


def test_fixed_point_quantize_stochastic_no_subset():
    torch.manual_seed(0)
    a = torch.tensor([0.25, 0.75, -1.2, 2.9], dtype=torch.float32)
    fmt = FixedPoint(wl=8, fl=4, subset_fxp=None)

    q = fixed_point_quantize(a, wl=fmt.wl, fl=fmt.fl,
                             clamp=fmt.clamp, symmetric=fmt.symmetric,
                             rounding="stochastic", subset_fxp=fmt.subset_fxp)

    scale = 2 ** fmt.fl
    q_scaled = torch.round(q * scale).to(torch.int32)
    expected = (q * scale).round().to(torch.int32)
    assert torch.equal(q_scaled, expected)


def test_fixed_point_quantize_nearest_with_subset():
    a = torch.tensor([0.1, 0.9, -1.7, 3.3], dtype=torch.float32)
    fl = 4
    scale = 2 ** fl
    subset_vals = [-2, -1, 0, 1, 2]
    fmt = FixedPoint(wl=8, fl=fl, subset_fxp=torch.tensor(subset_vals, dtype=torch.int32))

    q = fixed_point_quantize(a, wl=fmt.wl, fl=fmt.fl,
                             clamp=fmt.clamp, symmetric=fmt.symmetric,
                             rounding="nearest", subset_fxp=fmt.subset_fxp)

    def quantize_to_subset(x):
        int_x = int(round(x * scale))
        diffs = [abs(s - int_x) for s in subset_vals]
        closest = subset_vals[diffs.index(min(diffs))]
        return closest / scale

    expected = torch.tensor([quantize_to_subset(v.item()) for v in a],
                            dtype=torch.float32)
    assert torch.allclose(q, expected, atol=1e-6)


def test_fixed_point_quantize_stochastic_with_subset():
    torch.manual_seed(42)
    a = torch.tensor([0.1, 0.9, -1.7, 3.3], dtype=torch.float32)
    fl = 4
    scale = 2 ** fl
    subset_vals = [-2, -1, 0, 1, 2]
    fmt = FixedPoint(wl=8, fl=fl, subset_fxp=torch.tensor(subset_vals, dtype=torch.int32))

    q = fixed_point_quantize(a, wl=fmt.wl, fl=fmt.fl,
                             clamp=fmt.clamp, symmetric=fmt.symmetric,
                             rounding="stochastic", subset_fxp=fmt.subset_fxp)

    q_scaled = torch.round(q * scale).to(torch.int32)
    for v in q_scaled.tolist():
        assert v in subset_vals


def test_fixed_point_quantize_empty_subset_behaves_like_no_subset():
    a = torch.tensor([0.3, -0.3, 1.2], dtype=torch.float32)
    fmt_none = FixedPoint(wl=8, fl=4, subset_fxp=None)
    fmt_empty = FixedPoint(wl=8, fl=4, subset_fxp=torch.empty(0))

    q_none = fixed_point_quantize(a, wl=fmt_none.wl, fl=fmt_none.fl,
                                  clamp=fmt_none.clamp, symmetric=fmt_none.symmetric,
                                  rounding="nearest", subset_fxp=fmt_none.subset_fxp)
    q_empty = fixed_point_quantize(a, wl=fmt_empty.wl, fl=fmt_empty.fl,
                                   clamp=fmt_empty.clamp, symmetric=fmt_empty.symmetric,
                                   rounding="nearest", subset_fxp=fmt_empty.subset_fxp)

    assert torch.allclose(q_none, q_empty, atol=1e-6)
