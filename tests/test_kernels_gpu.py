"""Parity tests: every Triton kernel must match its torch reference within fp tolerance.

Marked `gpu` and skipped automatically on CPU hosts (see conftest). Triton kernel modules
are imported *inside* the tests so collection never fails on a machine without Triton.
"""

import pytest
import torch

from tessera.kernels.reference import (
    attention_ref,
    dequant_matmul_ref,
    rmsnorm_ref,
    swiglu_ref,
)

pytestmark = pytest.mark.gpu


def test_triton_rmsnorm_parity():
    from tessera.kernels.triton.rmsnorm import rmsnorm_triton

    torch.manual_seed(0)
    x = torch.randn(64, 1024, device="cuda", dtype=torch.float16)
    w = torch.randn(1024, device="cuda", dtype=torch.float16)
    out = rmsnorm_triton(x, w, 1e-5)
    ref = rmsnorm_ref(x, w, 1e-5)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_triton_rmsnorm_backward_parity():
    from tessera.kernels.triton.rmsnorm import rmsnorm_triton

    torch.manual_seed(0)
    x = torch.randn(32, 512, device="cuda", requires_grad=True)
    w = torch.randn(512, device="cuda", requires_grad=True)
    rmsnorm_triton(x, w, 1e-5).square().sum().backward()
    gx, gw = x.grad.clone(), w.grad.clone()

    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    rmsnorm_ref(x2, w2, 1e-5).square().sum().backward()
    torch.testing.assert_close(gx, x2.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gw, w2.grad, atol=1e-4, rtol=1e-4)


def test_triton_swiglu_parity():
    from tessera.kernels.triton.swiglu import swiglu_triton

    torch.manual_seed(0)
    x = torch.randn(128, 512, device="cuda", dtype=torch.float16)
    wg = torch.randn(1376, 512, device="cuda", dtype=torch.float16)
    wu = torch.randn(1376, 512, device="cuda", dtype=torch.float16)
    out = swiglu_triton(x, wg, wu)
    ref = swiglu_ref(x, wg, wu)
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("n_kv_heads", [8, 2])
def test_triton_flash_attention_parity(causal, n_kv_heads):
    from tessera.kernels.triton.flash_attention import flash_attention_triton

    torch.manual_seed(0)
    b, h, t, d = 2, 8, 256, 64
    q = torch.randn(b, h, t, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, n_kv_heads, t, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, n_kv_heads, t, d, device="cuda", dtype=torch.float16)
    out = flash_attention_triton(q, k, v, causal=causal)
    ref = attention_ref(q, k, v, causal=causal)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_triton_quant_matmul_parity():
    from tessera.kernels.triton.quant_matmul import quant_matmul_triton

    torch.manual_seed(0)
    m, n, k, group = 64, 512, 256, 128
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    qw = torch.randint(-127, 127, (n, k), device="cuda", dtype=torch.int8)
    scales = torch.rand(n, k // group, device="cuda", dtype=torch.float16) * 0.02
    out = quant_matmul_triton(x, qw, scales, group_size=group)
    ref = dequant_matmul_ref(x, qw, scales, None, group)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)
