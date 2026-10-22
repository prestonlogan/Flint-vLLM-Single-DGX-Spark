"""Batch-1 NVFP4 GEMVs (raw ModelOpt layout) for the Flint attention block on sm_121. Enabled by FLINT_ATTN_FUSE=1."""
import os
import torch

_ext = None


def get_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_gemv2.cu")
        build = os.environ.get("FLINT_ATTN_GEMV_BUILD", "/tmp/flint_attn_gemv_build")
        os.makedirs(build, exist_ok=True)
        _ext = load(name="flint_attn_gemv2", sources=[src], build_directory=build, verbose=False,
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_121,code=sm_121", "-std=c++20"])
    return _ext


R_QKV = int(os.environ.get("FLINT_ATTN_R_QKV", "1"))
R_O = int(os.environ.get("FLINT_ATTN_R_O", "1"))


def qkv_gemv(x, res, nw, eps, W, S, gs, pdl=True):
    """x, res [1, 2048] bf16 (res may be None). Returns (qkv [1, N] bf16 = GEMV(rmsnorm(x + res)), new residual [1, 2048] bf16)."""
    y = torch.empty((1, W.shape[0]), dtype=torch.bfloat16, device=x.device)
    res_out = torch.empty((1, 2048), dtype=torch.bfloat16, device=x.device)
    get_ext().qkv_gemv(x, res, nw, eps, W, S, gs, res_out, y, pdl, R_QKV)
    return y, res_out


_counters = None


def counters(device):
    """Per-head arrival counters of qkv_head (zero at allocation, self-resetting). Allocate before CUDA graph capture."""
    global _counters
    if _counters is None:
        assert not torch.cuda.is_current_stream_capturing()
        _counters = torch.zeros(64, dtype=torch.int32, device=device)
    return _counters


def qkv_head(x, res, nw, eps, W, S, gs, q_w, k_w, positions, cos_sin, k_cache, v_cache, slots, trigger=True, pdl=True):
    """Fused add+RMSNorm + qkv GEMV + per-head q/k RMSNorm + RoPE + KV write. Returns (q [1, 4096], qkv scratch [1, 5120], new residual)."""
    dev = x.device
    y = torch.empty((1, 5120), dtype=torch.bfloat16, device=dev)
    q = torch.empty((1, 4096), dtype=torch.bfloat16, device=dev)
    res_out = torch.empty((1, 2048), dtype=torch.bfloat16, device=dev)
    get_ext().qkv_head(x, res, nw, eps, W, S, gs, res_out, y, q_w, k_w, positions, cos_sin, k_cache, v_cache, slots, q, counters(dev), pdl, trigger, R_QKV)
    return q, y, res_out


def o_gemv(a, W, S, gs, pdl=True, trigger=0):
    """trigger: 1 = release the PDL successor right after the wait, 2 = at kernel start (only when the successor is a PDL-aware kernel that waits before reading)."""
    y = torch.empty((1, W.shape[0]), dtype=torch.bfloat16, device=a.device)
    get_ext().o_gemv(a, W, S, gs, y, pdl, R_O, trigger)
    return y


if __name__ == "__main__":  # build at image build time (no GPU needed)
    get_ext()
    print("built")
