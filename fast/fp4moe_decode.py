"""Decode-time NVFP4 MoE for Flint on sm_121 (GB10): own CUDA kernels (no cvt.e2m1x2), M == 1 only, reading the Marlin-repacked
routed-expert and shared-expert tensors in place (no second copy of the expert weights).  Enabled with VLLM_FLINT_MOEDEC=1."""
import os
import torch

_ext = None
UCFG = int(os.environ.get("FLINT_MOEDEC_UCFG", "4"))
DCFG = int(os.environ.get("FLINT_MOEDEC_DCFG", "3"))


def get_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp4moe_mar.cu")
        build = os.environ.get("FLINT_FP4MOE_BUILD", "/tmp/flint_fp4moe_build")
        os.makedirs(build, exist_ok=True)
        _ext = load(name="flint_fp4moe_mar", sources=[src], build_directory=build, verbose=False,
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_121,code=sm_121", "-std=c++20"])
    return _ext


def _u8(t):
    return t.view(torch.uint8)


class MarlinDecodeWeights:
    """Views (no copies) of the Marlin-format tensors vLLM already holds for the routed experts and the shared expert."""

    def __init__(self, w13, s13, g13, w2, s2, g2, sh_gu, sh_dn):
        E = 256
        assert w13.dtype == torch.int32 and tuple(w13.shape) == (E, 128, 2048), (w13.dtype, w13.shape)
        assert w2.dtype == torch.int32 and tuple(w2.shape) == (E, 32, 4096), (w2.dtype, w2.shape)
        assert tuple(s13.shape) == (E, 128, 1024) and tuple(s2.shape) == (E, 32, 2048), (s13.shape, s2.shape)
        assert tuple(sh_gu[0].shape) == (128, 4096) and tuple(sh_gu[1].shape) == (128, 2048), (sh_gu[0].shape, sh_gu[1].shape)
        assert tuple(sh_dn[0].shape) == (64, 4096) and tuple(sh_dn[1].shape) == (64, 2048), (sh_dn[0].shape, sh_dn[1].shape)
        for t in (w13, s13, w2, s2, sh_gu[0], sh_gu[1], sh_dn[0], sh_dn[1]):
            assert t.is_contiguous()
        self.w13, self.s13 = w13, _u8(s13)
        self.w2, self.s2 = w2, _u8(s2)
        self.g13 = g13.float().reshape(E, -1)[:, 0].contiguous()   # Marlin-processed global scales (2^119 / scale_factor folded in)
        self.g2 = g2.float().reshape(E, -1)[:, 0].contiguous()
        self.sw13, self.ss13, self.sg13 = sh_gu[0], _u8(sh_gu[1]), sh_gu[2].float().reshape(-1)[:1].contiguous()
        self.sw2, self.ss2, self.sg2 = sh_dn[0], _u8(sh_dn[1]), sh_dn[2].float().reshape(-1)[:1].contiguous()


_scratch = {}


def _sc(dev):
    sc = _scratch.get(dev)
    if sc is None:
        assert not torch.cuda.is_current_stream_capturing()
        sc = dict(lg=torch.zeros(256, dtype=torch.float32, device=dev), rc=torch.zeros(1, dtype=torch.int32, device=dev),
                  cu=torch.zeros(64, dtype=torch.int32, device=dev), cd=torch.zeros(32, dtype=torch.int32, device=dev),
                  pu=torch.empty(64 * 16 * 128, dtype=torch.float32, device=dev), pd=torch.empty(256 * 64, dtype=torch.float32, device=dev))
        _scratch[dev] = sc
    return sc


def route(x, gate_w):
    """Fused router (gate GEMV + top-6 sigmoid + renorm), one launch -> (tw [1,6] f32, ti [1,6] i32)."""
    sc = _sc(x.device)
    ti = torch.empty((1, 6), dtype=torch.int32, device=x.device)
    tw = torch.empty((1, 6), dtype=torch.float32, device=x.device)
    get_ext().moe_route(x, gate_w, sc["lg"], sc["rc"], ti, tw, True)
    return tw, ti


def route_norm(x, res, nw, eps, gate_w, trigger=0):
    """Router with the post-attention residual add + RMSNorm fused in (one launch).  x = attention output, res = residual stream.
    -> (tw [1,6] f32, ti [1,6] i32, xn [1,2048] bf16 normalized hidden state, new residual [1,2048] bf16)."""
    sc = _sc(x.device)
    ti = torch.empty((1, 6), dtype=torch.int32, device=x.device)
    tw = torch.empty((1, 6), dtype=torch.float32, device=x.device)
    xn = torch.empty((1, 2048), dtype=torch.bfloat16, device=x.device)
    res_out = torch.empty((1, 2048), dtype=torch.bfloat16, device=x.device)
    get_ext().moe_route_norm(x, res, nw, eps, gate_w, sc["lg"], sc["rc"], ti, tw, xn, res_out, True, trigger)
    return tw, ti, xn, res_out


def moe_up(x, ti, w: MarlinDecodeWeights):
    sc = _sc(x.device)
    act = torch.empty((6, 512), dtype=torch.bfloat16, device=x.device)
    sact = torch.empty((1024,), dtype=torch.bfloat16, device=x.device)
    get_ext().moe_up_m(UCFG, x, ti, w.w13, w.s13, w.g13, act, w.sw13, w.ss13, w.sg13, sact, sc["pu"], sc["cu"], True)
    return act, sact


def moe_down(act, sact, ti, tw, w: MarlinDecodeWeights, trigger=0):
    """-> shared_expert(x) + sum_j tw_j * expert_{ti_j}(x) (bf16), everything in the two kernels."""
    sc = _sc(act.device)
    out = torch.empty((1, 2048), dtype=torch.bfloat16, device=act.device)
    get_ext().moe_down_m(DCFG, act, sact, ti, tw, w.w2, w.s2, w.g2, w.sw2, w.ss2, w.sg2, out, sc["pd"], sc["cd"], True, trigger)
    return out


if __name__ == "__main__":  # build at image build time (no GPU needed)
    get_ext()
    print("built")
