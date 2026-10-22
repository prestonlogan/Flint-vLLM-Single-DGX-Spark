"""Fast LM head for Flint decode (env-gated by FLINT_FAST_HEAD=1; FLINT_FAST_HEAD_MODE=int2|nv4).

Pipeline per decode step (M <= FLINT_FAST_HEAD_MAX_M rows):
  1. coarse logits for the whole vocab from an INT2 copy of the head (per-16 e4m3 group scales, fp32 row scale),
  2. top-C candidate ids per row (C = FLINT_FAST_HEAD_C, default 512),
  3. exact logits for the candidates from the BF16 head rows (fp32 accumulate, bf16 round, global scale as in
     LogitsProcessor), scattered into a full-vocab bf16 tensor that is -inf elsewhere.
Prefill / prompt_logprobs / large batches use the stock BF16 path.
"""
import os
import torch
import triton
import triton.language as tl

import sys
ENABLED = os.environ.get("FLINT_FAST_HEAD", "0") == "1"
VERIFY = os.environ.get("FLINT_FAST_HEAD_VERIFY", "0") == "1"      # also run the stock path every call and compare
CAPTURE_DIR = os.environ.get("FLINT_HEAD_CAPTURE_DIR", "")         # dump real hidden states (+ head) for offline tests

TOPC = int(os.environ.get("FLINT_FAST_HEAD_C", "512"))
MAX_M = int(os.environ.get("FLINT_FAST_HEAD_MAX_M", "8"))
MODE = os.environ.get("FLINT_FAST_HEAD_MODE", "int2")   # int2 (coarse + exact top-C) | nv4 (exact NVFP4 GEMV)
GROUP = 16
ORDER = (0, 2, 1, 3)


@triton.jit
def _int2_head(X, W, S, O, RS, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               XM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               G: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=STAGES):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XM + rk[None, :], (rm[:, None] < M), other=0.0)
        rkq = k0 // 4 + tl.arange(0, BK // 4)
        p = tl.load(W + rn[:, None] * (K // 4) + rkq[None, :], (rn[:, None] < N), other=0)
        w = tl.reshape(tl.join(tl.join(p & 3, (p >> 2) & 3), tl.join((p >> 4) & 3, p >> 6)), (BN, BK)).to(tl.float32) - 1.5
        rg = k0 // G + tl.arange(0, BK // G)
        s = tl.load(S + rn[:, None] * (K // G) + rg[None, :], (rn[:, None] < N), other=0.0).to(tl.float32)
        w = tl.reshape(tl.reshape(w, (BN, BK // G, G)) * s[:, :, None], (BN, BK))
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    acc = acc * tl.load(RS + rn, rn < N, other=0.0)[None, :]
    tl.store(O + rm[:, None] * N + rn[None, :], acc.to(O.dtype.element_ty), (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _refine(X, IDX, W, O, scale, C: tl.constexpr, K: tl.constexpr, V: tl.constexpr, XM: tl.constexpr,
            BC: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0)
    cb = tl.program_id(1)
    rc = cb * BC + tl.arange(0, BC)
    rows = tl.load(IDX + m * C + rc, rc < C, other=0).to(tl.int64)
    acc = tl.zeros((BC,), tl.float32)
    for k0 in tl.range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + m * XM + rk).to(tl.float32)
        w = tl.load(W + rows[:, None] * K + rk[None, :], (rc < C)[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    # stock path (head_dtype float32): fp32 GEMM output, then `logits *= scale` in fp32
    tl.store(O + m * V + rows, acc * scale, rc < C)



@triton.jit
def _nv4_head(X, W, S, O, gs, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, XM: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr):
    # exact NVFP4 head: W [N, K/2] uint8 (low nibble = even k), S [N, K/16] e4m3, gs fp32 global scale; fp32 logits [M, N]
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=STAGES):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * XM + rk[None, :], rm[:, None] < M, other=0.0)
        rb = k0 // 2 + tl.arange(0, BK // 2)
        q = tl.load(W + rn[:, None] * (K // 2) + rb[None, :], rn[:, None] < N, other=0).to(tl.int32)
        qq = tl.reshape(tl.join(q & 0xF, (q >> 4) & 0xF), (BN, BK))
        e = (qq >> 1) & 3
        mt = (qq & 1).to(tl.float32)
        mag = tl.where(e == 0, mt * 0.5, (1.0 + mt * 0.5) * tl.exp2((e - 1).to(tl.float32)))
        v = tl.where((qq & 8) != 0, -mag, mag)
        rs = k0 // 16 + tl.arange(0, BK // 16)
        sc = tl.load(S + rn[:, None] * (K // 16) + rs[None, :], rn[:, None] < N, other=0.0).to(tl.float32)
        w = tl.reshape(tl.reshape(v, (BN, BK // 16, 16)) * sc[:, :, None], (BN, BK)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    acc = acc * gs
    tl.store(O + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))



@triton.jit
def _nv4_fma(X, W, S, O, gs, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr):
    # M=1 exact NVFP4 head on CUDA cores: fp32 FMA accumulation (no tensor-core accumulate truncation), closest to the stock fp32 GEMM
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    acc = tl.zeros((BN, BK), tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=STAGES):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rk).to(tl.float32)
        rb = k0 // 2 + tl.arange(0, BK // 2)
        q = tl.load(W + rn[:, None] * (K // 2) + rb[None, :], rn[:, None] < N, other=0).to(tl.int32)
        qq = tl.reshape(tl.join(q & 0xF, (q >> 4) & 0xF), (BN, BK))
        e = (qq >> 1) & 3
        mt = (qq & 1).to(tl.float32)
        mag = tl.where(e == 0, mt * 0.5, (1.0 + mt * 0.5) * tl.exp2((e - 1).to(tl.float32)))
        v = tl.where((qq & 8) != 0, -mag, mag)
        rs = k0 // 16 + tl.arange(0, BK // 16)
        sc = tl.load(S + rn[:, None] * (K // 16) + rs[None, :], rn[:, None] < N, other=0.0).to(tl.float32)
        w = tl.reshape(tl.reshape(v, (BN, BK // 16, 16)) * sc[:, :, None], (BN, BK))
        acc += w * x[None, :]
    r = tl.sum(acc, axis=1) * gs
    tl.store(O + rn, r, rn < N)


def nv4_logits_fma(x, P, S, gs, out, BN=32, BK=128, nw=4, stages=3):
    n = P.shape[0]
    _nv4_fma[(triton.cdiv(n, BN),)](x, P, S, out, gs, n, x.shape[1], BN, BK, stages, num_warps=nw)
    return out


def nv4_logits(x, P, S, gs, out, BN=64, BK=256, nw=4, stages=3):
    m, k = x.shape
    n = P.shape[0]
    _nv4_head[(triton.cdiv(n, BN),)](x, P, S, out, gs, m, n, k, x.stride(0), max(16, triton.next_power_of_2(m)),
                                     BN, BK, stages, num_warps=nw)
    return out


@torch.no_grad()
def quantize_int2(weight, group=GROUP):
    n, k = weight.shape
    packed = torch.empty((n, k // 4), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((n, k // group), dtype=torch.float8_e4m3fn, device=weight.device)
    rows = torch.empty((n,), dtype=torch.float32, device=weight.device)
    for a in range(0, n, 8192):
        f = weight[a:a + 8192].float().view(-1, k // group, group)
        s = f.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 1.5
        rs = s.amax(1, keepdim=True) / 448.0
        s = (s / rs).to(torch.float8_e4m3fn)
        sf = s.float() * rs
        rows[a:a + 8192] = rs.view(-1)
        idx = (torch.floor(f / sf) + 2).clamp(0, 3).to(torch.uint8).view(-1, k // 4, 4)
        b = torch.zeros(idx.shape[:2], dtype=torch.uint8, device=weight.device)
        for j in range(4):
            b |= idx[..., j] << (2 * ORDER[j])
        packed[a:a + 8192] = b
        scales[a:a + 8192] = s.squeeze(-1)
    return packed, scales, rows


def coarse_logits(x, P, S, RS, out, BN=64, BK=512, nw=4, stages=3):
    m, k = x.shape
    n = P.shape[0]
    _int2_head[(triton.cdiv(n, BN),)](x, P, S, out, RS, m, n, k, x.stride(0),
                                      max(16, triton.next_power_of_2(m)), BN, BK, GROUP, stages, num_warps=nw)
    return out


def refine_into(out, x, idx, W, scale, BC=8, BK=512, nw=4):
    m, c = idx.shape
    _refine[(m, triton.cdiv(c, BC))](x, idx, W, out, scale, c, x.shape[1], out.shape[1], x.stride(0), BC, BK, num_warps=nw)
    return out


_topk_impl = None


def topk_ids(scores, c):
    global _topk_impl
    if _topk_impl is None:
        try:
            from flashinfer import top_k as _fi
            _topk_impl = lambda s, k: _fi(s, k, sorted=False, deterministic=False)[1]
        except Exception:
            _topk_impl = lambda s, k: torch.topk(s, k, dim=-1, sorted=False)[1]
    return _topk_impl(scores, c)


class FastHead:
    """mode 'int2': INT2 coarse + exact top-C refine (inexact if a true top token is outside the candidates);
    mode 'nv4': exact full-vocab logits from the head's native NVFP4 representation (0.17 GB instead of 0.62 GB)."""

    def __init__(self, weight, scale, mode="int2", nv4=None):
        self.W = weight
        self.scale = float(scale)
        self.V, self.K = weight.shape
        self.mode = mode
        self.bufs = {}
        if mode == "int2":
            self.P, self.S, self.RS = quantize_int2(weight)
        else:
            self.P, self.S = nv4
        self.cfg_nv4 = (32, 256, 4, 4)
        self.cfg_nv4_fma = (64, 256, 8, 4)
        self.cfg_int2 = (128, 1024, 8, 2)

    def __call__(self, x):
        m = x.shape[0]
        x = x.contiguous()
        if self.mode == "nv4":
            out = torch.empty((m, self.V), dtype=torch.float32, device=x.device)
            if m == 1:   # CUDA-core fp32 FMA GEMV: same speed as the tensor-core version, but stock-grade fp32 accuracy
                return nv4_logits_fma(x, self.P, self.S, self.scale, out, *self.cfg_nv4_fma)
            return nv4_logits(x, self.P, self.S, self.scale, out, *self.cfg_nv4)
        buf = self.bufs.get(m)
        if buf is None:
            buf = self.bufs[m] = torch.empty((m, self.V), dtype=torch.float32, device=x.device)
        coarse_logits(x, self.P, self.S, self.RS, buf, *self.cfg_int2)
        idx = topk_ids(buf, TOPC)
        out = torch.full((m, self.V), float("-inf"), dtype=torch.float32, device=x.device)
        refine_into(out, x, idx, self.W, self.scale)
        return out

    @property
    def coarse_buf(self):
        return self.bufs


# ---------------------------------------------------------------------------------------------------------------
# integration: compute_logits hook, prompt-logprob guard, verify + capture helpers
# ---------------------------------------------------------------------------------------------------------------
def _log(msg):
    print(f"[flint-fast-head] {msg}", file=sys.stderr, flush=True)


_state = {"head": None, "failed": False, "bypass": 0, "calls": 0, "fast_calls": 0}
_cap = {"buf": [], "n": 0, "part": 0}
_ver = {"n": 0, "rows": 0, "argmax_mis": None, "top20_miss": None, "val_mis": None, "val_cnt": None,
        "maxdiff": None, "lse_maxdiff": None, "cand_has_top20": None}


def _wrap_bypass(cls, name):
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_fh_wrapped", False):
        return

    def wrapper(*a, **k):
        _state["bypass"] += 1
        try:
            return orig(*a, **k)
        finally:
            _state["bypass"] -= 1
    wrapper._fh_wrapped = True
    setattr(cls, name, wrapper)


def install_guards():
    """prompt_logprobs (and prompt token id logprobs) call compute_logits on prompt rows: keep those exact (BF16)."""
    n = 0
    try:
        from vllm.v1.worker.gpu.sample import prompt_logprob as pl
        for nm in ("compute_prompt_logprobs", "compute_prompt_token_id_logprobs"):
            _wrap_bypass(pl.PromptLogprobsWorker, nm); n += 1
    except Exception as e:
        _log(f"v2 prompt_logprob guard not installed: {type(e).__name__}: {e}")
    try:
        from vllm.v1.worker import gpu_model_runner as gm
        _wrap_bypass(gm.GPUModelRunner, "_get_prompt_logprobs_dict"); n += 1
    except Exception as e:
        _log(f"v1 prompt_logprobs guard not installed: {type(e).__name__}: {e}")
    _log(f"prompt-logprob guards installed: {n}")


if ENABLED or CAPTURE_DIR:
    install_guards()


def _nv4_from_gguf(model, W):
    """Native NVFP4 representation of the lm_head, straight from the GGUF (lossless repack), checked against the loaded BF16."""
    import gguf
    from vllm.models.flint.gguf_nvfp4 import _repack_nvfp4, dequantize_nvfp4, _as_tensor
    reader = gguf.GGUFReader(model.gguf_path)
    tensors = {t.name: t for t in reader.tensors}
    t = tensors["output.weight"]
    P, S = _repack_nvfp4(_as_tensor(t.data).to(W.device), int(t.shape[0]))
    ok = True
    for a in range(0, W.shape[0], 8192):                      # bit-exact check vs what the stock loader produced
        ok &= bool(torch.equal(dequantize_nvfp4(P[a:a + 8192], S[a:a + 8192], out_dtype=torch.bfloat16), W[a:a + 8192]))
    return (P.contiguous(), S.contiguous().view(torch.float8_e4m3fn)) if ok else None


def _build(model):
    lm = model.lm_head
    lp = model.logits_processor
    w = getattr(lm, "weight", None)
    ok = (w is not None and w.dim() == 2 and w.dtype == torch.bfloat16 and getattr(lm, "tp_size", 1) == 1
          and lp.soft_cap is None and not getattr(lp, "logits_as_input", False)
          and lp.head_dtype == torch.float32)
    if not ok:
        _log(f"fast head disabled: unsupported lm_head (weight={None if w is None else (tuple(w.shape), w.dtype)}, head_dtype={lp.head_dtype})")
        _state["failed"] = True
        return
    scale = float(lp.scale)
    nv4 = None
    mode = MODE
    if mode == "nv4":
        try:
            nv4 = _nv4_from_gguf(model, w.data)
        except Exception as e:
            _log(f"nv4 repack failed: {type(e).__name__}: {e}")
        if nv4 is None:
            _log("nv4 mode unavailable (repack mismatch/failure) -> fast head disabled")
            _state["failed"] = True
            return
    _state["head"] = FastHead(w.data, scale, mode, nv4)
    _state["org_vocab"] = lp.org_vocab_size
    h = _state["head"]
    _log(f"built mode={mode}: weight {tuple(w.shape)} scale {scale:.6g} org_vocab {_state['org_vocab']} C={TOPC} max_m={MAX_M} "
         f"compressed head {(h.P.numel() + h.S.numel()) / 1e6:.0f} MB verify={VERIFY}")


def _flush_capture(model, label):
    if not _cap["buf"]:
        return
    h = torch.cat(_cap["buf"], 0).cpu()
    path = os.path.join(CAPTURE_DIR, f"part{_cap['part']:03d}_{label}.pt")
    torch.save({"hidden": h, "label": label}, path)
    head_path = os.path.join(CAPTURE_DIR, "head.pt")
    if not os.path.exists(head_path):
        torch.save({"weight": model.lm_head.weight.data.cpu(), "scale": float(model.logits_processor.scale)}, head_path)
    _log(f"captured {h.shape[0]} hidden rows -> {path}")
    _cap["buf"] = []; _cap["n"] = 0; _cap["part"] += 1


def _capture(model, x):
    flag = os.path.join(CAPTURE_DIR, "FLUSH")
    if os.path.exists(flag):
        label = open(flag).read().strip() or "x"
        os.remove(flag)
        _flush_capture(model, label)
    _cap["buf"].append(x.detach().clone()); _cap["n"] += x.shape[0]


def _verify(model, x, fast_out):
    stock = model.logits_processor(model.lm_head, x)      # stock BF16 path (same tensor the sampler would see)
    fo = fast_out[:, :stock.shape[1]]
    sf, ff = stock.float(), fo.float()
    am_s, am_f = sf.argmax(-1), ff.argmax(-1)
    t20 = sf.topk(20, dim=-1).indices
    miss20 = (ff.gather(1, t20) == float("-inf")).sum()
    fin = ff > float("-inf")
    d = (ff - sf).abs().masked_fill(~fin, 0.0)
    lse_d = (torch.logsumexp(sf, -1) - torch.logsumexp(ff, -1)).abs().max()
    v = _ver
    def acc(name, val, op="sum"):
        v[name] = val if v[name] is None else (v[name] + val if op == "sum" else torch.maximum(v[name], val))
    acc("argmax_mis", (am_s != am_f).sum()); acc("top20_miss", miss20)
    acc("val_mis", ((d != 0) & fin).sum()); acc("val_cnt", fin.sum())
    acc("maxdiff", d.max(), "max"); acc("lse_maxdiff", lse_d, "max")
    v["rows"] += x.shape[0]; v["n"] += 1
    if v["n"] % 400 == 0:
        _log("VERIFY rows=%d argmax_mismatch=%d top20_missing_from_candidates=%d cand_value_mismatches=%d/%d "
             "max|fast-stock|=%.4g max|lse_stock-lse_fast|=%.4g" % (v["rows"], int(v["argmax_mis"]), int(v["top20_miss"]),
             int(v["val_mis"]), int(v["val_cnt"]), float(v["maxdiff"]), float(v["lse_maxdiff"])))


def compute_logits(model, hidden_states):
    """Returns logits (fp32, [M, org_vocab]) or None to let the caller use the stock path."""
    if _state["bypass"] > 0:
        return None
    m = hidden_states.shape[0]
    if m < 1 or m > MAX_M:
        return None
    if CAPTURE_DIR and not torch.cuda.is_current_stream_capturing():
        _capture(model, hidden_states)
    if not ENABLED or _state["failed"]:
        return None
    if _state["head"] is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        _build(model)
        if _state["head"] is None:
            return None
    out = _state["head"](hidden_states)
    out = out[:, :_state["org_vocab"]]
    if VERIFY and not torch.cuda.is_current_stream_capturing():
        _verify(model, hidden_states, out)
    return out
