# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Flint model (BF16 safetensors or NVFP4 GGUF checkpoints).

Flint is a Qwen3-MoE-style decoder with learned attention sinks, hybrid
sliding-window/full attention (full layers use no RoPE), dense SwiGLU MLPs on
``mlp_only_layers``, sigmoid top-k routing with renormalized weights, and an
ungated shared expert.
"""

from collections.abc import Iterable

import os

import torch
from torch import nn

from vllm.config import (
    CacheConfig,
    ParallelConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.platforms import current_platform

from . import fast_head as _fast_head
from . import attn_gemv as _ag
from .ops import decode_attention as _da
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import (
    unified_attention_with_output,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import make_attention_sink
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.flashinfer import (
    FlashInferImpl,
    FlashInferMetadata,
    FlashInferTrtllmAPIDecode,
)

from .configs import FlintConfig
from .gguf_nvfp4 import find_gguf_file, gguf_nvfp4_weights_iterator
from .ops import linear as flint_linear
from .ops.decode_attention import flint_decode_attention
from .ops.moe import (
    BF16_DECODE_MAX_TOKENS,
    LL_MOE_MAX_TOKENS,
    FlintBf16DecodeMoE,
    FlintNvfp4Experts,
    LLMoEWeights,
    flint_moe_ll,
    ll_weights_from_layers,
    make_route_counter,
)
from .ops.norm import RMSNorm, silu_and_mul
from . import fp4moe_decode
from .ops.qk_norm_rope import qk_norm_rope_cache
from .ops.qkv_fused import MAX_M as LL_MAX_TOKENS
from .ops.qkv_fused import flint_qkv_fused


# FLINT_ATTN_FUSE=1 (sm_121 batch-1 decode only): own NVFP4 GEMV kernels for qkv (with the input RMSNorm fused) and
# o_proj on the raw NVFP4 tensors (kept next to the Marlin copy, ~0.5 GB) + Flint's CuTe split-KV decode attention.
# FLINT_ATTN_GEMV / FLINT_ATTN_DEC select the parts (default: both when FLINT_ATTN_FUSE=1).
_AF = os.environ.get("FLINT_ATTN_FUSE", "") == "1"
_AF_GEMV = _AF and os.environ.get("FLINT_ATTN_GEMV", "1") == "1"
_AF_DEC = _AF and os.environ.get("FLINT_ATTN_DEC", "1") == "1"
_AF_FULL_CFG = tuple(int(v) for v in os.environ.get("FLINT_ATTN_FULL_CFG", "6,4,3,1").split(","))
# Merged-chain switches (flint-exp:all; each one can be turned off for ablation):
#   FLINT_MERGE_ROUTENORM=1  the MoE route kernel does the post-attention residual add + RMSNorm itself (drops the separate norm launch)
#   FLINT_MERGE_OTRIG=1      the o GEMV releases its PDL successor (route_norm) right after its wait, so the router prefetches its gate rows early
#                            (only effective together with FLINT_MERGE_ROUTENORM=1); 2 = release at kernel start, 0 = off
#   FLINT_MERGE_RTRIG=1      route_norm releases the up kernel after its wait (2 = at kernel start, 0 = off)
#   FLINT_MERGE_DTRIG=1      the MoE down kernel releases the next layer's qkv_head kernel early (weight prefetch overlaps the down tail); 2 = at start
_MG_RN = os.environ.get("FLINT_MERGE_ROUTENORM", "1") == "1"
_MG_OTRIG = int(os.environ.get("FLINT_MERGE_OTRIG", "1")) if _MG_RN else 0
_MG_RTRIG = int(os.environ.get("FLINT_MERGE_RTRIG", "1")) if _MG_RN else 0
_MG_DTRIG = int(os.environ.get("FLINT_MERGE_DTRIG", "1"))
if _AF_DEC:
    _orig_launch_config = _da._launch_config

    def _af_launch_config(batch, num_kv_heads, sliding, num_sms):
        # sm_121 has 99 KB of shared memory per block: the 8-warp / 2-stage full-attention config needs 128 KB.
        if not sliding and batch <= 2:
            return (*_AF_FULL_CFG, False)
        return _orig_launch_config(batch, num_kv_heads, sliding, num_sms)

    _da._launch_config = _af_launch_config


def _flint_kernels(
    quant_config: QuantizationConfig | None,
    parallel: ParallelConfig | None = None,
    part: str = "",
) -> bool:
    """Flint's kernels serve the BF16 and NVFP4 checkpoints on a single GPU
    (TP=1, and for the MoE no EP/EPLB); other configurations use vLLM's stock
    layers."""
    if get_tensor_model_parallel_world_size() != 1 or not (
        quant_config is None or isinstance(quant_config, ModelOptNvFp4Config)
    ):
        return False
    if not current_platform.is_device_capability_family(100):
        # Outside SM100 (e.g. DGX Spark's GB10, sm_121) only the parts listed in
        # VLLM_FLINT_KERNELS (moe, attn, dense) use Flint's kernels; the rest run
        # vLLM's stock layers (Marlin W4A16 for the NVFP4 GGUF).
        if part not in os.environ.get("VLLM_FLINT_KERNELS", "").split(","):
            return False
    return parallel is None or not (
        parallel.enable_expert_parallel or parallel.enable_eplb
    )


# Dequantized to BF16 (without the global scale, so exactly) at load so logits
# can be accumulated in fp32; the global scale is applied to the fp32 logits.
_GGUF_DEQUANTIZED = frozenset({"lm_head"})


def _nvfp4_gguf_quant_config() -> QuantizationConfig:
    # The GGUF reference runtime keeps activations in 16 bits (W4A16).
    return ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        exclude_modules=["model.embed_tokens", "*.mlp.gate", *_GGUF_DEQUANTIZED],
        group_size=16,
    )


class FlintMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = flint_linear.gate_up_silu_mul(self.gate_up_proj, x)
        if act is None:
            gate_up, _ = self.gate_up_proj(x)
            act = silu_and_mul(gate_up)
        x, _ = self.down_proj(act)
        return x


class FlintSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: FlintConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.shared_expert = FlintMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_expert",
        )
        vllm_config = get_current_vllm_config()
        # NVFP4: Flint's MoE for all batch sizes. BF16: Flint's decode kernels
        # for small batches on the TRTLLM-layout experts, TRTLLM MoE otherwise.
        is_nvfp4 = isinstance(quant_config, ModelOptNvFp4Config)
        use_flint = _flint_kernels(quant_config, vllm_config.parallel_config, "moe")
        if use_flint:
            assert config.router_score_function == "sigmoid"
            assert config.norm_topk_prob
        self.use_flint_kernel = use_flint and is_nvfp4
        self.bf16_decode: FlintBf16DecodeMoE | None = None
        self._ll_weights: LLMoEWeights | None = None
        if self.use_flint_kernel:
            self.register_buffer("ll_counter", make_route_counter(), persistent=False)
            self.experts = FlintNvfp4Experts(
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
                config.num_experts_per_tok,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )
            return
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            scoring_func=config.router_score_function,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )
        if use_flint:
            self.bf16_decode = FlintBf16DecodeMoE(
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
                config.num_experts_per_tok,
            )
        # VLLM_FLINT_MOEDEC=1 (NVFP4 outside SM100): own CUDA decode kernels (M == 1) that read the Marlin-repacked expert and shared-expert
        # tensors in place (no second copy of the weights); stock Marlin MoE for every other batch size and for prefill.
        self._dec_w = None
        self._dec_route = None
        self._dec_check = int(os.environ.get("VLLM_FLINT_MOEDEC_CHECK", "0"))
        self._dec_on = bool(
            is_nvfp4
            and not use_flint
            and os.environ.get("VLLM_FLINT_MOEDEC", "") == "1"
            and config.hidden_size == 2048
            and config.moe_intermediate_size == 512
            and config.num_experts_per_tok == 6
            and config.num_experts == 256
            and config.shared_expert_intermediate_size == 1024
        )
        if self._dec_on:
            fp4moe_decode.get_ext()

    def _build_decode_weights(self):
        assert not torch.cuda.is_current_stream_capturing()
        experts = next(
            m for m in self.experts.modules()
            if hasattr(m, "w13_weight") and hasattr(m, "quant_method")
        )
        gu, dn = self.shared_expert.gate_up_proj, self.shared_expert.down_proj
        return fp4moe_decode.MarlinDecodeWeights(
            experts.w13_weight.data, experts.w13_weight_scale.data, experts.w13_weight_scale_2.data,
            experts.w2_weight.data, experts.w2_weight_scale.data, experts.w2_weight_scale_2.data,
            (gu.weight.data, gu.weight_scale.data, gu.weight_global_scale.data),
            (dn.weight.data, dn.weight_scale.data, dn.weight_global_scale.data),
        )

    def _init_decode(self, x: torch.Tensor):
        self._dec_w = self._build_decode_weights()
        gw = self.gate.weight
        assert gw.dtype == torch.bfloat16 and tuple(gw.shape) == (256, 2048) and gw.is_contiguous() and x.dtype == torch.bfloat16

    def can_route_norm(self, hidden_states: torch.Tensor, residual: torch.Tensor | None) -> bool:
        return (
            _MG_RN
            and self._dec_on
            and self.bf16_decode is None
            and not self.use_flint_kernel
            and self._dec_check == 0
            and residual is not None
            and hidden_states.shape[0] == 1
            and hidden_states.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
        )

    def _forward_decode_rn(self, attn_out: torch.Tensor, residual: torch.Tensor, norm: RMSNorm) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch-1 decode MoE with the post-attention residual add + RMSNorm folded into the router kernel.
        attn_out: o_proj output, residual: residual stream. Returns (moe_out, new_residual)."""
        if self._dec_w is None:
            self._init_decode(attn_out)
        tw, ti, xn, residual = fp4moe_decode.route_norm(attn_out, residual, norm.weight, norm.variance_epsilon, self.gate.weight, trigger=_MG_RTRIG)
        act, sact = fp4moe_decode.moe_up(xn, ti, self._dec_w)
        out = fp4moe_decode.moe_down(act, sact, ti, tw, self._dec_w, trigger=_MG_DTRIG)
        return out, residual

    def _forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        if self._dec_w is None:
            self._init_decode(x)
        tw, ti = fp4moe_decode.route(x, self.gate.weight)
        act, sact = fp4moe_decode.moe_up(x, ti, self._dec_w)
        out = fp4moe_decode.moe_down(act, sact, ti, tw, self._dec_w, trigger=_MG_DTRIG)
        if self._dec_check > 0 and not torch.cuda.is_current_stream_capturing():
            # debug (eager only): compare against the stock Marlin MoE path when both pick the same experts (dummy warm-up rows have tied logits)
            lg, _ = self.gate(x)
            sw, si, _ = fused_topk(x, lg, self.top_k, True, scoring_func="sigmoid")
            torch.cuda.synchronize()
            same_set = set(ti[0].tolist()) == set(si[0].tolist())
            if same_set:
                self._dec_check -= 1
                ref = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
                d = (ref.float() - out.float()).norm() / ref.float().norm().clamp(min=1e-20)
                print(f"[moedec-check] marlin-direct rel_err={d.item():.3e} bit_identical={(ref == out).float().mean().item():.4f} same_order={ti.tolist() == si.tolist()} max_dw={(tw - sw).abs().max().item():.2e}", flush=True)
        return out

    def forward_ll(
        self, attn_out: torch.Tensor, residual: torch.Tensor, norm: RMSNorm
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(mlp(norm(attn_out + residual)), attn_out + residual)`` via the
        fused decode MoE block (NVFP4, M <= LL_MOE_MAX_TOKENS), else None."""
        if not self.use_flint_kernel or attn_out.shape[0] > LL_MOE_MAX_TOKENS:
            return None
        if self._ll_weights is None:
            assert self.experts.weights is not None
            self._ll_weights = ll_weights_from_layers(
                self.gate.weight,
                self.shared_expert.gate_up_proj,
                self.shared_expert.down_proj,
                self.experts.weights,
            )
        return flint_moe_ll(
            attn_out,
            residual,
            norm.weight,
            norm.variance_epsilon,
            self._ll_weights,
            self.top_k,
            self.ll_counter,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_flint_kernel:
            return self.experts(hidden_states, self.gate.weight, self.shared_expert)
        if (
            self.bf16_decode is not None
            and hidden_states.shape[0] <= BF16_DECODE_MAX_TOKENS
        ):
            return self.bf16_decode(
                hidden_states,
                self.gate.weight,
                self.experts.routed_experts,
                self.shared_expert,
            )
        if self._dec_on and hidden_states.shape[0] == 1:
            return self._forward_decode(hidden_states)
        return self.experts(hidden_states=hidden_states, router_logits=hidden_states)


class FlintAttention(nn.Module):
    def __init__(
        self,
        config: FlintConfig,
        layer_idx: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.af_gemv = (
            _AF_GEMV
            and isinstance(quant_config, ModelOptNvFp4Config)
            and tp_size == 1
            and config.hidden_size == 2048
            and self.head_dim == 128
            and self.total_num_heads == 32
            and self.total_num_kv_heads == 4
            and not config.attention_bias
        )
        if self.af_gemv:
            # vLLM keeps clones of the raw NVFP4 tensors (weight, weight_scale, weight_global_scale) next to the
            # Marlin repack: used by the batch-1 decode GEMVs.
            self.qkv_proj._retain_weight_for_gather = True
            self.o_proj._retain_weight_for_gather = True
        self._af_w = None
        self._af_head_ok = False
        if self.af_gemv:
            _ag.counters(torch.device("cuda", torch.cuda.current_device()))  # arrival counters must exist before graph capture
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = None
        if layer_idx not in config.no_rope_layer_indices:
            # Only its cos|sin table is used, by the fused kernels. Kept in
            # fp32 like the reference implementation; get_rope memoizes, so
            # all layers share one table.
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=config.rope_parameters,
                dtype=torch.float32,
            )
            assert self.rotary_emb.is_neox_style
            assert self.rotary_emb.rotary_dim == self.head_dim
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.softmax_offset = make_attention_sink(self.num_heads, torch.float32)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=config.sliding_window if is_sliding else None,
            prefix=f"{prefix}.attn",
            sinks=self.softmax_offset,
        )
        # The fused kernel writes the KV cache itself (in the FlashInfer layout)
        # and then calls the attention backend directly.
        impl = self.attn.impl
        assert isinstance(impl, FlashInferImpl), "Flint requires FlashInfer attention"
        assert get_current_vllm_config().model_config.dtype == torch.bfloat16
        assert self.attn.kv_cache_dtype in ("auto", "bfloat16")
        assert impl.kv_sharing_target_layer_name is None
        self.impl = impl
        # Flint decode attention and the fused low-latency decode path.
        self.flint_decode = _flint_kernels(quant_config, part="attn")
        # Largest pure-decode batch served by the Flint kernel; beyond it
        # FlashInfer is faster on the full-attention layers at long context.
        self.max_decode_batch = 64 if is_sliding else 8
        self.af_dec = _AF_DEC
        if self.af_dec:
            self.max_decode_batch = 1  # only batch 1 runs the Flint kernel; other sizes stay on FlashInfer
        if self.flint_decode:
            assert not config.attention_bias

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        k_cache, v_cache, layer_slots = self._cache_views()
        plan = self._decode_plan() if (self.flint_decode or self.af_dec) else None
        q = qk_norm_rope_cache(
            qkv,
            positions,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.variance_epsilon,
            self._cos_sin(),
            k_cache,
            v_cache,
            layer_slots,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            # Flint decode attention waits on the grid dependency before
            # reading q / K / V, so it may start its prologue early.
            launch_dependents=plan is not None,
        )
        num_tokens = qkv.shape[0]
        attn_output = torch.empty_like(q)
        if plan is not None:
            self._run_decode_attention(q, attn_output, *plan)
            output, _ = self.o_proj(attn_output)
            return output
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        unified_attention_with_output(
            q.view(num_tokens, self.num_heads, self.head_dim),
            k.view(num_tokens, self.num_kv_heads, self.head_dim),
            v.view(num_tokens, self.num_kv_heads, self.head_dim),
            attn_output.view(num_tokens, self.num_heads, self.head_dim),
            self.attn.layer_name,
        )
        output, _ = self.o_proj(attn_output)
        return output

    def _cos_sin(self) -> torch.Tensor | None:
        return None if self.rotary_emb is None else self.rotary_emb.cos_sin_cache

    def _cache_views(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """(k_cache, v_cache, slot_mapping) for the fused cache write, with the
        caches as [blocks, block_size, kv_heads, head_dim] views; all None while
        the KV cache is unallocated (memory profiling)."""
        kv_cache = self.attn.kv_cache
        if kv_cache.numel() == 0:
            return None, None, None
        # As in FlashInferImpl.do_kv_cache_update: (B, H, N, 2D) -> (B, N, H, D)
        k_cache, v_cache = kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)
        slot_mapping = get_forward_context().slot_mapping
        assert isinstance(slot_mapping, dict)
        return k_cache, v_cache, slot_mapping.get(self.attn.layer_name)

    def _decode_plan(self) -> tuple | None:
        """(decode metadata, num_decodes, window) when the batch is pure
        single-token decode of at most ``max_decode_batch`` requests (the Flint
        decode attention kernel), else None. K/V are already in the cache."""
        forward_metadata = get_forward_context().attn_metadata
        attn_metadata = (
            forward_metadata.get(self.attn.layer_name)
            if isinstance(forward_metadata, dict)
            else forward_metadata
        )
        if (
            not isinstance(attn_metadata, FlashInferMetadata)  # profiling run
            or attn_metadata.num_prefills != 0
            or not 0 < attn_metadata.num_decodes <= self.max_decode_batch
            or attn_metadata.num_decode_tokens != attn_metadata.num_decodes
        ):
            return None
        decode = attn_metadata.decode
        assert isinstance(decode, FlashInferTrtllmAPIDecode)
        assert decode.q_len_per_req == 1 and decode.dcp_query_start_loc is None
        left = self.impl.window_left
        window = left + 1 if left >= 0 else None
        return decode, attn_metadata.num_decodes, window

    def _run_decode_attention(
        self, q: torch.Tensor, out: torch.Tensor, decode, n: int, window: int | None
    ) -> None:
        shape = (-1, self.num_heads, self.head_dim)
        flint_decode_attention(
            q.view(shape)[:n],
            self.attn.kv_cache,
            decode.block_tables,
            decode.seq_lens,
            self.impl.sinks,
            window,
            self.impl.scale,
            out.view(shape)[:n],
        )

    def _af_weights(self):
        w = self._af_w
        if w is None:
            q, o = self.qkv_proj, self.o_proj
            for m in (q, o):
                if not hasattr(m, "_nvfp4_weight_for_gather"):
                    self._af_w = False
                    return False
            w = (
                q._nvfp4_weight_for_gather, q._nvfp4_weight_scale_for_gather, q._nvfp4_weight_global_scale_for_gather.float().reshape(1).contiguous(),
                o._nvfp4_weight_for_gather, o._nvfp4_weight_scale_for_gather, o._nvfp4_weight_global_scale_for_gather.float().reshape(1).contiguous(),
            )
            assert w[0].shape == (5120, 1024) and w[3].shape == (2048, 2048), (w[0].shape, w[3].shape)
            assert w[0].dtype == torch.uint8 and w[1].dtype == torch.float8_e4m3fn
            self._af_head_ok = (
                self.q_norm.weight.dtype == torch.bfloat16 and self.k_norm.weight.dtype == torch.bfloat16
                and (self.rotary_emb is None or self.rotary_emb.cos_sin_cache.dtype == torch.float32)
            )
            self._af_w = w
        return w

    def forward_af(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        input_layernorm: RMSNorm,
        o_trigger: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Batch-1 decode: fused residual-add RMSNorm + qkv GEMV (raw NVFP4), q/k norm + RoPE + KV write, Flint decode
        attention (or FlashInfer), o_proj GEMV. Returns None to fall back to the regular path."""
        if hidden_states.shape[0] != 1 or hidden_states.dtype != torch.bfloat16:
            return None
        k_cache, v_cache, layer_slots = self._cache_views()
        plan = self._decode_plan() if self.af_dec else None
        if k_cache is None or v_cache is None or layer_slots is None:
            return None
        if self.af_dec and plan is None:
            return None
        w = self._af_weights() if self.af_gemv else False
        if not w:
            return None
        nw = input_layernorm.weight
        if plan is not None and self._af_head_ok:
            # v2: one kernel for add+RMSNorm, qkv GEMV, per-head q/k norm + RoPE + KV write; releases the Flint attention (PDL) early
            q, qkv, residual = _ag.qkv_head(
                hidden_states, residual, nw, input_layernorm.variance_epsilon, w[0], w[1], w[2],
                self.q_norm.weight, self.k_norm.weight, positions, self._cos_sin(), k_cache, v_cache, layer_slots,
            )
            attn_output = torch.empty_like(q)
            self._run_decode_attention(q, attn_output, *plan)
            output = _ag.o_gemv(attn_output, w[3], w[4], w[5], trigger=o_trigger)
            return output, residual
        qkv, residual = _ag.qkv_gemv(hidden_states, residual, nw, input_layernorm.variance_epsilon, w[0], w[1], w[2])
        q = qk_norm_rope_cache(
            qkv,
            positions,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.variance_epsilon,
            self._cos_sin(),
            k_cache,
            v_cache,
            layer_slots,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            launch_dependents=plan is not None,
        )
        attn_output = torch.empty_like(q)
        if plan is not None:
            self._run_decode_attention(q, attn_output, *plan)
        else:
            _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            unified_attention_with_output(
                q.view(1, self.num_heads, self.head_dim),
                k.view(1, self.num_kv_heads, self.head_dim),
                v.view(1, self.num_kv_heads, self.head_dim),
                attn_output.view(1, self.num_heads, self.head_dim),
                self.attn.layer_name,
            )
        output = _ag.o_gemv(attn_output, w[3], w[4], w[5], trigger=o_trigger)
        return output, residual

    def forward_ll(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        input_layernorm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Small pure-decode batches: two fused kernels for the residual-add
        RMSNorm, qkv GEMV, q/k norm, RoPE and KV-cache write, then the Flint
        decode attention. Returns None to fall back to the regular path."""
        if hidden_states.shape[0] > LL_MAX_TOKENS:
            return None
        k_cache, v_cache, layer_slots = self._cache_views()
        plan = self._decode_plan()
        if k_cache is None or v_cache is None or layer_slots is None or plan is None:
            return None
        layer = self.qkv_proj
        bf16w = layer.weight.dtype == torch.bfloat16
        q, residual = flint_qkv_fused(
            hidden_states,
            residual,
            input_layernorm.weight,
            input_layernorm.variance_epsilon,
            layer.weight,
            None if bf16w else layer.flint_weight_scale,
            None if bf16w else layer.weight_global_scale,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.variance_epsilon,
            self._cos_sin(),
            positions,
            k_cache,
            v_cache,
            layer_slots,
            self.num_heads,
            self.num_kv_heads,
        )
        attn_output = torch.empty_like(q)
        self._run_decode_attention(q, attn_output, *plan)
        output, _ = self.o_proj(attn_output)
        return output, residual


class FlintDecoderLayer(nn.Module):
    def __init__(
        self,
        config: FlintConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.self_attn = FlintAttention(
            config,
            layer_idx,
            cache_config,
            quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp: FlintMLP | FlintSparseMoeBlock
        if layer_idx in config.mlp_only_layers:
            self.mlp = FlintMLP(
                config.hidden_size,
                config.intermediate_size,
                quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = FlintSparseMoeBlock(config, quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.self_attn.af_gemv and hidden_states.shape[0] == 1:
            af = self.self_attn.forward_af(
                positions, hidden_states, residual, self.input_layernorm,
                o_trigger=_MG_OTRIG if (isinstance(self.mlp, FlintSparseMoeBlock) and self.mlp.can_route_norm(hidden_states, hidden_states)) else 0,
            )
            if af is not None:
                return self._post_attention(*af)
        if self.self_attn.flint_decode:
            ll = self.self_attn.forward_ll(
                positions, hidden_states, residual, self.input_layernorm
            )
            if ll is not None:
                return self._post_attention(*ll)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        return self._post_attention(hidden_states, residual)

    def _post_attention(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.mlp, FlintSparseMoeBlock):
            ll = self.mlp.forward_ll(
                hidden_states, residual, self.post_attention_layernorm
            )
            if ll is not None:
                return ll
            if self.mlp.can_route_norm(hidden_states, residual):
                return self.mlp._forward_decode_rn(hidden_states, residual, self.post_attention_layernorm)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class FlintModel(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: FlintConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: FlintDecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        else:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        if self.end_layer != len(self.layers):
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class FlintForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        }
    )
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: FlintConfig = vllm_config.model_config.hf_config
        self.config = config

        model_config = vllm_config.model_config
        self.gguf_path = find_gguf_file(
            model_config.model,
            model_config.revision,
            vllm_config.load_config.download_dir,
        )
        quant_config = vllm_config.quant_config
        if self.gguf_path is not None:
            if quant_config is not None:
                raise ValueError(
                    "Flint NVFP4 GGUF checkpoints carry their own "
                    "quantization; do not pass --quantization."
                )
            quant_config = _nvfp4_gguf_quant_config()
            vllm_config.quant_config = quant_config
            # load_weights reads the GGUF itself and never consumes the default
            # loader's (lazy) iterator; the pattern just names the right file.
            self.allow_patterns_overrides = ["*.gguf"]
        if quant_config is not None:
            quant_config.packed_modules_mapping = self.packed_modules_mapping
        self.quant_config = quant_config

        self.model = FlintModel(
            vllm_config, quant_config, prefix=maybe_prefix(prefix, "model")
        )
        self.flint_dense = _flint_kernels(quant_config, part="dense")
        if self.flint_dense:
            flint_linear.install_flint_dense_kernels(self.model)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.moe_layers = [
            layer.mlp.experts
            for layer in self.model.layers[
                self.model.start_layer : self.model.end_layer
            ]
            if isinstance(layer.mlp, FlintSparseMoeBlock)
        ]
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 1
        self.num_logical_experts = config.num_experts
        self.num_physical_experts = config.num_experts
        self.num_local_physical_experts = config.num_experts
        self.num_routed_experts = config.num_experts
        self.num_redundant_experts = 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if _fast_head.ENABLED or _fast_head.CAPTURE_DIR:
            fast = _fast_head.compute_logits(self, hidden_states)
            if fast is not None:
                return fast
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.gguf_path is not None:
            device = next(self.parameters()).device
            weights = self._apply_dequantized_scales(
                gguf_nvfp4_weights_iterator(self.gguf_path, device, _GGUF_DEQUANTIZED),
            )
            if self.flint_dense:
                weights = flint_linear.check_merged_global_scales(weights)
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def _apply_dequantized_scales(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if name == "lm_head.weight_scale_2":
                self.logits_processor.scale = float(tensor)
            else:
                yield name, tensor
