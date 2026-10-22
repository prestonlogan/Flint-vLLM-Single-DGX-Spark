# Fast decode build for one DGX Spark (GB10, sm_121), layered on the base recipe image (build ../Dockerfile first).
# Batch-1 decode only: exact NVFP4 LM head, own CUDA MoE decode kernels on the Marlin-packed experts, Flint's CuTe decode
# attention + fused qkv/o GEMVs, and a deterministic moe_align_block_size. Prefill and batched decode keep vLLM's stock paths.
ARG BASE=flint-vllm-spark:pr62938
FROM ${BASE}
ARG D=/usr/local/lib/python3.12/dist-packages/vllm/models/flint
COPY patch_align.py /tmp/patch_align.py
RUN python3 /tmp/patch_align.py \
    && python3 -c "import ast; ast.parse(open('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/moe_align_block_size.py').read())"
COPY model.py fast_head.py fp4moe_decode.py fp4moe_mar.cu attn_gemv.py attn_gemv2.cu ${D}/
ENV FLINT_FP4MOE_BUILD=/opt/flint_fp4moe_build FLINT_ATTN_GEMV_BUILD=/opt/flint_attn_gemv_build \
    FLINT_FAST_HEAD=1 FLINT_FAST_HEAD_MODE=nv4 \
    VLLM_FLINT_MOEDEC=1 FLINT_MOEDEC_UCFG=5 FLINT_MOEDEC_DCFG=3 \
    FLINT_ATTN_FUSE=1 VLLM_KV_CACHE_LAYOUT=HND \
    VLLM_DET_MOE_ALIGN=1
# Prebuild the CUDA extensions (nvcc in the image; no GPU needed).
RUN python3 ${D}/fp4moe_decode.py && python3 ${D}/attn_gemv.py
