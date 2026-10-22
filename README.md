# Flint on one DGX Spark with vLLM: ~135 tok/s

This recipe serves [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint) on a single DGX Spark, using its 22.9 GB NVFP4 GGUF ([`RecursiveSI/flint-nvfp4-gguf`](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf)) and vLLM. Flint is a 39.8B-total / 2.5B-active MoE with 262K context. It supports reasoning and tool calls, under Apache 2.0.

Flint support is not in vLLM yet. The recipe is a thin image on top of a pinned vLLM nightly. It adds:
- [vLLM PR #62938](https://github.com/vllm-project/vllm/pull/62938), which adds the Flint model and kernels. The PR is open and unmerged; this repo vendors it at head `db02bc9f`.
- A small gate patch so the model runs on the Spark's GB10 (sm_121), where the PR's own SM100 kernels do not run.
- An optional **fast build** (`fast/`) with our own batch-1 decode kernels for GB10, which takes single-stream decode from ~85 to ~135 tok/s.

## Results

Measured on one DGX Spark (GB10, 128 GB), single stream (c1), with `run.sh` defaults (262K context, `GPU_UTIL=0.40`):

| | Fast build (`fast/`) | Stock recipe |
|---|---|---|
| Decode, greedy | **134.6 tok/s** | 84.0 – 86.2 tok/s |
| Decode, sampled (T 1.0, top-p 0.95, top-k 20) | **132.5 tok/s** | 83.2 – 85.2 tok/s |
| `smoke.py` | 134.3 tok/s, tool call OK | 84.9 – 85.2 tok/s, tool call OK |
| Time to first token | 0.11 s (short prompt) | 0.11 s |
| Weights in memory | 22.4 GiB | 21.9 GiB |
| KV cache | 824,573 tokens (3.15 full 262K-token requests) | 881,138 tokens (3.4 requests) |
| Load time | about 2 minutes to ready with warm caches | same |

- Decode is median tok/s over 8 prose prompts × 800 tokens (reasoning included), computed as (completion tokens − 1) / (last chunk − first chunk).
- Quality of the fast build is within measured noise: decode-path perplexity −0.04% to −0.29% vs stock, and the LM head is exact. Details in [Fast build](#fast-build-135-toks-experimental).
- Both builds return tool calls (`get_weather {"city": "Paris"}`, `finish_reason: tool_calls`, glm47 parser) and reasoning as `reasoning_content` (glm47 reasoning parser).

## Quick start

Requirements: a DGX Spark with Docker and the NVIDIA container runtime, and about 35 GB of free disk (22 GB of weights plus the 10.7 GB base image).

```bash
git clone https://github.com/prestonlogan/Flint-vLLM-Single-DGX-Spark && cd Flint-vLLM-Single-DGX-Spark

# Weights (GGUF) plus config, tokenizer and chat template from the BF16 repo (no BF16 weights needed).
# Needs the hf CLI: pip install -U huggingface_hub (tested with GGUF @ 6ec43e19, config @ 2a247be0)
hf download RecursiveSI/flint-nvfp4-gguf
hf download RecursiveSI/flint --include "*.json" "*.jinja"

docker build -t flint-vllm-spark:pr62938 .
docker build -t flint-vllm-spark:fast fast/      # optional fast build (layered on the image above)

# Detached container "flint-vllm", OpenAI API on :8000, model name "flint". Plain ./run.sh 8000 runs the stock recipe.
IMAGE=flint-vllm-spark:fast ./run.sh 8000
docker logs -f flint-vllm          # wait for "Application startup complete"
python3 smoke.py 8000              # decode rate + a tool call
```

`run.sh` runs with `HF_HUB_OFFLINE=1`, so download first (or pass `HF_HUB_OFFLINE=0`). Settings are read from env:

| Variable | Default | |
|---|---|---|
| `MAX_LEN` | 262144 | `--max-model-len` |
| `MAX_SEQS` | 8 | `--max-num-seqs` |
| `GPU_UTIL` | 0.40 | `--gpu-memory-utilization`; 0.40 leaves room for other work on the Spark |
| `NAME` | flint | served model name |
| `IMAGE`, `CONTAINER` | `flint-vllm-spark:pr62938`, `flint-vllm` | |
| `HF_CACHE`, `CACHE` | `~/.cache/huggingface`, `~/.cache/flint-vllm` | Hugging Face cache; JIT caches (vLLM, Triton, CuTe DSL) |
| `FOREGROUND=1` | off | stay attached to the logs and stop the container on exit (for llama-swap and similar) |
| `EXTRA_ARGS` | | appended to `vllm serve` |
| `FLINT_KERNELS` | empty | see below |

The server applies the model's sampling defaults (temperature 1.0, top-p 0.95, top-k 20) when a request omits them. The model card recommends an output budget of at least 32K tokens for hard problems, and keeping returned `reasoning_content` in the conversation history.

## Fast build: ~135 tok/s (experimental)

`fast/` layers our own batch-1 decode kernels on top of the base image. Single-stream decode goes from about 85 to about 135 tok/s, and quality stays within measured noise.

```bash
docker build -t flint-vllm-spark:pr62938 .                       # base image first
docker build -t flint-vllm-spark:fast fast/
IMAGE=flint-vllm-spark:fast ./run.sh 8000
python3 smoke.py 8000
```

### Speed
c1 prose decode, 8 prompts × 800 tokens, median tok/s. Metric = (completion tokens − 1) / (last chunk − first chunk).
Rows were measured by alternating A/B runs on fresh prompts, with one server start per run.

| Build | Greedy | Sampled (T 1.0, top-p 0.95, top-k 20) |
|---|---|---|
| stock (this recipe) | 83.9 – 84.6 | 83.0 – 83.7 |
| + exact NVFP4 LM head | 107.5 | 106.0 |
| + MoE decode kernels | 117.7 | 116.1 |
| + head + decode attention / fused qkv-o (no MoE kernels) | 122.5 | 120.8 |
| **all of it (`fast/`)** | **134.3 – 135.0** | **132.4 – 132.9** |

A clean build from this repo was then run with `run.sh` defaults (262K context, `GPU_UTIL=0.40`):
- 134.6 tok/s greedy and 132.5 tok/s sampled, on the same 8-prompt set;
- weights 22.4 GiB;
- KV cache 824,573 tokens (3.15 full-length requests);
- `smoke.py` 134.3 tok/s with a correct tool call.

The three middle rows were each checked independently over 3 alternating rounds. The `fast/` row comes from 2 rounds plus the clean-build run above; it has not had a separate independent check.

### What it does
Decode at batch 1 (one sequence, one token) takes these paths. Prefill and batched decode keep vLLM's stock paths.

- **LM head (3.22 → 0.79 ms per token).** Flint's BF16 `lm_head` holds NVFP4 values (e2m1 × e4m3) widened to BF16, so it can be read losslessly as 4-bit. An M=1 fp32-FMA GEMV does this (`fast_head.py`).
  - Measured on 12,830 real decode positions: argmax and top-20 order identical to the stock fp32 head every time, max |Δlogit| 3e-5.
  - `prompt_logprobs` keeps the stock head.
- **MoE (47 layers).** Own CUDA kernels in `fp4moe_mar.cu` replace vLLM's Marlin MoE chain (router GEMV + top-k, align/sort, 2 GEMMs, activation, sum) for one token:
  - a fused router (sigmoid top-6 + renorm);
  - an up kernel (6 experts + shared expert, SiLU·mul);
  - a down kernel (weighted top-6 sum + shared add);
  - programmatic dependent launch (PDL) between them.
  - The kernels read the Marlin-packed expert tensors vLLM already holds, so there is no second copy of the weights.
  - About 65 µs per layer against a ~59 µs DRAM floor (Marlin's chain: ~74 µs).
- **Attention (49 layers).** PR #62938's own CuTe split-KV decode attention with sinks, made to run on sm_121:
  - KV cache in HND layout (`VLLM_KV_CACHE_LAYOUT=HND`);
  - a full-attention launch config that fits sm_121's 99 KB of shared memory.
  - Plus own NVFP4 GEMVs (`attn_gemv2.cu`). The qkv kernel fuses residual add, input RMSNorm, GEMV, q/k RMSNorm, RoPE and the KV-cache write. The o_proj kernel uses PDL weight prefetch.
- **Deterministic MoE align (`patch_align.py`).** Stock greedy output was not reproducible run to run.
  - Cause: vLLM's `moe_align_block_size` assigns slots with `atomicAdd`, and in batches of 9 or more tokens that reorders Marlin's split-K accumulation. M=1 decode was already reproducible.
  - The patch makes greedy text bitwise reproducible. It costs about 3.7% prefill and nothing in decode.

### Quality
Gate: decode-path perplexity, with every token run through the M=1 path (`--max-num-seqs 1 --max-num-batched-tokens 1`, 9,075 tokens), within +0.3% of stock. Stock reruns are bitwise identical, so the noise floor is 0.

| Build | PPL | Top-1 agreement with stock |
|---|---|---|
| stock | 7.0728 | — |
| + MoE kernels | 7.0522 (−0.29%) | 97.16% |
| + attention | 7.0702 (−0.04%) | 97.09% |

For scale, stock batched vs stock M=1 agree on 97.11% of tokens. The new kernels move numerics by about one BF16 ulp, the same amount as stock's own batch-size dependence.
The fast build passes `smoke.py` (reasoning + tool call).

### Switches and limits
- Kill switches (`docker run -e` / edit `run.sh`):
  - `FLINT_FAST_HEAD=0`
  - `VLLM_FLINT_MOEDEC=0`
  - `FLINT_ATTN_FUSE=0` (also drop `VLLM_KV_CACHE_LAYOUT=HND`)
  - `VLLM_DET_MOE_ALIGN=0`
- The kernels are written for Flint's shapes (256 experts, top-6, hidden 2048, expert width 512, shared 1024). The code checks the shapes and falls back to stock otherwise.
- Multiple concurrent streams fall back to the stock kernels for decode steps with more than one token, so aggregate throughput gains less than single-stream.
- Memory: weights 22.4 GiB vs 21.9 GiB (raw qkv/o copies for the GEMVs); KV cache 824,573 vs 881,138 tokens at `GPU_UTIL=0.40` (about 6% smaller).
- Not yet re-checked at very long context. The decode attention gains more as context grows, but the full-attention launch config was tuned at 1–4K tokens.

## What the image changes

The base is `vllm/vllm-openai:nightly-8e2beaa3bff595577a069d49d96bde9074294ec2` (vLLM 0.32.1rc1.dev197, torch 2.14.0+cu130, digest `sha256:167ccd8696bc09f1b63301185134e04b883623f8adcbd8e3a0ec3140a193abcf`).

- **`patches/flint-vllm-pr62938.patch`**: the `vllm/` files of PR #62938 at `db02bc9f`. This covers the Flint model, the NVFP4 GGUF loader, its kernels, and the config and registry entries. It is applied to the installed package.
- **`patches/flint-sm12x-gates.patch`**: the PR turns on Flint's own kernels whenever TP=1, without checking the GPU. On anything other than SM100 (B200/GB200), this patch switches them off and vLLM's stock layers run instead:
  - Marlin W4A16 for the NVFP4 linears and MoE experts. vLLM serves the GGUF weight-only.
  - FlashInfer attention with sinks for the sliding-window and full-attention layers.
- **`gguf>=0.19.0`**, which the GGUF loader needs.

### Why Flint's kernels are off on GB10

Each part was tried on the Spark:

| Part | What happens on sm_121 |
|---|---|
| MoE, grouped (prefill, more than 6 tokens) | FlashInfer's CuTe DSL `blackwell` grouped GEMM: "No supported CUDA architectures found for major versions [9, 10]" |
| MoE, decode (6 tokens or fewer) | Triton NVFP4 GEMV fails in `ptxas` for `sm_121a` |
| Decode attention / fused QKV | needs Flint's dense kernel install, which only replaces FlashInfer CuTe DSL linears. Marlin is picked on GB10, so it fails with `'QKVParallelLinear' object has no attribute 'flint_weight_scale'` |
| Dense GEMV | does nothing for the same reason; no speed change (84.5 vs 85.2 tok/s) |

`FLINT_KERNELS=moe,attn,dense` (any subset) turns the parts back on, for anyone porting them to sm_12x. As of this commit each of them fails or does nothing on a Spark, as shown above.

The stock path reads about 1.89 GB of weights per decode token, against a measured read ceiling of about 239 GB/s, so its floor is about 7.9 ms (~127 tok/s). The fast build gets under that by reading the LM head as NVFP4 (0.62 → 0.17 GB) and by running near the floor in its own decode kernels. See above.

## Known issues

- **Large agent tool lists.** In an agent harness with a big prompt (about 37K tokens and 135 tools), Flint often leaves a required argument out of its tool calls, even at temperature 0. In our case that was `bash`'s `description`. With a small tool list it fills the same schema correctly. The parser is not the cause: the raw model text is missing the argument. If your client rejects such calls, it will loop. Fill the argument in client- or gateway-side, or trim the tool list.
- **Unmerged PR.** The vendored patch tracks one PR commit against one nightly. Once Flint lands in vLLM, use an official image plus the gate patch, or drop the gate patch once vLLM checks GPU capability itself.

## Files

| File | |
|---|---|
| `Dockerfile` | base image + the two patches + `gguf` |
| `run.sh` | start the server (see the table above) |
| `smoke.py` | streamed chat with decode tok/s, then a tool call: `python3 smoke.py <port>` |
| `patches/` | the PR (vendored) and the GB10 gate |
| `fast/` | optional fast decode build (Dockerfile, kernels, patched `model.py`), see "Fast build" |

For llama-swap, use `cmd: env FOREGROUND=1 /path/to/run.sh ${PORT}` and `cmdStop: docker stop -t 30 flint-vllm`, with `checkEndpoint: /health`.

## License

Apache 2.0. The vendored patch is vLLM code (Apache 2.0) from PR #62938. Flint's weights are Apache 2.0 from RecursiveSI.
