# Flint on one DGX Spark with vLLM: ~85 tok/s

This recipe serves [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint) on a single DGX Spark, using its 22.9 GB NVFP4 GGUF ([`RecursiveSI/flint-nvfp4-gguf`](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf)) and vLLM. Flint is a 39.8B-total / 2.5B-active MoE with 262K context. It supports reasoning and tool calls, under Apache 2.0.

Flint support is not in vLLM yet. The recipe is a thin image on top of a pinned vLLM nightly. It adds:
- [vLLM PR #62938](https://github.com/vllm-project/vllm/pull/62938), which adds the Flint model and kernels. The PR is open and unmerged; this repo vendors it at head `db02bc9f`.
- A small gate patch so the model runs on the Spark's GB10 (sm_121), where the PR's own SM100 kernels do not run.

## Results

Measured on one DGX Spark (GB10, 128 GB), single stream, using `smoke.py`:

| | |
|---|---|
| Decode | **84.9 – 85.2 tok/s** (307-token reply, reasoning included) |
| Time to first token | 0.11 s (short prompt) |
| Weights in memory | 21.9 GiB |
| KV cache at `GPU_UTIL=0.40` | 24.8 GiB = 881K tokens (3.4 full 262K-token requests) |
| Load time | about 100 s to load weights; about 2 minutes to ready with warm caches |
| Tool calls | `get_weather {"city": "Paris"}`, `finish_reason: tool_calls` (glm47 parser) |
| Reasoning | returned separately as `reasoning_content` (glm47 reasoning parser) |

## Quick start

Requirements: a DGX Spark with Docker and the NVIDIA container runtime, and about 35 GB of free disk (22 GB of weights plus the 10.7 GB base image).

```bash
git clone https://github.com/prestonlogan/Flint-vLLM-Single-DGX-Spark && cd Flint-vLLM-Single-DGX-Spark

# Weights (GGUF) plus config, tokenizer and chat template from the BF16 repo (no BF16 weights needed)
hf download RecursiveSI/flint-nvfp4-gguf
hf download RecursiveSI/flint --include "*.json" "*.jinja"

docker build -t flint-vllm-spark:pr62938 .
./run.sh 8000                      # detached container "flint-vllm", OpenAI API on :8000, model name "flint"
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

A rough byte count puts the decode floor near 150 tok/s at GB10 bandwidth: about 1.8 GB read per token across routed and shared experts, attention, and the BF16 LM head. Native sm_12x kernels have room to beat the current 85.

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

For llama-swap, use `cmd: env FOREGROUND=1 /path/to/run.sh ${PORT}` and `cmdStop: docker stop -t 30 flint-vllm`, with `checkEndpoint: /health`.

## License

Apache 2.0. The vendored patch is vLLM code (Apache 2.0) from PR #62938. Flint's weights are Apache 2.0 from RecursiveSI.
