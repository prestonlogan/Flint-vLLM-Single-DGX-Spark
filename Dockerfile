# Flint (RecursiveSI) on vLLM for one DGX Spark: the vLLM nightly image, vLLM PR #62938 (Flint model and kernels),
# and a gate so GB10 (sm_121) runs vLLM stock layers where Flint's SM100 kernels do not build.
ARG BASE=vllm/vllm-openai:nightly-8e2beaa3bff595577a069d49d96bde9074294ec2
FROM ${BASE}
COPY patches/*.patch /tmp/patches/
RUN cd /usr/local/lib/python3.12/dist-packages \
    && patch -p1 --forward < /tmp/patches/flint-vllm-pr62938.patch \
    && patch -p1 --forward < /tmp/patches/flint-sm12x-gates.patch \
    && rm -r /tmp/patches \
    && pip install --no-cache-dir "gguf>=0.19.0"
