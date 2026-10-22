import re, sys
p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/moe_align_block_size.py"
s = open(p).read()
helper = '''

import os as _os
_DET_ALIGN = _os.environ.get("VLLM_DET_MOE_ALIGN", "0") == "1"
_DET_MIN_TOKENS = int(_os.environ.get("VLLM_DET_MOE_ALIGN_MIN_TOKENS", "9"))


def _deterministic_align(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad):
    """Same outputs as the CUDA moe_align_block_size except the order of routes inside an expert is ascending
    (route index) instead of atomicAdd-arrival order. The atomic order decides which m-block of an expert a route lands in,
    and Marlin's split-K stripe boundaries depend on the m-block position -> run-to-run 1-ulp differences. Sync-free."""
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    n = flat.numel()
    counts = torch.zeros(num_experts, dtype=torch.int64, device=dev).scatter_add_(0, flat, torch.ones_like(flat))
    padded = ((counts + block_size - 1) // block_size) * block_size
    pad_end = padded.cumsum(0)
    pad_start = pad_end - padded
    cnt_start = counts.cumsum(0) - counts
    sv, si = torch.sort(flat, stable=True)
    rank = torch.arange(n, device=dev) - cnt_start[sv]
    sorted_ids.fill_(n)
    sorted_ids[pad_start[sv] + rank] = si.to(torch.int32)
    blk_start = torch.arange(expert_ids.numel(), device=dev, dtype=torch.int64) * block_size
    e = torch.searchsorted(pad_end, blk_start, right=True)
    expert_ids.copy_(torch.where(e >= num_experts, torch.full_like(e, -1), e).to(torch.int32))
    num_tokens_post_pad.copy_(pad_end[-1:].to(torch.int32))
'''
anchor = "\n\n@overload\ndef moe_align_block_size("
assert anchor in s
s = s.replace(anchor, helper + anchor, 1)
old = "    ops.moe_align_block_size(\n        topk_ids,\n        num_experts,\n        block_size,\n        sorted_ids,\n        expert_ids,\n        num_tokens_post_pad,\n        expert_map if ignore_invalid_experts else None,\n        scatter_idx,\n    )\n"
assert old in s
new = ("    if (_DET_ALIGN and topk_ids.shape[0] >= _DET_MIN_TOKENS and scatter_idx is None\n"
       "            and not (expert_map is not None and ignore_invalid_experts)):\n"
       "        _deterministic_align(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)\n"
       "    else:\n    " + old.replace("\n    ", "\n        ").rstrip(" ") )
new = new.replace("    else:\n        ops.moe", "    else:\n        ops.moe")
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched")
