import json, sys, time, urllib.request
URL = f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions"
def post(body, stream=False):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)
# 1. plain chat, streamed: reasoning + answer, decode rate
body = {"model": "flint", "messages": [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation. Then give two example calls."}],
        "max_tokens": 1500, "stream": True, "stream_options": {"include_usage": True}}
t0, first, times, rc, ct, usage = time.time(), None, [], "", "", None
for line in post(body):
    line = line.decode().strip()
    if not line.startswith("data: ") or line == "data: [DONE]": continue
    d = json.loads(line[6:])
    if d.get("usage"): usage = d["usage"]
    for c in d.get("choices", []):
        dl = c.get("delta") or {}
        piece = (dl.get("reasoning_content") or dl.get("reasoning") or "") + (dl.get("content") or "")
        if piece:
            now = time.time(); first = first or now; times.append(now)
        rc += dl.get("reasoning_content") or dl.get("reasoning") or ""; ct += dl.get("content") or ""
n = (usage or {}).get("completion_tokens")
print(json.dumps({"ttft_s": round(first - t0, 2), "completion_tokens": n, "decode_tok_s": round((n - 1) / (times[-1] - first), 1) if n and len(times) > 1 else None,
                  "reasoning_chars": len(rc), "answer": ct[:300]}))
# 2. tool call
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
d = json.load(post({"model": "flint", "messages": [{"role": "user", "content": "What's the weather in Paris right now?"}], "tools": tools, "max_tokens": 1000}))
m = d["choices"][0]["message"]
print(json.dumps({"finish": d["choices"][0]["finish_reason"], "tool_calls": m.get("tool_calls"), "content": (m.get("content") or "")[:200]}))
