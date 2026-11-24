#!/usr/bin/env python3
"""Smoke tests: chat template + tool calling via llama-server OpenAI API."""
import json, sys, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8181

def post(path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)

print("== models ==")
print(json.dumps(post("/models", {}), indent=1)[:400])

print("\n== chat: basic ==")
r = post("/chat/completions", {
    "messages": [{"role": "user", "content": "Reply with exactly one word: what is 2+2?"}],
    "temperature": 0, "max_tokens": 64})
print(json.dumps(r["choices"][0]["message"], ensure_ascii=False)[:600])

print("\n== chat: tool call ==")
r = post("/chat/completions", {
    "messages": [{"role": "user", "content": "What's the weather in Tokyo right now? Use a tool."}],
    "temperature": 0, "max_tokens": 200,
    "tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"}},
            "required": ["city"]}}}]})
msg = r["choices"][0]["message"]
print(json.dumps(msg, ensure_ascii=False)[:600])
tc = msg.get("tool_calls") or []
ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "tokyo" in tc[0]["function"]["arguments"].lower()
print("TOOL_CALL_OK" if ok else "TOOL_CALL_FAIL")

print("\n== chat: structured JSON out ==")
r = post("/chat/completions", {
    "messages": [{"role": "user", "content": 'Output only JSON: {"name": <your name>, "params": <param count as int string?>} for a model named LFM3-1.7B with 1.73 billion parameters.'}],
    "temperature": 0, "max_tokens": 120})
print(json.dumps(r["choices"][0]["message"], ensure_ascii=False)[:400])
