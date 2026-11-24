#!/usr/bin/env python3
"""Fetch a page through the CloakBrowser CDP endpoint and dump rendered text.

Usage: python3 scripts/cdp_fetch.py <url> [outfile] [--wait SECONDS] [--raw-html]
Requires PYTHONPATH to include the vendored websockets package (see scripts/env.sh).
"""
import asyncio
import json
import os
import re
import sys
import urllib.request

MANAGER = os.environ.get("CLOAKBROWSER", "http://localhost:18080")
PROFILE = os.environ.get("CLOAK_PROFILE", "8f3fe7a3-89ff-4017-8b63-bf2e2b14589c")
CDP = f"{MANAGER}/api/profiles/{PROFILE}/cdp"


async def send(ws, cid, method, params=None, session=None):
    msg = {"id": cid, "method": method, "params": params or {}}
    if session:
        msg["sessionId"] = session
    await ws.send(json.dumps(msg))
    while True:
        reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
        if reply.get("id") == cid:
            if "error" in reply:
                raise RuntimeError(f"{method}: {reply['error']}")
            return reply.get("result", {})


async def fetch(url, wait, raw_html):
    import websockets
    cid = 0

    def nid():
        nonlocal cid
        cid += 1
        return cid

    async with websockets.connect(CDP.replace("http", "ws"),
                                  max_size=100 * 1024 * 1024) as ws:
        # create our own tab at browser level (HTTP /json/new is proxied read-only)
        tgt = await send(ws, nid(), "Target.createTarget", {"url": "about:blank"})
        target_id = tgt["targetId"]
        session = (await send(ws, nid(), "Target.attachToTarget",
                              {"targetId": target_id, "flatten": True}))["sessionId"]

        async def cdp(method, params=None):
            return await send(ws, nid(), method, params, session)

        async def ev(expr):
            r = await cdp("Runtime.evaluate", {"expression": expr,
                                               "returnByValue": True, "awaitPromise": True})
            return r.get("result", {}).get("value")

        try:
            await cdp("Page.enable")
            await cdp("Page.navigate", {"url": url})
            for _ in range(60):  # wait for load event
                if await ev("document.readyState") == "complete":
                    break
                await asyncio.sleep(1)
            await asyncio.sleep(wait)  # settle time for JS rendering
            title = await ev("document.title") or ""
            if raw_html:
                body = await ev("document.documentElement.outerHTML") or ""
            else:
                body = await ev("""
                    (() => {
                      for (const s of document.querySelectorAll('script,style,noscript,svg')) s.remove();
                      const main = document.querySelector('main, article, [role=main]') || document.body;
                      return main.innerText;
                    })()
                """) or ""
        finally:
            await cdp("Page.close")
        return title, body


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    url = args[0]
    outfile = args[1] if len(args) > 1 else None
    wait = 3.0
    raw = "--raw-html" in sys.argv
    for i, a in enumerate(sys.argv):
        if a == "--wait":
            wait = float(sys.argv[i + 1])
    title, body = asyncio.run(fetch(url, wait, raw))
    body = re.sub(r"\n{3,}", "\n\n", body)
    text = f"SOURCE: {url}\nTITLE: {title}\n{'=' * 80}\n{body}"
    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[saved {outfile}] {len(body)} chars — title: {title}")
    else:
        print(text[:20000])


if __name__ == "__main__":
    main()
