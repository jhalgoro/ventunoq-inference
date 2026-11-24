#!/usr/bin/env python3
"""Extract selected entries from a remote zip via HTTP Range requests (no full download)."""
import os, struct, sys, zlib, requests

URL = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.51.200.261021/v2.51.200.261021.zip"
DESIRED = sys.argv[1:]  # substrings of zip entry paths to extract
DEST = os.environ.get("EXTRACT_DEST", ".")

def rng(start, end):  # inclusive
    r = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    r.raise_for_status()
    return r.content

r = requests.get(URL, headers={"Range": "bytes=0-0"}, timeout=60)
size = int(r.headers["Content-Range"].split("/")[-1])
tail = rng(size - 1_000_000, size - 1)
i = tail.rfind(b"PK\x05\x06")
_, _, _, _, cd_count, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i+22])
cd = rng(cd_off, cd_off + cd_size - 1)

entries, p = [], 0
while p < len(cd):
    if cd[p:p+4] != b"PK\x01\x02": break
    (ver, verv, flag, method, _, _, _, csize, uncomp, nlen, elen, clen,
     _, _, ext, lho) = struct.unpack("<HHHHHHIIIHHHHHII", cd[p+4:p+46])
    name = cd[p+46:p+46+nlen].decode()
    if any(d in name for d in DESIRED):
        entries.append((name, method, csize, uncomp, lho))
    p += 46 + nlen + elen + clen
print(f"{len(entries)} entries to extract from {size/1e9:.1f} GB remote zip")
for name, method, csize, uncomp, lho in entries:
    lh = rng(lho, lho + 29)
    n, e = struct.unpack("<HH", lh[26:30])
    data = rng(lho + 30 + n + e, lho + 30 + n + e + csize - 1)
    if method == 8:
        data = zlib.decompress(data, -15)
    out = os.path.join(DEST, os.path.basename(name))
    open(out, "wb").write(data)
    print(f"  {name} -> {out} ({len(data)} bytes)")
