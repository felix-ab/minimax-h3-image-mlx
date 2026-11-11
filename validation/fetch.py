#!/usr/bin/env python3
"""Selective, resumable download of MiniMax-H3-Image tensors with HTTP range reads.

The full upstream checkpoint is ~135 GB and the Mac has ~124 GB free, so we never mirror whole shards. Each
needed tensor is range-read out of the pinned upstream shards and written into small per-component safetensors
files (original tensor names kept, so the torch reference loads them unchanged).

Groups (see plan()):
  dit/core.safetensors        proj_in, context_embedder, time_embedder, norm_out, pdd_proj_out.*
  dit/refiner.safetensors     token_refiner.*
  dit/blocks/NN.safetensors   transformer_blocks.NN.* except adaln_proj
  dit/adaln/NN.safetensors    transformer_blocks.NN.adaln_proj.* (temporary: consumed by make_modulation.py)
  te/embed.safetensors        model.language_model.embed_tokens.weight
  te/layers/NN.safetensors    model.language_model.layers.NN.* for NN < 50 (DiT reads hidden_states[50])
  te/visual.safetensors       model.visual.* (vision tower, edit only)
  vae/                        vae_diffusers/diffusion_pytorch_model.safetensors (whole file, 2.3 GB)

A group is complete when its `.json` sidecar exists (per-tensor sha256 of what we wrote, plus source offsets).

Usage: fetch.py OUT_DIR [--only REGEX] [--workers 16]
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "MiniMaxAI/MiniMax-H3-Image"
REVISION = "492fd8c4eca666f1085f57e30738c1e0d86b3439"  # main on 2026-11-10 16:21 UTC
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
TOKEN = (Path.home() / ".cache/huggingface/token").read_text().strip()
CHUNK = 32 << 20
TE_LAYERS = 50


def request(path, start, end, attempts=8):
    """Bytes [start, end) of a repo file."""
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(f"{BASE}/{path}", headers={
                "Authorization": f"Bearer {TOKEN}", "Range": f"bytes={start}-{end - 1}",
                "User-Agent": "selective-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=120) as response:
                data = response.read()
            if len(data) != end - start:
                raise IOError(f"short read {len(data)} != {end - start}")
            return data
        except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as error:
            wait = min(60, 2 ** attempt)
            print(f"  retry {attempt + 1} {path} [{start},{end}) after {error!r}; sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {path} [{start},{end})")


_headers = {}


def shard_header(path):
    if path not in _headers:
        (length,) = struct.unpack("<Q", request(path, 0, 8))
        header = json.loads(request(path, 8, 8 + length))
        header.pop("__metadata__", None)
        _headers[path] = (8 + length, header)
    return _headers[path]


def plan(upstream: Path):
    dit_map = json.loads((upstream / "transformer_diffusers/diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    te_map = json.loads((upstream / "text_encoder/model.safetensors.index.json").read_text())["weight_map"]
    groups = {}

    def add(group, name, shard):
        groups.setdefault(group, []).append((name, shard))

    for name, shard in dit_map.items():
        shard = f"transformer_diffusers/{shard}"
        block = re.match(r"transformer_blocks\.(\d+)\.", name)
        if block:
            index = int(block.group(1))
            add(f"dit/adaln/{index:02d}" if ".adaln_proj." in name else f"dit/blocks/{index:02d}", name, shard)
        elif name.startswith("token_refiner."):
            add("dit/refiner", name, shard)
        else:
            add("dit/core", name, shard)
    for name, shard in te_map.items():
        shard = f"text_encoder/{shard}"
        layer = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
        if layer:
            if int(layer.group(1)) < TE_LAYERS:
                add(f"te/layers/{int(layer.group(1)):02d}", name, shard)
        elif name == "model.language_model.embed_tokens.weight":
            add("te/embed", name, shard)
        elif name.startswith("model.visual."):
            add("te/visual", name, shard)
        # lm_head, model.language_model.norm and layers >= 50 are never read
    return groups


def fetch_group(out: Path, group: str, members, pool):
    target = out / f"{group}.safetensors"
    sidecar = out / f"{group}.json"
    if sidecar.exists():
        return 0
    adaln = re.match(r"dit/adaln/(\d+)$", group)
    if adaln and (out / f"dit/modulation/{adaln.group(1)}.safetensors").exists():
        return 0  # already consumed into the modulation table by make_modulation.py
    target.parent.mkdir(parents=True, exist_ok=True)
    entries, offset = [], 0
    for name, shard in sorted(members):
        data_start, header = shard_header(shard)
        info = header[name]
        begin, end = info["data_offsets"]
        size = end - begin
        entries.append(dict(name=name, dtype=info["dtype"], shape=info["shape"], shard=shard,
                            src=data_start + begin, size=size, dst=offset))
        offset += size
    header = {"__metadata__": {"format": "pt", "source": f"{REPO}@{REVISION}"}}
    for e in entries:
        header[e["name"]] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [e["dst"], e["dst"] + e["size"]]}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    base = 8 + len(blob)
    partial = target.with_suffix(".partial")
    with open(partial, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)) + blob)
        handle.truncate(base + offset)
    fd = os.open(partial, os.O_WRONLY)
    jobs = []
    for e in entries:
        for start in range(0, e["size"], CHUNK):
            stop = min(e["size"], start + CHUNK)
            jobs.append((e["shard"], e["src"] + start, e["src"] + stop, base + e["dst"] + start))

    def run(job):
        shard, src_start, src_end, dst = job
        data = request(shard, src_start, src_end)
        os.pwrite(fd, data, dst)
        return len(data)

    done = sum(pool.map(run, jobs))
    os.fsync(fd)
    os.close(fd)
    # Hash what we wrote (our own manifest; upstream publishes only whole-shard hashes).
    with open(partial, "rb") as handle:
        for e in entries:
            handle.seek(base + e["dst"])
            digest = hashlib.sha256()
            remaining = e["size"]
            while remaining:
                piece = handle.read(min(remaining, 64 << 20))
                digest.update(piece)
                remaining -= len(piece)
            e["sha256"] = digest.hexdigest()
    os.replace(partial, target)
    sidecar.write_text(json.dumps({"source": f"{REPO}@{REVISION}", "tensors": entries}, indent=1))
    return done


def fetch_whole(out: Path, path: str, dest: str, pool):
    target = out / dest
    sidecar = target.with_name(target.name + ".json")
    if sidecar.exists():
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{BASE}/{path}", method="HEAD", headers={"Authorization": f"Bearer {TOKEN}"})
    size = None
    for name in ("x-linked-size", "content-length"):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                size = int(r.headers.get(name) or 0) or size
        except Exception:
            pass
        if size:
            break
    partial = target.with_suffix(".partial")
    with open(partial, "wb") as handle:
        handle.truncate(size)
    fd = os.open(partial, os.O_WRONLY)
    jobs = [(s, min(size, s + CHUNK)) for s in range(0, size, CHUNK)]
    done = sum(pool.map(lambda j: os.pwrite(fd, request(path, *j), j[0]), jobs))
    os.fsync(fd)
    os.close(fd)
    digest = hashlib.sha256()
    with open(partial, "rb") as handle:
        while piece := handle.read(64 << 20):
            digest.update(piece)
    os.replace(partial, target)
    sidecar.write_text(json.dumps({"source": f"{REPO}@{REVISION}/{path}", "size": size, "sha256": digest.hexdigest()}))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--upstream", type=Path, default=Path(__file__).resolve().parent.parent / "upstream")
    parser.add_argument("--only", default=".*")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    groups = plan(args.upstream)
    def rank(g):
        # core first (make_modulation.py needs time_embedder), then each block's adaln right before its block so
        # the temporary adaln files are consumed as they land, then the text encoder.
        head = ["dit/core", "dit/refiner", "te/embed"]
        if g in head:
            return (0, head.index(g), "")
        m = re.match(r"dit/(adaln|blocks)/(\d+)", g)
        if m:
            return (1, int(m.group(2)), m.group(1))
        return (2, 0, g)
    order = sorted(groups, key=rank)
    order = [g for g in order if re.search(args.only, g)]
    total_bytes = 0
    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        if re.search(args.only, "vae"):
            n = fetch_whole(args.out, "vae_diffusers/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors", pool)
            total_bytes += n
            print(f"vae {n / 1e9:.2f} GB", flush=True)
        for group in order:
            n = fetch_group(args.out, group, groups[group], pool)
            total_bytes += n
            if n:
                rate = total_bytes / max(1e-9, time.time() - start) / 1e6
                print(f"{group}: {n / 1e9:.2f} GB  (total {total_bytes / 1e9:.1f} GB, {rate:.0f} MB/s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
