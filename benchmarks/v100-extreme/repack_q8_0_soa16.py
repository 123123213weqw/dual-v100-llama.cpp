#!/usr/bin/env python3
"""Repack GGUF Q8_0 tensors into the zero-overhead CUDA SoA16 layout.

Within each group of 16 standard 34-byte blocks, the bytes change from
    [d0,q0, d1,q1, ... d15,q15]
to
    [d0..d15, q0..q15].
The transformation preserves tensor/file sizes and makes every 32-byte quant
payload naturally aligned.  The resulting GGUF requires the matching V100-X1
CUDA kernels and is intentionally not a standard portable GGUF.
"""
from __future__ import annotations

import argparse
import mmap
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Run with PYTHONPATH=<llama.cpp>/gguf-py when gguf is not installed.
from gguf import GGMLQuantizationType, GGUFReader

GROUP = 16
BLOCK_BYTES = 34
SCALE_BYTES = 2
GROUP_BYTES = GROUP * BLOCK_BYTES


def copy_if_needed(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.stat().st_size != src.stat().st_size:
            raise RuntimeError(f"existing destination has wrong size: {dst}")
        return
    print(f"copying {src} -> {dst}", flush=True)
    shutil.copyfile(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--chunk-groups", type=int, default=4096)
    args = ap.parse_args()

    copy_if_needed(args.src, args.dst)
    reader = GGUFReader(args.dst, "r")
    # token_embd is intentionally left in the portable AoS layout: llama.cpp
    # keeps it in the CPU_Mapped buffer even when every transformer layer is
    # offloaded.  All GPU-resident Q8_0 weights use SoA16.
    tensors = [
        t for t in reader.tensors
        if t.tensor_type == GGMLQuantizationType.Q8_0 and t.name != "token_embd.weight"
    ]
    meta = [(t.name, int(t.data_offset), int(t.n_bytes), int(t.shape[0])) for t in tensors]
    del reader

    total = sum(nbytes for _, _, nbytes, _ in meta)
    done = 0
    with args.dst.open("r+b", buffering=0) as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
        for ti, (name, offset, nbytes, ne0) in enumerate(meta, 1):
            nblocks = nbytes // BLOCK_BYTES
            if nbytes % BLOCK_BYTES or nblocks % GROUP or ne0 % (32 * GROUP):
                raise RuntimeError(f"Q8 tensor is not SoA16-compatible: {name}, ne0={ne0}, bytes={nbytes}")

            ngroups = nblocks // GROUP
            for g0 in range(0, ngroups, args.chunk_groups):
                ng = min(args.chunk_groups, ngroups - g0)
                begin = offset + g0 * GROUP_BYTES
                end = begin + ng * GROUP_BYTES
                aos = np.frombuffer(mm[begin:end], dtype=np.uint8).reshape(ng, GROUP, BLOCK_BYTES)
                soa = np.empty((ng, GROUP_BYTES), dtype=np.uint8)
                soa[:, :GROUP * SCALE_BYTES] = aos[:, :, :SCALE_BYTES].reshape(ng, -1)
                soa[:, GROUP * SCALE_BYTES:] = aos[:, :, SCALE_BYTES:].reshape(ng, -1)
                mm[begin:end] = soa.tobytes()

            done += nbytes
            print(f"[{ti:3d}/{len(meta)}] {done/total:7.2%} {name}", flush=True)
        mm.flush()

    print(f"packed {len(meta)} Q8_0 tensors, {total} bytes", flush=True)


if __name__ == "__main__":
    main()
