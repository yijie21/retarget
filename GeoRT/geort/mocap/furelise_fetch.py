# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
    Fetch individual pieces from the FürElise dataset (rcwang/for_elise on Hugging Face)
    WITHOUT downloading the full 47 GB dataset.zip.

    The dataset ships as one monolithic zip; this streams only the central directory plus
    the requested entry via HTTP range requests (the CDN supports them), so a single piece
    costs a few MB instead of 47 GB.

    Usage:
        python geort/mocap/furelise_fetch.py --list                 # list entries
        python geort/mocap/furelise_fetch.py --piece 0 --out <dir>  # extract one piece
'''

import argparse
import io
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://huggingface.co/datasets/rcwang/for_elise/resolve/main/dataset.zip"
API_URL = "https://huggingface.co/api/datasets/rcwang/for_elise"


class HTTPRangeFile(io.RawIOBase):
    '''A read-only, seekable file backed by HTTP range requests (via curl, which follows
    HF's redirect to the signed CDN URL on every call).'''

    def __init__(self, url, size):
        self.url = url
        self.size = size
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        start = self.pos
        end = min(self.size, start + n) - 1
        out = subprocess.run(
            ["curl", "-s", "-m", "120", "-r", f"{start}-{end}", "-L", self.url],
            capture_output=True,
        )
        data = out.stdout
        self.pos = start + len(data)
        return data


def get_zip_size():
    with urllib.request.urlopen(API_URL, timeout=30) as r:
        meta = json.load(r)
    for s in meta.get("siblings", []):
        if s["rfilename"] == "dataset.zip":
            # size not always in siblings; fall back to a HEAD-style range probe
            break
    # Authoritative size via content-range probe.
    out = subprocess.run(["curl", "-sI", "-r", "0-0", "-L", ZIP_URL], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.lower().startswith("content-range:"):
            return int(line.split("/")[-1].strip())
    raise RuntimeError("could not determine dataset.zip size")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="list zip entries and exit")
    p.add_argument("--piece", type=int, default=None, help="piece_id to extract")
    p.add_argument("--out", type=str, default="/mnt/yijie/storage/hf_datasets/geort/furelise")
    p.add_argument("--max_list", type=int, default=60)
    args = p.parse_args()

    size = get_zip_size()
    print(f"dataset.zip size: {size/1e9:.1f} GB")
    rf = HTTPRangeFile(ZIP_URL, size)
    zf = zipfile.ZipFile(rf)
    names = zf.namelist()
    print(f"{len(names)} entries in zip.")

    if args.list or args.piece is None:
        for info in zf.infolist()[: args.max_list]:
            print(f"  {info.file_size:>12d}  {info.filename}")
        if len(names) > args.max_list:
            print(f"  ... ({len(names) - args.max_list} more)")
        return

    # Find entries for this piece. FürElise pieces live under dataset/<zero-padded id>/;
    # we only need motion.pkl. Match the directory for both padded and unpadded ids.
    pids = {str(args.piece), f"{args.piece:03d}"}
    candidates = [n for n in names if not n.endswith("/")
                  and any(f"dataset/{p}/" in n or n.startswith(f"{p}/") for p in pids)
                  and n.endswith("motion.pkl")]
    if not candidates:  # fall back to all files for the piece if motion.pkl naming differs
        candidates = [n for n in names if not n.endswith("/")
                      and any(f"dataset/{p}/" in n or n.startswith(f"{p}/") for p in pids)]
    if not candidates:
        raise SystemExit(f"No entries matched piece {args.piece}. Re-run with --list to inspect names.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {len(candidates)} entries -> {out_dir}:")
    for n in candidates:
        info = zf.getinfo(n)
        dst = out_dir / Path(n).name
        print(f"  {info.file_size/1e6:.2f} MB  {n} -> {dst}")
        with zf.open(n) as src, open(dst, "wb") as f:
            f.write(src.read())
    print("Done.")


if __name__ == "__main__":
    main()
