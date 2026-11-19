#!/usr/bin/env python
"""Download PTB-XL v1.0.3 from PhysioNet and unpack it to <dest> (default: data/ptb-xl).

Pure Python (no curl/unzip), so it works the same on Windows, macOS and Linux.

    python scripts/download_data.py                      # -> data/ptb-xl
    python scripts/download_data.py --dest D:/datasets/ptb-xl
    python scripts/download_data.py --verify-only        # re-check an existing copy

After extraction every file is checked against PhysioNet's SHA256SUMS.txt.
PTB-XL is licensed CC BY 4.0 (Wagner et al., Scientific Data 2020).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import resolve_path  # noqa: E402

URL = "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
ZIP_SIZE = 1_839_504_686


def download(url: str, out: Path) -> None:
    """Stream to out, resuming a partial download with an HTTP Range request."""
    have = out.stat().st_size if out.exists() else 0
    if have == ZIP_SIZE:
        print(f"zip already downloaded: {out}")
        return
    req = urllib.request.Request(url, headers={"User-Agent": "ptbxl-downloader"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req) as r:
        if have and r.status != 206:      # server ignored the Range header: start over
            have = 0
        total = have + int(r.headers.get("Content-Length", 0))
        with open(out, "ab" if have else "wb") as f:
            done = have
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done / 1e9:5.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()


def verify(root: Path) -> int:
    sums = root / "SHA256SUMS.txt"
    if not sums.exists():
        sys.exit(f"{sums} not found; is {root} a PTB-XL folder?")
    lines = [l.split(maxsplit=1) for l in sums.read_text().splitlines() if l.strip()]
    bad = 0
    for i, (digest, name) in enumerate(lines, 1):
        p = root / name.strip().lstrip("*")
        if not p.exists():
            print(f"\nMISSING {name}")
            bad += 1
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            print(f"\nBAD HASH {name}")
            bad += 1
        if i % 2000 == 0 or i == len(lines):
            print(f"\r  verified {i}/{len(lines)} files", end="", flush=True)
    print()
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/ptb-xl", help="target folder (relative = repo root)")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    dest = resolve_path(args.dest)

    if not args.verify_only:
        if (dest / "ptbxl_database.csv").exists():
            print(f"{dest} already contains PTB-XL; verifying only.")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            zpath = dest.parent / "ptb-xl-1.0.3.zip"
            print(f"downloading {URL}\n  -> {zpath}")
            download(URL, zpath)
            print(f"extracting to {dest}")
            tmp = dest.parent / "_ptbxl_extract"
            shutil.rmtree(tmp, ignore_errors=True)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(tmp)
            inner = [p for p in tmp.iterdir() if p.is_dir()]
            if len(inner) != 1:
                sys.exit(f"unexpected zip layout in {tmp}: {inner}")
            inner[0].rename(dest)         # flatten PhysioNet's long folder name
            tmp.rmdir()
            if not args.keep_zip:
                zpath.unlink()

    print(f"verifying checksums in {dest}")
    bad = verify(dest)
    if bad:
        sys.exit(f"{bad} files missing or corrupted; delete {dest} and re-run.")
    print(f"OK. PTB-XL ready at {dest}")


if __name__ == "__main__":
    main()
