#!/usr/bin/env python3
"""A1: download negative datasets (no GPU needed).

  Google Speech Commands v2  -> ext/gsc/  (35 words, CC-BY-4.0)
  MUSAN (noise/music/speech) -> ext/musan/ (CC-BY-4.0)

  python3 scripts/download_negatives.py [--which gsc|musan|all]

Resume-capable; verifies marker files; writes ext/SOURCES.md.
~3.5 GB total. Run once.
"""
import argparse, os, sys, tarfile, urllib.request

GSC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"

def fetch(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except Exception as e:
        sys.exit(f"download failed: {e}")
    if have and r.status == 200:
        print("server ignored Range; restarting from 0"); have = 0
        mode = "wb"
    else:
        mode = "ab" if have else "wb"
        if have:
            print(f"resuming at {have / 1e6:.0f} MB")
    total = r.getheader("Content-Length")
    total = (int(total) + have) if total else None
    got, last = have, 0
    with open(dest, mode) as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total and got - last > 100e6:
                last = got
                print(f"  {got / 1e6:.0f}/{total / 1e6:.0f} MB")
    print(f"saved {dest} ({got / 1e6:.0f} MB)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", "gsc", "musan"])
    ap.add_argument("--ext", default="ext")
    a = ap.parse_args()
    jobs = []
    if a.which in ("all", "gsc"):
        jobs.append((GSC_URL, f"{a.ext}/gsc.tar.gz", f"{a.ext}/gsc",
                     "testing_list.txt", "Google Speech Commands v2 (CC-BY-4.0)"))
    if a.which in ("all", "musan"):
        jobs.append((MUSAN_URL, f"{a.ext}/musan.tar.gz", f"{a.ext}/musan",
                     "music", "MUSAN OpenSLR-17 (CC-BY-4.0)"))
    for url, arc, out, marker, label in jobs:
        if os.path.exists(os.path.join(out, marker)):
            print(f"{label}: already extracted at {out}, skipping")
            continue
        if not os.path.exists(arc):
            print(f"{label}: downloading ...")
            fetch(url, arc)
        print(f"{label}: extracting ...")
        with tarfile.open(arc) as t:
            t.extractall(out, filter="data")
        assert os.path.exists(os.path.join(out, marker)), f"marker {marker} missing!"
        print(f"{label}: OK")
    with open(f"{a.ext}/SOURCES.md", "w") as f:
        f.write("# Negative-data sources\n\n"
                f"- {GSC_URL} (Speech Commands v2, CC-BY-4.0)\n"
                f"- {MUSAN_URL} (MUSAN, CC-BY-4.0)\n")
    print("wrote ext/SOURCES.md")

if __name__ == "__main__":
    main()
