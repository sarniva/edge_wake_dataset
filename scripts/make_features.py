#!/usr/bin/env python3
"""A2: wav -> 61x40 log-mel .npy using the BIT-EXACT S3 reference pipeline.

Reuses ref_logmel() from scripts/frontend_check.py (the same code the
bit-match checker validates against firmware), so training features are
provably what the chip computes. Handles any input SR/channels/length:
mono-mix, numpy-linear resample, center-crop/pad to exactly 1.0 s.

  python3 scripts/make_features.py --manifest data/manifest.csv \
      --wavdir data --outdir features/our
  python3 scripts/make_features.py --negatives --outdir features/neg \
      --max-per-word 300 --musan-slices 3000
"""
import argparse, csv, glob, os, struct, sys
import wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
# NOTE: frontend_check.py is vendored from edge_wake/scripts/frontend_check.py
# (same repo family). The S3 bit-match is validated from the edge_wake side;
# this copy only reuses its ref_logmel(). If upstream params change, re-copy
# and re-run the bit-match checker before training.
from frontend_check import ref_logmel  # noqa: E402  (bit-exact S3 pipeline)

SR = 16000
N = SR  # exactly 1.0 s

def load_any_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = w.readframes(n)
    width = len(raw) // max(n * ch, 1)
    fmt = {1: "b", 2: "h", 4: "i"}[width]
    s = np.array(struct.unpack(f"<{n * ch}{fmt}", raw), dtype=np.float64)
    if width == 1:  # unsigned 8-bit PCM
        s -= 128.0
    s *= 32768.0 / (2 ** (8 * width - 1))
    if ch > 1:
        s = s.reshape(n, ch).mean(axis=1)
    if sr != SR:  # linear resample (fine for features; inputs are clean)
        t_old = np.linspace(0, 1, len(s), endpoint=False)
        t_new = np.linspace(0, 1, int(round(len(s) * SR / sr)), endpoint=False)
        s = np.interp(t_new, t_old, s)
    if len(s) > N:  # center crop
        st = (len(s) - N) // 2
        s = s[st:st + N]
    elif len(s) < N:  # zero pad (logs the fact)
        s = np.pad(s, (0, N - len(s)))
    return np.clip(np.round(s), -32768, 32767).astype(np.int16).tolist()

def feat_of_wav(path):
    pcm = load_any_wav(path)
    return np.asarray(ref_logmel(pcm), dtype=np.float32)  # 61x40

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--wavdir", default="data")
    ap.add_argument("--outdir", default="features/our")
    ap.add_argument("--negatives", action="store_true")
    ap.add_argument("--ext", default="ext")
    ap.add_argument("--max-per-word", type=int, default=300)
    ap.add_argument("--musan-slices", type=int, default=3000)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(7)

    jobs = []  # (wav_path, label, key)
    if a.negatives:
        gsc = os.path.join(a.ext, "gsc")
        words = sorted(d for d in os.listdir(gsc)
                       if os.path.isdir(os.path.join(gsc, d))
                       and not d.startswith("_") and d != "LICENSE")
        for wd in words:
            fs = sorted(glob.glob(os.path.join(gsc, wd, "*.wav")))[:a.max_per_word]
            jobs += [(f, "unknown", f"gsc/{wd}/{os.path.basename(f)}") for f in fs]
        # MUSAN long files -> random 1 s slices (speech=unknown, music/noise=silence)
        mus = []
        for sub, lab in (("speech", "unknown"), ("music", "silence"), ("noise", "silence")):
            mus += [(p, lab) for p in glob.glob(os.path.join(a.ext, "musan", sub, "**", "*.wav"), recursive=True)]
        per = max(1, a.musan_slices // max(len(mus), 1))
        for p, lab in mus:
            try:
                s = np.array(load_any_wav(p), dtype=np.float64)
            except (wave.Error, struct.error, KeyError):
                continue
            for _ in range(per):
                if len(s) < N:
                    continue
                st = int(rng.integers(0, len(s) - N + 1))
                jobs.append((None, lab, None))  # placeholder, sliced below
                jobs[-1] = (s[st:st + N].astype(np.int16).tolist(), lab,
                            f"musan/{sub}/{os.path.basename(p)}#{st}")
    else:
        if not a.manifest or not os.path.exists(a.manifest):
            sys.exit("need --manifest data/manifest.csv (run clip review first)")
        with open(a.manifest) as f:
            for row in csv.DictReader(f):
                sub = "wake" if row["label"] == "wake" else "silence"
                jobs.append((os.path.join(a.wavdir, sub, row["clip"]),
                             row["label"], row["clip"]))

    os.makedirs(a.outdir, exist_ok=True)
    idx_path = os.path.join(a.outdir, "features.csv")
    have = set()
    if a.resume and os.path.exists(idx_path):
        with open(idx_path) as f:
            for row in csv.DictReader(f):
                have.add(row["key"])
    mode = "a" if os.path.exists(idx_path) else "w"
    n_ok = n_skip = 0
    with open(idx_path, mode, newline="") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["key", "npy", "label"])
        for j, (src, label, key) in enumerate(jobs):
            out = os.path.join(a.outdir, f"f{j:06d}.npy")
            if key in have and os.path.exists(out):
                n_skip += 1
                continue
            try:
                if isinstance(src, list):
                    feat = np.asarray(ref_logmel(src), dtype=np.float32)
                else:
                    feat = feat_of_wav(src)
                assert feat.shape == (61, 40) and np.isfinite(feat).all()
            except Exception as e:
                print(f"SKIP {key}: {e}")
                continue
            np.save(out, feat)
            w.writerow([key, os.path.basename(out), label])
            n_ok += 1
            if n_ok % 2000 == 0:
                print(f"  {n_ok} features ...")
    print(f"done: {n_ok} new, {n_skip} resumed in {a.outdir}")

if __name__ == "__main__":
    main()
