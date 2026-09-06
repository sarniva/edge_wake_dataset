#!/usr/bin/env python3
"""Phase 3a: cut 30 s dataset takes into labeled 1 s clips.

VAD (adaptive per take) finds speech bursts; each burst becomes a 1.0 s
wake-word candidate centered on its energy centroid; low-energy regions
become silence clips. You review wake candidates with one keystroke
(energy contour shown + optional playback).

  python3 scripts/clip_cutter.py --takes-dir takes --outdir data
  python3 scripts/clip_cutter.py --take "sp01*" --no-play     # quiet review
  python3 scripts/clip_cutter.py --auto                        # accept all

Output: data/wake/*.wav, data/silence/*.wav, data/manifest.csv
Clip names: sp01_quiet_30cm_take01_c01.wav (typo 'quite' normalized).
"""
import argparse, csv, glob, os, struct, subprocess, sys
import wave
import numpy as np

SR = 16000

# ---------------------------------------------------------------- parsing

def load_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, path
        n = w.getnframes()
        s = np.array(struct.unpack("<%dh" % n, w.readframes(n)),
                     dtype=np.float64)
    return s

def save_wav(path, s):
    s = np.clip(np.round(s), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(s.tobytes())

def parse_session(d):
    # 'sp01_quite_30cm' -> speaker sp01, room quiet (typo fixed), dist 30cm
    parts = d.split("_")
    dist = parts[-1] if parts[-1] in ("30cm", "1m", "3m") else "?"
    rest = parts[:-1] if dist != "?" else parts
    speaker = rest[0]
    room = {"quite": "quiet"}.get("_".join(rest[1:]), "_".join(rest[1:]))
    return speaker, room, dist

# ---------------------------------------------------------------- VAD

def frame_rms(s, frame=480, hop=160):
    n = 1 + (len(s) - frame) // hop
    out = np.empty(n)
    for i in range(n):
        seg = s[i * hop:i * hop + frame]
        out[i] = np.sqrt((seg ** 2).mean())
    return out

def find_bursts(e, hop_s, a):
    floor = float(np.percentile(e, 10))
    enter = max(a.abs_min_enter, a.enter_mult * floor)
    exit = max(a.abs_min_exit, a.exit_mult * floor)
    hang = int(round(0.30 / hop_s))
    merge = int(round(a.merge_gap_ms / 1000 / hop_s))
    minlen = int(round(a.min_burst_ms / 1000 / hop_s))
    state, below, bursts, start = False, 0, [], 0
    for i, v in enumerate(e):
        if not state:
            if v >= enter:
                state, start, below = True, i, 0
        elif v < exit and (below := below + 1) >= hang:
            state = False
            bursts.append((start, i - hang))
    if state:
        bursts.append((start, len(e) - 1))
    # merge across word gaps ("Ja-go ... Gu-ru"), drop blips
    merged = []
    for b in bursts:
        if merged and b[0] - merged[-1][1] <= merge:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    info = {"floor": floor, "enter": enter, "exit": exit}
    return [(s0 * hop_s, (s1 + 1) * hop_s) for s0, s1 in merged
            if s1 + 1 - s0 >= minlen], info

# ---------------------------------------------------------------- review UI

def contour(e, hop_s, t0, t1, w0=None, w1=None, width=60):
    i0, i1 = max(0, int(t0 / hop_s)), min(len(e), int(t1 / hop_s))
    seg = e[i0:i1]
    if len(seg) == 0:
        return ""
    blk = np.array_split(seg, width)
    chars = []
    peak = max(seg.max(), 1.0)
    for j, b in enumerate(blk):
        v = b.max() / peak
        c = "#" if v > 0.66 else ("+" if v > 0.33 else ("." if v > 0.1 else " "))
        tc = (i0 + j * len(seg) / width) * hop_s
        c = "[" if (w0 is not None and abs(tc - w0) < (t1 - t0) / width) else c
        c = "]" if (w1 is not None and abs(tc - w1) < (t1 - t0) / width) else c
        chars.append(c)
    return "".join(chars)

def play(path, player):
    if player == "none":
        return
    try:
        subprocess.run([player, "-nodisp", "-autoexit", "-loglevel", "quiet", path]
                       if player == "ffplay" else [player, "-q", path],
                       timeout=5, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

def detect_player():
    for p in ("ffplay", "aplay", "paplay"):
        try:
            subprocess.run([p, "-h" if p == "ffplay" else "--help"],
                           capture_output=True, timeout=5)
            return p
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return "none"

def review(clip_path, meta, e, hop_s, player):
    print(f"\n--- {os.path.basename(clip_path)} ---")
    print(f"src {meta['src']} @ {meta['t0']:.2f}s  burst {meta['dur']:.2f}s  "
          f"peak {meta['peak']} rms {meta['rms']:.0f}")
    print("2 s context, [ ] = kept 1 s window:")
    print(contour(e, hop_s, meta['t0'] - 0.5, meta['t0'] + 1.5,
                  meta['w0'], meta['w1']))
    save_wav(clip_path + ".tmp", meta["audio"])
    os.replace(clip_path + ".tmp", clip_path)
    if player != "none":
        play(clip_path, player)
    while True:
        r = input("[y]eep wake / [n]o drop / [s]ilence / [q]uit? ").strip().lower()
        if r in ("y", "n", "s", "q"):
            return r

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes-dir", default="takes")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--take", default="*", help="glob on session dir")
    ap.add_argument("--enter-mult", type=float, default=2.2)
    ap.add_argument("--exit-mult", type=float, default=1.5)
    ap.add_argument("--abs-min-enter", type=float, default=2500)
    ap.add_argument("--abs-min-exit", type=float, default=1200)
    ap.add_argument("--merge-gap-ms", type=float, default=500)
    ap.add_argument("--min-burst-ms", type=float, default=250)
    ap.add_argument("--win-s", type=float, default=1.0)
    ap.add_argument("--silence-per-take", type=int, default=3)
    ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="accept all wake candidates without review")
    ap.add_argument("--resume", action="store_true",
                    help="skip takes already in manifest")
    a = ap.parse_args()

    player = "none" if a.no_play else detect_player()
    print(f"playback: {player}")

    man_path = os.path.join(a.outdir, "manifest.csv")
    os.makedirs(a.outdir, exist_ok=True)
    done_takes = set()
    if a.resume and os.path.exists(man_path):
        with open(man_path) as f:
            for row in csv.DictReader(f):
                done_takes.add(row["src_take"])
    man_exists = os.path.exists(man_path)
    man = open(man_path, "a", newline="")
    writer = csv.writer(man)
    if not man_exists:
        writer.writerow(["clip", "src_take", "src_start_s", "speaker", "room",
                         "dist", "label", "peak", "rms", "review"])

    sessions = sorted(glob.glob(os.path.join(a.takes_dir, a.take)))
    n_wake = n_sil = n_drop = 0
    try:
        for sess in sessions:
            if not os.path.isdir(sess):
                continue
            speaker, room, dist = parse_session(os.path.basename(sess))
            for wav in sorted(glob.glob(os.path.join(sess, "take_*.wav"))):
                tag = f"{os.path.basename(sess)}/{os.path.basename(wav)}"
                if tag in done_takes:
                    print(f"skip {tag} (in manifest)"); continue
                s = load_wav(wav)
                e = frame_rms(s)
                hop_s = 160 / SR
                bursts, info = find_bursts(e, hop_s, a)
                print(f"\n== {tag}: floor={info['floor']:.0f} "
                      f"thr={info['enter']:.0f}/{info['exit']:.0f} "
                      f"bursts={len(bursts)}")
                used = []  # kept 1 s windows (avoid overlap + silence clash)
                for bi, (b0, b1) in enumerate(bursts):
                    seg = e[int(b0 / hop_s):max(int(b1 / hop_s), int(b0 / hop_s) + 1)]
                    cen = (b0 + np.average(
                        np.arange(len(seg)), weights=seg + 1e-9) * hop_s
                        if len(seg) else (b0 + b1) / 2)
                    w0 = min(max(cen - a.win_s / 2, 0), len(s) / SR - a.win_s)
                    w1 = w0 + a.win_s
                    if any(not (w1 <= u0 or w0 >= u1) for u0, u1 in used):
                        print(f"  burst {bi}: overlaps kept window, skipped"); continue
                    audio = s[int(w0 * SR):int(w1 * SR)]
                    name = (f"{speaker}_{room}_{dist}_"
                            f"{os.path.basename(wav)[:-4]}_c{bi + 1:02d}.wav")
                    meta = {"src": tag, "t0": round(w0, 2),
                            "dur": round(b1 - b0, 2),
                            "peak": int(np.abs(audio).max()),
                            "rms": float(np.sqrt((audio ** 2).mean())),
                            "w0": w0, "w1": w1, "audio": audio}
                    if a.auto:
                        r = "y"
                        os.makedirs(f"{a.outdir}/wake", exist_ok=True)
                        save_wav(f"{a.outdir}/wake/{name}", audio)
                    else:
                        os.makedirs(f"{a.outdir}/wake", exist_ok=True)
                        tmp = f"{a.outdir}/wake/{name}"
                        r = review(tmp, meta, e, hop_s, player)
                        if r == "q":
                            if os.path.exists(tmp):
                                os.remove(tmp)
                            print("quit."); man.close(); summary(n_wake, n_sil, n_drop); return
                        if r == "n":
                            os.remove(tmp); n_drop += 1; continue
                        if r == "s":
                            os.makedirs(f"{a.outdir}/silence", exist_ok=True)
                            os.replace(tmp, f"{a.outdir}/silence/{name}")
                    label = "wake" if r == "y" else "silence"
                    writer.writerow([name, tag, meta["t0"], speaker, room,
                                     dist, label, meta["peak"],
                                     round(meta["rms"]), "auto" if a.auto else "human"])
                    man.flush()
                    used.append((w0, w1))
                    if r == "y":
                        n_wake += 1
                    else:
                        n_sil += 1
                # silence: quietest 1 s grid windows clear of wake windows
                sils = []
                t = 0.0
                while t + a.win_s <= len(s) / SR:
                    if all(t + a.win_s <= u0 - 0.5 or t >= u1 + 0.5 for u0, u1 in used):
                        seg = e[int(t / hop_s):int((t + a.win_s) / hop_s)]
                        sils.append((seg.max() if len(seg) else 1e18, t))
                    t += 0.5
                sils.sort()
                os.makedirs(f"{a.outdir}/silence", exist_ok=True)
                for k, (_, t0_) in enumerate(sils[:a.silence_per_take]):
                    audio = s[int(t0_ * SR):int((t0_ + a.win_s) * SR)]
                    name = (f"{speaker}_{room}_{dist}_"
                            f"{os.path.basename(wav)[:-4]}_sil{k + 1}.wav")
                    save_wav(f"{a.outdir}/silence/{name}", audio)
                    writer.writerow([name, tag, round(t0_, 2), speaker, room,
                                     dist, "silence",
                                     int(np.abs(audio).max()),
                                     round(float(np.sqrt((audio ** 2).mean())), 1),
                                     "auto-sil"])
                    man.flush()
                    n_sil += 1
    finally:
        man.close()
    summary(n_wake, n_sil, n_drop)

def summary(n_wake, n_sil, n_drop):
    print(f"\ndone: wake={n_wake} silence={n_sil} dropped={n_drop}")

if __name__ == "__main__":
    main()
