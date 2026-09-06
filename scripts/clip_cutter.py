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

# ---------------------------------------------------------------- peaks
# Peak-picked candidates (robust in noise): smooth the energy, take local
# maxima above the enter threshold with >=700 ms separation (syllables of one
# "Jago Guru" are ~150-250 ms apart and merge; utterances are ~3 s apart and
# split). One 1 s window per peak. This replaces burst-merge logic, which
# glued whole takes together in noisy rooms (exit thr below noise floor).

def find_peaks(e, hop_s, enter, min_sep_s=0.7, smooth_s=0.15):
    w = max(1, int(round(smooth_s / hop_s)))
    sm = np.convolve(e, np.ones(w) / w, mode="same")
    loc = [i for i in range(1, len(sm) - 1)
           if sm[i] >= enter and sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]]
    loc.sort(key=lambda i: sm[i], reverse=True)
    kept = []
    for i in loc:
        if all(abs(i - j) * hop_s >= min_sep_s for j in kept):
            kept.append(i)
    kept.sort()
    return kept, sm

def prominence(sm, hop_s, i, half_s=1.5, guard_s=0.2):
    # Peak height above local background (median of surroundings, excluding
    # the peak's own neighbourhood). Loud horns score high too - the human
    # review drops those; this only removes flat noise bumps.
    w = int(round(half_s / hop_s))
    g = int(round(guard_s / hop_s))
    lo, hi = max(0, i - w), min(len(sm), i + w + 1)
    left = sm[lo:max(i - g, lo)]
    right = sm[min(i + g + 1, hi):hi]
    bg = np.median(np.concatenate([left, right])) if left.size + right.size else np.median(sm[lo:hi])
    return float(sm[i] - bg)

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

NUDGE_S = 0.10  # window slide step for a/d keys

def review(clip_path, meta, e, hop_s, player, full, take_dur):
    # Interactive verdict with re-positioning: [a]/[d] slide the 1 s window
    # ∓100 ms (repeatable), re-rendering + replaying each time. Returns
    # (verdict, used_merge) where verdict is y/n/s/q/m (m = merge with next).
    w0, w1 = meta["w0"], meta["w1"]
    win = w1 - w0  # constant 1 s window; nudges slide it rigidly
    while True:
        meta["w0"], meta["w1"] = w0, w1
        meta["t0"] = round(w0, 2)
        audio = full[int(w0 * SR):int(w1 * SR)]
        meta["audio"] = audio
        meta["peak"] = int(np.abs(audio).max())
        meta["rms"] = float(np.sqrt((audio ** 2).mean()))
        print(f"\n--- {os.path.basename(clip_path)} ---")
        print(f"src {meta['src']} @ {w0:.2f}s  peak@{meta['pk']:.2f}s  "
              f"prom {meta['prom']:.0f}  clip peak {meta['peak']} rms {meta['rms']:.0f}")
        print("3 s context (one cell = 50 ms), [ ] = kept 1 s window:")
        print(contour(e, hop_s, w0 - 1.0, w0 + 2.0, w0, w1))
        print("0.0s         0.5s         1.0s         1.5s         2.0s         2.5s")
        save_wav(clip_path + ".tmp", audio)
        os.replace(clip_path + ".tmp", clip_path)
        if player != "none":
            play(clip_path, player)
        r = input("[y]eep / [n]o / [s]ilence / [a]back 100ms / [d]fwd 100ms / "
                  "[p]lay again / [m]erge next / [q]uit? ").strip().lower()
        if r == "p":
            continue  # loop re-renders the contour and replays the clip
        if r == "a":
            w0 = max(0.0, w0 - NUDGE_S); w1 = w0 + win
            continue
        if r == "d":
            w1 = min(take_dur, w1 + NUDGE_S); w0 = w1 - win
            continue
        if r in ("y", "n", "s", "q", "m"):
            return r
        print("unknown key")

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes-dir", default="takes")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--take", default="*", help="glob on session dir")
    ap.add_argument("--enter-mult", type=float, default=2.2)
    ap.add_argument("--abs-min-enter", type=float, default=2500)
    ap.add_argument("--min-sep-ms", type=float, default=700,
                    help="min separation between utterance peaks")
    ap.add_argument("--min-prom", type=float, default=1500,
                    help="drop peaks standing out less than this (noise bumps)")
    ap.add_argument("--top-k", type=int, default=0,
                    help="per take, review only the K most prominent peaks (0=all)")
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
                floor = float(np.percentile(e, 10))
                enter = max(a.abs_min_enter, a.enter_mult * floor)
                peaks, _sm = find_peaks(e, hop_s, enter,
                                         min_sep_s=a.min_sep_ms / 1000)
                # Acceptance: most prominent first, so when two candidate
                # windows overlap (syllable split / dense speech) the
                # stronger peak wins instead of whichever came first.
                scored = [(pi, prominence(_sm, hop_s, pi)) for pi in peaks]
                n_bump = sum(1 for _, pr in scored if pr < a.min_prom)
                cands = []
                for pi, pr in scored:
                    if pr < a.min_prom:
                        continue
                    cen = pi * hop_s
                    w0 = min(max(cen - a.win_s / 2, 0), len(s) / SR - a.win_s)
                    cands.append((pr, cen, w0, w0 + a.win_s, pi))
                if a.top_k > 0:
                    cands = sorted(cands, reverse=True)[:a.top_k]
                accepted, n_overlap = [], 0
                for pr, cen, w0, w1, pi in sorted(cands, reverse=True):
                    if any(not (w1 <= u0 or w0 >= u1) for _, _, u0, u1, _
                           in accepted):
                        n_overlap += 1
                        continue
                    accepted.append((pr, cen, w0, w1, pi))
                accepted.sort(key=lambda c: c[2])  # review in time order
                print(f"\n== {tag}: floor={floor:.0f} "
                      f"thr={enter:.0f} peaks={len(peaks)} "
                      f"(+{n_bump} noise bumps auto-dropped, "
                      f"+{n_overlap} overlaps lost to stronger peaks)")
                used = []  # kept 1 s windows (avoid overlap + silence clash)
                skip, ci, take_dur = set(), 0, len(s) / SR
                for bi, (pr, cen, w0, w1, pi) in enumerate(accepted):
                    if bi in skip:
                        print(f"  candidate {bi + 1}: merged into previous, skipped")
                        continue
                    cur = [pr, cen, w0, w1, pi]
                    merged_flag = False
                    while True:
                        pr_, cen_, w0_, w1_, pi_ = cur
                        ci += 1
                        audio = s[int(w0_ * SR):int(w1_ * SR)]
                        name = (f"{speaker}_{room}_{dist}_"
                                f"{os.path.basename(wav)[:-4]}_c{ci:02d}.wav")
                        meta = {"src": tag, "t0": round(w0_, 2),
                                "dur": round(2 * (cen_ - w0_), 2),
                                "peak": int(np.abs(audio).max()),
                                "rms": float(np.sqrt((audio ** 2).mean())),
                                "w0": w0_, "w1": w1_, "audio": audio,
                                "pk": round(cen_, 2), "prom": round(pr_)}
                        if a.auto:
                            r = "y"
                            os.makedirs(f"{a.outdir}/wake", exist_ok=True)
                            save_wav(f"{a.outdir}/wake/{name}", audio)
                            break
                        os.makedirs(f"{a.outdir}/wake", exist_ok=True)
                        tmp = f"{a.outdir}/wake/{name}"
                        r = review(tmp, meta, e, hop_s, player, s, take_dur)
                        if r != "m":
                            break
                        if bi + 1 >= len(accepted) or (bi + 1) in skip:
                            print("  nothing to merge with (last candidate)")
                            if os.path.exists(tmp):
                                os.remove(tmp)
                            ci -= 1
                            continue
                        npr, ncen, _nw0, _nw1, _npi = accepted[bi + 1]
                        mid = (cen_ + ncen) / 2
                        w0_ = min(max(mid - a.win_s / 2, 0), take_dur - a.win_s)
                        skip.add(bi + 1)
                        merged_flag = True
                        print(f"  merged with candidate {bi + 2}; "
                              f"re-reviewing joint window")
                        if os.path.exists(tmp):
                            os.remove(tmp)
                        ci -= 1
                        cur = [max(pr_, npr), mid, w0_, w0_ + a.win_s, pi_]
                    if r == "q":
                        if not a.auto and os.path.exists(tmp):
                            os.remove(tmp)
                        print("quit."); man.close(); summary(n_wake, n_sil, n_drop); return
                    if r == "n":
                        if not a.auto and os.path.exists(tmp):
                            os.remove(tmp)
                        ci -= 1  # dropped clips leave no numbering gaps
                        n_drop += 1; continue
                    if r == "s":
                        if not a.auto:
                            os.makedirs(f"{a.outdir}/silence", exist_ok=True)
                            os.replace(tmp, f"{a.outdir}/silence/{name}")
                    label = "wake" if r == "y" else "silence"
                    tag3 = ("auto" if a.auto else
                            ("human-merge" if merged_flag else "human"))
                    writer.writerow([name, tag, meta["t0"], speaker, room,
                                     dist, label, meta["peak"],
                                     round(meta["rms"]), tag3])
                    man.flush()
                    used.append((meta["w0"], meta["w1"]))
                    if r == "y":
                        n_wake += 1
                    else:
                        n_sil += 1
                # silence: quietest 1 s grid windows clear of wake windows.
                # Two passes: strict 0.5 s margin first, then relaxed 0.1 s
                # (dense takes have no wide gaps; flagged auto-sil-loose).
                sils, seen = [], set()
                for margin, tag2 in ((0.5, "auto-sil"), (0.1, "auto-sil-loose")):
                    cands = []
                    t = 0.0
                    while t + a.win_s <= len(s) / SR:
                        if (round(t, 2) not in seen and
                            all(t + a.win_s <= u0 - margin or t >= u1 + margin
                                for u0, u1 in used)):
                            seg = e[int(t / hop_s):int((t + a.win_s) / hop_s)]
                            cands.append((seg.max() if len(seg) else 1e18, t))
                        t += 0.5
                    cands.sort()
                    for vmax, tt in cands:
                        if len(sils) >= a.silence_per_take:
                            break
                        sils.append((vmax, tt, tag2))
                        seen.add(round(tt, 2))
                    if len(sils) >= a.silence_per_take:
                        break
                sils.sort()
                os.makedirs(f"{a.outdir}/silence", exist_ok=True)
                for k, (_, t0_, tag2) in enumerate(sils[:a.silence_per_take]):
                    audio = s[int(t0_ * SR):int((t0_ + a.win_s) * SR)]
                    name = (f"{speaker}_{room}_{dist}_"
                            f"{os.path.basename(wav)[:-4]}_sil{k + 1}.wav")
                    save_wav(f"{a.outdir}/silence/{name}", audio)
                    writer.writerow([name, tag, round(t0_, 2), speaker, room,
                                     dist, "silence",
                                     int(np.abs(audio).max()),
                                     round(float(np.sqrt((audio ** 2).mean())), 1),
                                     tag2])
                    man.flush()
                    n_sil += 1
    finally:
        man.close()
    summary(n_wake, n_sil, n_drop)

def summary(n_wake, n_sil, n_drop):
    print(f"\ndone: wake={n_wake} silence={n_sil} dropped={n_drop}")

if __name__ == "__main__":
    main()
