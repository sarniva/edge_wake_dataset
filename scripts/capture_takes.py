#!/usr/bin/env python3
"""Convert dataset-recorder logs (one or MANY takes per file) to WAVs.

Live:   python3 scripts/capture_takes.py --port /dev/ttyACM0 [--takes 1]
          (DTR-resets the board; its boot auto-take is captured. For more
          takes in one session, hold BOOT 1.5 s per take. NO host writes:
          some USB-CDC hosts stall on OUT transfers, so triggering is
          deliberately read-only.)
Log:    python3 scripts/capture_takes.py --log phone_session.txt --outdir takes/
          -> takes/take_01.wav, take_02.wav, ... (+ CRC check per take)

Phone flow (Android OTG serial app): open port (any baud), type r, wait for
TAKE_END, repeat, SAVE the whole session log, transfer to laptop, run --log.
"""
import argparse, os, struct, sys, time, wave, zlib

def parse_takes(lines):
    takes, cur, in_dump = [], None, False
    bad = {"lines": 0, "chars": 0}
    for ln in lines:
        s = ln.strip()
        if s.startswith("TAKE_START"):
            cur = {"n": None, "samples": [], "crc": None}
            for tok in s.split():
                if tok.startswith("n="):
                    try: cur["n"] = int(tok[2:])
                    except ValueError: pass
            in_dump = False
        elif s.startswith("AUD_DUMP_START") and cur is not None:
            in_dump = True
        elif s.startswith("AUD_DUMP_END"):
            in_dump = False
        elif s.startswith("AUD_CRC") and cur is not None:
            try: cur["crc"] = int(s.split()[1], 16)
            except (ValueError, IndexError): pass
        elif s.startswith("TAKE_END") and cur is not None:
            takes.append(cur); cur, in_dump = None, False
        elif in_dump and cur is not None and "AUD_DATA:" in s:
            hx = s.split("AUD_DATA:", 1)[1].strip()
            if len(hx) % 4:
                # Truncated/inserted line: realign by dropping the ragged tail
                # (CRC at take end is the final arbiter; counts reported).
                bad["lines"] += 1
            for i in range(0, len(hx) - 3, 4):
                g = hx[i:i + 4]
                try:
                    w = int(g, 16)
                except ValueError:
                    # Substituted char: skip just this group (1 sample =
                    # 62 us glitch), keep alignment for the rest of the line.
                    bad["chars"] += 1
                    continue
                cur["samples"].append(w - 0x10000 if w >= 0x8000 else w)
    # tolerate missing TAKE_END at end of truncated log
    if cur is not None and cur["samples"]:
        takes.append(cur)
    return takes, bad

def write_take(samples, path, sr=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % len(samples), *samples))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--log", default=None)
    ap.add_argument("--outdir", default="takes")
    ap.add_argument("--takes", type=int, default=1,
                    help="how many TAKE_ENDs to collect in live mode "
                         "(extra takes via BOOT-hold)")
    ap.add_argument("--timeout", type=float, default=180.0)
    a = ap.parse_args()

    if a.log:
        with open(a.log, errors="ignore") as f:
            lines = f.read().splitlines()
    else:
        import serial
        ser = serial.Serial(a.port, a.baud, timeout=1)
        # Control-transfer reset only (proven reliable); then read-only.
        # The board auto-records one 30 s take after boot. More takes:
        # hold BOOT 1.5 s each while this runs.
        ser.dtr = False; time.sleep(0.3); ser.dtr = True
        t_reset = time.time()
        el = lambda: time.time() - t_reset
        print(f"reset {a.port}; collecting {a.takes} take(s).")
        print("Each take: ~35 s recording (SPEAK when told) + ~40 s dump. "
              "Hold BOOT 1.5 s only AFTER a take completes.")
        buf, linebuf, t0 = "", "", time.time()
        exp_lines, got_lines, last_pct = 0, 0, -1
        try:
            while time.time() - t0 < a.timeout:
                chunk = ser.read(8192).decode("ascii", "ignore")
                if not chunk:
                    continue
                buf += chunk
                linebuf += chunk
                while "\n" in linebuf:
                    line, linebuf = linebuf.split("\n", 1)
                    s = line.strip()
                    if s.startswith("TAKE_START"):
                        print(f"  [+{el():5.1f}s] take started: RECORDING "
                              f"-- speak now (5-8x JAGO GURU)", flush=True)
                    elif s.startswith("REC_END"):
                        print(f"  [+{el():5.1f}s] recorded, dumping "
                              f"(don't press BOOT yet) ...", flush=True)
                    elif s.startswith("AUD_DUMP_START"):
                        for tok in s.split():
                            if tok.startswith("samples="):
                                try:
                                    exp_lines = (int(tok[8:]) + 255) // 256
                                except ValueError:
                                    pass
                        got_lines, last_pct = 0, -1
                    elif "AUD_DATA:" in s:
                        got_lines += 1
                        if exp_lines:
                            pct = 100 * got_lines // exp_lines
                            if pct // 10 > last_pct // 10:
                                last_pct = pct
                                print(f"  [+{el():5.1f}s] ... dump {pct}% "
                                      f"({got_lines}/{exp_lines})", flush=True)
                    elif s.startswith("TAKE_END"):
                        print(f"  [+{el():5.1f}s] take complete. "
                              f"{'Hold BOOT 1.5 s for the next take.' if buf.count('TAKE_END') < a.takes else ''}",
                              flush=True)
                if buf.count("TAKE_END") >= a.takes:
                    break
        except KeyboardInterrupt:
            print("(stopped by user - parsing what arrived)")
        ser.close()
        lines = buf.splitlines()
        raw_path = os.path.join(a.outdir, "session.log")
        os.makedirs(a.outdir, exist_ok=True)
        with open(raw_path, "w", errors="ignore") as f:
            f.write(buf)
        print(f"raw log saved to {raw_path}")

    takes, bad = parse_takes(lines)
    print(f"found {len(takes)} take(s), malformed dump lines: "
          f"{bad['lines']}, bad chars: {bad['chars']}")
    if not takes:
        sys.exit("No takes found. Tip: save the RAW serial log (not a copy "
                 "of the terminal with wrapped lines).")
    os.makedirs(a.outdir, exist_ok=True)
    for i, t in enumerate(takes, 1):
        n = t["n"] if t["n"] is not None else i
        path = os.path.join(a.outdir, f"take_{n:02d}.wav")
        if not t["samples"]:
            print(f"take {n}: EMPTY, skipped"); continue
        write_take(t["samples"], path)
        crc = zlib.crc32(struct.pack("<%dh" % len(t["samples"]), *t["samples"])) & 0xFFFFFFFF
        tag = "CRC-OK" if (t["crc"] is None or t["crc"] == crc) else \
            f"CRC-MISMATCH(dev {t['crc']:08x} vs host {crc:08x})"
        print(f"take {n}: {len(t['samples'])/16000:.1f}s -> {path} [{tag}]")
    print("Next: name takes per convention spXX_<room>_<dist>_takeNN.wav, "
          "then run the clip tool (Phase 3a).")

if __name__ == "__main__":
    main()
