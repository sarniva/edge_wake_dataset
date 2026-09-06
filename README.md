# edge_wake_dataset — field recorder for KWS dataset collection

Records 30 s (or 5 s) takes of 16 kHz mono PCM from an INMP441 and dumps them
as hex over serial. Works over **native USB** (USB-Serial/JTAG): any serial
terminal — laptop or Android phone with OTG — can collect takes.

## Wiring (same as edge_wake)

| INMP441 | ESP32-S3 |
|---|---|
| VDD | 3V3 (NEVER 5V) |
| GND | GND |
| L/R | GND (LEFT slot) |
| SD | GPIO10 |
| WS | GPIO11 |
| SCK | GPIO12 |

Use the board's **native USB** port (Espressif `303a:1001`, shows as
`/dev/ttyACM0`), not the UART bridge.

## Laptop flow

```bash
source ~/.espressif/tools/activate_idf_v6.1.sh
idf.py set-target esp32s3 && idf.py build && idf.py -p /dev/ttyACM0 flash
python scripts/capture_takes.py --port /dev/ttyACM0 --takes 1 --outdir takes/
# more takes in one session: hold BOOT 1.5 s per take, --takes N
```

The board auto-records one 30 s take after every (re)boot; `--takes` counts
`TAKE_END` markers. Each take is CRC-checked (`CRC-OK` required). Timing per
take: ~35 s recording + ~40 s dump. **Hold BOOT only after the previous take
completes (`TAKE_END`)** — presses during recording/dumping are ignored
(the main loop is busy), which is also why an early BOOT hold seems "dead".

## Phone flow (Android + OTG)

1. Connect phone to the **native USB** port via OTG adapter (board powers up).
2. Open a serial-terminal app (e.g. Serial USB Terminal), open the ESP port
   (any baud — USB-CDC ignores it), start logging to a file.
3. Wait: boot auto-take records 30 s and dumps (~1 min total). For more takes,
   hold the BOOT button 1.5 s per take (watch for `TAKE_END n=N`).
4. Save the log, transfer to laptop:
   `python scripts/capture_takes.py --log session.txt --outdir takes/`
5. Takes with `CRC-MISMATCH` or malformed lines: re-take (hold BOOT again).

Tip: keep takes short of phone-log limits; one log file per speaker/room.

## Clip cutting (takes -> labeled 1 s clips)

```bash
python scripts/clip_cutter.py --takes-dir takes --outdir data
```

Adaptive per-take VAD finds utterance peaks (smoothed energy maxima >=700 ms
apart, so one "Jago Guru" = one peak even with its internal word gap, while
3 s-spaced utterances split even in noisy rooms where old burst-merging glued
takes together); each peak becomes a 1 s window. Up-to-3 auto silence clips
per take. You review wake candidates one keystroke each: `y` keep, `n` drop,
`s` file as silence, `p` hear the clip again, `a`/`d` slide the 1 s window ∓100 ms and re-listen
(repeatable — fixes silence clips catching a wake-word tail, or off-center
windows), `m` merge with the next candidate (for speakers pausing long
between "Jago" and "Guru" — joint window is re-reviewed, tagged
`human-merge`), `q` quit. Each candidate shows an energy contour, time
ruler, and plays back (`--no-play` to skip audio, `--auto` to accept all,
`--resume` to continue, `--take 'sp01*'` for a subset). Output:
`data/wake/`, `data/silence/`, `data/manifest.csv` (`quite`→`quiet`
normalized in clip names; source take kept per row). Tune with
`--enter-mult` (peak height), `--min-sep-ms` (utterance spacing),
`--min-prom` (noise-bump rejection) and `--top-k` (cap candidates per
take). Overlaps resolve to the stronger peak; silence falls back to a
looser margin (`auto-sil-loose`) in dense takes.

## Session protocol (per speaker)

- Distances: 30 cm, 1 m, 3 m — one take each, per room.
- Rooms: quiet + one noisy (fan / street / TV).
- Per 30 s take: 5–8× "JAGO GURU", 3–4 s apart, varied speed/pitch/loudness.
- Plus 2–3 background-only takes per room (no speech).
- Name takes: `sp03_quiet_1m_take02.wav` (speaker / room / distance / take).

## Known quirks

- **Host USB writes stall this link**: laptop/phone-to-board commands (`r`/`q`)
  may hang the session on some hosts (writes block forever host-side; even a
  single ignored byte kills subsequent reads). Triggering is therefore
  read-only by design: boot auto-take + BOOT button. Serial commands remain
  in firmware and work where the host stack behaves (e.g. UART port).
- **Task watchdog during dumps**: the ~2 MB dumps starve IDLE, so the IDLE
  watchdog checks are disabled for this tool firmware (the MAIN task stays
  subscribed + fed, so real hangs still trip it). If you ever see `task_wdt`
  text inside a dump again, that take is corrupt (CRC will flag it).
- Re-flash via native USB works with plain `idf.py flash`.
