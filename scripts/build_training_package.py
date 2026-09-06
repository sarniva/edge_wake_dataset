#!/usr/bin/env python3
"""A3: assemble versioned training package from features.

Speaker-disjoint splits (explicit speaker lists - no silent magic), negatives
stratified randomly with a fixed seed. Output is self-contained for Kaggle.

  python3 scripts/build_training_package.py \
      --our features/our --neg features/neg --manifest data/manifest.csv \
      --train sp01,sp02,sp03,sp04,sp05,sp06 --val sp07,sp08 --test sp09 \
      --out edgewake-v1

Needs >=2 distinct OUR speakers across splits used; empty splits are fatal.
Writes edgewake-v1/{train,val,test}.csv + features/*.npy + datasheet.md
"""
import argparse, csv, os, shutil, sys
import numpy as np

def read_idx(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our", default="features/our")
    ap.add_argument("--neg", default="features/neg")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--train", default="")
    ap.add_argument("--val", default="")
    ap.add_argument("--test", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="edgewake-v1")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    tr = set(filter(None, a.train.split(",")))
    va = set(filter(None, a.val.split(",")))
    te = set(filter(None, a.test.split(",")))
    if not (tr and va and te):
        sys.exit("give explicit --train/--val/--test speaker lists")
    if tr & va or tr & te or va & te:
        sys.exit("speaker overlap between splits!")

    clip_speaker = {}
    if os.path.exists(a.manifest):
        with open(a.manifest) as f:
            for row in csv.DictReader(f):
                clip_speaker[row["clip"]] = row["speaker"]

    ours = read_idx(os.path.join(a.our, "features.csv"))
    negs = read_idx(os.path.join(a.neg, "features.csv"))
    # our clip key == clip filename; find its speaker via manifest
    rows = {"train": [], "val": [], "test": []}
    missing_spk = 0
    for r in ours:
        spk = None
        for clip, s in clip_speaker.items():
            if r["key"] == clip:
                spk = s
                break
        if spk is None:
            missing_spk += 1
            continue
        split = ("train" if spk in tr else "val" if spk in va
                 else "test" if spk in te else None)
        if split is None:
            continue
        rows[split].append((r["npy"], r["label"], spk, r["key"], a.our))
    # negatives: stratified random split matching our speaker proportions
    n = {k: len(v) for k, v in rows.items()}
    tot = sum(n.values()) or 1
    frac = {k: n[k] / tot for k in rows}
    order = np.arange(len(negs))
    rng.shuffle(order)
    cuts = [int(len(negs) * frac["train"]), int(len(negs) * frac["val"])]
    parts = {"train": order[:cuts[0]], "val": order[cuts[0]:cuts[0] + cuts[1]],
             "test": order[cuts[0] + cuts[1]:]}
    for k, idx in parts.items():
        for i in idx:
            r = negs[int(i)]
            rows[k].append((r["npy"], r["label"], "-", r["key"], a.neg))

    for k, v in rows.items():
        if not v:
            sys.exit(f"split '{k}' is EMPTY - fix speaker lists "
                     f"(train={sorted(tr)} val={sorted(va)} test={sorted(te)})")
    os.makedirs(os.path.join(a.out, "features"), exist_ok=True)
    stats = {}
    for k, v in rows.items():
        with open(os.path.join(a.out, f"{k}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["npy", "label", "speaker", "src"])
            for npy, label, spk, key, srcdir in v:
                dst = f"{k}_{os.path.basename(npy)}"
                shutil.copyfile(os.path.join(srcdir, npy),
                                os.path.join(a.out, "features", dst))
                w.writerow([f"features/{dst}", label, spk, key])
        import collections
        stats[k] = collections.Counter(x[1] for x in v)
    with open(os.path.join(a.out, "datasheet.md"), "w") as f:
        f.write("# edgewake-v1 datasheet\n\n"
                "Wake word: JAGO GURU. Features: 61x40 log-mel, 16 kHz, "
                "512-frame/256-hop, 40 mels 20-4000 Hz, pre-emph 0.97 "
                "(bit-exact with ESP32-S3 firmware, see frontend_check).\n\n"
                f"Speaker splits: train={sorted(tr)} val={sorted(va)} "
                f"test={sorted(te)} (disjoint).\n\n"
                "Class counts per split:\n")
        for k in ("train", "val", "test"):
            f.write(f"- {k}: {dict(stats[k])}\n")
        f.write("\nLicenses: own clips (project), "
                "Speech Commands v2 (CC-BY-4.0), MUSAN (CC-BY-4.0).\n")
        if missing_spk:
            f.write(f"\nWARNING: {missing_spk} feature rows had no manifest "
                    f"speaker and were dropped.\n")
    print("splits:", {k: dict(v) for k, v in stats.items()})
    print(f"package ready at {a.out}/ (+ datasheet.md)")

if __name__ == "__main__":
    main()
