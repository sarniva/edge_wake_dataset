#!/usr/bin/env python3
"""A4 (CPU smoke gate): integrity + numpy centroid baseline on a package.

No TF/torch needed. Loads edgewake-v1 (or raw features dirs), checks shapes /
finite values / split disjointness, then nearest-class-mean accuracy.
A sane pipeline scores far above chance; anything near chance means broken
labels or features - fix BEFORE spending GPU hours.

  python3 scripts/smoke_features.py --pkg edgewake-v1
"""
import argparse, csv, os
import numpy as np

def load_split(pkg, name):
    xs, ys, spks = [], [], []
    with open(os.path.join(pkg, f"{name}.csv")) as f:
        for row in csv.DictReader(f):
            xs.append(np.load(os.path.join(pkg, row["npy"])).astype(np.float64))
            ys.append(row["label"])
            spks.append(row["speaker"])
    return np.array(xs), np.array(ys), np.array(spks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="edgewake-v1")
    a = ap.parse_args()
    tr_x, tr_y, tr_s = load_split(a.pkg, "train")
    va_x, va_y, va_s = load_split(a.pkg, "val")
    assert tr_x.ndim == 3 and tr_x.shape[1:] == (61, 40), tr_x.shape
    assert np.isfinite(tr_x).all() and np.isfinite(va_x).all(), "NaN/Inf!"
    overlap = (set(tr_s) - {"-"}) & (set(va_s) - {"-"})
    print(f"train {tr_x.shape} val {va_x.shape} | "
          f"train speakers leaking into val: {sorted(overlap) or 'none'}")
    import collections
    print("train classes:", dict(collections.Counter(tr_y)),
          "| val classes:", dict(collections.Counter(va_y)))
    flat_tr, flat_va = tr_x.reshape(len(tr_x), -1), va_x.reshape(len(va_x), -1)
    # per-feature z-norm stats from train (also what the notebook will use)
    mu, sd = flat_tr.mean(0), flat_tr.std(0) + 1e-6
    zt, zv = (flat_tr - mu) / sd, (flat_va - mu) / sd
    classes = sorted(set(tr_y))
    means = {c: zt[tr_y == c].mean(0) for c in classes}
    pred = [min(classes, key=lambda c: ((zv[i] - means[c]) ** 2).sum())
            for i in range(len(zv))]
    pred = np.array(pred)
    acc = (pred == va_y).mean()
    print(f"centroid baseline val accuracy: {acc:.3f} "
          f"(chance ~{1 / len(classes):.3f}, classes={classes})")
    for c in classes:
        m = va_y == c
        print(f"  {c}: recall {(pred[m] == c).mean():.3f} (n={m.sum()})")
    print("GATE:", "PASS - signal present, safe for GPU training"
          if acc > 0.60 else "WEAK - investigate labels/features first")

if __name__ == "__main__":
    main()
