"""Generate a synthetic surface-inspection dataset (brushed-metal parts with defects).

Each 256x256 grayscale image shows a metal surface with fine texture and uneven lighting.
Defective images contain 1-3 defects of type: scratch, crack or dent.

NOTE: This is simulated data created for demonstration. It is NOT real production imagery.
Usage:  python src/generate_dataset.py
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SIZE = 256
DEFECT_TYPES = ("scratch", "crack", "dent")


def make_surface(rng: np.random.Generator) -> np.ndarray:
    streaks = cv2.GaussianBlur(rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32), (0, 0), sigmaX=10, sigmaY=0.8)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32), (0, 0), 0.8)
    img = 128 + 5 * streaks / streaks.std() + 3 * fine / fine.std()
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    a = rng.uniform(0, 2 * np.pi)
    img += rng.uniform(10, 28) * ((xx / SIZE - 0.5) * np.cos(a) + (yy / SIZE - 0.5) * np.sin(a))   # lighting
    return img.astype(np.float32)


def _bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0.15)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def add_scratch(img, rng):
    mask = np.zeros((SIZE, SIZE), np.float32)
    p0 = rng.uniform(30, SIZE - 30, 2)
    ang, length = rng.uniform(0, np.pi), rng.uniform(45, 110)
    pts = []
    for t in np.linspace(0, 1, 5):
        bend = rng.normal(0, 3)
        pts.append([p0[0] + np.cos(ang) * length * t - np.sin(ang) * bend,
                    p0[1] + np.sin(ang) * length * t + np.cos(ang) * bend])
    cv2.polylines(mask, [np.array(pts, np.int32)], False, 1.0, int(rng.integers(1, 3)))
    mask = cv2.GaussianBlur(mask, (0, 0), 0.7)
    img += mask * rng.choice([-1, 1]) * rng.uniform(40, 70)
    return _bbox(mask)


def add_crack(img, rng):
    mask = np.zeros((SIZE, SIZE), np.float32)
    p = rng.uniform(40, SIZE - 40, 2)
    ang = rng.uniform(0, 2 * np.pi)
    for _ in range(int(rng.integers(10, 18))):            # jagged random walk
        ang += rng.normal(0, 0.7)
        q = p + rng.uniform(4, 8) * np.array([np.cos(ang), np.sin(ang)])
        cv2.line(mask, tuple(p.astype(int)), tuple(q.astype(int)), 1.0, 1)
        if rng.random() < 0.25:                           # small branch
            b = q + rng.uniform(4, 9) * np.array([np.cos(ang + 1.0), np.sin(ang + 1.0)])
            cv2.line(mask, tuple(q.astype(int)), tuple(b.astype(int)), 1.0, 1)
        p = q
    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), 0.6) * 1.6, 0, 1)
    img -= mask * rng.uniform(55, 90)
    return _bbox(mask)


def add_dent(img, rng):
    c = rng.uniform(30, SIZE - 30, 2)
    r = rng.uniform(6, 15)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    ex = rng.uniform(0.7, 1.3)
    blob = np.exp(-(((xx - c[0]) * ex) ** 2 + ((yy - c[1]) / ex) ** 2) / (2 * (r / 1.8) ** 2))
    img += blob * rng.choice([-1, 1]) * rng.uniform(28, 48)
    return _bbox(blob)


ADDERS = {"scratch": add_scratch, "crack": add_crack, "dent": add_dent}


def generate_split(out_dir: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    labels = {}
    for i in range(n):
        img = make_surface(rng)
        defects = []
        if rng.random() < 0.6:                            # 60 % defective parts
            for _ in range(int(rng.integers(1, 4))):
                kind = str(rng.choice(DEFECT_TYPES))
                defects.append({"type": kind, "bbox": ADDERS[kind](img, rng)})
        img += rng.normal(0, 1.2, img.shape)              # sensor noise
        name = f"img_{i:04d}.png"
        cv2.imwrite(str(out_dir / "images" / name), np.clip(img, 0, 255).astype(np.uint8))
        labels[name] = defects
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"{out_dir}: {n} images, {sum(bool(v) for v in labels.values())} defective")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    generate_split(Path(a.out) / "train", a.train, seed=1)
    generate_split(Path(a.out) / "test", a.test, seed=2)
