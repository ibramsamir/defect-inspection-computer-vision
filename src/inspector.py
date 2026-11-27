"""Core inspection logic: candidate detection (OpenCV), feature extraction, annotation."""
import cv2
import numpy as np

K_SIGMA = 4.0          # robust z-score threshold for a pixel to be "abnormal"
MIN_AREA = 12          # ignore candidate regions smaller than this (pixels)
COLORS = {"scratch": (0, 165, 255), "crack": (0, 0, 255), "dent": (255, 0, 0), "none": (0, 200, 0)}
FEATURE_NAMES = ["area", "w", "h", "extent", "aspect", "solidity", "compactness",
                 "mean_absz", "max_absz", "polarity", "mean_dir"]


def find_candidates(gray: np.ndarray):
    """Flat-field correction -> robust z-score -> threshold -> connected components."""
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    flat = g - cv2.GaussianBlur(g, (0, 0), 25)                   # remove uneven lighting
    med = np.median(flat)
    mad = np.median(np.abs(flat - med)) * 1.4826
    z = (flat - med) / (mad + 1e-6)
    mask = (np.abs(z) > K_SIGMA).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        cands.append({"bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
                      "mask": (labels == i), "z": z})
    return cands


def features(c: dict) -> list:
    m, z = c["mask"], c["z"]
    ys, xs = np.where(m)
    area = float(m.sum())
    x0, y0, x1, y1 = c["bbox"]
    w, h = x1 - x0 + 1, y1 - y0 + 1
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    aspect = max(rw, rh) / (min(rw, rh) + 1.0)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(cnt)) + 1e-6
    perim = cv2.arcLength(cnt, True)
    vals = z[m]
    return [area, w, h, area / (w * h), aspect, min(area / hull_area, 1.5),
            perim ** 2 / (4 * np.pi * area), float(np.abs(vals).mean()),
            float(np.abs(vals).max()), float(np.mean(vals > 0)), float(vals.mean())]


def overlap_label(c: dict, gt: list, pad: int = 3) -> str:
    """Ground-truth type if >50 % of the candidate lies inside a labelled defect box, else 'none'."""
    ys, xs = np.where(c["mask"])
    for d in gt:
        x0, y0, x1, y1 = d["bbox"]
        inside = (xs >= x0 - pad) & (xs <= x1 + pad) & (ys >= y0 - pad) & (ys <= y1 + pad)
        if inside.mean() > 0.5:
            return d["type"]
    return "none"


def annotate(gray: np.ndarray, dets: list, verdict: str) -> np.ndarray:
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for d in dets:
        x0, y0, x1, y1 = d["bbox"]
        cv2.rectangle(out, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), COLORS[d["type"]], 1)
        cv2.putText(out, d["type"], (x0, max(y0 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS[d["type"]], 1)
    color = (0, 0, 255) if verdict == "DEFECTIVE" else (0, 160, 0)
    cv2.rectangle(out, (0, 0), (out.shape[1], 16), (255, 255, 255), -1)
    cv2.putText(out, verdict, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out
