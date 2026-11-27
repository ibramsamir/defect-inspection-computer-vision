"""Inspect one image or a folder of images and save annotated results.

Usage:  python src/inspect_image.py data/test/images/img_0001.png
        python src/inspect_image.py data/test/images --out results/inspected
"""
import argparse
import sys
from pathlib import Path

import cv2
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from inspector import annotate, features, find_candidates  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path"); ap.add_argument("--out", default="results/inspected")
    a = ap.parse_args()
    clf = joblib.load("models/defect_classifier.joblib")
    p = Path(a.path)
    files = sorted(p.glob("*.png")) if p.is_dir() else [p]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for f in files:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        dets = []
        for c in find_candidates(gray):
            kind = clf.predict([features(c)])[0]
            if kind != "none":
                dets.append({"type": kind, "bbox": c["bbox"]})
        verdict = "DEFECTIVE" if dets else "OK"
        cv2.imwrite(str(Path(a.out) / f.name), annotate(gray, dets, verdict))
        print(f"{f.name}: {verdict}" + (f" ({', '.join(d['type'] for d in dets)})" if dets else ""))


if __name__ == "__main__":
    main()
