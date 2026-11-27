"""Train the defect-type classifier on candidate regions and evaluate on unseen test images.

Pipeline: OpenCV candidate detection -> shape/contrast features -> Random Forest
(scratch / crack / dent / none). Candidates classified as 'none' are treated as false alarms.
Run from the project root:  python src/train.py
"""
import json
import sys
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

sys.path.insert(0, str(Path(__file__).parent))
from inspector import annotate, features, find_candidates, overlap_label  # noqa: E402

DATA, RESULTS, MODELS = Path("data"), Path("results"), Path("models")
CLASSES = ["none", "scratch", "crack", "dent"]
SEED = 42


def load_split(split: str):
    labels = json.loads((DATA / split / "labels.json").read_text())
    for name, gt in labels.items():
        gray = cv2.imread(str(DATA / split / "images" / name), cv2.IMREAD_GRAYSCALE)
        yield name, gray, gt


def box_hit(a, b, pad=3):
    return not (a[2] < b[0] - pad or b[2] < a[0] - pad or a[3] < b[1] - pad or b[3] < a[1] - pad)


def main() -> None:
    RESULTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True); (RESULTS / "examples").mkdir(exist_ok=True)

    X, y = [], []
    for _, gray, gt in load_split("train"):
        for c in find_candidates(gray):
            X.append(features(c)); y.append(overlap_label(c, gt))
    print(f"Training candidates: {len(X)}  ->", {k: y.count(k) for k in CLASSES})
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                 random_state=SEED, n_jobs=-1).fit(X, y)
    joblib.dump(clf, MODELS / "defect_classifier.joblib")

    # ---------------- evaluation on unseen test images ----------------
    img_true, img_pred, reg_true, reg_pred = [], [], [], []
    found, total_gt, saved = 0, 0, {"DEFECTIVE": 0, "OK": 0}
    for name, gray, gt in load_split("test"):
        dets = []
        for c in find_candidates(gray):
            kind = clf.predict([features(c)])[0]
            reg_true.append(overlap_label(c, gt)); reg_pred.append(kind)
            if kind != "none":
                dets.append({"type": kind, "bbox": c["bbox"]})
        verdict = "DEFECTIVE" if dets else "OK"
        img_true.append(int(bool(gt))); img_pred.append(int(bool(dets)))
        for d in gt:
            total_gt += 1
            found += any(box_hit(d["bbox"], p["bbox"]) for p in dets)
        if saved[verdict] < 4:
            saved[verdict] += 1
            cv2.imwrite(str(RESULTS / "examples" / f"{verdict.lower()}_{name}"), annotate(gray, dets, verdict))

    tp = sum(t and p for t, p in zip(img_true, img_pred)); fp = sum((not t) and p for t, p in zip(img_true, img_pred))
    fn = sum(t and (not p) for t, p in zip(img_true, img_pred)); tn = len(img_true) - tp - fp - fn
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    metrics = {
        "test_images": len(img_true), "image_accuracy": round((tp + tn) / len(img_true), 3),
        "image_precision": round(prec, 3), "image_recall": round(rec, 3),
        "image_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
        "false_alarms_on_good_parts": int(fp), "missed_defective_parts": int(fn),
        "defect_localisation_recall": round(found / max(total_gt, 1), 3),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(classification_report(reg_true, reg_pred, labels=CLASSES, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(reg_true, reg_pred, labels=CLASSES, xticks_rotation=30)
    plt.title("Candidate-region classification (test images)")
    plt.tight_layout(); plt.savefig(RESULTS / "confusion_matrix.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
