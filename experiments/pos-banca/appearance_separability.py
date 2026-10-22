"""Can appearance tell the two robots apart? A direct test of premise C2.

The tracker comparison found that ReID trackers (DeepOCSORT, BoT-SORT) do not beat
motion-only ones, and the paper explains it with C2: the two robots look alike, so
appearance carries little identity signal. That explanation was inferred, not
measured. This measures it.

Every identity-labelled box of the BR gold (identity.txt, robots A and B) is cropped
from the clip the pipeline sees, embedded, and the cosine similarity of every pair is
split into intra-robot (same id, different frames) and inter-robot (different ids).
If appearance identified the robots, intra pairs would score clearly above inter
pairs. Reported per extractor:

  * mean and std of each distribution and the overlap coefficient of their
    histograms (1 = identical distributions, 0 = disjoint);
  * AUC of separating intra from inter by similarity (0.5 = chance, 1 = perfect);
  * the same AUC restricted to intra pairs at least MIN_GAP_FRAMES apart, because
    near-adjacent frames of the same robot are similar for reasons (same pose, same
    blur) that do not help after an occlusion;
  * leave-one-out nearest-neighbour identity accuracy, excluding neighbours closer
    than MIN_GAP_FRAMES in time.

Extractors: the tracker's own ReID (boxmot ReID backend, osnet_x0_25_msmt17, the
weights DeepOCSORT/BoT-SORT used) and a generic one (torchvision ResNet-50,
ImageNet weights, global-average-pooled), to separate "the robots look alike" from
"this ReID model was trained on people". A colour histogram (HSV, 8x8x8) is a third,
non-learned reference.

Usage:
    uv run --project ../pre-banca python appearance_separability.py
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "pre-banca"))

MIN_SIDE_PX = 12
MIN_GAP_FRAMES = 30  # 0.5 s at 60 fps
HIST_BINS = 50
OUT = ROOT / "results" / "E6_final_measurements" / "appearance"


def pair_similarities(embeddings: np.ndarray, ids: np.ndarray, frames: np.ndarray) -> dict[str, np.ndarray]:
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = normed @ normed.T
    intra, intra_far, inter = [], [], []
    for i, j in combinations(range(len(ids)), 2):
        if ids[i] == ids[j]:
            intra.append(sims[i, j])
            if abs(frames[i] - frames[j]) >= MIN_GAP_FRAMES:
                intra_far.append(sims[i, j])
        else:
            inter.append(sims[i, j])
    return {"intra": np.array(intra), "intra_far": np.array(intra_far), "inter": np.array(inter), "matrix": sims}


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(positive > negative) with ties counted half (Mann-Whitney U / n1 n2)."""
    greater = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * ties) / (positive.size * negative.size))


def overlap_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    ha, _ = np.histogram(a, bins=HIST_BINS, range=(lo, hi))
    hb, _ = np.histogram(b, bins=HIST_BINS, range=(lo, hi))
    return float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum())


def nearest_neighbour_accuracy(sims: np.ndarray, ids: np.ndarray, frames: np.ndarray) -> float:
    hits = 0
    for i in range(len(ids)):
        allowed = np.abs(frames - frames[i]) >= MIN_GAP_FRAMES
        if not allowed.any():
            continue
        candidates = np.where(allowed, sims[i], -np.inf)
        hits += ids[int(np.argmax(candidates))] == ids[i]
    return hits / len(ids)


def summarize(embeddings: np.ndarray, ids: np.ndarray, frames: np.ndarray) -> dict:
    s = pair_similarities(embeddings, ids, frames)
    stat = lambda x: {"n": int(x.size), "mean": round(float(x.mean()), 4), "std": round(float(x.std()), 4)}  # noqa: E731
    return {
        "intra": stat(s["intra"]),
        "intra_far": stat(s["intra_far"]),
        "inter": stat(s["inter"]),
        "auc_intra_vs_inter": round(auc(s["intra"], s["inter"]), 4),
        "auc_intra_far_vs_inter": round(auc(s["intra_far"], s["inter"]), 4),
        "overlap_intra_vs_inter": round(overlap_coefficient(s["intra"], s["inter"]), 4),
        "overlap_intra_far_vs_inter": round(overlap_coefficient(s["intra_far"], s["inter"]), 4),
        "nn_identity_accuracy_gap_ge_min": round(nearest_neighbour_accuracy(s["matrix"], ids, frames), 4),
    }


def load_crops(video: Path, gold: Path):
    from evaluate import _load_mot
    from final_tracking import clip_gold_to_frame
    from infer import load_frames

    frames, _ = load_frames(video)
    height, width = frames[0].shape[:2]
    rows = clip_gold_to_frame(_load_mot(gold), width, height)
    boxes, ids, frame_idx, images = [], [], [], []
    for frame, by_id in sorted(rows.items()):
        for oid, (x, y, w, h) in by_id.items():
            if min(w, h) < MIN_SIDE_PX:
                continue
            boxes.append([x, y, x + w, y + h])
            ids.append(oid)
            frame_idx.append(frame)
            images.append(frames[frame])
    return np.array(boxes), np.array(ids), np.array(frame_idx), images


def osnet_embeddings(boxes, images, device: str) -> np.ndarray:
    from boxmot.reid.core import ReID
    from boxmot.utils import WEIGHTS

    model = ReID(weights=Path(WEIGHTS) / "osnet_x0_25_msmt17.pt", device=device, half=False).model
    return np.vstack([model.get_features(np.array([b]), img) for b, img in zip(boxes, images)])


def resnet_embeddings(boxes, images, device: str) -> np.ndarray:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).eval().to(device)
    model.fc = torch.nn.Identity()
    preprocess = weights.transforms()
    out = []
    with torch.no_grad():
        for (x0, y0, x1, y1), img in zip(boxes.astype(int), images):
            crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            tensor = preprocess(torch.from_numpy(crop).permute(2, 0, 1)).unsqueeze(0).to(device)
            out.append(model(tensor).cpu().numpy()[0])
    return np.vstack(out)


def hsv_histograms(boxes, images) -> np.ndarray:
    out = []
    for (x0, y0, x1, y1), img in zip(boxes.astype(int), images):
        hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
        out.append(hist / hist.sum())
    return np.vstack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "data/processed/clips/br/gold_zb01.mp4")
    ap.add_argument("--gold", type=Path, default=ROOT / "data/annotations/gold_crop/identity.txt")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    boxes, ids, frames, images = load_crops(args.video, args.gold)
    report = {
        "gold": str(args.gold.relative_to(ROOT)),
        "video": str(args.video.relative_to(ROOT)),
        "n_crops": int(len(ids)),
        "crops_per_id": {int(k): int((ids == k).sum()) for k in np.unique(ids)},
        "min_side_px": MIN_SIDE_PX,
        "min_gap_frames": MIN_GAP_FRAMES,
        "extractors": {},
    }
    for name, embed in (
        ("osnet_x0_25_msmt17 (boxmot ReID)", lambda: osnet_embeddings(boxes, images, device)),
        ("resnet50_imagenet (torchvision IMAGENET1K_V2)", lambda: resnet_embeddings(boxes, images, device)),
        ("hsv_histogram_8x8x8", lambda: hsv_histograms(boxes, images)),
    ):
        report["extractors"][name] = summarize(embed(), ids, frames)
        print(name, json.dumps(report["extractors"][name]), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "appearance_separability.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
