"""Run SAM 3 over many prepared clips with one model load.

Loading the video predictor takes about 30 s on the RTX 4070 laptop, longer than SAM
spends on a typical 5 s round, so the harvest annotates clips in one process. A clip
that already has a `done` line in the status file is skipped, so a run can be
stopped and resumed. A clip that fails is recorded and the batch moves on.

With `--fallback-thresh`, a clip whose labelled frames rarely hold both robots is run
again at that lower threshold and the run with more two-robot frames is kept. The
harvest pilot showed why: the default thresholds lost every robot in the small, dark
Robotex 2013 overhead footage, while 0.15 on handheld side views tracked hands and
mat texture until 8 GB of VRAM ran out.

Usage:
    uv run python annotate_batch.py clips.txt --split web_candidates --status status.tsv \\
        [--score-thresh 0.15] [--fallback-thresh 0.15] [--masks-dir DIR] [--max-tracked 8]
"""

import argparse
import shutil
import time
import traceback
from pathlib import Path

import torch

from annotate_to_yolo import ANN_DIR, PROMPT, _set_detection_threshold, annotate_clip, build_predictor

FALLBACK_BELOW_PAIR_SHARE = 0.3


def finished_clips(status: Path) -> set[str]:
    if not status.exists():
        return set()
    return {line.split("\t")[0] for line in status.read_text().splitlines() if "\tdone\t" in line}


def clip_labels(split: str, clip: str) -> list[Path]:
    return sorted((ANN_DIR / split / "labels").glob(f"{clip}_[0-9]*.txt"))


def pair_frames(split: str, clip: str) -> int:
    return sum(1 for label in clip_labels(split, clip) if len(label.read_text().split()) == 10)


def remove_clip(split: str, clip: str) -> None:
    for label in clip_labels(split, clip):
        label.unlink()
        (ANN_DIR / split / "images" / f"{label.stem}.jpg").unlink(missing_ok=True)


def move_clip(source_split: str, target_split: str, clip: str) -> None:
    for kind in ("labels", "images"):
        (ANN_DIR / target_split / kind).mkdir(parents=True, exist_ok=True)
    for label in clip_labels(source_split, clip):
        shutil.move(label, ANN_DIR / target_split / "labels" / label.name)
        image = ANN_DIR / source_split / "images" / f"{label.stem}.jpg"
        shutil.move(image, ANN_DIR / target_split / "images" / image.name)


def detection_thresholds(predictor) -> tuple[float, float]:
    return predictor.model.new_det_thresh, predictor.model.score_threshold_detection


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", type=Path, help="text file, one clip id per line")
    ap.add_argument("--split", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--score-thresh", type=float, default=None)
    ap.add_argument("--fallback-thresh", type=float, default=None)
    ap.add_argument("--status", type=Path, required=True, help="TSV: clip, done|failed, frames, seconds, threshold")
    ap.add_argument("--masks-dir", type=Path, default=None)
    ap.add_argument("--max-tracked", type=int, default=None, help="cap on masklets SAM 3 tracks at once")
    args = ap.parse_args()

    done = finished_clips(args.status)
    clips = [c for c in args.clips.read_text().split() if c not in done]
    print(f"{len(clips)} clips to annotate ({len(done)} already done)", flush=True)
    predictor = build_predictor()
    if args.score_thresh is not None:
        _set_detection_threshold(predictor, args.score_thresh)
    primary = detection_thresholds(predictor)
    fallback_split = f"{args.split}__fallback"
    fallback_masks = args.masks_dir / "fallback" if args.masks_dir else None

    for position, clip in enumerate(clips, 1):
        started = time.monotonic()
        used = "primary"
        try:
            frames = annotate_clip(clip, args.split, args.prompt, None, predictor, args.masks_dir, args.max_tracked)
            pairs = pair_frames(args.split, clip)
            if args.fallback_thresh is not None and pairs < FALLBACK_BELOW_PAIR_SHARE * max(frames, 1):
                _set_detection_threshold(predictor, args.fallback_thresh)
                try:
                    low = annotate_clip(clip, fallback_split, args.prompt, None, predictor, fallback_masks, args.max_tracked)
                finally:
                    predictor.model.new_det_thresh, predictor.model.score_threshold_detection = primary
                if pair_frames(fallback_split, clip) > pairs:
                    remove_clip(args.split, clip)
                    move_clip(fallback_split, args.split, clip)
                    if fallback_masks is not None and (fallback_masks / f"{clip}.npz").exists():
                        shutil.move(fallback_masks / f"{clip}.npz", args.masks_dir / f"{clip}.npz")
                    frames, used = low, f"fallback {args.fallback_thresh}"
                else:
                    remove_clip(fallback_split, clip)
                    if fallback_masks is not None:
                        (fallback_masks / f"{clip}.npz").unlink(missing_ok=True)
            outcome = f"{clip}\tdone\t{frames}"
        except Exception:
            traceback.print_exc()
            torch.cuda.empty_cache()
            outcome = f"{clip}\tfailed\t0"
        elapsed = time.monotonic() - started
        with args.status.open("a") as status:
            status.write(f"{outcome}\t{elapsed:.1f}\t{used}\n")
        print(f"[{position}/{len(clips)}] {outcome} {elapsed:.1f}s {used}", flush=True)


if __name__ == "__main__":
    main()
