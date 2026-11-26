"""Run SAM 3 over many prepared clips with one model load.

Loading the video predictor takes about 30 s on the RTX 4070 laptop, longer than SAM
spends on a typical 5 s round, so the harvest annotates clips in one process. A clip
that already has a `done` line in the status file is skipped, so a run can be
stopped and resumed. A clip that fails is recorded and the batch moves on.

Usage:
    uv run python annotate_batch.py clips.txt --split web_candidates --score-thresh 0.15 \\
        --status status.tsv --masks-dir ../../data/processed/sam_masks
"""

import argparse
import time
import traceback
from pathlib import Path

import torch

from annotate_to_yolo import PROMPT, annotate_clip, build_predictor


def finished_clips(status: Path) -> set[str]:
    if not status.exists():
        return set()
    return {line.split("\t")[0] for line in status.read_text().splitlines() if "\tdone\t" in line}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", type=Path, help="text file, one clip id per line")
    ap.add_argument("--split", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--score-thresh", type=float, default=None)
    ap.add_argument("--status", type=Path, required=True, help="TSV: clip, done|failed, frames, seconds")
    ap.add_argument("--masks-dir", type=Path, default=None)
    ap.add_argument("--max-tracked", type=int, default=None, help="cap on masklets SAM 3 tracks at once")
    args = ap.parse_args()

    done = finished_clips(args.status)
    clips = [c for c in args.clips.read_text().split() if c not in done]
    print(f"{len(clips)} clips to annotate ({len(done)} already done)", flush=True)
    predictor = build_predictor()
    for position, clip in enumerate(clips, 1):
        started = time.monotonic()
        try:
            frames = annotate_clip(
                clip, args.split, args.prompt, args.score_thresh, predictor, args.masks_dir, args.max_tracked
            )
            outcome = f"{clip}\tdone\t{frames}"
        except Exception:
            traceback.print_exc()
            torch.cuda.empty_cache()
            outcome = f"{clip}\tfailed\t0"
        elapsed = time.monotonic() - started
        with args.status.open("a") as status:
            status.write(f"{outcome}\t{elapsed:.1f}\n")
        print(f"[{position}/{len(clips)}] {outcome} {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
