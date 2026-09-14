"""Label-free structural audit of pipeline output.

Some footage has no gold: annotating a round frame by frame costs hours, so the RC
(radio-controlled) category and other out-of-distribution clips are used today only
as qualitative evidence, with no number attached. The domain, however, constrains
what a correct output must look like without any annotation at all: exactly two
robots are on the dohyo for the whole round, a 3 kg robot's motion is bounded by
physics, and each robot's trajectory is continuous. Every departure from those
constraints is observable from the pipeline's own JSON.

That makes the indicators here a proxy, not ground truth, and a proxy is only worth
reporting if it is anchored. The anchor is the gold round, where MOTA and IDF1 were
measured against a human annotation: reading a clip's violation rate next to the
gold's says whether it sits in a regime whose true tracking quality is known. The
numbers below are therefore always reported alongside the gold row, never alone.

Indicators are computed inside the round window (round_start to round_end), because
frames before the release and after the ring-out legitimately show fewer than two
robots on the dohyo; counting those as failures would inflate every rate.

Usage:
    uv run python audit.py results/examples/rc/*.json --csv results/audit/rc.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# The pre-banca pipeline is the delivered artifact and stays frozen; this study reads
# its output rather than forking it, so the dohyo diameter keeps a single definition.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pre-banca"))

from schema import DOHYO_DIAMETER_CM  # noqa: E402

# A 3 kg sumo robot peaks around 3 to 4 m/s; 5 m/s is a generous ceiling, so anything
# above it is a tracking artifact rather than a robot.
MAX_PLAUSIBLE_SPEED_CMS = 500.0

# A quarter of the dohyo crossed between two consecutive frames. At 60 fps that is
# 23 m/s, which no robot does: it is an identity swap or a detection jumping to
# another object.
TELEPORT_STEP_CM = DOHYO_DIAMETER_CM / 4


@dataclass(frozen=True)
class Audit:
    """Structural indicators for one clip, all inside the round window."""

    clip: str
    fps: float
    n_frames: int
    round_start: int
    round_end: int
    round_frames: int
    both_present_rate: float
    one_present_rate: float
    none_present_rate: float
    cardinality_violation_rate: float
    longest_gap_frames: int
    longest_gap_ms: float
    max_step_cm: float
    n_teleports: int
    implausible_speed_rate: float
    max_speed_cms: float
    ring_out_detected: bool
    round_end_note: str | None


def _round_window(report: dict) -> tuple[int, int]:
    """Frames spanned by the round, falling back to the whole clip.

    A clip with no detected round_start never produced two moving robots, which is
    itself a failure; auditing the whole clip then reports it as such instead of
    silently skipping the clip.
    """
    by_kind = {e["kind"]: e for e in report.get("events", [])}
    last = report["n_frames"] - 1
    start = int(by_kind["round_start"]["frame"]) if "round_start" in by_kind else 0
    end = int(by_kind["round_end"]["frame"]) if "round_end" in by_kind else last
    if end <= start:
        start, end = 0, last
    return start, end


def _presence(report: dict, start: int, end: int) -> np.ndarray:
    """How many robots are tracked at each frame of the round window."""
    counts = np.zeros(end - start + 1, dtype=int)
    for traj in report["trajectories"]:
        frames = np.asarray(traj["frames"], dtype=int)
        inside = frames[(frames >= start) & (frames <= end)]
        counts[inside - start] += 1
    return counts


def _longest_run(mask: np.ndarray) -> int:
    """Longest run of True in mask."""
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def _steps_cm(traj: dict, start: int, end: int) -> np.ndarray:
    """Displacement between consecutive frames, in cm.

    Only true neighbours count: across a tracking gap the robot really did move
    further, so including that step would fabricate teleports out of dropouts.
    """
    frames = np.asarray(traj["frames"], dtype=int)
    keep = (frames >= start) & (frames <= end)
    frames = frames[keep]
    if frames.size < 2:
        return np.empty(0)
    x = np.asarray(traj["x_cm"], dtype=float)[keep]
    y = np.asarray(traj["y_cm"], dtype=float)[keep]
    adjacent = np.diff(frames) == 1
    return np.hypot(np.diff(x), np.diff(y))[adjacent]


def _speeds(traj: dict, start: int, end: int) -> np.ndarray:
    frames = np.asarray(traj["frames"], dtype=int)
    keep = (frames >= start) & (frames <= end)
    return np.asarray(traj["speed_cms"], dtype=float)[keep]


def audit_report(report: dict, clip: str) -> Audit:
    start, end = _round_window(report)
    counts = _presence(report, start, end)
    total = counts.size

    steps = np.concatenate([_steps_cm(t, start, end) for t in report["trajectories"]] or [np.empty(0)])
    speeds = np.concatenate([_speeds(t, start, end) for t in report["trajectories"]] or [np.empty(0)])
    fps = float(report["fps"])
    gap = _longest_run(counts < 2)
    notes = {e["kind"]: e.get("note") for e in report.get("events", [])}

    return Audit(
        clip=clip,
        fps=round(fps, 2),
        n_frames=int(report["n_frames"]),
        round_start=start,
        round_end=end,
        round_frames=total,
        both_present_rate=round(float(np.mean(counts == 2)), 4),
        one_present_rate=round(float(np.mean(counts == 1)), 4),
        none_present_rate=round(float(np.mean(counts == 0)), 4),
        cardinality_violation_rate=round(float(np.mean(counts != 2)), 4),
        longest_gap_frames=gap,
        longest_gap_ms=round(gap / fps * 1000, 1) if fps else 0.0,
        max_step_cm=round(float(steps.max()), 2) if steps.size else 0.0,
        n_teleports=int(np.sum(steps > TELEPORT_STEP_CM)) if steps.size else 0,
        implausible_speed_rate=round(float(np.mean(speeds > MAX_PLAUSIBLE_SPEED_CMS)), 4) if speeds.size else 0.0,
        max_speed_cms=round(float(speeds.max()), 1) if speeds.size else 0.0,
        ring_out_detected=any(e["kind"] == "ring_out" for e in report.get("events", [])),
        round_end_note=notes.get("round_end"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", type=Path, nargs="+", help="inference JSONs to audit")
    ap.add_argument("--csv", type=Path, help="also write the table here")
    args = ap.parse_args()

    audits = [audit_report(json.loads(p.read_text()), p.stem) for p in args.reports]

    header = f"{'clip':<22} {'round':>10} {'2 robos':>8} {'viol':>7} {'gap ms':>7} {'salto cm':>9} {'tp':>3} {'ringout':>8}"
    print(header)
    print("-" * len(header))
    for a in audits:
        print(
            f"{a.clip:<22} {a.round_frames:>10} {a.both_present_rate:>8.3f} "
            f"{a.cardinality_violation_rate:>7.3f} {a.longest_gap_ms:>7.0f} "
            f"{a.max_step_cm:>9.1f} {a.n_teleports:>3} {str(a.ring_out_detected):>8}"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(audits[0])))
            writer.writeheader()
            writer.writerows(asdict(a) for a in audits)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
