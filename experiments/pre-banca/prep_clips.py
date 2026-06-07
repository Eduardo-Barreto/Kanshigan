"""Cut raw videos into clips, cropped to the dohyo, for SAM 3 annotation.

Reads spec from configs/clips.yaml (source, source_fps, start, end, subset).
Each clip is cut to its round window, then cropped to the dohyo (the robots are a
small fraction of the full frame; cropping zooms them in and drops the background).
For each clip, emits:
  data/processed/clips/<subset>/<clip_id>.mp4   cropped, native fps (inference + frame dump)
  data/processed/sam_input/<clip_id>.mp4        cropped, 10 fps, ~480 wide (for SAM 3)
  data/processed/clips/<subset>/<clip_id>.roi.json   crop rectangle in native pixels

Usage:
    uv run python prep_clips.py
"""

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import yaml

from crop import clip_roi, crop_frame

ROOT = Path(__file__).resolve().parents[2]
CLIPS_YAML = Path(__file__).parent / "configs" / "clips.yaml"
CLIPS_DIR = ROOT / "data" / "processed" / "clips"
SAM_DIR = ROOT / "data" / "processed" / "sam_input"

SAM_FPS = 10
SAM_WIDTH = 480


def _cut(src: Path, start: float, end: float, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-to", f"{end}", "-i", str(src), "-c", "copy", str(dst)],
        check=True,
    )


def _read_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def _write(frames, fps, size, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    if not CLIPS_YAML.exists():
        raise SystemExit(f"missing {CLIPS_YAML}; copy configs/clips.example.yaml and fill in.")

    spec = yaml.safe_load(CLIPS_YAML.read_text())
    for clip in spec["clips"]:
        clip_id, subset = clip["id"], clip["subset"]
        src = Path(clip["source"])
        if not src.is_absolute():
            src = ROOT / src

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "cut.mp4"
            _cut(src, clip["start"], clip["end"], raw)
            frames, fps = _read_frames(raw)

        sample = frames[:: max(1, len(frames) // 12)]
        roi = clip_roi(sample)
        x0, y0, w, h = roi
        cropped = [crop_frame(f, roi) for f in frames]
        print(f"[{clip_id}] {clip['start']}-{clip['end']} from {src.name}; dohyo crop {w}x{h}")

        native_path = CLIPS_DIR / subset / f"{clip_id}.mp4"
        _write(cropped, fps, (w, h), native_path)
        (native_path.with_suffix(".roi.json")).write_text(json.dumps({"x0": x0, "y0": y0, "w": w, "h": h}))

        sam_h = int(round(SAM_WIDTH * h / w))
        sam_h -= sam_h % 2
        step = max(1, round(fps / SAM_FPS))
        sam_frames = [cv2.resize(f, (SAM_WIDTH, sam_h)) for f in cropped[::step]]
        _write(sam_frames, SAM_FPS, (SAM_WIDTH, sam_h), SAM_DIR / f"{clip_id}.mp4")


if __name__ == "__main__":
    main()
