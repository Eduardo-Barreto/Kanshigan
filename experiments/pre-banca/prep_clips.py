"""Cut and decimate raw videos into clips for SAM 3 annotation.

Reads spec from configs/clips.yaml (source, source_fps, start, end, subset).
For each clip, emits two outputs:
  data/processed/clips/<subset>/<clip_id>.mp4         native fps, full res (for inference + review)
  data/processed/sam_input/<clip_id>.mp4              10 fps + 480x270 (for SAM 3)

Usage:
    uv run python prep_clips.py
"""

import subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CLIPS_YAML = Path(__file__).parent / "configs" / "clips.yaml"
CLIPS_DIR = ROOT / "data" / "processed" / "clips"
SAM_DIR = ROOT / "data" / "processed" / "sam_input"

SAM_FPS = 10
SAM_W, SAM_H = 480, 270


def cut_native(src: Path, start: float, end: float, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start}", "-to", f"{end}",
        "-i", str(src),
        "-c", "copy", str(dst),
    ]
    subprocess.run(cmd, check=True)


def cut_sam_input(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"fps={SAM_FPS},scale={SAM_W}:{SAM_H}:flags=lanczos"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-an", str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    if not CLIPS_YAML.exists():
        raise SystemExit(f"missing {CLIPS_YAML}; copy configs/clips.example.yaml and fill in.")

    spec = yaml.safe_load(CLIPS_YAML.read_text())
    for clip in spec["clips"]:
        clip_id = clip["id"]
        subset = clip["subset"]
        src = Path(clip["source"])
        if not src.is_absolute():
            src = ROOT / src

        native_path = CLIPS_DIR / subset / f"{clip_id}.mp4"
        sam_path = SAM_DIR / f"{clip_id}.mp4"

        print(f"[{clip_id}] cut native {clip['start']}-{clip['end']} from {src.name}")
        cut_native(src, clip["start"], clip["end"], native_path)

        print(f"[{clip_id}] build SAM input ({SAM_FPS} fps, {SAM_W}x{SAM_H})")
        cut_sam_input(native_path, sam_path)


if __name__ == "__main__":
    main()
