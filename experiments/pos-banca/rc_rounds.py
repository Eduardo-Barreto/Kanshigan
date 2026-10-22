"""Run the full pipeline on every RC round, for the label-free structural audit.

Diary 20 left this pending: the RC corpus (ThundeRatz phone footage, 25 files, 3 exact
duplicates) was segmented into rounds but only one round went through infer.py,
because the trained weights were on another machine. This closes it:

  1. drop byte-identical duplicates (md5);
  2. segment each video with segment_rounds.round_windows;
  3. cut each window with ffmpeg; 4K sources are scaled to 1920 px wide (H.264,
     CRF 18), because infer.py holds every decoded frame in RAM and a 4K round of
     ~15 s would need over 20 GB. rc1591 in results/examples was run at 1080p too;
  4. infer.run (YOLOv8s + OC-SORT, the reported pipeline), JSON and MOT kept.

The audit itself is audit.py over the JSONs this writes.

Usage:
    uv run --project ../pre-banca python rc_rounds.py --videos "/path/to/3kg RC"
    uv run python audit.py <gold json> ../../results/E6_final_measurements/rc/*.json --csv ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "pre-banca"))

WEIGHTS = ROOT / "results/training/yolov8s_kanshigan/weights/best.pt"
OUT = ROOT / "results" / "E6_final_measurements" / "rc"
WORK = Path("/tmp/kanshigan_e6/rc")
MAX_WIDTH = 1920


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:stream_side_data=rotation", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"][0]


def cut(src: Path, start: float, end: float, dst: Path, width: int) -> None:
    scale = ["-vf", f"scale={MAX_WIDTH}:-2"] if width > MAX_WIDTH else []
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-to", f"{end}", "-i", str(src), *scale,
                    "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast", str(dst)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from events import EventConfig
    from infer import run
    from segment_rounds import round_windows

    args.out.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    manifest = {"weights": str(WEIGHTS.relative_to(ROOT)), "tracker": "ocsort", "duplicates": {}, "rounds": []}
    for video in sorted(args.videos.glob("*.MOV")):
        digest = md5(video)
        if digest in seen:
            manifest["duplicates"][video.name] = seen[digest]
            continue
        seen[digest] = video.name
        info = probe(video)
        windows = round_windows(video)
        if not windows:
            manifest["rounds"].append({"video": video.name, "error": "segment_rounds found no round"})
            continue
        for k, (start, end) in enumerate(windows, 1):
            clip_id = f"rc_{video.stem.lower()}_r{k}"
            clip = WORK / f"{clip_id}.mp4"
            cut(video, start, end, clip, int(info["width"]))
            entry = {"clip": clip_id, "video": video.name, "start_s": round(start, 2), "end_s": round(end, 2),
                     "source_size": [info["width"], info["height"]], "source_fps": info["r_frame_rate"],
                     "scaled_to_1920": int(info["width"]) > MAX_WIDTH}
            try:
                payload = run(clip, WEIGHTS, "ocsort", WORK, EventConfig())
            except RuntimeError as exc:  # e.g. dohyo never detected: record it, keep going
                entry["error"] = str(exc)
                manifest["rounds"].append(entry)
                continue
            for suffix in (".json", "_tracks.txt"):
                shutil.copy(WORK / f"{clip_id}{suffix}", args.out / f"{clip_id}{suffix}")
            entry.update({"n_frames": payload["n_frames"], "fps": payload["fps"],
                          "events": [(e["kind"], e["frame"]) for e in payload["events"]]})
            manifest["rounds"].append(entry)
            print(json.dumps(entry), flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
