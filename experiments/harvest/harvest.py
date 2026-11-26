"""Harvest Robot Sumo rounds from public videos into SAM 3 candidates for the studio.

Each stage reads the manifest and what earlier stages left on disk, so any stage can be
rerun or resumed. The heavy lifting is done by the same scripts the studio's ingest
jobs call: `segment_rounds.py`, `prep_clips.py` and SAM 3 (`annotate_batch.py`).

    scout      long streams: find rounds on a 360p copy, keep only those stretches
    download   yt-dlp each manifest row (whole video, or only its sections)
    segment    propose round windows per raw video        -> data/harvest/windows/<raw>.json
    cut        cut and crop each window to the dohyo       -> data/processed/clips/<source>/
    annotate   SAM 3 boxes (and masks) for every clip      -> data/annotations/<dataset>/
    register   empty studio sidecars + provenance table    -> data/harvest/provenance.tsv
    report     per-source counts of videos, rounds and labelled frames

Manifest (TSV, header required): video_id, source (br|jp|intl), event, sections, channel,
title. `sections` is empty for the whole video, `start-end` pairs in seconds joined by `;`,
or `scout` to take the sections the scout stage found.

Usage:
    python3 harvest.py <stage> --manifest manifest.tsv [--jobs 6]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("KANSHIGAN_DATA_ROOT", REPO / "data"))
PRE_BANCA = REPO / "experiments" / "pre-banca"
SAM3_POC = REPO / "experiments" / "sam3-poc"
HARVEST = DATA / "harvest"
WINDOWS = HARVEST / "windows"
LOGS = HARVEST / "logs"

DATASET = "web_candidates"
SAM_PROMPT = "toy"
# SAM 3's built-in thresholds miss the small dark robots of the JP overhead view
# (diário 14), so JP runs at 0.15 as the studio does. Side views keep the defaults.
SAM_SCORE_THRESHOLD: dict[str, float] = {"jp": 0.15}
# Side views start at the defaults and retry at 0.15 when few frames hold both robots:
# in the pilot the defaults missed the small robots of overhead-ish footage (Robotex
# 2013), while 0.15 everywhere ran out of VRAM on handheld footage.
SAM_FALLBACK_THRESHOLD = 0.15
# At that threshold SAM can track dozens of spurious masklets and overflow 8 GB of VRAM;
# only two boxes per frame are kept, so 8 live masklets leave room for false positives.
SAM_MAX_TRACKED = 8
YOUTUBE_FORMAT = "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]"
SCOUT_FORMAT = "bv*[height<=360][vcodec^=avc1]/bv*[height<=360]"
SCOUT_MERGE_GAP_S = 30.0  # rounds of one match sit seconds apart; merging saves yt-dlp calls
SCOUT_PAD_S = 5.0
# A regional stream holds 100+ matches; 40 stretches spread over the day keep many
# different robots without one stream taking most of the GPU time.
SCOUT_MAX_SECTIONS = 40
# YouTube answers a burst of requests with a bot check that blocks the whole IP.
POLITE = ["--sleep-requests", "1", "--sleep-interval", "3", "--max-sleep-interval", "8"]
MIN_ROUND_S = 2.0
MAX_ROUND_S = 30.0  # longer windows merge rounds with resets and people on the dohyo
SOURCES = ("br", "jp", "intl")


@dataclass(frozen=True)
class Row:
    video_id: str
    source: str
    event: str
    sections: tuple[tuple[int, int], ...] | None  # None: scout decides
    channel: str
    title: str


@dataclass(frozen=True)
class RawVideo:
    row: Row
    name: str
    offset: int  # seconds into the original video where this file starts

    @property
    def path(self) -> Path:
        return DATA / "raw" / self.row.source / f"{self.name}.mp4"


def read_manifest(path: Path) -> list[Row]:
    rows = []
    with path.open(encoding="utf8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            if record["source"] not in SOURCES:
                raise SystemExit(f"{record['video_id']}: source must be one of {SOURCES}")
            sections = None if record["sections"] == "scout" else tuple(
                (int(start), int(end))
                for start, end in (pair.split("-") for pair in record["sections"].split(";") if pair)
            )
            rows.append(
                Row(record["video_id"], record["source"], record["event"], sections, record["channel"], record["title"])
            )
    return rows


def scouted_sections(row: Row) -> tuple[tuple[int, int], ...]:
    path = HARVEST / "scout" / f"{row.video_id}.json"
    if not path.exists():
        return ()
    sections = [tuple(pair) for pair in json.loads(path.read_text())]
    if len(sections) > SCOUT_MAX_SECTIONS:
        step = len(sections) / SCOUT_MAX_SECTIONS
        sections = [sections[int(i * step)] for i in range(SCOUT_MAX_SECTIONS)]
    return tuple(sections)


def sections_of(row: Row) -> tuple[tuple[int, int], ...]:
    return scouted_sections(row) if row.sections is None else row.sections


def raw_videos(row: Row) -> list[RawVideo]:
    # The studio names downloads yt_<id>; the prefix also keeps ids starting with '-' path-safe.
    if row.sections == ():
        return [RawVideo(row, f"yt_{row.video_id}", 0)]
    return [RawVideo(row, f"yt_{row.video_id}_s{start}", start) for start, _end in sections_of(row)]


def merge_windows(windows: list[list[float]]) -> list[tuple[int, int]]:
    """Round windows of a low-res copy -> padded, merged download sections in whole seconds."""
    sections: list[list[float]] = []
    for start, end in windows:
        if not MIN_ROUND_S <= end - start <= MAX_ROUND_S:
            continue
        if sections and start - sections[-1][1] <= SCOUT_MERGE_GAP_S:
            sections[-1][1] = end
        else:
            sections.append([start, end])
    return [(max(0, int(start - SCOUT_PAD_S)), int(end + SCOUT_PAD_S + 1)) for start, end in sections]


def scout(row: Row) -> str:
    out = HARVEST / "scout" / f"{row.video_id}.json"
    if out.exists():
        return "kept"
    low_res = HARVEST / "scout" / f"{row.video_id}.mp4"
    low_res.parent.mkdir(parents=True, exist_ok=True)
    if not low_res.exists():
        command = [
            "uvx", "--from", "yt-dlp[default]", "yt-dlp", "--no-playlist", "--no-warnings", *POLITE,
            "--format", SCOUT_FORMAT, "--remux-video", "mp4", "-N", "4",
            "--output", str(low_res.with_suffix(".%(ext)s")),
            "--", f"https://www.youtube.com/watch?v={row.video_id}",
        ]
        if not run(command, low_res.parent, LOGS / "scout" / f"{row.video_id}.download.log"):
            return "download failed"
    log = LOGS / "scout" / f"{row.video_id}.segment.log"
    if not run(["uv", "run", "python", "segment_rounds.py", str(low_res)], PRE_BANCA, log):
        return "segment failed"
    windows = [
        [float(t) for t in line.split(":")[1].split()[0].split("-")]
        for line in log.read_text().splitlines() if line.startswith("round ")
    ]
    sections = merge_windows(windows)
    out.write_text(json.dumps(sections))
    return f"{len(windows)} windows -> {len(sections)} sections, {sum(e - s for s, e in sections)} s"


# Each OpenCV process otherwise spawns a thread per core for its loops and for ffmpeg
# decoding; with a dozen stages in parallel the load reached 180 on 22 cores and starved
# the GPU stage of CPU time.
WORKER_THREADS = {"OMP_NUM_THREADS": "2", "OPENCV_FOR_THREADS_NUM": "2", "OPENCV_FFMPEG_CAPTURE_OPTIONS": "threads;2"}


def run(command: list[str], cwd: Path, log: Path) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(
            command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, **WORKER_THREADS, "KANSHIGAN_DATA_ROOT": str(DATA)},
        )
    return result.returncode == 0


def download(video: RawVideo) -> str:
    if video.path.exists():
        return "kept"
    video.path.parent.mkdir(parents=True, exist_ok=True)
    section = []
    if video.row.sections != ():
        start, end = next(pair for pair in sections_of(video.row) if pair[0] == video.offset)
        section = ["--download-sections", f"*{start}-{end}"]
    command = [
        "uvx", "--from", "yt-dlp[default]", "yt-dlp", "--no-playlist", "--no-warnings", *POLITE,
        "--format", YOUTUBE_FORMAT, "--remux-video", "mp4", *section,
        "--output", str(video.path.with_suffix(".%(ext)s")),
        "--", f"https://www.youtube.com/watch?v={video.row.video_id}",
    ]
    ok = run(command, video.path.parent, LOGS / "download" / f"{video.name}.log")
    return "downloaded" if ok and video.path.exists() else "failed"


def segment(video: RawVideo) -> str:
    out = WINDOWS / f"{video.name}.json"
    if out.exists():
        return "kept"
    if not video.path.exists():
        return "no video"
    log = LOGS / "segment" / f"{video.name}.log"
    if not run(["uv", "run", "python", "segment_rounds.py", str(video.path)], PRE_BANCA, log):
        return "failed"
    windows = []
    for line in log.read_text().splitlines():
        if line.startswith("round "):
            start, end = line.split(":")[1].split()[0].split("-")
            windows.append([float(start), float(end)])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(windows))
    return f"{len(windows)} windows"


def clip_specs(video: RawVideo) -> list[dict]:
    path = WINDOWS / f"{video.name}.json"
    if not path.exists():
        return []
    kept = [w for w in json.loads(path.read_text()) if MIN_ROUND_S <= w[1] - w[0] <= MAX_ROUND_S]
    return [
        {"id": f"{video.name}_r{position:02d}", "source": str(video.path), "subset": video.row.source,
         "start": start, "end": end}
        for position, (start, end) in enumerate(kept, 1)
    ]


def clip_path(spec: dict) -> Path:
    return DATA / "processed" / "clips" / spec["subset"] / f"{spec['id']}.mp4"


def cut(video: RawVideo) -> str:
    pending = [spec for spec in clip_specs(video) if not clip_path(spec).exists()]
    if not pending:
        return "kept"
    spec_path = HARVEST / "specs" / f"{video.name}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps({"clips": pending}))
    ok = run(["uv", "run", "python", "prep_clips.py", "--config", str(spec_path)], PRE_BANCA,
             LOGS / "cut" / f"{video.name}.log")
    return f"{len(pending)} clips" if ok else "failed"


def all_videos(rows: list[Row]) -> list[RawVideo]:
    return [video for row in rows for video in raw_videos(row)]


def all_clips(rows: list[Row]) -> list[tuple[RawVideo, dict]]:
    return [(video, spec) for video in all_videos(rows) for spec in clip_specs(video) if clip_path(spec).exists()]


def annotate(rows: list[Row]) -> None:
    by_threshold: dict[float | None, list[str]] = {}
    # BR first: it is the footage the project is about, so it reaches the studio first.
    clips = sorted(all_clips(rows), key=lambda clip: SOURCES.index(clip[0].row.source))
    for video, spec in clips:
        by_threshold.setdefault(SAM_SCORE_THRESHOLD.get(video.row.source), []).append(spec["id"])
    for threshold, clips in by_threshold.items():
        listing = HARVEST / f"annotate-clips-{threshold or 'default'}.txt"
        listing.write_text("\n".join(clips) + "\n")
        command = [
            "uv", "run", "python", "annotate_batch.py", str(listing), "--split", DATASET,
            "--prompt", SAM_PROMPT, "--status", str(HARVEST / "annotate-status.tsv"),
            "--masks-dir", str(DATA / "processed" / "sam_masks"),
            "--max-tracked", str(SAM_MAX_TRACKED),
            *(["--score-thresh", str(threshold)] if threshold is not None else ["--fallback-thresh", str(SAM_FALLBACK_THRESHOLD)]),
        ]
        subprocess.run(command, cwd=SAM3_POC, check=True, env={**os.environ, "KANSHIGAN_DATA_ROOT": str(DATA)})
    fallback = DATA / "annotations" / f"{DATASET}__fallback"
    if fallback.exists() and not any(fallback.rglob("*.*")):
        shutil.rmtree(fallback)


def labelled_frames(clip: str) -> int:
    labels = DATA / "annotations" / DATASET / "labels"
    return sum(1 for _ in labels.glob(f"{clip}_[0-9]*.txt"))


def register(rows: list[Row]) -> None:
    sidecars = DATA / "annotations" / DATASET / "studio"
    sidecars.mkdir(parents=True, exist_ok=True)
    empty = {"version": 1, "review": {}, "identity": {}, "events": [], "rounds": []}
    lines = ["clip\tsource\tevent\tchannel\turl\tstart_s\tend_s\tframes\ttitle"]
    for video, spec in all_clips(rows):
        frames = labelled_frames(spec["id"])
        if frames == 0:
            continue
        sidecar = sidecars / f"{spec['id']}.json"
        if not sidecar.exists():
            sidecar.write_text(json.dumps(empty, indent=1) + "\n")
        start = video.offset + spec["start"]
        url = f"https://www.youtube.com/watch?v={video.row.video_id}&t={int(start)}s"
        lines.append("\t".join([
            spec["id"], video.row.source, video.row.event, video.row.channel, url,
            f"{start:.1f}", f"{video.offset + spec['end']:.1f}", str(frames), video.row.title,
        ]))
    (HARVEST / "provenance.tsv").write_text("\n".join(lines) + "\n")
    print(f"{len(lines) - 1} clips registered in {DATASET}")


def report(rows: list[Row]) -> None:
    by_source: dict[str, list[int]] = {}
    for video, spec in all_clips(rows):
        totals = by_source.setdefault(video.row.source, [0, 0, 0])
        frames = labelled_frames(spec["id"])
        totals[0] += 1
        totals[1] += frames > 0
        totals[2] += frames
    print("source\tclips\twith_labels\tframes")
    for source, (clips, with_labels, frames) in sorted(by_source.items()):
        print(f"{source}\t{clips}\t{with_labels}\t{frames}")


def parallel(stage, videos: list[RawVideo], jobs: int) -> None:
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for video, outcome in zip(videos, pool.map(stage, videos)):
            print(f"{stage.__name__}\t{video.name}\t{outcome}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["scout", "download", "segment", "cut", "annotate", "register", "report"])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    rows = read_manifest(args.manifest)
    stages = {"download": download, "segment": segment, "cut": cut}
    if args.stage == "scout":
        scouted = [row for row in rows if row.sections is None]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for row, outcome in zip(scouted, pool.map(scout, scouted)):
                print(f"scout\t{row.video_id}\t{outcome}", flush=True)
    elif args.stage in stages:
        parallel(stages[args.stage], all_videos(rows), args.jobs)
    else:
        {"annotate": annotate, "register": register, "report": report}[args.stage](rows)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
