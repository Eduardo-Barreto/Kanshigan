"""Generate paper figures from an inference JSON: trajectories and speed profiles.

Reads the structured output of infer.py and renders publication figures into
results/figures/. Kept separate from inference so figures can be regenerated and
restyled without re-running the GPU pipeline.

Usage:
    uv run python figures.py results/E2_yolo_oc_vs_gold/<id>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from schema import DOHYO_DIAMETER_CM

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "figures"
COLORS = {"A": "#00b300", "B": "#ff6a00"}


def plot_trajectories(payload: dict, out: Path) -> None:
    radius = DOHYO_DIAMETER_CM / 2
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="gray", lw=1.5))
    for traj in payload["trajectories"]:
        ax.plot(traj["x_cm"], traj["y_cm"], color=COLORS.get(traj["robot_id"], "k"), lw=1.8, label=f"Robô {traj['robot_id']}")
    ax.set_aspect("equal")
    ax.set_xlim(-radius * 1.1, radius * 1.1)
    ax.set_ylim(-radius * 1.1, radius * 1.1)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(f"Trajetórias --- {payload['video']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_speed(payload: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for traj in payload["trajectories"]:
        ax.plot(traj["t_s"], traj["speed_cms"], color=COLORS.get(traj["robot_id"], "k"), lw=1.5, label=f"Robô {traj['robot_id']}")
    for event in payload["events"]:
        if event["kind"] in ("first_contact", "ring_out"):
            ax.axvline(event["t_ms"] / 1000, color="red", ls="--", lw=1, alpha=0.7)
            ax.text(event["t_ms"] / 1000, ax.get_ylim()[1] * 0.95, event["kind"], rotation=90, fontsize=7, va="top")
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("velocidade (cm/s)")
    ax.set_title(f"Perfil de velocidade --- {payload['video']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=Path)
    args = ap.parse_args()
    payload = json.loads(args.json.read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = args.json.stem
    plot_trajectories(payload, FIGURES / f"{stem}_trajectories.png")
    plot_speed(payload, FIGURES / f"{stem}_speed.png")
    print(f"wrote figures for {stem} to {FIGURES}")


if __name__ == "__main__":
    main()
