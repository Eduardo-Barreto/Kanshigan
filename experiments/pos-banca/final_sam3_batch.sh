#!/usr/bin/env bash
set -u
# Needs a data root whose annotations/ is scratch, so the gold folders are never written:
#   mkdir -p /tmp/kanshigan_e6/sam_root/annotations && ln -sfn <repo>/data/processed /tmp/kanshigan_e6/sam_root/processed
cd /home/barreto/tcc/Kanshigan/experiments/sam3-poc
A=/home/barreto/tcc/Kanshigan/data/annotations
# item 9: video predictor on the gold JP clip, JP parameters (prompt toy, thresh 0.15)
KANSHIGAN_DATA_ROOT=/tmp/kanshigan_e6/sam_root uv run python annotate_to_yolo.py gold_jp25 --split sam3_video_gold_jp --prompt toy --score-thresh 0.15
# E1: image predictor on the gold frames
uv run python annotate_gold_images.py --images $A/gold_crop/images --out /tmp/kanshigan_e6/sam_e1/gold_br --prompt robot
uv run python annotate_gold_images.py --images $A/gold_jp/images --out /tmp/kanshigan_e6/sam_e1/gold_jp --prompt toy --score-thresh 0.15
echo SAM_DONE
