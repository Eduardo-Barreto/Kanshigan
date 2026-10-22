#!/usr/bin/env bash
set -u
cd /home/barreto/tcc/Kanshigan/experiments/pos-banca
# Input for the "native" rows: the same gold round cut from the raw video, not cropped.
mkdir -p /tmp/kanshigan_e6
[ -f /tmp/kanshigan_e6/gold_zb01_native.mp4 ] || ffmpeg -y -loglevel error -ss 7.8 -to 14.2 -i ../../data/raw/br/ZB4dF1ub5QM.mp4 -c copy /tmp/kanshigan_e6/gold_zb01_native.mp4
R=../../results/training
CROP=../../data/processed/clips/br/gold_zb01.mp4
NATIVE=/tmp/kanshigan_e6/gold_zb01_native.mp4
RC4K="/home/barreto/tcc/videos/3kg RC/IMG_1594.MOV"
V8=$R/yolov8s_kanshigan/weights/best.pt
N26=$R/yolo26n_kanshigan/weights/best.pt
run() { uv run --project ../pre-banca --with nvidia-ml-py python final_viability.py "$@"; }
for det in yolov8s yolo26n; do
  W=$([ $det = yolov8s ] && echo $V8 || echo $N26)
  run --video $CROP --weights $W --label gpu_crop_${det}_ocsort --note "gold_zb01 pre-cropped clip 1558x506 (prep_clips.py crop done offline)"
  run --video $NATIVE --weights $W --label gpu_native_${det}_ocsort --note "same round cut from raw ZB4dF1ub5QM.mp4 7.8-14.2 s with ffmpeg -c copy, 1920x1080 H.264, crop inside timer"
  for trk in bytetrack deepocsort botsort; do
    run --video $CROP --weights $W --tracker $trk --label gpu_crop_${det}_${trk} --note "gold_zb01 pre-cropped clip, tracker $trk"
  done
  run --video "$RC4K" --weights $W --label gpu_rc4k_${det}_ocsort --note "RC IMG_1594.MOV 3840x2160 HEVC 59.94 fps, native, crop inside timer"
done
for det in yolo26n yolov8s; do
  W=$([ $det = yolov8s ] && echo $V8 || echo $N26)
  CUDA_VISIBLE_DEVICES= run --video $CROP --weights $W --label cpu_crop_${det}_ocsort --note "CPU only, pre-cropped clip"
  CUDA_VISIBLE_DEVICES= run --video $NATIVE --weights $W --label cpu_native_${det}_ocsort --note "CPU only, native 1920x1080 cut, crop inside timer"
done
echo ALL_DONE
