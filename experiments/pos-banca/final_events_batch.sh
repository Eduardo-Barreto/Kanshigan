#!/usr/bin/env bash
# Item 4: regenerate the audited inference JSONs with the current events.py
# (YOLOv8s + OC-SORT, plain infer.py), then rebuild the audit table.
set -eu
cd /home/barreto/tcc/Kanshigan/experiments/pre-banca
W=../../results/training/yolov8s_kanshigan/weights/best.pt
WORK=/tmp/kanshigan_e6/events
OUT=../../results/E6_final_measurements/events
mkdir -p $WORK/in $OUT
# Inputs. demo_br = same round as gold_zb01, uncropped (README: ZB4dF1ub5QM 7.8-14.2 s).
[ -f $WORK/in/gold_zb01_native.mp4 ] || ffmpeg -y -loglevel error -ss 7.8 -to 14.2 -i ../../data/raw/br/ZB4dF1ub5QM.mp4 -c copy $WORK/in/gold_zb01_native.mp4
[ -f $WORK/in/atena.mp4 ] || cp /home/barreto/tcc/atenavsbullbassauro.mp4 $WORK/in/atena.mp4
[ -f $WORK/in/worlds_jE4E-XhiyjU_12401.mp4 ] || cp ../../data/raw/jp/worlds_jE4E-XhiyjU_12401.mp4 $WORK/in/
[ -f $WORK/in/rc1591.mp4 ] || ffmpeg -y -loglevel error -i "/home/barreto/tcc/videos/3kg RC/IMG_1591.MOV" -vf scale=1920:-2 -an -c:v libx264 -crf 18 -preset fast $WORK/in/rc1591.mp4
for v in ../../data/processed/clips/br/gold_zb01.mp4 $WORK/in/gold_zb01_native.mp4 $WORK/in/atena.mp4 $WORK/in/worlds_jE4E-XhiyjU_12401.mp4 $WORK/in/rc1591.mp4; do
  uv run python infer.py $v --weights $W --tracker ocsort --out $WORK/out
  stem=$(basename $v .mp4)
  cp $WORK/out/$stem.json $WORK/out/${stem}_tracks.txt $OUT/
done
echo EVENTS_DONE
