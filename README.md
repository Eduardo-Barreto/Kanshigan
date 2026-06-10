# Kanshigan
Automated Analysis of Robot Sumo Matches through Computer Vision

## Documents

- [Paper (IEEE)](https://nightly.link/Eduardo-Barreto/Kanshigan/workflows/build-paper/main/kanshigan-paper.zip)
- [Projeto de Pesquisa (ABNT)](https://nightly.link/Eduardo-Barreto/Kanshigan/workflows/build-paper/main/projeto-de-pesquisa.zip)
- [Project Plan (Inteli)](https://nightly.link/Eduardo-Barreto/Kanshigan/workflows/build-paper/main/project-plan.zip)

## Reproducibility

Pinned versions and checkpoints behind the reported results. Exact resolutions live
in each subproject's `uv.lock`.

### Inference pipeline (`experiments/pre-banca`)

| Component | Version / artifact |
| --- | --- |
| torch | `2.12.0+cu126` |
| ultralytics | `8.4.55` |
| boxmot | `19.0.0` (OC-SORT, ByteTrack, DeepOCSORT, BoT-SORT) |
| Detector base weights | `yolov8s.pt`, `yolo26n.pt` (COCO, Ultralytics) |
| Trained detectors | `results/training/{yolov8s,yolo26n}_kanshigan/weights/best.pt` |
| ReID (appearance trackers) | `osnet_x0_25_msmt17` |
| GPU | NVIDIA RTX 4070 Laptop, 8 GB |

### Annotation (`experiments/sam3-poc`)

SAM 3 is used only as an offline annotator (stage E1), never in the inference path.
Its output (the dataset labels) is frozen under DVC, so the reported results do not
depend on re-running it. The source is pinned in `experiments/sam3-poc/uv.lock` at
commit [`f6e51f5`](https://github.com/facebookresearch/sam3/commit/f6e51f59500a87c576c2df2323ce56b9fd7a12de);
re-annotation from raw video is the only step that needs it.
