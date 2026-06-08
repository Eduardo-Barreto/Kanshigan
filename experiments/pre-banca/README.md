# Pre-banca pipeline

Pipeline da pré-banca: SAM 3 anota → YOLOv8s treina → OC-SORT rastreia → métricas no gold set.

## Setup

Dois envs (sam3 conflita com boxmot/ultralytics):

```bash
# env pra anotacao (SAM 3 -> YOLO bbox)
cd experiments/sam3-poc && uv sync

# env pra editor, treino, tracking, eval
cd experiments/pre-banca && uv sync
```

## Fluxo

### 1. Definir clips

Copie `configs/clips.example.yaml` para `configs/clips.yaml` e edite. Cada round vira 1 clip. `clips.yaml` fica fora do git (segue padrão `*spec.md`/dados versionados via DVC).

### 2. Cortar e decimar

```bash
uv run python prep_clips.py
```

Gera dois outputs por clip:
- `data/processed/clips/<subset>/<id>.mp4` (fps nativo, full res, alimenta inferência e review)
- `data/processed/sam_input/<id>.mp4` (10 fps, 480x270, alimenta SAM)

### 3. Anotar via SAM 3 (env sam3-poc)

```bash
cd ../sam3-poc
uv run python annotate_to_yolo.py <clip_id> --split <train|val|gold>
cd -
```

Roda SAM 3 no clip pequeno, escala bboxes pra resolução nativa, salva frames + labels YOLO em `data/annotations/<split>/`.

### 4. Editar / validar (env pre-banca)

```bash
uv run python editor.py --split gold --clip-id <clip_id>
```

Use o editor pra revisar/corrigir cada frame. Recomendado pelo menos pro gold; opcional pra train/val (revisão rápida).

Editor:
- click numa box: seleciona
- drag dentro: move
- drag num canto: redimensiona
- `n` + drag: cria nova box
- `d`: deleta selecionada
- `c`: limpa todas as do frame
- setas / espaço: navega (auto-save)
- `s`: força save
- `r`: recarrega do disco
- `q`: sai (auto-save)

Para revisar a anotação em vídeo (qualidade do SAM antes de treinar):

```bash
uv run python preview_annotations.py --split train      # gera results/preview/*.mp4
```

### 5. Treinar (E2)

```bash
uv run python train.py --epochs 100        # YOLOv8s, seed 42, logs em results/training/
```

Equivale a `yolo train data=configs/kanshigan.yaml model=yolov8s.pt epochs=100 imgsz=640 seed=42`,
com hiperparâmetros versionados e resumo em JSON.

### 6. Inferência

```bash
uv run python infer.py data/processed/clips/br/<id>.mp4 \
    --weights results/training/yolov8s_kanshigan/weights/best.pt \
    --tracker ocsort --out results/E2_yolo_oc_vs_gold
```

Emite `<id>.json` (trajetórias, métricas, eventos, FPS), `<id>_tracks.txt` (MOT) e
`<id>_overlay.mp4` (bbox + ID A/B + eventos).

### 7. Avaliar contra o gold

```bash
uv run python evaluate.py --weights <best.pt>                          # mAP do detector
uv run python evaluate.py --pred-mot <id>_tracks.txt --gold-mot gold.txt   # MOTA, IDF1
uv run python evaluate.py --pred <id>.json --gold-events gold.json     # eventos P/R
```

## Módulos da pipeline

| Arquivo | Responsabilidade |
|---------|------------------|
| `schema.py` | Tipos compartilhados (Calibration, Track, Event) |
| `dohyo.py` | Detecção da arena por visão clássica + calibração px→cm |
| `tracking.py` | Wrapper OC-SORT/ByteTrack + convenção de identidade A/B |
| `metrics.py` | Cinemática (posição, velocidade, aceleração via Savitzky-Golay) |
| `events.py` | Detecção determinística de eventos (início, contato, ring-out) |
| `infer.py` | Orquestra a pipeline ponta a ponta |
| `train.py` | Treino do YOLOv8s |
| `evaluate.py` | mAP, MOTA/IDF1, erro de métricas, P/R de eventos vs gold |
| `preview_annotations.py` | Vídeo de revisão das anotações |

Testes da lógica pura (sem GPU): `uv run --group dev pytest`.

## Exemplos

Saída da pipeline em footage real (detecção do dohyo + bbox + ID A/B), uma fonte por
exemplo: `results/figures/qualitative_br_jp.png` (frame BR + JP lado a lado, versionado).

Vídeos de overlay (não versionados; regenere com o comando abaixo) ficam em
`results/examples/`:

```bash
# BR (câmera de mão) e JP (cenital fixa), round real cada
ffmpeg -ss 7.8 -to 14.2 -i data/raw/br/ZB4dF1ub5QM.mp4 -c copy /tmp/demo_br.mp4
uv run python infer.py /tmp/demo_br.mp4 \
    --weights results/training/yolov8s_kanshigan/weights/best.pt \
    --tracker ocsort --out results/examples
```

Cada execução gera `<id>_overlay.mp4` (vídeo anotado), `<id>.json` (métricas) e
`<id>_tracks.txt` (MOT).

## Layout de dados

```
data/
├── raw/{jp,br}/                 vídeos brutos (DVC)
├── processed/
│   ├── clips/{jp,br}/<id>.mp4   clip nativo
│   └── sam_input/<id>.mp4       clip decimado 480x270
└── annotations/
    ├── train/{images,labels}/
    ├── val/{images,labels}/
    └── gold/{images,labels}/
```
