import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _imports():
    from pathlib import Path
    import time

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    return Path, cv2, np, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Alternativa leve ao SAM 3: YOLO + ByteTrack

    Este notebook é uma alternativa "meia bomba", mas defensável, ao pipeline com SAM 3.
    A ideia não é provar que o modelo já está bom. A ideia é mostrar um caminho mais viável
    para a entrega curta:

    1. detectar o dohyo com visão clássica;
    2. usar a ROI do dohyo para reduzir falsos positivos;
    3. rodar YOLO frame a frame;
    4. usar ByteTrack para manter identidade dos robôs;
    5. extrair métricas simples de FPS e quantidade de tracks.

    O notebook usa `YOLO(...).track(..., tracker="bytetrack.yaml")`, que é o caminho mais curto
    com Ultralytics. Com pesos COCO (`yolo11n.pt` ou `yolov8n.pt`) ele provavelmente vai detectar
    mal os robôs, porque a classe "sumo robot" não existe no COCO. Mesmo assim, o código demonstra
    a arquitetura que depois deve receber um modelo treinado no dataset do projeto.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuração

    Entrada esperada:

    - `notebooks/data/IMG_1573.MOV`

    Saída:

    - `notebooks/output/yolo_baseline_preview.mp4`

    Dependência opcional para realmente rodar a célula de YOLO:

    ```bash
    cd notebooks
    uv add ultralytics
    ```
    """)
    return


@app.cell
def _paths(Path, mo):
    here = Path(__file__).parent.resolve()
    video_path = here / "data" / "IMG_1573.MOV"
    output_dir = here / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / "yolo_baseline_preview.mp4"

    mo.md(
        f"""
        **Vídeo:** `{video_path}`

        **Saída:** `{output_video_path}`
        """
    )
    return output_video_path, video_path


@app.cell
def _helpers(cv2, np):
    def detect_dohyo(frame_bgr, threshold=200, padding=40):
        """Detecta uma ROI aproximada do dohyo usando a borda branca (tawara)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=3)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = frame_bgr.shape[:2]
        x, y, rw, rh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        x = max(0, x - padding)
        y = max(0, y - padding)
        rw = min(w - x, rw + 2 * padding)
        rh = min(h - y, rh + 2 * padding)
        return x, y, rw, rh

    def box_center_xyxy(box):
        x1, y1, x2, y2 = box
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)

    def keep_boxes_near_roi_center(boxes_xyxy, roi, max_boxes=2):
        """Mantém no máximo dois boxes mais próximos do centro do dohyo."""
        if roi is None or len(boxes_xyxy) <= max_boxes:
            return list(range(len(boxes_xyxy)))

        rx, ry, rw, rh = roi
        cx, cy = rx + rw / 2.0, ry + rh / 2.0

        distances = []
        for idx, box in enumerate(boxes_xyxy):
            bx, by = box_center_xyxy(box)
            distances.append((idx, float(np.hypot(bx - cx, by - cy))))

        return [idx for idx, _ in sorted(distances, key=lambda item: item[1])[:max_boxes]]

    def draw_tracks(frame_bgr, boxes_xyxy, track_ids, scores, roi=None):
        """Desenha ROI, boxes e IDs em um frame BGR."""
        out = frame_bgr.copy()
        if roi is not None:
            rx, ry, rw, rh = roi
            cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (0, 220, 255), 2)

        for box, track_id, score in zip(boxes_xyxy, track_ids, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            label = f"id={track_id} conf={score:.2f}" if track_id is not None else f"conf={score:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (80, 255, 80), 2)
            cv2.putText(
                out,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (80, 255, 80),
                2,
                cv2.LINE_AA,
            )
        return out

    return detect_dohyo, draw_tracks, keep_boxes_near_roi_center


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preview da ROI

    Esta célula só testa a parte que reaproveitamos do experimento com SAM 3: detectar o dohyo
    e desenhar uma região de interesse. Ela independe de YOLO e deve rodar com OpenCV puro.
    """)
    return


@app.cell
def _(cv2, detect_dohyo, mo, plt, video_path):
    mo.stop(
        not video_path.exists(),
        mo.callout(mo.md(f"Vídeo não encontrado: `{video_path}`"), kind="warn"),
    )

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    mo.stop(not ok, mo.callout(mo.md("Não foi possível ler o primeiro frame."), kind="danger"))

    roi = detect_dohyo(frame)
    preview = frame.copy()
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 220, 255), 2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Frame 0 - {frame_count} frames @ {fps:.1f} fps")
    ax.axis("off")
    plt.tight_layout()
    out = mo.vstack([mo.md(f"ROI detectada: `{roi}`"), mo.as_html(fig)])
    plt.close(fig)
    out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## YOLO + ByteTrack

    Esta célula é o baseline alternativo. Ela está escrita para compilar mesmo sem `ultralytics`
    instalado; se a dependência não existir, o notebook para aqui com uma mensagem.

    Por padrão usa `yolo11n.pt`, que é pequeno. Em uma versão séria, esse caminho vira algo como:

    ```python
    model = YOLO("runs/detect/sumo-robots-yolo/weights/best.pt")
    ```

    O resultado esperado com peso COCO é ruim, mas a arquitetura é a que interessa para a entrega.
    """)
    return


@app.cell
def _yolo_baseline(
    cv2,
    detect_dohyo,
    draw_tracks,
    keep_boxes_near_roi_center,
    mo,
    output_video_path,
    time,
    video_path,
):
    try:
        from ultralytics import YOLO
    except ImportError:
        mo.stop(
            True,
            mo.callout(
                mo.md("`ultralytics` não está instalado. Para rodar: `cd notebooks && uv add ultralytics`."),
                kind="warn",
            ),
        )

    mo.stop(
        not video_path.exists(),
        mo.callout(mo.md(f"Vídeo não encontrado: `{video_path}`"), kind="warn"),
    )

    model_name = "yolo11n.pt"
    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(video_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_fps = 10
    max_frames = 120
    stride = max(1, round(source_fps / target_fps))

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        target_fps,
        (source_w, source_h),
    )

    seen_tracks = set()
    processed = 0
    read_index = 0
    last_roi = None
    started = time.perf_counter()

    while processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if read_index % stride != 0:
            read_index += 1
            continue

        roi = detect_dohyo(frame)
        if roi is None:
            roi = last_roi
        else:
            last_roi = roi

        # O ByteTrack fica dentro do Ultralytics. persist=True mantém o estado entre frames.
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.15,
            iou=0.5,
            verbose=False,
        )

        result = results[0]
        boxes_xyxy = []
        track_ids = []
        scores = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            if result.boxes.id is None:
                track_ids = [None] * len(boxes_xyxy)
            else:
                track_ids = [int(x) for x in result.boxes.id.cpu().numpy()]

        keep = keep_boxes_near_roi_center(boxes_xyxy, roi, max_boxes=2)
        boxes_xyxy = [boxes_xyxy[i] for i in keep]
        track_ids = [track_ids[i] for i in keep]
        scores = [float(scores[i]) for i in keep]

        for track_id in track_ids:
            if track_id is not None:
                seen_tracks.add(track_id)

        annotated = draw_tracks(frame, boxes_xyxy, track_ids, scores, roi=roi)
        writer.write(annotated)

        processed += 1
        read_index += 1

    cap.release()
    writer.release()

    elapsed = max(time.perf_counter() - started, 1e-9)
    measured_fps = processed / elapsed

    mo.md(
        f"""
        ## Resultado do baseline

        - Modelo: `{model_name}`
        - Frames processados: `{processed}`
        - FPS medido: `{measured_fps:.2f}`
        - Tracks únicos vistos: `{len(seen_tracks)}`
        - Vídeo anotado: `{output_video_path}`

        Observação: com pesos COCO, esse resultado serve só como demonstração de pipeline.
        O próximo passo real é trocar o peso por um YOLO treinado em robôs de sumô.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Por que esta alternativa é aceitável como entrega

    Mesmo que o resultado visual seja ruim com pesos genéricos, este notebook mostra uma decisão
    técnica coerente:

    - o SAM 3 sai do caminho crítico;
    - a parte boa do experimento anterior, a ROI dinâmica do dohyo, continua sendo usada;
    - o detector final vira treinável no domínio;
    - o tracking usa uma ferramenta padrão e mensurável;
    - as métricas de viabilidade ficam fáceis de reportar: FPS, quantidade de tracks, ID switches
      observados manualmente e, depois, MOTA/IDF1 quando houver ground truth.
    """)
    return


if __name__ == "__main__":
    app.run()
