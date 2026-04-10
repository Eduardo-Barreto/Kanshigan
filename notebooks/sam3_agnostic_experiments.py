import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _imports():
    import cv2
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from PIL import Image
    from transformers import Sam3Processor, Sam3Model, Sam3VideoModel

    return Image, Path, Sam3Model, Sam3Processor, Sam3VideoModel, cv2, np, plt, torch


@app.cell
def _device(torch):
    """Device-agnostic resolution: cuda > mps > cpu. No platform checks needed."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return (device,)


@app.cell
def _auth(Path):
    import os
    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv(Path(__file__).parent / ".env")
    login(token=os.environ["HF_KEY"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SAM 3 – Segmentação de Robôs de Sumô (HuggingFace Transformers)

    Este notebook é a versão migrada do experimento original com o SAM 3. Em vez de depender
    do repositório oficial (`facebookresearch/sam3`) — que tem CUDA acoplado em todo o codebase —,
    usamos a reimplementação device-agnostic do [HuggingFace Transformers](https://huggingface.co/docs/transformers/model_doc/sam3).

    O objetivo continua o mesmo: segmentar robôs de sumô automaticamente usando prompts de texto,
    propagando as segmentações ao longo do vídeo.

    ---

    ### O que está aqui

    | # | Seção | Modelo | O que faz |
    |---|-------|--------|-----------|
    | 1 | Teste de prompts | `Sam3Model` (imagem) | Compara diferentes descrições de texto no frame 0 |
    | 2 | Vídeo baseline | `Sam3VideoModel` | Inferência direta no vídeo completo, sem pré-processamento |
    | 3 | Vídeo avançado | `Sam3VideoModel` | Detecta o dohyô por visão computacional → recorta → infere → remapeia |
    | 4 | Por frame | `Sam3Model` (imagem) | Inferência independente em cada frame, sem propagação temporal |

    ---

    ### Sobre o ambiente

    Esta versão roda em CUDA (Linux/Windows), MPS (Apple Silicon) e CPU — sem forks,
    sem stub de decord, sem patches de triton. O device é detectado automaticamente na
    célula `_device` acima.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Configuração inicial

    Os caminhos abaixo são relativos ao arquivo do notebook. Os resultados são salvos em `output/`.
    """)
    return


@app.cell
def _setup(Path, mo):
    _here = Path(__file__).parent.resolve()
    video_path = _here / "data" / "IMG_1573.MOV"
    output_dir = _here / "output"
    mo.stop(
        not video_path.exists(),
        mo.callout(mo.md(f"Vídeo não encontrado: `{video_path}`"), kind="danger"),
    )
    mo.md(f"**Vídeo:** `{video_path.name}` — **Saída:** `{output_dir}`")
    return output_dir, video_path


@app.cell
def _preview(cv2, mo, plt, video_path):
    _cap = cv2.VideoCapture(str(video_path))
    _ret, _bgr = _cap.read()
    _fps = _cap.get(cv2.CAP_PROP_FPS)
    _n = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _cap.release()
    mo.stop(not _ret, mo.callout(mo.md("Não foi possível ler o vídeo."), kind="danger"))
    _rgb = cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.imshow(_rgb)
    _ax.set_title(f"Frame 0 — {_n} frames @ {_fps:.0f} fps  ({_w}×{_h})")
    _ax.axis("off")
    plt.tight_layout()
    _out = mo.vstack([mo.md("### Preview do vídeo"), mo.as_html(_fig)])
    plt.close(_fig)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Funções auxiliares

    Abaixo estão as funções usadas por todas as seções do notebook. Estão agrupadas aqui
    pra evitar repetição e facilitar a leitura das células de experimento.

    - **`detect_dohyo`** — detecta o dohyô (plataforma branca circular) via limiarização OpenCV
    - **`preprocess_video`** — extrai frames, detecta ROI por frame, recorta e salva vídeo menor
    - **`masks_to_full_frame`** — remapeia as máscaras do espaço recortado pro frame original
    - **`filter_closest_to_center`** — filtra máscaras muito grandes e mantém as mais próximas do centro
    - **`overlay_masks`** — renderiza as máscaras coloridas sobre o frame
    - **`overlay_masks_from_output`** — igual, mas recebe diretamente o dict `{"out_binary_masks": ...}`
    """)
    return


@app.cell
def _helpers(cv2, np, torch):
    COLORS = np.array(
        [[255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 50], [255, 50, 255]],
        dtype=np.uint8,
    )
    ALPHA = 0.45
    ROI_COLOR = (0, 255, 200)

    def detect_dohyo(frame_bgr):
        """Detecta a plataforma branca do dohyô e retorna seu bounding rect (x,y,w,h)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k, iterations=3)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k, iterations=2)
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
        h, w = frame_bgr.shape[:2]
        pad = int(0.05 * max(w, h))
        x, y = max(0, x - pad), max(0, y - pad)
        cw, ch = min(w - x, cw + 2 * pad), min(h - y, ch + 2 * pad)
        return x, y, cw, ch

    def preprocess_video(video_path, output_path, fps, crop_size):
        """Extrai frames no fps alvo, detecta ROI por frame, recorta e salva vídeo."""
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        skip = max(1, round(src_fps / fps))
        raw, i = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % skip == 0:
                raw.append(frame)
            i += 1
        cap.release()

        rois, fallback = [], None
        for frame in raw:
            roi = detect_dohyo(frame)
            if roi is not None:
                fallback = roi
            rois.append(roi if roi is not None else fallback)
        if fallback is None:
            raise RuntimeError("Não foi possível detectar o dohyô em nenhum frame.")
        for j in range(len(rois)):
            if rois[j] is None:
                rois[j] = fallback

        cw, ch = crop_size
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, crop_size)
        for frame, roi in zip(raw, rois):
            rx, ry, rw, rh = roi
            writer.write(cv2.resize(frame[ry:ry+rh, rx:rx+rw], crop_size, interpolation=cv2.INTER_LINEAR))
        writer.release()

        return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in raw], rois

    def masks_to_full_frame(output, roi, crop_size, full_h, full_w):
        """Remapeia as máscaras do espaço recortado de volta às coordenadas do frame original."""
        masks = output.get("out_binary_masks")
        if masks is None:
            return []
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if masks.ndim == 4:
            masks = masks[:, 0]
        rx, ry, rw, rh = roi
        result = []
        for i in range(masks.shape[0]):
            resized = cv2.resize(masks[i].astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
            full = np.zeros((full_h, full_w), dtype=np.uint8)
            full[ry:min(ry+rh, full_h), rx:min(rx+rw, full_w)] = resized[:min(rh, full_h-ry), :min(rw, full_w-rx)]
            result.append(full)
        return result

    def filter_closest_to_center(full_masks, roi, max_objects):
        """Filtra máscaras muito grandes e mantém as mais próximas do centro do ROI."""
        rx, ry, rw, rh = roi
        max_area = rw * rh * 0.15
        valid = [m for m in full_masks if np.count_nonzero(m) <= max_area]
        if len(valid) <= max_objects:
            return valid
        cx, cy = rx + rw / 2, ry + rh / 2
        def _dist(m):
            ys, xs = np.where(m > 0)
            return float("inf") if len(xs) == 0 else np.sqrt((xs.mean()-cx)**2 + (ys.mean()-cy)**2)
        return sorted(valid, key=_dist)[:max_objects]

    def overlay_masks(frame, masks_list, roi=None):
        """Renderiza uma lista de máscaras sobre um frame RGB, com retângulo do ROI opcional."""
        result = frame.copy()
        for idx, mask in enumerate(masks_list):
            binary = mask > 0
            if not binary.any():
                continue
            color = COLORS[idx % len(COLORS)]
            result[binary] = (ALPHA * color + (1 - ALPHA) * result[binary]).astype(np.uint8)
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color.tolist(), 2)
        if roi:
            rx, ry, rw, rh = roi
            cv2.rectangle(result, (rx, ry), (rx+rw, ry+rh), ROI_COLOR, 2)
        return result

    def overlay_masks_from_output(frame, output):
        """Renderiza as máscaras do dict {"out_binary_masks": ...} diretamente sobre o frame."""
        masks = output.get("out_binary_masks") or output.get("out_masks")
        if masks is None:
            return frame.copy()
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if masks.ndim == 4:
            masks = masks[:, 0]
        result = frame.copy()
        for idx in range(masks.shape[0]):
            binary = masks[idx] > 0
            if binary.shape != frame.shape[:2]:
                continue
            color = COLORS[idx % len(COLORS)]
            result[binary] = (ALPHA * color + (1 - ALPHA) * result[binary]).astype(np.uint8)
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color.tolist(), 2)
        return result

    return (
        filter_closest_to_center,
        masks_to_full_frame,
        overlay_masks,
        overlay_masks_from_output,
        preprocess_video,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1. Teste de Prompts

    Antes de rodar o pipeline completo, é importante saber qual descrição de texto funciona melhor
    pra detectar os robôs. O SAM 3 usa um encoder de texto (tipo CLIP) pra encontrar regiões da
    imagem que correspondam ao prompt.

    Esta seção roda o **modelo de imagem** (`Sam3Model`) no frame 0 do vídeo selecionado, testando
    cada prompt da lista e mostrando quantas detecções cada um produziu.

    **Por que isso importa?** A escolha do prompt afeta diretamente a qualidade do tracking no vídeo.
    Um prompt vago como `"object"` tende a detectar tudo. Um muito específico como `"sumo robot"`
    pode não generalizar bem. O ideal é testar antes.
    """)
    return


@app.cell
def _pt_ui(mo):
    pt_prompts_ui = mo.ui.text_area(
        value="\n".join([
            "robot", "sumo robot", "black box", "machine", "electronic device",
            "object on metal platform", "small box on circular platform",
            "dark object", "vehicle", "toy",
        ]),
        label="Prompts para testar (um por linha)",
        rows=10,
    )
    pt_run_btn = mo.ui.run_button(label="Rodar teste de prompts")
    mo.vstack([pt_prompts_ui, pt_run_btn])
    return pt_prompts_ui, pt_run_btn


@app.cell
def _pt_run(
    Image,
    Sam3Model,
    Sam3Processor,
    cv2,
    device,
    mo,
    np,
    output_dir,
    plt,
    pt_prompts_ui,
    pt_run_btn,
    torch,
    video_path,
):
    mo.stop(not pt_run_btn.value, mo.md("*Clique em **Rodar teste de prompts** para começar.*"))

    _out = output_dir / "prompts"
    _out.mkdir(parents=True, exist_ok=True)

    _cap = cv2.VideoCapture(str(video_path))
    _, _bgr = _cap.read()
    _cap.release()
    _frame_rgb = cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB)
    _pil = Image.fromarray(_frame_rgb)

    _processor = Sam3Processor.from_pretrained("facebook/sam3")
    _model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    _model.eval()

    _prompts = [p.strip() for p in pt_prompts_ui.value.splitlines() if p.strip()]
    _figs, _rows = [], []

    for _prompt in _prompts:
        _inputs = _processor(
            images=_pil,
            text=[_prompt],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            _outputs = _model(**_inputs)

        # pred_masks: (1, num_masks, H, W) — convert to helper-compatible dict
        _masks = _outputs.pred_masks[0].cpu().numpy()       # (num_masks, H, W)
        _scores = _outputs.pred_iou_scores[0].cpu().numpy() # (num_masks,)

        _n = _masks.shape[0] if _masks is not None else 0
        _top = str(_scores[:3].round(3).tolist()) if _n > 0 else "—"
        _rows.append({"Prompt": _prompt, "Detecções": _n, "Top-3 scores": _top})

        _fig, _ax = plt.subplots(figsize=(5, 3))
        _ax.imshow(_frame_rgb)
        _ax.set_title(f'"{_prompt}"  ({_n} detecções)', fontsize=9)
        _ax.axis("off")
        if _n > 0:
            _cm = plt.cm.tab10(np.linspace(0, 1, max(_n, 1)))
            for _i in range(min(_n, 5)):
                _rgba = np.zeros((*(_masks[_i] > 0).shape, 4))
                _rgba[_masks[_i] > 0] = [*_cm[_i][:3], 0.45]
                _ax.imshow(_rgba)
        plt.tight_layout()
        _fig.savefig(_out / f"{_prompt.replace(' ', '_')}.png", dpi=100, bbox_inches="tight")
        _figs.append(mo.as_html(_fig))
        plt.close(_fig)

    mo.vstack([
        mo.md("### Resultados por prompt"),
        mo.ui.table(data=_rows, label="Resumo"),
        mo.md("### Visualizações"),
        mo.hstack(_figs, wrap=True),
        mo.callout(mo.md(f"Imagens salvas em `{_out}`"), kind="success"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2. Vídeo Baseline

    Com o melhor prompt em mãos, o próximo passo é rodar o **modelo de vídeo** (`Sam3VideoModel`)
    no vídeo completo.

    O modelo de vídeo do SAM 3 funciona diferente do de imagem: ele processa o frame 0, detecta
    os objetos via prompt de texto, e depois **propaga** as segmentações ao longo do tempo.
    Isso significa que ele não precisa re-detectar o robô a cada frame — ele rastreia.

    Esta seção é o baseline: sem nenhum pré-processamento específico do domínio. O vídeo vai
    direto pro modelo, frame a frame no tamanho original.

    **Limitação esperada:** sem recorte do dohyô, o modelo vê muito contexto irrelevante
    (chão, público, etc.) o que pode confundir o tracking.

    > **Nota MPS:** se rodar em Apple Silicon e encontrar um erro de device mismatch durante
    > a inferência de vídeo, é o bug de `.pin_memory()` em `processing_sam3_video.py`
    > (issue conhecido no transformers). O fix é remover o `.pin_memory()` na linha
    > `keep_idx_gpu = keep_idx.to(device=..., non_blocking=True)`.
    """)
    return


@app.cell
def _bv_ui(mo):
    bv_prompt_ui = mo.ui.text(value="object on metal platform", label="Prompt")
    bv_run_btn = mo.ui.run_button(label="Rodar vídeo baseline")
    mo.vstack([bv_prompt_ui, bv_run_btn])
    return bv_prompt_ui, bv_run_btn


@app.cell
def _bv_run(
    Image,
    Sam3Processor,
    Sam3VideoModel,
    bv_prompt_ui,
    bv_run_btn,
    cv2,
    device,
    mo,
    output_dir,
    overlay_masks_from_output,
    plt,
    torch,
    video_path,
):
    mo.stop(not bv_run_btn.value, mo.md("*Clique em **Rodar vídeo baseline** para começar.*"))

    _out = output_dir / "baseline"
    _out.mkdir(parents=True, exist_ok=True)

    _cap = cv2.VideoCapture(str(video_path))
    _fps = _cap.get(cv2.CAP_PROP_FPS)
    _frames_bgr = []
    while True:
        _ret, _fr = _cap.read()
        if not _ret:
            break
        _frames_bgr.append(_fr)
    _cap.release()

    _frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in _frames_bgr]
    _pil_frames = [Image.fromarray(f) for f in _frames_rgb]

    _processor = Sam3Processor.from_pretrained("facebook/sam3")
    _model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device)
    _model.eval()

    # Sam3VideoModel accepts a list of PIL frames + a text prompt and returns
    # per-frame masks via temporal propagation from frame 0.
    _inputs = _processor(
        videos=_pil_frames,
        text=[bv_prompt_ui.value],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        _outputs = _model(**_inputs)

    # pred_masks: (num_frames, num_objects, H, W)
    _all_masks = _outputs.pred_masks.cpu().numpy()
    _n0 = _all_masks.shape[1]

    _h, _w = _frames_rgb[0].shape[:2]
    _vid_path = str(_out / "annotated.mp4")
    _writer = cv2.VideoWriter(_vid_path, cv2.VideoWriter_fourcc(*"mp4v"), _fps, (_w, _h))
    _annotated = []

    for _i, _frame in enumerate(_frames_rgb):
        _frame_masks = _all_masks[_i] if _i < len(_all_masks) else None
        _output_dict = {"out_binary_masks": _frame_masks} if _frame_masks is not None else {}
        _ann = overlay_masks_from_output(_frame, _output_dict)
        _writer.write(cv2.cvtColor(_ann, cv2.COLOR_RGB2BGR))
        _annotated.append(_ann)
    _writer.release()

    _stride = max(1, int(_fps))
    _figs = []
    for _i in range(0, len(_annotated), _stride):
        _fig, _ax = plt.subplots(figsize=(5, 3))
        _ax.imshow(_annotated[_i])
        _ax.set_title(f"Frame {_i}")
        _ax.axis("off")
        plt.tight_layout()
        _figs.append(mo.as_html(_fig))
        plt.close(_fig)

    mo.vstack([
        mo.md(f"**{len(_frames_rgb)} frames** processados — **{_n0} objetos** detectados"),
        mo.md("### Frames-chave (1 por segundo)"),
        mo.hstack(_figs, wrap=True),
        mo.callout(mo.md(f"Vídeo salvo em `{_vid_path}`"), kind="success"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3. Vídeo Avançado — Pipeline com Detecção do Dohyô

    A abordagem baseline sofre com contexto irrelevante. A solução foi criar um pipeline
    em três etapas que aproveita o conhecimento do domínio:

    **Etapa 1 — Detecção do dohyô:**
    O dohyô (plataforma de sumô) é branco e circular. Dá pra detectá-lo com limiarização
    simples no OpenCV: threshold em pixels muito brancos (>200), morfologia pra fechar buracos,
    e pegar o maior contorno. Resultado: bounding box do dohyô por frame.

    **Etapa 2 — Recorte e redimensionamento:**
    Cada frame é recortado pro bounding box do dohyô e redimensionado pra um tamanho fixo
    (padrão 320×180). Isso reduz o contexto irrelevante e diminui drasticamente o custo
    computacional. O vídeo recortado vai pro SAM 3.

    **Etapa 3 — Remapeamento das máscaras:**
    As máscaras que o SAM 3 retorna estão no espaço do vídeo recortado. Precisamos
    remapeá-las de volta pro frame original pra gerar a visualização final.

    Adicionalmente, filtramos máscaras muito grandes (provavelmente o próprio dohyô sendo
    segmentado) e priorizamos as mais próximas do centro da arena.
    """)
    return


@app.cell
def _av_ui(mo):
    av_prompt_ui = mo.ui.text(value="toy", label="Prompt")
    av_fps_ui = mo.ui.number(start=1, stop=30, value=5, label="FPS alvo")
    av_crop_w_ui = mo.ui.number(start=64, stop=1920, value=320, label="Largura do recorte")
    av_crop_h_ui = mo.ui.number(start=64, stop=1080, value=180, label="Altura do recorte")
    av_max_robots_ui = mo.ui.number(start=1, stop=50, value=20, label="Máx. robôs")
    av_run_btn = mo.ui.run_button(label="Rodar vídeo avançado")
    mo.vstack([
        mo.hstack([av_prompt_ui, av_fps_ui, av_max_robots_ui], justify="start"),
        mo.hstack([av_crop_w_ui, av_crop_h_ui], justify="start"),
        av_run_btn,
    ])
    return (
        av_crop_h_ui,
        av_crop_w_ui,
        av_fps_ui,
        av_max_robots_ui,
        av_prompt_ui,
        av_run_btn,
    )


@app.cell
def _av_run(
    Image,
    Sam3Processor,
    Sam3VideoModel,
    av_crop_h_ui,
    av_crop_w_ui,
    av_fps_ui,
    av_max_robots_ui,
    av_prompt_ui,
    av_run_btn,
    cv2,
    device,
    filter_closest_to_center,
    masks_to_full_frame,
    mo,
    output_dir,
    overlay_masks,
    plt,
    preprocess_video,
    torch,
    video_path,
):
    mo.stop(not av_run_btn.value, mo.md("*Clique em **Rodar vídeo avançado** para começar.*"))

    _out = output_dir / "advanced"
    _out.mkdir(parents=True, exist_ok=True)

    _crop_size = (int(av_crop_w_ui.value), int(av_crop_h_ui.value))
    _fps = int(av_fps_ui.value)
    _cropped_path = str(_out / "cropped_input.mp4")

    _orig_frames, _rois = preprocess_video(str(video_path), _cropped_path, _fps, _crop_size)
    _orig_h, _orig_w = _orig_frames[0].shape[:2]

    # Preview do ROI detectado
    _prev = _orig_frames[0].copy()
    _rx, _ry, _rw, _rh = _rois[0]
    cv2.rectangle(_prev, (_rx, _ry), (_rx+_rw, _ry+_rh), (0, 255, 200), 3)
    _fig_roi, _ax_roi = plt.subplots(figsize=(7, 4))
    _ax_roi.imshow(_prev)
    _ax_roi.set_title("Dohyô detectado (frame 0) — área que vai pro SAM 3")
    _ax_roi.axis("off")
    plt.tight_layout()

    # Load cropped frames for the model
    _cap = cv2.VideoCapture(_cropped_path)
    _cropped_frames = []
    while True:
        _ret, _fr = _cap.read()
        if not _ret:
            break
        _cropped_frames.append(Image.fromarray(cv2.cvtColor(_fr, cv2.COLOR_BGR2RGB)))
    _cap.release()

    _processor = Sam3Processor.from_pretrained("facebook/sam3")
    _model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device)
    _model.eval()

    _inputs = _processor(
        videos=_cropped_frames,
        text=[av_prompt_ui.value],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        _outputs = _model(**_inputs)

    # pred_masks: (num_frames, num_objects, H, W)
    _all_masks = _outputs.pred_masks.cpu().numpy()
    _n0 = _all_masks.shape[1]

    _vid_path = str(_out / "annotated.mp4")
    _writer = cv2.VideoWriter(_vid_path, cv2.VideoWriter_fourcc(*"mp4v"), _fps, (_orig_w, _orig_h))
    _annotated = []

    for _i in range(len(_orig_frames)):
        _roi = _rois[_i]
        if _i < len(_all_masks):
            _frame_masks = _all_masks[_i]          # (num_objects, H, W)
            _output_dict = {"out_binary_masks": _frame_masks}
            _full_masks = masks_to_full_frame(_output_dict, _roi, _crop_size, _orig_h, _orig_w)
            _full_masks = filter_closest_to_center(_full_masks, _roi, int(av_max_robots_ui.value))
        else:
            _full_masks = []
        _ann = overlay_masks(_orig_frames[_i], _full_masks, _roi)
        _writer.write(cv2.cvtColor(_ann, cv2.COLOR_RGB2BGR))
        _annotated.append(_ann)
    _writer.release()

    _stride = max(1, _fps)
    _figs = []
    for _i in range(0, len(_annotated), _stride):
        _fig, _ax = plt.subplots(figsize=(5, 3))
        _ax.imshow(_annotated[_i])
        _ax.set_title(f"Frame {_i}")
        _ax.axis("off")
        plt.tight_layout()
        _figs.append(mo.as_html(_fig))
        plt.close(_fig)

    mo.vstack([
        mo.md(f"**{len(_orig_frames)} frames** processados — **{_n0} objetos** detectados no frame 0"),
        mo.md("### ROI do dohyô"),
        mo.as_html(_fig_roi),
        mo.md("### Frames-chave (1 por segundo)"),
        mo.hstack(_figs, wrap=True),
        mo.callout(mo.md(f"Vídeo salvo em `{_vid_path}`"), kind="success"),
    ])
    plt.close(_fig_roi)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 4. Segmentação Por Frame (sem propagação temporal)

    Esta seção serve como controle experimental: roda o **modelo de imagem** (`Sam3Model`) de forma
    independente em cada frame subamostrado, sem nenhuma propagação temporal.

    A diferença fundamental em relação ao modelo de vídeo é que aqui **não existe memória
    entre frames**. O modelo trata cada frame como uma imagem isolada e re-detecta os
    objetos do zero usando o prompt.

    **Por que isso é útil?** Comparar os resultados daqui com o vídeo avançado ajuda a
    quantificar o ganho real da propagação temporal. Se o modelo de imagem já acerta bem,
    a complexidade do pipeline de vídeo pode não se justificar. Se ele perde tracking com
    frequência, o vídeo vale a pena.
    """)
    return


@app.cell
def _pf_ui(mo):
    pf_prompt_ui = mo.ui.text(value="object on metal platform", label="Prompt")
    pf_fps_ui = mo.ui.number(start=1, stop=30, value=5, label="FPS alvo (subamostragem)")
    pf_run_btn = mo.ui.run_button(label="Rodar por frame")
    mo.vstack([mo.hstack([pf_prompt_ui, pf_fps_ui], justify="start"), pf_run_btn])
    return pf_fps_ui, pf_prompt_ui, pf_run_btn


@app.cell
def _pf_run(
    Image,
    Sam3Model,
    Sam3Processor,
    cv2,
    device,
    mo,
    output_dir,
    overlay_masks_from_output,
    pf_fps_ui,
    pf_prompt_ui,
    pf_run_btn,
    plt,
    torch,
    video_path,
):
    mo.stop(not pf_run_btn.value, mo.md("*Clique em **Rodar por frame** para começar.*"))

    _out = output_dir / "per_frame"
    _out.mkdir(parents=True, exist_ok=True)

    _cap = cv2.VideoCapture(str(video_path))
    _src_fps = _cap.get(cv2.CAP_PROP_FPS)
    _skip = max(1, round(_src_fps / int(pf_fps_ui.value)))
    _frames, _i = [], 0
    while True:
        _ret, _fr = _cap.read()
        if not _ret:
            break
        if _i % _skip == 0:
            _frames.append(cv2.cvtColor(_fr, cv2.COLOR_BGR2RGB))
        _i += 1
    _cap.release()

    _processor = Sam3Processor.from_pretrained("facebook/sam3")
    _model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    _model.eval()

    _annotated = []
    for _frame in _frames:
        _pil = Image.fromarray(_frame)
        _inputs = _processor(
            images=_pil,
            text=[pf_prompt_ui.value],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            _outputs = _model(**_inputs)

        # pred_masks: (1, num_masks, H, W)
        _masks = _outputs.pred_masks[0].cpu().numpy()  # (num_masks, H, W)
        _output_dict = {"out_binary_masks": _masks} if len(_masks) > 0 else {}
        _ann = overlay_masks_from_output(_frame, _output_dict)
        _annotated.append(_ann)

    _h, _w = _frames[0].shape[:2]
    _vid_path = str(_out / "annotated.mp4")
    _writer = cv2.VideoWriter(_vid_path, cv2.VideoWriter_fourcc(*"mp4v"), int(pf_fps_ui.value), (_w, _h))
    for _ann_fr in _annotated:
        _writer.write(cv2.cvtColor(_ann_fr, cv2.COLOR_RGB2BGR))
    _writer.release()

    _stride = max(1, len(_annotated) // 10)
    _figs = []
    for _i in range(0, len(_annotated), _stride):
        _fig, _ax = plt.subplots(figsize=(5, 3))
        _ax.imshow(_annotated[_i])
        _ax.set_title(f"Frame {_i}")
        _ax.axis("off")
        plt.tight_layout()
        _figs.append(mo.as_html(_fig))
        plt.close(_fig)

    mo.vstack([
        mo.md(f"**{len(_frames)} frames** processados independentemente"),
        mo.md("### Frames-chave"),
        mo.hstack(_figs, wrap=True),
        mo.callout(mo.md(f"Vídeo salvo em `{_vid_path}`"), kind="success"),
    ])
    return


if __name__ == "__main__":
    app.run()
