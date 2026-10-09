"""Browser bbox editor for reviewing SAM 3 labels from another machine.

Same job as editor.py (fix YOLO boxes frame by frame), served over HTTP so it works
remotely. Binds to one host only (default: this machine's Tailscale IP), so it is
reachable inside the tailnet and nowhere else. Stdlib only.

Each frame also gets a review status (approved / rejected) in
data/annotations/<split>/review.json. Rejecting never deletes files; the status is
what decides later which frames move to train.

The last box of each frame starts selected, so the common fixes (delete, nudge,
resize) need no mouse. Playback steps through frames at the SAM annotation rate
and pauses at each new clip.

Mouse:
    click a box               select it
    drag a selected box       move it
    drag a corner             resize
    drag on empty area        draw a new box

Keys:
    space            play / pause                 right / left   next / prev frame (auto-saves)
    tab / shift+tab  select next / prev box       w a s d        move selected box
    shift+w a s d    shorter / narrower / taller / wider
    Delete / Bksp    delete selected box          c              clear all boxes
    Enter            approve + next               x              reject + next
    y                copy boxes                   v              paste boxes (replaces)
    r                reload from disk             g              next unreviewed frame

Usage:
    uv run python web_editor.py --split candidates_jp [--host 100.85.135.81] [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = ROOT / "data" / "annotations"
SAFE_NAME = re.compile(r"^[\w-]+$")
REVIEW_STATUSES = {"approved", "rejected", "unreviewed"}
MAX_BODY_BYTES = 64_000
IMAGE_CACHE_CONTROL = "private, max-age=3600"  # frames never change; caching lets playback prefetch


def tailscale_ip() -> str:
    out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def read_boxes(label: Path) -> list[list[float]]:
    if not label.exists():
        return []
    return [[float(v) for v in line.split()[1:5]] for line in label.read_text().splitlines() if line.strip()]


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def valid_box(box: object) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(v, (int, float)) for v in box)
        and 0 <= box[0] <= 1
        and 0 <= box[1] <= 1
        and 0 < box[2] <= 1
        and 0 < box[3] <= 1
    )


class Split:
    def __init__(self, name: str):
        self.dir = ANN_DIR / name
        self.images = self.dir / "images"
        self.labels = self.dir / "labels"
        self.review_path = self.dir / "review.json"
        self.stems = sorted(p.stem for p in self.images.glob("*.jpg"))
        self.review: dict[str, str] = json.loads(self.review_path.read_text()) if self.review_path.exists() else {}

    def has(self, stem: str) -> bool:
        return SAFE_NAME.match(stem) is not None and (self.images / f"{stem}.jpg").exists()

    def frames(self) -> list[dict]:
        return [
            {"stem": s, "status": self.review.get(s, "unreviewed"), "boxes": len(read_boxes(self.labels / f"{s}.txt"))}
            for s in self.stems
        ]

    def save_boxes(self, stem: str, boxes: list[list[float]]) -> None:
        self.labels.mkdir(parents=True, exist_ok=True)
        lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes]
        write_atomic(self.labels / f"{stem}.txt", "".join(f"{line}\n" for line in lines))

    def set_status(self, stem: str, status: str) -> None:
        if status == "unreviewed":
            self.review.pop(stem, None)
        else:
            self.review[stem] = status
        write_atomic(self.review_path, json.dumps(self.review, indent=1, sort_keys=True))


def make_handler(split: Split):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if self.command != "GET":
                super().log_message(fmt, *args)

        def send(self, status: HTTPStatus, body: bytes, content_type: str, cache_control: str = "no-store") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send(status, json.dumps(payload).encode(), "application/json")

        def fail(self, status: HTTPStatus, message: str) -> None:
            self.send_json({"error": message}, status)

        def read_json(self) -> object:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_BYTES:
                raise ValueError("body too large")
            return json.loads(self.rfile.read(length))

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self.send(HTTPStatus.OK, PAGE.replace("__SPLIT__", split.dir.name).encode(), "text/html; charset=utf-8")
            if path == "/api/frames":
                return self.send_json(split.frames())
            parts = path.strip("/").split("/")
            if len(parts) != 2 or not split.has(parts[1].removesuffix(".jpg")):
                return self.fail(HTTPStatus.NOT_FOUND, "not found")
            kind, stem = parts[0], parts[1].removesuffix(".jpg")
            if kind == "img":
                return self.send(HTTPStatus.OK, (split.images / f"{stem}.jpg").read_bytes(), "image/jpeg", IMAGE_CACHE_CONTROL)
            if kind == "labels":
                return self.send_json({"boxes": read_boxes(split.labels / f"{stem}.txt"), "status": split.review.get(stem, "unreviewed")})
            return self.fail(HTTPStatus.NOT_FOUND, "not found")

        def do_PUT(self):
            parts = urlparse(self.path).path.strip("/").split("/")
            if len(parts) != 2 or not split.has(parts[1]):
                return self.fail(HTTPStatus.NOT_FOUND, "not found")
            kind, stem = parts
            try:
                body = self.read_json()
            except (ValueError, json.JSONDecodeError) as error:
                return self.fail(HTTPStatus.BAD_REQUEST, str(error))
            if not isinstance(body, dict):
                return self.fail(HTTPStatus.BAD_REQUEST, "expected object")
            if kind == "labels":
                boxes = body.get("boxes")
                if not isinstance(boxes, list) or not all(valid_box(b) for b in boxes):
                    return self.fail(HTTPStatus.BAD_REQUEST, "boxes must be [[cx, cy, w, h], ...] normalized")
                split.save_boxes(stem, boxes)
                return self.send_json({"ok": True})
            if kind == "review":
                status = body.get("status")
                if status not in REVIEW_STATUSES:
                    return self.fail(HTTPStatus.BAD_REQUEST, f"status must be one of {sorted(REVIEW_STATUSES)}")
                split.set_status(stem, status)
                return self.send_json({"ok": True})
            return self.fail(HTTPStatus.NOT_FOUND, "not found")

    return Handler


PAGE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Kanshigan editor: __SPLIT__</title>
<style>
  :root { color-scheme: dark; font-family: system-ui, sans-serif; }
  body { margin: 0; display: grid; grid-template-columns: 260px 1fr; height: 100vh; background: #111; color: #eee; }
  aside { overflow-y: auto; border-right: 1px solid #333; padding: 8px; }
  main { display: flex; flex-direction: column; min-width: 0; }
  header { padding: 8px 12px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; border-bottom: 1px solid #333; }
  #stage { flex: 1; display: flex; align-items: center; justify-content: center; min-height: 0; padding: 8px; }
  canvas { max-width: 100%; max-height: 100%; cursor: crosshair; }
  select, button { background: #222; color: #eee; border: 1px solid #555; padding: 4px 8px; border-radius: 4px; }
  button:focus-visible, select:focus-visible { outline: 2px solid #4af; }
  ul { list-style: none; margin: 0; padding: 0; font: 12px ui-monospace, monospace; }
  li { padding: 2px 6px; cursor: pointer; border-radius: 3px; display: flex; justify-content: space-between; }
  li.current { background: #2a4a7a; }
  li.approved::before { content: "✓ "; color: #4c4; }
  li.rejected { color: #888; text-decoration: line-through; }
  li.rejected::before { content: "✗ "; color: #c44; }
  .status-approved { color: #4c4; } .status-rejected { color: #c44; } .status-unreviewed { color: #ca4; }
  #help { font-size: 12px; color: #aaa; }
  #saveState { font-size: 12px; }
</style>
</head>
<body>
<aside>
  <label>Clip <select id="clipFilter"><option value="">todos</option></select></label>
  <p id="progress"></p>
  <ul id="frameList"></ul>
</aside>
<main>
  <header>
    <strong id="stemLabel"></strong>
    <span id="statusLabel"></span>
    <span id="saveState" role="status"></span>
    <button id="playButton">▶ Play (espaço)</button>
    <label>Velocidade <select id="speedSelect">
      <option value="0.5">0,5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option>
    </select></label>
    <label><input type="checkbox" id="autoApprove"> aprovar ao avançar</label>
    <button id="approveButton">Aprovar (Enter)</button>
    <button id="rejectButton">Rejeitar (x)</button>
    <button id="rejectClipButton">Rejeitar clip inteiro</button>
    <span id="help">← → quadro · Tab troca caixa · WASD move · Shift+WASD tamanho · Del apaga · arrastar no vazio cria · c limpa · y/v copia/cola · r recarrega · g próximo pendente</span>
  </header>
  <div id="stage"><canvas id="canvas"></canvas></div>
</main>
<script>
const CORNER_HIT_PX = 14;
const MIN_BOX_PX = 4;
const PLAYBACK_BASE_FPS = 10;
const NUDGE_FRACTION = 0.004;
const PREFETCH_COUNT = 8;
const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const image = new Image();

let frames = [];
let visibleFrames = [];
let currentIndex = 0;
let boxes = [];
let selected = null;
let drag = null;
let dirty = false;
let clipboard = null;
let playing = false;

const clipOf = (stem) => stem.replace(/_\\d+$/, "");
const toPixels = ([cx, cy, w, h]) => ({ x: (cx - w / 2) * canvas.width, y: (cy - h / 2) * canvas.height, w: w * canvas.width, h: h * canvas.height });
const clamp01 = (v) => Math.min(1, Math.max(0, v));

function toNormalized({ x, y, w, h }) {
  const x0 = clamp01(Math.min(x, x + w) / canvas.width), x1 = clamp01(Math.max(x, x + w) / canvas.width);
  const y0 = clamp01(Math.min(y, y + h) / canvas.height), y1 = clamp01(Math.max(y, y + h) / canvas.height);
  return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0];
}

async function request(method, url, body) {
  const response = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: body && JSON.stringify(body) });
  if (!response.ok) throw new Error(`${method} ${url}: ${response.status} ${await response.text()}`);
  return response.json();
}

function showSaveState(text, isError = false) {
  const saveState = document.getElementById("saveState");
  saveState.textContent = text;
  saveState.style.color = isError ? "#f66" : "#8a8";
}

const currentFrame = () => visibleFrames[currentIndex];

async function saveIfDirty() {
  if (!dirty) return;
  const frame = currentFrame();
  try {
    await request("PUT", `/labels/${frame.stem}`, { boxes });
    frame.boxes = boxes.length;
    dirty = false;
    showSaveState("salvo");
  } catch (error) {
    console.error(error);
    showSaveState(`erro ao salvar: ${error.message}`, true);
    throw error;
  }
}

async function setStatus(frame, status) {
  await request("PUT", `/review/${frame.stem}`, { status });
  frame.status = status;
}

async function loadFrame(index) {
  if (!visibleFrames.length) return;
  currentIndex = Math.max(0, Math.min(visibleFrames.length - 1, index));
  const frame = currentFrame();
  const labels = await request("GET", `/labels/${frame.stem}`);
  boxes = labels.boxes;
  frame.status = labels.status;
  selected = boxes.length ? boxes.length - 1 : null;
  dirty = false;
  await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = `/img/${frame.stem}.jpg`; });
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  render();
  renderList();
  prefetchImages();
}

function prefetchImages() {
  visibleFrames.slice(currentIndex + 1, currentIndex + 1 + PREFETCH_COUNT).forEach((frame) => { new Image().src = `/img/${frame.stem}.jpg`; });
}

async function goTo(index) {
  try { await saveIfDirty(); } catch { return false; }
  await loadFrame(index);
  return true;
}

async function advance() {
  const frame = currentFrame();
  if (document.getElementById("autoApprove").checked && frame.status === "unreviewed") {
    try { await saveIfDirty(); await setStatus(frame, "approved"); } catch (error) { console.error(error); showSaveState(`erro: ${error.message}`, true); return false; }
  }
  return goTo(currentIndex + 1);
}

function updatePlayButton() {
  document.getElementById("playButton").textContent = playing ? "⏸ Pausar (espaço)" : "▶ Play (espaço)";
}

async function play() {
  playing = true;
  updatePlayButton();
  while (playing && currentIndex < visibleFrames.length - 1) {
    const startedAt = performance.now();
    const clipBefore = clipOf(currentFrame().stem);
    if (!(await advance())) break;
    if (clipOf(currentFrame().stem) !== clipBefore) break;
    const speed = Number(document.getElementById("speedSelect").value);
    const remainingMs = 1000 / (PLAYBACK_BASE_FPS * speed) - (performance.now() - startedAt);
    await new Promise((resolve) => setTimeout(resolve, Math.max(0, remainingMs)));
  }
  playing = false;
  updatePlayButton();
}

const togglePlay = () => { if (playing) { playing = false; return; } play(); };

function render() {
  context.drawImage(image, 0, 0);
  const lineWidth = Math.max(2, canvas.width / 400);
  boxes.forEach((box, i) => {
    const { x, y, w, h } = toPixels(box);
    context.lineWidth = lineWidth;
    context.strokeStyle = i === selected ? "#0ff" : "#0c0";
    context.strokeRect(x, y, w, h);
    if (i !== selected) return;
    context.fillStyle = "#0ff";
    [[x, y], [x + w, y], [x, y + h], [x + w, y + h]].forEach(([cx, cy]) => context.fillRect(cx - lineWidth * 2, cy - lineWidth * 2, lineWidth * 4, lineWidth * 4));
  });
  if (drag?.kind === "new") {
    context.strokeStyle = "#f0f";
    context.strokeRect(drag.box.x, drag.box.y, drag.box.w, drag.box.h);
  }
  const frame = currentFrame();
  document.getElementById("stemLabel").textContent = `${frame.stem} (${currentIndex + 1}/${visibleFrames.length}) · ${boxes.length} caixa(s)`;
  const statusLabel = document.getElementById("statusLabel");
  statusLabel.textContent = frame.status;
  statusLabel.className = `status-${frame.status}`;
}

function renderList() {
  const list = document.getElementById("frameList");
  list.replaceChildren(...visibleFrames.map((frame, i) => {
    const item = document.createElement("li");
    item.className = `${frame.status}${i === currentIndex ? " current" : ""}`;
    item.innerHTML = `<span>${frame.stem}</span><span>${frame.boxes}</span>`;
    item.onclick = () => goTo(i);
    return item;
  }));
  list.children[currentIndex]?.scrollIntoView({ block: "nearest" });
  const count = (status) => visibleFrames.filter((f) => f.status === status).length;
  document.getElementById("progress").textContent = `${count("approved")} aprovados · ${count("rejected")} rejeitados · ${count("unreviewed")} pendentes`;
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  const scale = canvas.width / rect.width;
  return { x: (event.clientX - rect.left) * scale, y: (event.clientY - rect.top) * scale, scale };
}

function cornerHit(box, point) {
  const { x, y, w, h } = toPixels(box);
  const radius = CORNER_HIT_PX * point.scale;
  const corners = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]];
  const hit = corners.findIndex(([cx, cy]) => Math.abs(point.x - cx) <= radius && Math.abs(point.y - cy) <= radius);
  return hit === -1 ? null : hit;
}

const contains = (box, point) => { const { x, y, w, h } = toPixels(box); return point.x >= x && point.x <= x + w && point.y >= y && point.y <= y + h; };

canvas.addEventListener("pointerdown", (event) => {
  playing = false;
  const point = pointerPosition(event);
  canvas.setPointerCapture(event.pointerId);
  if (selected !== null) {
    const corner = cornerHit(boxes[selected], point);
    if (corner !== null) {
      const { x, y, w, h } = toPixels(boxes[selected]);
      const opposite = [[x + w, y + h], [x, y + h], [x + w, y], [x, y]][corner];
      drag = { kind: "resize", anchor: opposite };
      return;
    }
  }
  const hitIndex = boxes.findLastIndex((box) => contains(box, point));
  if (hitIndex !== -1) {
    selected = hitIndex;
    drag = { kind: "move", start: point, original: toPixels(boxes[hitIndex]) };
  } else {
    selected = null;
    drag = { kind: "new", start: point, box: { x: point.x, y: point.y, w: 0, h: 0 } };
  }
  render();
});

canvas.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const point = pointerPosition(event);
  if (drag.kind === "move") {
    const { original, start } = drag;
    boxes[selected] = toNormalized({ ...original, x: original.x + point.x - start.x, y: original.y + point.y - start.y });
    dirty = true;
  }
  if (drag.kind === "resize") {
    const [ax, ay] = drag.anchor;
    boxes[selected] = toNormalized({ x: ax, y: ay, w: point.x - ax, h: point.y - ay });
    dirty = true;
  }
  if (drag.kind === "new") drag.box = { x: drag.start.x, y: drag.start.y, w: point.x - drag.start.x, h: point.y - drag.start.y };
  render();
});

canvas.addEventListener("pointerup", () => {
  if (drag?.kind === "new" && Math.abs(drag.box.w) >= MIN_BOX_PX && Math.abs(drag.box.h) >= MIN_BOX_PX) {
    boxes.push(toNormalized(drag.box));
    selected = boxes.length - 1;
    dirty = true;
  }
  drag = null;
  render();
});

async function review(status) {
  try {
    await saveIfDirty();
    await setStatus(currentFrame(), status);
  } catch (error) {
    console.error(error);
    showSaveState(`erro: ${error.message}`, true);
    return;
  }
  await loadFrame(currentIndex + 1);
}

async function rejectClip() {
  const clip = clipOf(currentFrame().stem);
  if (!confirm(`Rejeitar todos os quadros de ${clip}?`)) return;
  try {
    await saveIfDirty();
    for (const frame of frames.filter((f) => clipOf(f.stem) === clip)) await setStatus(frame, "rejected");
    showSaveState(`${clip} rejeitado`);
  } catch (error) {
    console.error(error);
    showSaveState(`erro: ${error.message}`, true);
  }
  renderList();
  render();
}

function transformSelected(transform) {
  if (selected === null) return;
  boxes[selected] = toNormalized(transform(toPixels(boxes[selected])));
  dirty = true;
  render();
}

const nudge = (dx, dy) => transformSelected((box) => ({ ...box, x: box.x + dx * canvas.width * NUDGE_FRACTION, y: box.y + dy * canvas.height * NUDGE_FRACTION }));

const grow = (dw, dh) => transformSelected((box) => {
  const w = Math.max(MIN_BOX_PX, box.w + 2 * dw * canvas.width * NUDGE_FRACTION);
  const h = Math.max(MIN_BOX_PX, box.h + 2 * dh * canvas.height * NUDGE_FRACTION);
  return { x: box.x + (box.w - w) / 2, y: box.y + (box.h - h) / 2, w, h };
});

function cycleSelection(step) {
  if (!boxes.length) return;
  selected = ((selected ?? -step) + step + boxes.length) % boxes.length;
  render();
}

function deleteSelected() {
  if (selected === null) return;
  boxes.splice(selected, 1);
  selected = boxes.length ? Math.min(selected, boxes.length - 1) : null;
  dirty = true;
  render();
}

const KEY_ACTIONS = {
  Space: () => togglePlay(),
  ArrowRight: () => advance(),
  ArrowLeft: () => goTo(currentIndex - 1),
  Tab: () => cycleSelection(1),
  KeyW: () => nudge(0, -1),
  KeyA: () => nudge(-1, 0),
  KeyS: () => nudge(0, 1),
  KeyD: () => nudge(1, 0),
  Delete: () => deleteSelected(),
  Backspace: () => deleteSelected(),
  Enter: () => review("approved"),
  KeyX: () => review("rejected"),
  KeyC: () => { boxes = []; selected = null; dirty = true; render(); },
  KeyY: () => { clipboard = structuredClone(boxes); showSaveState(`${boxes.length} caixa(s) copiada(s)`); },
  KeyV: () => { if (!clipboard) return; boxes = structuredClone(clipboard); selected = boxes.length ? boxes.length - 1 : null; dirty = true; render(); },
  KeyR: () => loadFrame(currentIndex),
  KeyG: () => { const next = visibleFrames.findIndex((f, i) => i > currentIndex && f.status === "unreviewed"); if (next !== -1) goTo(next); },
};

const SHIFT_KEY_ACTIONS = {
  Tab: () => cycleSelection(-1),
  KeyW: () => grow(0, -1),
  KeyA: () => grow(-1, 0),
  KeyS: () => grow(0, 1),
  KeyD: () => grow(1, 0),
};

document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLSelectElement || event.ctrlKey || event.metaKey || event.altKey) return;
  const action = (event.shiftKey ? SHIFT_KEY_ACTIONS : KEY_ACTIONS)[event.code];
  if (!action) return;
  event.preventDefault();
  if (event.code !== "Space") playing = false;
  action();
});

document.addEventListener("click", (event) => { if (event.target.matches("button, input")) event.target.blur(); });
document.getElementById("playButton").onclick = () => togglePlay();
document.getElementById("speedSelect").onchange = (event) => event.target.blur();
document.getElementById("approveButton").onclick = () => review("approved");
document.getElementById("rejectButton").onclick = () => review("rejected");
document.getElementById("rejectClipButton").onclick = rejectClip;
document.getElementById("clipFilter").onchange = async (event) => {
  try { await saveIfDirty(); } catch { return; }
  visibleFrames = event.target.value ? frames.filter((f) => clipOf(f.stem) === event.target.value) : frames;
  event.target.blur();
  await loadFrame(0);
};

window.addEventListener("beforeunload", (event) => { if (dirty) event.preventDefault(); });

(async () => {
  frames = await request("GET", "/api/frames");
  visibleFrames = frames;
  const clipFilter = document.getElementById("clipFilter");
  [...new Set(frames.map((f) => clipOf(f.stem)))].forEach((clip) => clipFilter.add(new Option(clip, clip)));
  await loadFrame(0);
})();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="candidates_jp", help="annotation subdir under data/annotations")
    ap.add_argument("--host", help="bind address (default: this machine's Tailscale IPv4)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if not SAFE_NAME.match(args.split) or not (ANN_DIR / args.split / "images").is_dir():
        raise SystemExit(f"split not found: {ANN_DIR / args.split}")
    host = args.host or tailscale_ip()
    split = Split(args.split)
    server = ThreadingHTTPServer((host, args.port), make_handler(split))
    print(f"[{args.split}] {len(split.stems)} frames at http://{host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
