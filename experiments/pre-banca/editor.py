"""Frame-by-frame bbox editor for building the gold set.

Loads images + YOLO labels (e.g. pseudo-labels emitted by annotate.py),
lets you fix them with the mouse, saves YOLO labels back to disk.

Mouse:
    left-click inside a box        select it
    left-drag inside selected box  move it
    left-drag on a corner          resize from that corner
    'n' then left-drag empty area  draw new box
    'd'                            delete selected box
    'c'                            clear all boxes on current frame
    'r'                            reload labels from disk (undo unsaved)

Keys:
    right / space / 'n'  next frame  (auto-saves current)
    left  / 'p'          prev frame  (auto-saves current)
    'g'                  jump to first unsaved-no-boxes frame
    's'                  force save
    'q' / ESC            quit  (auto-saves current)

Usage:
    uv run python editor.py --split gold [--clip-id <prefix>]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = ROOT / "data" / "annotations"

CORNER_PX = 14
COLOR_BOX = (0, 200, 0)
COLOR_SEL = (0, 255, 255)
COLOR_NEW = (255, 0, 255)
TXT_BG = (0, 0, 0)
TXT_FG = (255, 255, 255)


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    def corners(self):
        return [
            (self.x, self.y),
            (self.x + self.w, self.y),
            (self.x, self.y + self.h),
            (self.x + self.w, self.y + self.h),
        ]

    def contains(self, px, py) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def corner_hit(self, px, py) -> int | None:
        for i, (cx, cy) in enumerate(self.corners()):
            if abs(px - cx) <= CORNER_PX and abs(py - cy) <= CORNER_PX:
                return i
        return None


def load_yolo(path: Path, w: int, h: int) -> list[Box]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _cls, cx, cy, bw, bh = (float(v) for v in parts[1:5]) if False else (None, *[float(v) for v in parts[1:]])
        cx, cy, bw, bh = float(parts[1]) * w, float(parts[2]) * h, float(parts[3]) * w, float(parts[4]) * h
        boxes.append(Box(int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)))
    return boxes


def save_yolo(path: Path, boxes: list[Box], w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not boxes:
        path.write_text("")
        return
    lines = []
    for b in boxes:
        bx = max(0, min(b.x, w - 1))
        by = max(0, min(b.y, h - 1))
        bw = max(1, min(b.w, w - bx))
        bh = max(1, min(b.h, h - by))
        cx = (bx + bw / 2) / w
        cy = (by + bh / 2) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
    path.write_text("\n".join(lines) + "\n")


def normalize_box(b: Box) -> Box:
    x, y = b.x, b.y
    w, h = b.w, b.h
    if w < 0:
        x += w
        w = -w
    if h < 0:
        y += h
        h = -h
    return Box(x, y, max(1, w), max(1, h))


class EditorState:
    def __init__(self):
        self.boxes: list[Box] = []
        self.selected: int | None = None
        self.dragging = False
        self.draw_new = False
        self.drag_kind: str | None = None  # 'move' | 'corner-<i>' | 'new'
        self.drag_start = (0, 0)
        self.box_start: Box | None = None
        self.new_box: Box | None = None

    def reset_drag(self):
        self.dragging = False
        self.drag_kind = None
        self.box_start = None
        self.new_box = None

    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.draw_new:
                self.dragging = True
                self.drag_kind = "new"
                self.drag_start = (x, y)
                self.new_box = Box(x, y, 1, 1)
                return

            if self.selected is not None:
                b = self.boxes[self.selected]
                hit = b.corner_hit(x, y)
                if hit is not None:
                    self.dragging = True
                    self.drag_kind = f"corner-{hit}"
                    self.drag_start = (x, y)
                    self.box_start = Box(b.x, b.y, b.w, b.h)
                    return

            for i in range(len(self.boxes) - 1, -1, -1):
                if self.boxes[i].contains(x, y):
                    self.selected = i
                    self.dragging = True
                    self.drag_kind = "move"
                    self.drag_start = (x, y)
                    self.box_start = Box(self.boxes[i].x, self.boxes[i].y, self.boxes[i].w, self.boxes[i].h)
                    return

            self.selected = None

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]
            if self.drag_kind == "move" and self.selected is not None and self.box_start is not None:
                bs = self.box_start
                self.boxes[self.selected] = Box(bs.x + dx, bs.y + dy, bs.w, bs.h)
            elif self.drag_kind and self.drag_kind.startswith("corner-") and self.selected is not None and self.box_start is not None:
                idx = int(self.drag_kind.split("-")[1])
                bs = self.box_start
                x1, y1, x2, y2 = bs.x, bs.y, bs.x + bs.w, bs.y + bs.h
                if idx == 0:
                    x1, y1 = bs.x + dx, bs.y + dy
                elif idx == 1:
                    x2, y1 = bs.x + bs.w + dx, bs.y + dy
                elif idx == 2:
                    x1, y2 = bs.x + dx, bs.y + bs.h + dy
                elif idx == 3:
                    x2, y2 = bs.x + bs.w + dx, bs.y + bs.h + dy
                self.boxes[self.selected] = normalize_box(Box(x1, y1, x2 - x1, y2 - y1))
            elif self.drag_kind == "new" and self.new_box is not None:
                self.new_box = Box(self.drag_start[0], self.drag_start[1], dx, dy)

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            if self.drag_kind == "new" and self.new_box is not None:
                nb = normalize_box(self.new_box)
                if nb.w >= 4 and nb.h >= 4:
                    self.boxes.append(nb)
                    self.selected = len(self.boxes) - 1
                self.draw_new = False
            self.reset_drag()


def render(img, state: EditorState, info: str):
    out = img.copy()
    for i, b in enumerate(state.boxes):
        color = COLOR_SEL if i == state.selected else COLOR_BOX
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
        if i == state.selected:
            for cx, cy in b.corners():
                cv2.rectangle(out, (cx - 5, cy - 5), (cx + 5, cy + 5), color, -1)
    if state.new_box is not None:
        nb = normalize_box(state.new_box)
        cv2.rectangle(out, (nb.x, nb.y), (nb.x + nb.w, nb.y + nb.h), COLOR_NEW, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), TXT_BG, -1)
    cv2.putText(out, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TXT_FG, 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "gold"), default="gold")
    ap.add_argument("--clip-id", help="filter by stem prefix")
    args = ap.parse_args()

    img_dir = ANN_DIR / args.split / "images"
    lbl_dir = ANN_DIR / args.split / "labels"
    images = sorted(img_dir.glob("*.jpg"))
    if args.clip_id:
        images = [p for p in images if p.stem.startswith(args.clip_id)]
    if not images:
        raise SystemExit(f"no images in {img_dir}")

    cv2.namedWindow("editor", cv2.WINDOW_NORMAL)
    state = EditorState()
    cv2.setMouseCallback("editor", state.on_mouse)

    idx = 0
    current_path: Path | None = None
    img_shape = (0, 0)

    def load_frame(i: int):
        nonlocal current_path, img_shape
        path = images[i]
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        state.boxes = load_yolo(lbl_dir / f"{path.stem}.txt", w, h)
        state.selected = None
        state.reset_drag()
        state.draw_new = False
        current_path = path
        img_shape = (h, w)
        return img

    def save_current():
        if current_path is None:
            return
        h, w = img_shape
        save_yolo(lbl_dir / f"{current_path.stem}.txt", state.boxes, w, h)

    img = load_frame(idx)
    while True:
        mode = "DRAW" if state.draw_new else "EDIT"
        info = f"[{idx + 1}/{len(images)}] {current_path.stem}  boxes={len(state.boxes)}  sel={state.selected}  mode={mode}"
        cv2.imshow("editor", render(img, state, info))
        key = cv2.waitKey(20) & 0xFF

        if key == 255:
            continue
        if key in (ord("q"), 27):
            save_current()
            break
        if key == ord("s"):
            save_current()
            continue
        if key == ord("r"):
            img = load_frame(idx)
            continue
        if key == ord("c"):
            state.boxes = []
            state.selected = None
            continue
        if key == ord("d") and state.selected is not None:
            state.boxes.pop(state.selected)
            state.selected = None
            continue
        if key == ord("n"):
            state.draw_new = True
            continue
        if key in (ord("p"), 81):  # left arrow
            save_current()
            idx = max(0, idx - 1)
            img = load_frame(idx)
            continue
        if key in (ord(" "), 83, 13):  # space / right arrow / enter
            save_current()
            idx = min(len(images) - 1, idx + 1)
            img = load_frame(idx)
            continue
        if key == ord("g"):
            save_current()
            for j in range(len(images)):
                stem = images[j].stem
                lp = lbl_dir / f"{stem}.txt"
                if not lp.exists() or not lp.read_text().strip():
                    idx = j
                    break
            img = load_frame(idx)
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
