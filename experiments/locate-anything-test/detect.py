"""NVIDIA LocateAnything-3B open-vocab detection on Robot Sumo frames.

Tests whether an open-vocab grounding VLM can locate sumo robots on the dohyo,
to compare against the SAM3 text-prompt pipeline in experiments/sam3-test.
"""

import re
import sys
import time
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

MODEL_ID = "nvidia/LocateAnything-3B"
HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"

DETECT_TEMPLATE = "Locate all the instances that matches the following description: {desc}."

# Each entry is one detection prompt describing the target category set.
PROMPTS = {
    "robot": "robot",
    "sumo_robot": "sumo robot",
    "two_robots": "robot, robot",
    "robot_on_platform": "robot on a metal platform",
    "black_box_robot": "black robot, metal robot",
}

COLORS = ["#ff3232", "#32ff32", "#3296ff", "#ffff32", "#ff32ff", "#32ffff"]


def parse_boxes(answer: str, width: int, height: int) -> list[dict]:
    """Parse <box><x1><y1><x2><y2></box> (normalized 0-1000) to pixel coords."""
    boxes = []
    pattern = r"<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>"
    for m in re.finditer(pattern, answer):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        boxes.append(
            {
                "x1": x1 / 1000 * width,
                "y1": y1 / 1000 * height,
                "x2": x2 / 1000 * width,
                "y2": y2 / 1000 * height,
            }
        )
    return boxes


def build_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to("cuda")
        .eval()
    )
    return tokenizer, processor, model


def detect(tokenizer, processor, model, image: Image.Image, desc: str) -> tuple[str, list[dict]]:
    question = DETECT_TEMPLATE.format(desc=desc)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = processor.process_vision_info(messages)
    inputs = processor(
        text=[text], images=images, videos=videos, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        response = model.generate(
            pixel_values=inputs["pixel_values"].to(torch.bfloat16),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=tokenizer,
            max_new_tokens=4096,
            use_cache=True,
            generation_mode="hybrid",
            do_sample=False,
        )

    answer = response[0] if isinstance(response, (tuple, list)) else response
    answer = str(answer)
    w, h = image.size
    return answer, parse_boxes(answer, w, h)


def save_annotated(image: Image.Image, boxes: list[dict], title: str, path: Path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(image)
    ax.set_title(f"{title} ({len(boxes)} boxes)")
    ax.axis("off")
    for i, b in enumerate(boxes):
        rect = patches.Rectangle(
            (b["x1"], b["y1"]),
            b["x2"] - b["x1"],
            b["y2"] - b["y1"],
            linewidth=2.5,
            edgecolor=COLORS[i % len(COLORS)],
            facecolor="none",
        )
        ax.add_patch(rect)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = HERE / (sys.argv[1] if len(sys.argv) > 1 else "frame_000.jpg")
    image = Image.open(image_path).convert("RGB")
    print(f"Image: {image_path.name} {image.size}")

    print(f"Loading {MODEL_ID} (bf16)...")
    t0 = time.time()
    tokenizer, processor, model = build_model()
    print(f"  loaded in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(f"  VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    for name, desc in PROMPTS.items():
        t0 = time.time()
        answer, boxes = detect(tokenizer, processor, model, image, desc)
        dt = time.time() - t0
        print(f"\n[{name}] desc='{desc}' -> {len(boxes)} boxes in {dt:.1f}s")
        print(f"  raw: {answer[:300]}")
        save_annotated(
            image, boxes, f"{name}: '{desc}'", OUTPUT_DIR / f"{image_path.stem}_{name}.png"
        )

    print(f"\nSaved annotations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
