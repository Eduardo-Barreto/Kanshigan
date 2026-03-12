import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_PATH = "/home/barreto/tcc/experiments/sam3-test/frame_000.jpg"
OUTPUT_DIR = Path("/home/barreto/tcc/experiments/sam3-test/output/prompts")

PROMPTS = [
    "robot",
    "sumo robot",
    "black box",
    "machine",
    "electronic device",
    "object on metal platform",
    "small box on circular platform",
    "dark object",
    "vehicle",
    "toy",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building SAM 3 image model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    from PIL import Image

    image = Image.open(IMAGE_PATH)
    inference_state = processor.set_image(image)
    img_np = np.array(image)

    for prompt in PROMPTS:
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        n = len(scores) if scores is not None else 0
        score_str = ""
        if n > 0:
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            score_str = f", scores={scores[:5]}"

        print(f"  '{prompt}': {n} detections{score_str}")

        # Save visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(img_np)
        ax.set_title(f"prompt: '{prompt}' ({n} detections)")
        ax.axis("off")

        if n > 0 and masks is not None:
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if masks.ndim == 4:
                masks = masks[:, 0]
            colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))
            for i in range(min(n, 5)):
                binary = masks[i] > 0
                colored = np.zeros((*binary.shape, 4))
                colored[binary] = [*colors[i][:3], 0.45]
                ax.imshow(colored)

        safe_name = prompt.replace(" ", "_")
        fig.savefig(OUTPUT_DIR / f"{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
