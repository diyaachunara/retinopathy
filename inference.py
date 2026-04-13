import argparse
from pathlib import Path

import torch
from PIL import UnidentifiedImageError

from model import DEFAULT_LABELS, load_model, preprocess_image


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for retinal images.")
    parser.add_argument("--weights", default="classifier.pt", help="Path to model weights.")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS, help="Class labels in order.")
    return parser.parse_args()


def iter_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if path.is_file():
        yield path
    else:
        for p in path.rglob("*"):
            if p.suffix.lower() in exts:
                yield p


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path=args.weights, device=device, num_classes=len(args.labels))

    for img_path in iter_images(Path(args.input)):
        try:
            batch = preprocess_image(str(img_path)).to(device)
        except (FileNotFoundError, UnidentifiedImageError):
            print(f"[skip] {img_path}")
            continue
        with torch.inference_mode():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            idx = int(probs.argmax())
            print(f"{img_path}: {args.labels[idx]} ({probs[idx].item()*100:.1f}%)")


if __name__ == "__main__":
    main()
