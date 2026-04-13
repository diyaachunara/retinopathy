import os
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import models, transforms

# Default class order aligned with the PyTorch alphabetical ordering from training
DEFAULT_LABELS = ["Mild", "Moderate", "No DR", "Proliferative DR", "Severe"]


def build_model(num_classes: int = 5) -> torch.nn.Module:
    """
    Create a ResNet backbone with the final layer adapted for the target classes.
    Using ResNet-18 here to be lighter on CPU / RAM.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(
    weights_path: str = "classifier.pt",
    device: torch.device | None = None,
    num_classes: int = 5,
) -> torch.nn.Module:
    """
    Load the model; if weights are missing, return an untrained network so the app can still start.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes)
    if os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    model.to(device).eval()
    return model


def build_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Preprocessing used for both training validation and inference.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def preprocess_image(path: str, image_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = build_transforms(image_size)
    return t(img).unsqueeze(0)


@torch.inference_mode()
def predict(
    image_path: str,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    labels: List[str] = DEFAULT_LABELS,
) -> Tuple[str, List[float]]:
    """
    Run a single-image prediction and return (label, probabilities).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model or load_model(device=device, num_classes=len(labels))
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    batch = preprocess_image(image_path).to(device)
    logits = model(batch)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    top_idx = int(probs.argmax().item())
    return labels[top_idx], probs.cpu().tolist()
