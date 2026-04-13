import torch
from PIL import Image

from model import build_transforms


def test_inference_transform_shape():
    t = build_transforms(image_size=224)
    dummy = Image.fromarray((torch.zeros(3, 300, 300).permute(1, 2, 0).numpy() * 255).astype("uint8"))
    out = t(dummy)
    assert out.shape == (3, 224, 224)
