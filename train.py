import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import build_model
from utils import env_flag, get_device, save_json, seed_everything


def build_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
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


def split_dataset(dataset, val_ratio: float = 0.2, seed: int = 42):
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=seed, stratify=dataset.targets)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def save_checkpoint(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-152 for DR classification.")
    parser.add_argument("--config", help="Path to YAML config. CLI args override.", default=None)
    parser.add_argument("--data-dir", help="ImageFolder root (one subfolder per class).")
    parser.add_argument("--output", default=None, help="Where to save the weights.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", help="Path to existing weights to continue training.", default=None)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Class folder names in order.",
    )
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (epochs). 0 disables.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (CUDA only).")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_config(config_path: str | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_config(args) -> Dict[str, Any]:
    cfg = load_config(args.config)
    # project
    seed = args.seed or cfg.get("project", {}).get("seed", 42)
    output = args.output or cfg.get("project", {}).get("output_weights", "classifier.pt")
    metrics_path = cfg.get("project", {}).get("metrics_path", "artifacts/metrics.json")

    # data
    data_root = args.data_dir or cfg.get("data", {}).get("root")
    if not data_root:
        raise SystemExit("Please supply --data-dir or set data.root in config.")
    classes = args.classes or cfg.get("data", {}).get(
        "classes", ["no_dr", "mild", "moderate", "severe", "proliferative"]
    )
    val_ratio = args.val_ratio or cfg.get("data", {}).get("val_ratio", 0.2)
    num_workers = args.num_workers or cfg.get("data", {}).get("num_workers", 2)

    # training
    epochs = args.epochs or cfg.get("training", {}).get("epochs", 10)
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 16)
    lr = args.lr or cfg.get("training", {}).get("lr", 1e-4)
    patience = args.patience
    if patience is None:
        patience = cfg.get("training", {}).get("patience", 0)

    # compute
    image_size = cfg.get("compute", {}).get("image_size", 224)
    amp = args.amp or cfg.get("compute", {}).get("amp", False) or env_flag("AMP", False)

    return {
        "seed": seed,
        "output": output,
        "metrics_path": metrics_path,
        "data_root": data_root,
        "classes": classes,
        "val_ratio": val_ratio,
        "num_workers": num_workers,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
        "image_size": image_size,
        "amp": amp,
        "resume": args.resume,
    }


def main():
    args = parse_args()
    cfg = resolve_config(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("Config: %s", cfg)

    seed_everything(cfg["seed"])
    device = get_device()
    logging.info("Using device: %s", device)

    transform_train = build_transforms(image_size=cfg["image_size"], train=True)
    transform_val = build_transforms(image_size=cfg["image_size"], train=False)

    dataset = datasets.ImageFolder(root=cfg["data_root"], transform=transform_train)
    if cfg["classes"]:
        dataset.classes = cfg["classes"]
        dataset.class_to_idx = {c: i for i, c in enumerate(cfg["classes"])}

    train_set, val_set = split_dataset(dataset, val_ratio=cfg["val_ratio"], seed=cfg["seed"])
    val_set.dataset.transform = transform_val

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
    )

    model = build_model(num_classes=len(dataset.classes)).to(device)
    if cfg["resume"] and os.path.isfile(cfg["resume"]):
        logging.info("Resuming from %s", cfg["resume"])
        model.load_state_dict(torch.load(cfg["resume"], map_location=device))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        logging.info("Epoch %s/%s", epoch, cfg["epochs"])
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logging.info(
            "train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f",
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(model, Path(cfg["output"]))
            logging.info("Saved best model -> %s", cfg["output"])

        if cfg["patience"] and epoch - best_epoch >= cfg["patience"]:
            logging.info("Early stopping triggered at epoch %s.", epoch)
            break

    save_json(
        {"best_val_acc": best_val_acc, "best_epoch": best_epoch, "history": history},
        Path(cfg["metrics_path"]),
    )
    logging.info("Done. Best val acc: %.4f (epoch %s)", best_val_acc, best_epoch)


if __name__ == "__main__":
    main()
