from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.data import read_manifest_csv, remap_path_columns, write_json
from larvae_cv.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR


TARGET_COLUMNS = ["weight", "length"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPU image regression model for larvae weight and length.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default="resnet18_multitarget")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RegressionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        transform: transforms.Compose,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = target_std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(row["single_image_path"]).convert("RGB")
        image = self.transform(image)
        target = row[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        target = (target - self.target_mean) / self.target_std
        return image, torch.tensor(target, dtype=torch.float32)


def build_model() -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(TARGET_COLUMNS))
    return model


def denormalize(pred: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return pred * std + mean


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    loss_fn = nn.SmoothL1Loss()
    losses: list[float] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            losses.append(float(loss.item()))
            all_true.append(targets.cpu().numpy())
            all_pred.append(outputs.cpu().numpy())

    y_true = denormalize(np.concatenate(all_true, axis=0), target_mean, target_std)
    y_pred = denormalize(np.concatenate(all_pred, axis=0), target_mean, target_std)

    metrics: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []
    for idx, column in enumerate(TARGET_COLUMNS):
        metrics[column] = {
            "mae": float(mean_absolute_error(y_true[:, idx], y_pred[:, idx])),
            "rmse": float(root_mean_squared_error(y_true[:, idx], y_pred[:, idx])),
            "r2": float(r2_score(y_true[:, idx], y_pred[:, idx])),
        }
    metrics["avg_r2"] = float(np.mean([metrics[col]["r2"] for col in TARGET_COLUMNS]))
    metrics["loss"] = float(np.mean(losses))

    return float(np.mean(losses)), metrics, predictions


def gather_predictions(
    model: nn.Module,
    frame: pd.DataFrame,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    split_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    dataset = RegressionDataset(frame, transform, target_mean, target_std)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            all_true.append(targets.numpy())
            all_pred.append(outputs.cpu().numpy())

    y_true = denormalize(np.concatenate(all_true, axis=0), target_mean, target_std)
    y_pred = denormalize(np.concatenate(all_pred, axis=0), target_mean, target_std)

    metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for idx, column in enumerate(TARGET_COLUMNS):
        metrics[column] = {
            "mae": float(mean_absolute_error(y_true[:, idx], y_pred[:, idx])),
            "rmse": float(root_mean_squared_error(y_true[:, idx], y_pred[:, idx])),
            "r2": float(r2_score(y_true[:, idx], y_pred[:, idx])),
        }

    metrics["avg_r2"] = float(np.mean([metrics[col]["r2"] for col in TARGET_COLUMNS]))

    for image_id, true_values, pred_values in zip(frame["image_id"], y_true, y_pred):
        rows.append(
            {
                "split": split_name,
                "image_id": image_id,
                "weight_true": float(true_values[0]),
                "weight_pred": float(pred_values[0]),
                "length_true": float(true_values[1]),
                "length_pred": float(pred_values[1]),
            }
        )
    return metrics, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA tidak tersedia di environment ini.")

    device = torch.device(args.device if args.device != "cuda" else "cuda")
    manifest = read_manifest_csv(PROCESSED_DIR / "regression_manifest.csv")
    manifest = remap_path_columns(manifest, ["single_image_path"])
    manifest = manifest.loc[manifest["single_image_path"].notna()].copy()
    manifest = manifest.loc[manifest["single_image_path"].map(lambda value: Path(value).exists())].copy()

    train_frame = manifest.loc[manifest["split"] == "train"].copy()
    val_frame = manifest.loc[manifest["split"] == "val"].copy()
    test_frame = manifest.loc[manifest["split"] == "test"].copy()

    target_mean = train_frame[TARGET_COLUMNS].mean().to_numpy(dtype=np.float32)
    target_std = train_frame[TARGET_COLUMNS].std().replace(0, 1.0).to_numpy(dtype=np.float32)

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    train_dataset = RegressionDataset(train_frame, train_transform, target_mean, target_std)
    val_dataset = RegressionDataset(val_frame, eval_transform, target_mean, target_std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        _, val_metrics, _ = evaluate_model(model, val_loader, device, target_mean, target_std)
        train_loss = float(np.mean(train_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_avg_r2": val_metrics["avg_r2"],
                "val_weight_r2": val_metrics["weight"]["r2"],
                "val_length_r2": val_metrics["length"]["r2"],
            }
        )
        print(json.dumps(history[-1]))

        if val_metrics["avg_r2"] > best_score:
            best_score = val_metrics["avg_r2"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    val_metrics, val_predictions = gather_predictions(
        model,
        val_frame,
        eval_transform,
        args.batch_size,
        args.num_workers,
        device,
        target_mean,
        target_std,
        "val",
    )
    test_metrics, test_predictions = gather_predictions(
        model,
        test_frame,
        eval_transform,
        args.batch_size,
        args.num_workers,
        device,
        target_mean,
        target_std,
        "test",
    )

    output_dir = REPORTS_DIR / "regression_gpu" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{args.output_name}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
            "image_size": args.image_size,
            "weights": str(models.ResNet18_Weights.DEFAULT),
        },
        model_path,
    )

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.concat([val_predictions, test_predictions], ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)

    summary = {
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model_path": str(model_path.resolve()),
        "val": val_metrics,
        "test": test_metrics,
    }
    write_json(output_dir / "metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
