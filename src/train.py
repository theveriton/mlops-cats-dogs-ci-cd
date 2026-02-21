from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.data import build_preprocess_transform
from src.model import SimpleCNN, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Cats vs Dogs baseline CNN")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--experiment", type=str, default="cats-vs-dogs-baseline")
    parser.add_argument("--run-name", type=str, default="baseline-cnn")
    return parser.parse_args()


def make_loader(split_dir: Path, batch_size: int, train: bool) -> tuple[DataLoader, List[str]]:
    dataset = datasets.ImageFolder(split_dir, transform=build_preprocess_transform(train=train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader, dataset.classes


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total = 0
    correct = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y_batch.cpu().numpy().tolist())

    accuracy = correct / total if total else 0.0
    return accuracy, np.array(all_preds), np.array(all_labels)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, classes = make_loader(args.data_dir / "train", args.batch_size, train=True)
    val_loader, _ = make_loader(args.data_dir / "val", args.batch_size, train=False)

    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    mlflow.set_experiment(args.experiment)
    train_losses: list[float] = []

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "classes": classes,
            }
        )

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / max(len(train_loader), 1)
            train_losses.append(avg_loss)

            val_acc, _, _ = evaluate(model, val_loader, device)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        model_path, labels_path = save_checkpoint(model, classes, args.output_dir)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(labels_path))

        val_acc, preds, y_true = evaluate(model, val_loader, device)
        cm = confusion_matrix(y_true, preds)
        cm_fig = ConfusionMatrixDisplay(cm, display_labels=classes).plot().figure_
        cm_path = args.output_dir / "confusion_matrix.png"
        cm_fig.savefig(cm_path)
        plt.close(cm_fig)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_metric("final_val_accuracy", val_acc)

        loss_fig = plt.figure()
        plt.plot(train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        loss_path = args.output_dir / "loss_curve.png"
        loss_fig.savefig(loss_path)
        plt.close(loss_fig)
        mlflow.log_artifact(str(loss_path))


if __name__ == "__main__":
    train(parse_args())
