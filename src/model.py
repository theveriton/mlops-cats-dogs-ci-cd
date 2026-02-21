from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def predict_probabilities(
    model: nn.Module, image_tensor: torch.Tensor, labels: List[str]
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).tolist()
    return {label: float(prob) for label, prob in zip(labels, probabilities)}


def save_checkpoint(model: nn.Module, labels: List[str], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    labels_path = output_dir / "labels.json"

    torch.save(model.state_dict(), model_path)
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return model_path, labels_path


def load_checkpoint(model_path: Path, labels_path: Path, device: torch.device) -> Tuple[nn.Module, List[str]]:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    model = SimpleCNN(num_classes=len(labels))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, labels
