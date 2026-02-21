import torch

from src.model import predict_probabilities


class DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.2, 0.8]], dtype=torch.float32)


def test_predict_probabilities_returns_labeled_probs() -> None:
    model = DummyModel()
    sample = torch.zeros(3, 224, 224)
    labels = ["cat", "dog"]
    probs = predict_probabilities(model, sample, labels)

    assert set(probs.keys()) == {"cat", "dog"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["dog"] > probs["cat"]
