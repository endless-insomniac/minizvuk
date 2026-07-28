import torch
from torch import nn


class ExampleLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "class_weights",
            torch.tensor(
                [0.1, 0.9],  # spoof, bonafided
                dtype=torch.float32,
            ),
        )

        self.loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=0.05,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **batch,
    ):
        return {
            "loss": self.loss(logits, labels)
        }