from torch import nn


class ExampleMetric(nn.Module):
    def __init__(self, metric, device, name):
        super().__init__()

        self.metric = metric.to(device)
        self.name = name

    def forward(self, logits, labels, **batch):
        predictions = logits.argmax(dim=1)

        return self.metric(
            predictions,
            labels,
        ).item()