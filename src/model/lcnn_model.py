from torch import nn
from torch.nn import Sequential
import torch


class MFM(nn.Module):
    def __init__(self, dim=1):
        self.dim = dim
        super().__init__()

    def forward(self, x):
        if x.shape[self.dim] % 2 != 0:
            raise ValueError("MFM require even dimension!")
        first_part, second_part = torch.chunk(x, chunks=2, dim=self.dim)
        return torch.maximum(first_part, second_part)


class LCNNModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self):
        """
        Args:
        """
        super().__init__()

        self.net = Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=1),
            MFM(dim=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            MFM(dim=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            MFM(dim=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0),
            MFM(dim=1),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            MFM(dim=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            MFM(dim=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            MFM(dim=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MFM(dim=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            MFM(dim=1),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(53 * 37 * 32, 160),
            MFM(dim=1),
            nn.Dropout(0.75),
            nn.BatchNorm1d(80),
            nn.Linear(80, 2)







        )

    def forward(self, data_object, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.net(data_object)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
