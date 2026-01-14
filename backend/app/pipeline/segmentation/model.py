import torch
import torch.nn as nn

class UNetWrapper(nn.Module):
    """
    Wrapper for pretrained segmentation model.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)
