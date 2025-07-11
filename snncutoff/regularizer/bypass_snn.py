import torch
from torch import nn
import numpy as np


class BYPASSSNN(nn.Module):
    def __init__(self, beta: float = 0.3):
        """
        Initialize the RCSSNN module.

        Args:
            beta (float): The fraction of the input sequence to consider for the delayed response.
        """
        super().__init__()
        self.beta = beta
        self.add_loss = True

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        return x 