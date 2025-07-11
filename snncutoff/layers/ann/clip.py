import torch
from torch import nn
from .relu import ReLU
from typing import Callable, List, Type


class Clip(ReLU):
    def __init__(self, 
                 regularizer: Type[nn.Module] = None, 
                 neuron_params: dict = {'vthr': 8.0, 
                                        },
                ):
        super().__init__()
        self.vthr = nn.Parameter(torch.tensor([neuron_params['vthr']]), requires_grad=True)
        self.regularizer = regularizer
        self.relu = nn.ReLU(inplace=True)

    def _forward(self, x):
        x = self.relu(x)
        x = x / self.vthr
        x = torch.clamp(x, 0, 1)
        x = x * self.vthr
        return x

