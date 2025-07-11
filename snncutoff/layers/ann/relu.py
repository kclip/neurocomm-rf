import torch
from torch import nn
from typing import Callable, List, Type

class ReLU(nn.Module):
    def __init__(self,  
                 regularizer: Type[nn.Module] = None, 
                 neuron_params: dict = {'vthr': 8.0, 
                                        },
                 momentum=0.9):
        super().__init__()
        self.vthr = nn.Parameter(torch.tensor([neuron_params['vthr']]), requires_grad=False)
        self.regularizer = regularizer
        self.momentum = momentum
        self.relu = nn.ReLU(inplace=True)

    def _forward(self,x):
        if self.training:
            vthr = (1-self.momentum)*torch.max(x.detach())+self.momentum*self.vthr
            self.vthr.copy_(vthr)
        x = self.relu(x)
        return x

    def forward(self, x):
        if self.regularizer is not None:
            loss = self.regularizer(x.clone())
        x = self._forward(x)
        return x 