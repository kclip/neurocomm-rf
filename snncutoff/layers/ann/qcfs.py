import torch
from torch import nn
from torch.autograd import Function
from .relu import ReLU
from typing import Callable, List, Type
from snncutoff.gradients import GradFloor

class QCFS(ReLU):
    def __init__(self, 
                 regularizer: Type[nn.Module] = None, 
                 neuron_params: dict = {'vthr': 8.0, 
                                       'L': 4, 
                                        },
                 gradient: Type[Function] = GradFloor
                 ):
        super().__init__()
        self.vthr = nn.Parameter(torch.tensor([neuron_params['vthr']]), requires_grad=True)
        self.regularizer = regularizer
        self.L = neuron_params['L']
        self.gradient = gradient.apply
        self.relu = nn.ReLU(inplace=True)
        
    def _forward(self, x):
        x = self.relu(x)
        x = x / self.vthr
        x = self.gradient(x*self.L+0.5)/self.L
        x = torch.clamp(x, 0, 1)
        x = x * self.vthr
        return x
