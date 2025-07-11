import torch
import torch.nn as nn
from torch.autograd import Function
from snncutoff.gradients import PWG
from typing import Type
from snncutoff.gradients.zif import ZIF
import math

# default values for time constants
DEFAULT_ALIF_TAU_M = 20.
DEFAULT_ALIF_TAU_ADP = 20.

# base threshold
DEFAULT_ALIF_THETA = 0.01

DEFAULT_ALIF_BETA = 1.8

DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN = 20.
DEFAULT_ALIF_ADAPTIVE_TAU_M_STD = 5.
# rho parameter initialization
DEFAULT_ALIF_ADAPTIVE_TAU_ADP_MEAN = 150.
DEFAULT_ALIF_ADAPTIVE_TAU_ADP_STD = 10.


def quantize_tensor(tensor: torch.Tensor, f: int) -> torch.Tensor:
    # Quantization formula: tensor_q = round(2^f * tensor) * 2^(-f)
    return torch.round(2**f * tensor) * 0.5**f

decay = 0.1  # neuron decay rate
lens = 0.5  # hyper-parameters of approximate function
scale = 6.
hight = 0.15
gamma = 0.5
gradient_type = 'MG'

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma
# define approximate firing function

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
  
        if gradient_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        elif gradient_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif gradient_type =='linear':
            temp = F.relu(1-input.abs())
        elif gradient_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        return grad_input * temp.float() * gamma


class LIF(nn.Module):
    def __init__(self, T: int=4, 
                 vthr: float = 1.0, 
                 delta: float = 0.0, 
                 surogate: Type[Function] = PWG,
                 mem_init: float = 0.5,
                 multistep: bool = True,
                 reset_mode: str ='hard',
                 num_bit: int=16,
                 adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN,
                 adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD,
                 tau_mem: float = DEFAULT_ALIF_TAU_M,  # time constant for alpha
                 adaptive_tau_mem: bool = True,
                 adaptive_tau_adp: bool = True,
                 tau_adp: float = DEFAULT_ALIF_TAU_ADP,  # time constant for rho
                 adaptive_tau_adp_mean: float=DEFAULT_ALIF_ADAPTIVE_TAU_ADP_MEAN,
                 adaptive_tau_adp_std: float=DEFAULT_ALIF_ADAPTIVE_TAU_ADP_STD,
                 recurrent: bool=False,
                 **kwargs):
        
        super(LIF, self).__init__()
        
        """
        Initialize the multi-level or graded LIF neuron model.

        Args:
            vthr (float): The threshold voltage for spike generation.
            delta (float): The time constant of the membrane potential decay.
        """

        self.t = 0.0
        self.T = T
        self.vmem = 0.0
        self.delta = delta
        self.reset_mode = reset_mode
        self.mem_init = mem_init
        self.multistep = multistep
        self.num_bit = num_bit
        self.surogate = surogate.apply
        self.vthr = vthr
        self.adaptive_tau_mem = adaptive_tau_mem
        self.adaptive_tau_mem_mean = adaptive_tau_mem_mean
        self.adaptive_tau_mem_std = adaptive_tau_mem_std
        self.bit_precision = 32
        self.recurrent = recurrent
        self.gamma = 0.9
        self.tau_mem_init = tau_mem
        self.tau_adp_init = tau_adp
        self.adaptive_tau_adp = adaptive_tau_adp
        self.adaptive_tau_adp_mean = adaptive_tau_adp_mean
        self.adaptive_tau_adp_std = adaptive_tau_adp_std
        # hidden_size = 256
        # self.params_init(hidden_size)

    def params_init(self, hidden_size, device):
        tau_mem_init = self.tau_mem_init * torch.ones(hidden_size).to(device)

        if self.adaptive_tau_mem:
            tau_mem = torch.nn.Parameter(tau_mem_init)
            self.register_parameter('tau_mem',tau_mem)
            torch.nn.init.normal_(self.tau_mem, mean=self.adaptive_tau_mem_mean, std=self.adaptive_tau_mem_std)
        else:
            self.register_buffer("tau_mem", tau_mem_init)
        
        if self.recurrent: 
            # self.w = nn.Linear(hidden_size[0],hidden_size[0],bias=False).to(device)
            # torch.nn.init.xavier_uniform_(self.w.weight)
            self.w = nn.Linear(hidden_size[0],hidden_size[0]).to(device)
            torch.nn.init.orthogonal_(self.w.weight)
            torch.nn.init.constant_(self.w.bias,0)
    def neuron_update(self,
            x: torch.Tensor,
            z: torch.Tensor,
            u: torch.Tensor,
            # alpha: torch.Tensor,
            # rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        alpha = torch.exp(-1. * 1. / torch.abs(self.tau_mem))

        # membrane potential update
        u = u*(alpha) + x*(1.0 - alpha)-z*self.vthr

        # generate spike.
        # z = StepDoubleGaussianGrad.apply(u - theta_t)
        z = ActFun_adp.apply(u - self.vthr)

        # reset membrane potential.
        # soft reset (keeps remaining membrane potential)
        return z, u,  self.vthr


    def forward(self,x):
        if not hasattr(self, 'tau_mem'):
            self.params_init(x.shape[2:],x.device)
        # divergence boundary
        spike_post = []
        mem_post = []
        self.reset()
        u = torch.rand_like(x[0])#*0.0
        spike = torch.zeros_like(x[0])

        for t in range(x.shape[0]):
            if self.recurrent:
                r_x = self.w(spike)
                spike, u, theta_t = self.neuron_update(x[t]+r_x,
                                            z=spike,
                                            u=u,
                                            )
            else:
                spike, u, theta_t = self.neuron_update(x[t],
                                            z=spike,
                                            u=u,
                                            )
            mem = torch.clamp(u/theta_t,min=0)
            # mem_mask = torch.ones_like(u)
            # N = 20
            # mem_mask[:N] = mem_mask[:N]*0.0
            # mem = mem*mem_mask
            spike_post.append(spike)
            mem_post.append(mem)
        spike_post = torch.stack(spike_post,dim=0)
        mem_post = torch.stack(mem_post,dim=0)
        return spike_post, mem_post


    def reset(self):
        """
        Reset the membrane potential and time step to initial values.
        """
        self.t = 0.0
        self.real = 0.0
        self.imag = 0.0

    def initMem(self, x: complex):
        """
        Initialize the membrane potential with a given value.

        Args:
            x (float): The initial membrane potential.
        """
        self.vmem = x

    def updateMem(self, x: complex):
        """
        Update the membrane potential based on the input and time constant.

        Args:
            x (float): The input value to update the membrane potential.
        """
        self.vmem = x * self.delta
        self.t += 1

    def is_spike(self) -> bool:
        """
        Check if the membrane potential has reached the threshold.

        Returns:
            bool: True if the membrane potential has reached or exceeded the threshold, False otherwise.
        """
        return self.vmem >= self.vthr

    def vmem_reset(self, x, spike):
        if self.reset_mode == 'hard':
            return x * (1-spike)
        elif self.reset_mode == 'soft':  
            return x - self.vthr*spike
