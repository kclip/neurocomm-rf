import torch
import torch.nn as nn
from torch.autograd import Function
from snncutoff.gradients import PWG
from typing import Type
from snncutoff.gradients.zif import ZIF
import math
import numpy as np
from scipy import special as ss
from scipy import signal


# def transition(measure, N, **measure_args):
#     # Laguerre (translated)
#     if measure == 'lagt':
#         b = measure_args.get('beta', 1.0)
#         A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
#         B = b * np.ones((N, 1))
#     # Legendre (translated)
#     elif measure == 'legt':
#         Q = np.arange(N, dtype=np.float64)
#         R = (2*Q + 1) ** .5
#         j, i = np.meshgrid(Q, Q)
#         A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
#         B = R[:, None]
#         A = -A
#     # Legendre (scaled)
#     elif measure == 'legs':
#         q = np.arange(N, dtype=np.float64)
#         col, row = np.meshgrid(q, q)
#         r = 2 * q + 1
#         M = -(np.where(row >= col, r, 0) - np.diag(q))
#         T = np.sqrt(np.diag(2 * q + 1))
#         A = T @ M @ np.linalg.inv(T)
#         B = np.diag(T)[:, None]
#         B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
#     elif measure == 'fourier':
#         freqs = np.arange(N//2)
#         d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
#         A = 2*np.pi*(-np.diag(d, 1) + np.diag(d, -1))
#         B = np.zeros(N)
#         B[0::2] = 2
#         B[0] = 2**.5
#         A = A - B[:, None] * B[None, :]
#         # A = A - np.eye(N)
#         B *= 2**.5
#         B = B[:, None]

#     return A, B


# def measure(method, c=0.0):
#     if method == 'legt':
#         fn = lambda x: np.heaviside(x, 0.0) * np.heaviside(1.0-x, 0.0)
#     elif method == 'legs':
#         fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
#     elif method == 'lagt':
#         fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
#     elif method in ['fourier']:
#         fn = lambda x: np.heaviside(x, 1.0) * np.heaviside(1.0-x, 1.0)
#     else: raise NotImplementedError
#     fn_tilted = lambda x: np.exp(c*x) * fn(x)
#     return fn_tilted

# def basis(method, N, vals, c=0.0, truncate_measure=True):
#     """
#     vals: list of times (forward in time)
#     returns: shape (T, N) where T is length of vals
#     """
#     if method == 'legt':
#         eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 2*vals-1).T
#         eval_matrix *= (2*np.arange(N)+1)**.5 * (-1)**np.arange(N)
#     elif method == 'legs':
#         _vals = np.exp(-vals)
#         eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 1-2*_vals).T # (L, N)
#         eval_matrix *= (2*np.arange(N)+1)**.5 * (-1)**np.arange(N)
#     elif method == 'lagt':
#         vals = vals[::-1]
#         eval_matrix = ss.eval_genlaguerre(np.arange(N)[:, None], 0, vals)
#         eval_matrix = eval_matrix * np.exp(-vals / 2)
#         eval_matrix = eval_matrix.T
#     elif method == 'fourier':
#         cos = 2**.5 * np.cos(2*np.pi*np.arange(N//2)[:, None]*(vals)) # (N/2, T/dt)
#         sin = 2**.5 * np.sin(2*np.pi*np.arange(N//2)[:, None]*(vals)) # (N/2, T/dt)
#         cos[0] /= 2**.5
#         eval_matrix = np.stack([cos.T, sin.T], axis=-1).reshape(-1, N) # (T/dt, N)
# #     print("eval_matrix shape", eval_matrix.shape)

#     if truncate_measure:
#         eval_matrix[measure(method)(vals) == 0.0] = 0.0

#     p = torch.tensor(eval_matrix)
#     p *= np.exp(-c*vals)[:, None] # [::-1, None]
#     return p



DEFAULT_MASK_PROB = 0.

TRAIN_B_offset = True
DEFAULT_RF_B_offset = 1.

# here: Depends on the initialization
DEFAULT_RF_ADAPTIVE_B_offset_a = 2.
DEFAULT_RF_ADAPTIVE_B_offset_b = 3.

TRAIN_OMEGA = True
DEFAULT_RF_OMEGA = 10.

# here: Depends on the initialization
DEFAULT_RF_ADAPTIVE_OMEGA_a = 5.
DEFAULT_RF_ADAPTIVE_OMEGA_b = 10.

DEFAULT_RF_THETA = 1.0  # 1.0 # * 0.1

DEFAULT_DT = 0.01
FACTOR = 1 / (DEFAULT_DT * 2)

@torch.jit.script
def step(x: torch.Tensor) -> torch.Tensor:
    #
    # x.gt(0.0).float()
    # is slightly faster (but less readable) than
    # torch.where(x > 0.0, 1.0, 0.0)
    #
    return x.gt(0.0).float()

@torch.jit.script
def gaussian(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    return (1 / (sigma * torch.sqrt(2 * torch.tensor(math.pi)))) * torch.exp(
        -((x - mu) ** 2) / (2.0 * (sigma ** 2))
    )


class StepDoubleGaussianGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors

        p = 0.15
        scale = 6.
        len = 0.5

        sigma1 = len
        sigma2 = scale * len

        gamma = 0.5
        dfd = (1. + p) * gaussian(x, mu=0., sigma=sigma1) - 2. * p * gaussian(x, mu=0., sigma=sigma2)

        return grad_output * dfd * gamma


def sustain_osc(omega: torch.Tensor, dt: float = DEFAULT_DT) -> torch.Tensor:
    return (-1 + torch.sqrt(1 - torch.square(dt * omega))) / dt




class BRF(nn.Module):
    def __init__(self, T: int=4, 
                 vthr: float = 1.0, 
                 delta: float = 0.0, 
                 surogate: Type[Function] = PWG,
                 mem_init: float = 0.5,
                 multistep: bool = True,
                 reset_mode: str ='hard',
                 num_bit: int=16,
                 b_offset: float = DEFAULT_RF_B_offset,
                 omega: float = DEFAULT_RF_OMEGA,
                #  alpha: float=1.0,
                 period: float=1.0e3,
                 delay: float=0,
                 adaptive_omega: bool=True,
                 adaptive_b_offset: bool=True,
                 recurrent: bool=False,
                 **kwargs):
        
        super(BRF, self).__init__()
        
        """
        Initialize the multi-level or graded LIF neuron model.

        Args:
            vthr (float): The threshold voltage for spike generation.
            delta (float): The time constant of the membrane potential decay.
        """

        self.t = 0.0
        self.T = T
        self.vmem = 0.0
        # self.vthr = vthr
        self.delta = delta
        self.reset_mode = reset_mode
        self.mem_init = mem_init
        self.multistep = multistep
        self.num_bit = num_bit
        self.surogate = surogate.apply
        self.vthr = vthr
        self.gradient = ZIF.apply
        self.real = 0.0
        self.imag = 0.0
        self.adaptive_omega = adaptive_omega 
        self.adaptive_b_offset = adaptive_b_offset
        self.omega_init = omega
        self.b_offset_init = b_offset
        self.dt = 0.01
        self.recurrent = recurrent
        self.gamma = 0.9

    def params_init(self, hidden_size, device):
        omega_init = self.omega_init * torch.ones(hidden_size).to(device)
        adaptive_omega_a = DEFAULT_RF_ADAPTIVE_OMEGA_a
        adaptive_omega_b = DEFAULT_RF_ADAPTIVE_OMEGA_b
        if self.adaptive_omega:
            omega = torch.nn.Parameter(omega_init)
            self.register_parameter('omega',omega)
            torch.nn.init.uniform_(self.omega, adaptive_omega_a, adaptive_omega_b)

        adaptive_b_offset_a = DEFAULT_RF_ADAPTIVE_B_offset_a
        adaptive_b_offset_b = DEFAULT_RF_ADAPTIVE_B_offset_b

        b_offset_init = self.b_offset_init *  torch.ones(hidden_size).to(device)
        if self.adaptive_b_offset:
            b_offset = torch.nn.Parameter(b_offset_init)
            self.register_parameter('b_offset',b_offset)
            torch.nn.init.uniform_(self.b_offset, adaptive_b_offset_a, adaptive_b_offset_b)
        
        if self.recurrent: 
            self.w = nn.Linear(hidden_size[0],hidden_size[0],bias=False).to(device)
            torch.nn.init.xavier_uniform_(self.w.weight)


        # method = 'legs'
        # N = hidden_size
        # # self.dt = dt
        # # self.T = T
        # # self.c = c
        # c = 0.0
        # A, B = transition(method, N)
        # A = A + np.eye(N)*c
        # self.A = A
        # self.B = B.squeeze(-1)
        # self.measure_fn = measure(method)

        # C = np.ones((1, N))
        # D = np.zeros((1,))
        # dA, dB, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        # dB = dB.squeeze(-1)

        # self.register_buffer('dA', torch.Tensor(dA)) # (N, N)
        # self.register_buffer('dB', torch.Tensor(dB)) # (N,)

        # self.vals = np.arange(0.0, T, dt)
        # self.eval_matrix = basis(self.method, self.N, self.vals, c=self.c) # (T/dt, N)
        # self.measure = measure(self.method)(self.vals)



    def brf_update(
            self,
            x: torch.Tensor,  # injected current: input x weight
            u: torch.Tensor,  # membrane potential (real part)
            v: torch.Tensor,  # membrane potential (complex part)
            q: torch.Tensor,  # refractory period
            b: torch.Tensor,  # attraction to resting state
            omega: torch.Tensor,  # eigen ang. frequency of the neuron
            dt: float = DEFAULT_DT,  # 0.01
            theta: float = DEFAULT_RF_THETA,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # membrane update u dim = (batch_size, hidden_size)
        if not torch.is_complex(x):
            x = x.to(torch.complex64)
        u_complex = torch.complex(u,v)
        u_complex = u_complex + u_complex.mul(torch.complex(b, omega)).mul(dt) + x.mul(dt)
        # z = functional.FGI_DGaussian(u_ - theta - q)
        # u_ = u + b * u * dt - omega * v * dt + x * dt
        # v = v + omega * u * dt + b * v * dt
        theta_t = theta + q
        z = StepDoubleGaussianGrad.apply(u_complex.real - theta_t)
        q = self.gamma*q + z
        return z, q, u_complex.real,u_complex.imag, theta_t


    def forward(self,x):
        if not hasattr(self, 'omega'):
            self.params_init(x.shape[2:],x.device)
        # divergence boundary
        spike_post = []
        mem_post = []
        self.reset()
        q   = 0.0
        u,v,  = torch.zeros_like(x[0],dtype=torch.float32), torch.zeros_like(x[0],dtype=torch.float32)
        spike = torch.zeros_like(x[0],dtype=torch.float32)
        for t in range(x.shape[0]):
            omega = torch.abs(self.omega)
            b_offset = torch.abs(self.b_offset)
            p_omega = sustain_osc(omega)

            b = p_omega - b_offset - q
            if self.recurrent:
                r_x = self.w(spike)
                spike, q, u, v, theta_t = self.brf_update( x[t]+r_x,
                                            u=u,
                                            v=v,
                                            q=q,
                                            b=b,
                                            omega=omega,
                                            )
            else:
                spike, q, u, v, theta_t = self.brf_update( x[t],
                                            u=u,
                                            v=v,
                                            q=q,
                                            b=b,
                                            omega=omega,
                                            )
            mem = torch.clamp(u/theta_t,min=0)
            spike_post.append(spike)
            mem_post.append(mem)
        spike_post = torch.stack(spike_post,dim=0)
        mem_post = torch.stack(mem_post,dim=0)
        return spike_post, mem_post


    def resonator_dynamics(self, real_current, imag_current, real_last, imag_last):
        """Resonate and Fire real and imaginary voltage dynamics

        Parameters
        ----------
        a_real_in_data : np.ndarray
            Real component input current
        a_imag_in_data : np.ndarray
            Imaginary component input current
        real : np.ndarray
            Real component voltage to be updated
        imag : np.ndarray
            Imag component voltage to be updated

        Returns
        -------
        np.ndarray, np.ndarray
            updated real and imaginary components

        """
        decayed_real = self.cos_decay * real_last \
            - self.sin_decay * imag_last \
            + real_current
        decayed_imag = self.sin_decay * real_last \
            + self.cos_decay * imag_last \
            + imag_current

        return decayed_real, decayed_imag



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
