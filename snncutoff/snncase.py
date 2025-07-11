import torch
import torch.nn as nn
from snncutoff.API import get_loss, get_regularizer_loss
from snncutoff.utils import  OutputHook, sethook
from snncutoff.neuron.RF import RF
import numpy as np
import torch.nn.functional as F


class SNNCASE:
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        args: dict
    ) -> None:
        self.criterion = criterion
        self.snn_loss = get_loss(args.loss.name,method=args.snn_settings.method)(criterion, args.loss.means,args.loss.lamb)
        self.compute_reg_loss = get_regularizer_loss(args.regularizer.name,method=args.snn_settings.method)(args.regularizer).compute_reg_loss
        self.add_time_dim = args.snn_settings.add_time_dim
        self.T = args.neuron['T']
        self.alpha = args.snn_settings.alpha
        self.net = net
        self.loss_reg = torch.tensor([0.0],device=net.device)
        self.init = False
        self.RF_NEURON = None
        n_fft = 16  # Length of FFT
        num_lambda = 8 #m=3 and m=4
        self.n_fft=n_fft
        self.num_lambda=num_lambda
        # if not self.init:
        # F_range = np.linspace(1, -1, n_fft)  # Define F in (-1, 1)
        # alpha = 3
        # freq_list = self.generate_frequencies(F_range, alpha)
        # freq_list = torch.from_numpy(freq_list).to('cuda')
        # a_range = np.linspace(0.01, 0.5, num_lambda)  # Define F in (-1, 1) it was 0.9, 0.8
        # alpha = 1 # it was 2
        # lambda_list = self.generate_lambda(a_range, alpha)
        # lambda_list = torch.from_numpy(lambda_list).to('cuda')
        # self.RF_NEURON = RF(vthr=2, alpha=lambda_list, period=1/(freq_list)).to('cuda')
        # self.init = False



    def stft(self, x):
        # from snncutoff.neuron import RF

        n_fft = 16  # Length of FFT
        hop_length = n_fft // 2  # 50% overlap
        window = torch.hann_window(n_fft).cuda()  # Hann window
        # RF_NEURON = RF(vthr=1.0,alpha=0.07,period=1e2)
        # dt = 1e-6
        complex_signal = x[:,0] + 1j * x[:,1]

        stft_result = torch.stft(complex_signal, n_fft=n_fft, hop_length=hop_length, 
                         window=window, return_complex=True)
        # print(stft_result.shape)
        # amp = stft_result.abs()
        # angle = stft_result.angle()
        out = torch.stack([stft_result.real,stft_result.imag],dim=2)
        return out.transpose(1,3)


    def preprocess(self,x):
        return x.transpose(0,1)

    def _quantize(self, x):
        """
        Quantize a tensor into discrete levels.
        
        Args:
            values (torch.Tensor): Input tensor to be quantized.
            levels (int): Number of quantization levels.

        Returns:
            torch.Tensor: Quantized values.
        """
        # if x.ndim == 2:  # Shape is [L, 2]
        #     x = x.unsqueeze(0)  # Add batch dimension [1, L, 2]
        # x = x[:,:120]
        B,L, _ = x.shape
        T = self.T
        m=4
        levels = 2**m
        min_val = -1
        max_val = 1
        quantized = torch.round((x - min_val) / (max_val-min_val)*levels).clamp(0,levels-1)  #+ min_val
        quantized = quantized[...,0]+levels*quantized[...,1]
        L_slice = L // T     # Columns without padding
        remainder = L % T  # Find leftover elements
        # PADDING = 0 if L % T == 0 else 1  # Add 1 column if padding is required
        L_slice = L_slice + (1 if remainder > 0 else 0)
        total_L = T * L_slice
        padded_x = torch.ones((B,total_L),device=quantized.device)*(quantized[:,-2:-1])
        padded_x[:,:L] = quantized  # Copy original signal into padded version
        quantized = padded_x
        indices = quantized.to(int).reshape(B, self.T, L_slice)
        
        iq_grid = torch.zeros((B,self.T, levels ** 2), device=x.device)
        iq_grid.scatter_add_(
            dim=2,
            index=indices,
            src=torch.ones_like(indices, dtype=torch.float, device=x.device)
        )
        return iq_grid.reshape(B,self.T,1,levels,levels)
    
    def quantize(self, x):
        """
        Quantize a tensor into discrete levels.
        
        Args:
            values (torch.Tensor): Input tensor to be quantized.
            levels (int): Number of quantization levels.

        Returns:
            torch.Tensor: Quantized values.
        """
        # if x.ndim == 2:  # Shape is [L, 2]
        #     x = x.unsqueeze(0)  # Add batch dimension [1, L, 2]
        # x = x[:,:120]
        B,L, _ = x.shape
        m = 4
        levels = 2**m
        min_val = -1
        max_val = 1
        quantized = torch.round((x - min_val) / (max_val-min_val)*levels).clamp(0,levels-1)  #+ min_val
        mask = torch.nn.functional.one_hot(quantized.long(), num_classes=levels) 
        x = mask*x.unsqueeze(-1)
        return x.transpose(-2,-1)
    

    def generate_frequencies(self, F_range, alpha):
        """
        Generate frequency mapping based on the generalized equation.

        Parameters:
        - F_range: ndarray, input frequency range (e.g., np.linspace(-1, 1, num_points))
        - alpha: float, non-linearity parameter (alpha > 0)

        Returns:
        - f: ndarray, mapped frequencies
        """
        # Generalized frequency mapping
        f = 0.5 * np.sign(F_range) * np.abs(F_range) ** alpha
        return f

    def generate_lambda(self, F_range, alpha):
        """
        Generate frequency mapping based on the generalized equation.

        Parameters:
        - F_range: ndarray, input frequency range (e.g., np.linspace(-1, 1, num_points))
        - alpha: float, non-linearity parameter (alpha > 0)

        Returns:
        - f: ndarray, mapped frequencies
        """
        # Generalized frequency mapping
        f = np.sign(F_range) * np.abs(F_range) ** alpha
        return f
    
    def _forward(self,x,y):
        x = self.preprocess(x)
        x = self.net(x)
        return self.snn_loss(x,y)

    def _forward_regularization(self, x, y): 
        x = self.preprocess(x)
        output_hook =  OutputHook(output_type='reg_loss')
        self.net = sethook(output_hook, output_type='reg_loss')(self.net)
        x = self.net(x)
        cs_mean = torch.stack(output_hook,dim=2).flatten(0, 1).contiguous() 
        loss_reg = self.compute_reg_loss(x,y,cs_mean)
        self.loss_reg = loss_reg
        return self.snn_loss(x,y)[0], self.snn_loss(x,y)[1]+ loss_reg
    
    # def fft_reg_forward(self,x,y):
    #     x = self.preprocess(x)
    #     output_hook =  OutputHook(output_type='reg_loss')
    #     self.net = sethook(output_hook, output_type='reg_loss')(self.net)
    #     x = self.net(x)
    #     loss_reg = output_hook[0]
    #     self.loss_reg = loss_reg
    #     return self.snn_loss(x,y)[0], self.snn_loss(x,y)[1]+ self.alpha*loss_reg

    def forward(self, x, y, regularization):
        if regularization:
            return self._forward_regularization(x,y)
        else:
            return self._forward(x,y)

        # return self.fft_reg_forward(x,y)

    def get_loss_reg(self):
        return self.loss_reg

    def remove_hook(self):
        output_hook =  OutputHook(output_type='reg_loss')
        self.net  = sethook(output_hook,output_type='reg_loss')(self.net ,remove=True)