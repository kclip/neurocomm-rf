import torch
from torch import nn
import numpy as np


class FFTSNN(nn.Module):
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
        """
        Forward pass for the RCSSNN module.

        Args:
            x (torch.Tensor): The input tensor.
            mem (torch.Tensor): The membrane potential tensor.

        Returns:
            torch.Tensor: The computed cosine similarity inverse.
        """

        # time, batch, freq_bins = x.shape
        # reshaped = x.reshape(batch * time, freq_bins)  # Merge batch & time
        # normed_output = reshaped / (torch.norm(reshaped, dim=0, keepdim=True) + 1e-6)  # Normalize frequency bins
        # cos_sim_matrix = torch.mm(normed_output.T, normed_output)  # Cosine similarity between frequency bins
        # I = torch.eye(freq_bins, device=x.device)  # Identity matrix
        # return torch.norm(cos_sim_matrix - I)  # Penalize off-diagonal elements
        # # fft_output = torch.fft.fft(mem, dim=0)  # FFT along time
        # diff = fft_output[1:,...] - fft_output[:-1, ...]  # Difference in frequency domain
        # return torch.mean(diff.abs() ** 2)  # Penalize frequency dr

        # return loss 
        return self.spectral_whitening_loss(mem)
    def frequency_decorrelation_loss(self, freq_output):
        """
        Ensures different frequency bins in the output are independent.
        freq_output shape: (batch, time_steps, freq_bins)
        """
        time, batch, freq_bins = freq_output.shape
        reshaped = freq_output.reshape(batch * time, freq_bins)  # Merge batch & time
        normed_output = reshaped / (torch.norm(reshaped, dim=0, keepdim=True) + 1e-6)  # Normalize frequency bins
        cos_sim_matrix = torch.mm(normed_output.T, normed_output)  # Cosine similarity between frequency bins
        I = torch.eye(freq_bins, device=freq_output.device)  # Identity matrix
        return torch.norm(cos_sim_matrix - I)  # Penalize off-diagonal elements

    def spectral_whitening_loss(self, freq_output):
        """
        Ensures that the covariance matrix of frequency bins is close to diagonal.
        """
        time, batch, freq_bins = freq_output.shape
        reshaped = freq_output.reshape(batch * time, freq_bins)  # Flatten batch & time dimensions
        centered_output = reshaped - reshaped.mean(dim=0, keepdim=True)  # Zero-mean normalization
        cov_matrix = torch.mm(centered_output.T, centered_output) / (batch * time)  # Compute covariance
        I = torch.eye(freq_bins, device=freq_output.device)  # Identity matrix
        return torch.norm(cov_matrix - I, p="fro")  # Frobenius norm loss



class FFTSNNLoss:
    def __init__(self, config, *args, **kwargs):
        """
        Initialize the RCSSNNLoss module.

        Args:
            config: Configuration object containing the rcs_n attribute.
        """
        super().__init__()
        self.rcs_n = config.rcs_n

    def compute_reg_loss(self, x: torch.Tensor, y: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the regularization loss.

        Args:
            x (torch.Tensor): The predictions.
            y (torch.Tensor): The target labels.
            features (torch.Tensor): The feature activations.

        Returns:
            torch.Tensor: The computed regularization loss.
        """
        # _target = torch.unsqueeze(y,dim=0)  # T N C 
        # right_predict_mask = x.max(-1)[1].eq(_target).to(torch.float32)
        # right_predict_mask = torch.unsqueeze(right_predict_mask,dim=2).flatten(0, 1).contiguous().detach()
        # features = features*right_predict_mask

        # features_max = features.max(dim=0)[0]
        # features_min = (features+(1-right_predict_mask)*1000.0).min(dim=0)[0] # find min, exclude zero value
        # features_min = features_min*features_min.lt(torch.tensor(1000.0)).to(torch.float32) # set wrong prediction to zero
        # loss = (features_max.detach()-features_min).pow(2).mean() #change pow into abs
        loss= features[...,0].mean()
        return loss

# Example usage:
# config = type('config', (object,), {'rcs_n': 0.3})()  # Example configuration object
# model = RCSSNN()
# loss_fn = RCSSNNLoss(config)
# x = torch.randn(10, 5)  # Example input tensor
# mem = torch.randn(10, 5)  # Example membrane potential tensor
# y = torch.randint(0, 5, (10,))  # Example target tensor
# features = torch.randn(10, 5)  # Example features tensor
# output = model(x, mem)
# loss = loss_fn.compute_reg_loss(output, y, features)
# print(loss)
