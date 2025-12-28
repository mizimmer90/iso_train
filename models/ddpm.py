"""
Denoising Diffusion Probabilistic Model (DDPM) implementation.

This module implements a DDPM for learning the score function of a 2D distribution.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import math


class SinusoidalPositionalEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for time steps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for time steps.
        
        Args:
            time: Tensor of shape (batch_size,) with time steps
            
        Returns:
            Tensor of shape (batch_size, dim) with embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class MLPScoreNetwork(nn.Module):
    """MLP-based score network for 2D DDPM."""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list = [128, 256, 256, 128],
        time_embed_dim: int = 128,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1
    ):
        """
        Initialize the score network.
        
        Args:
            input_dim: Dimension of input data (2 for 2D)
            hidden_dims: List of hidden layer dimensions
            time_embed_dim: Dimension of time embeddings
            activation: Activation function
            dropout: Dropout rate
        """
        super().__init__()
        
        self.time_embed_dim = time_embed_dim
        self.time_embed = SinusoidalPositionalEmbeddings(time_embed_dim)
        
        # Build MLP layers
        dims = [input_dim + time_embed_dim] + hidden_dims + [input_dim]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the score network.
        
        Args:
            x: Tensor of shape (batch_size, input_dim) with noisy data
            t: Tensor of shape (batch_size,) with time steps (normalized to [0, 1])
            
        Returns:
            Tensor of shape (batch_size, input_dim) with predicted scores
        """
        # Get time embeddings
        t_embed = self.time_embed(t * 1000)  # Scale time for better embeddings
        
        # Concatenate input and time embeddings
        x_t = torch.cat([x, t_embed], dim=-1)
        
        # Forward through network
        score = self.network(x_t)
        return score


class DDPM:
    """Denoising Diffusion Probabilistic Model."""
    
    def __init__(
        self,
        score_network: nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu"
    ):
        """
        Initialize DDPM.
        
        Args:
            score_network: Neural network for predicting scores
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting noise schedule value
            beta_end: Ending noise schedule value
            device: Device to run on
        """
        self.score_network = score_network.to(device)
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device),
            self.alphas_cumprod[:-1]
        ])
        
        # Precompute constants for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from q(x_t | x_0).
        
        Args:
            x_start: Tensor of shape (batch_size, 2) with clean data
            t: Tensor of shape (batch_size,) with timesteps
            noise: Optional noise tensor, if None will be sampled
            
        Returns:
            Tuple of (noisy samples, noise used)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = False
    ) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t) using the learned score.
        
        Args:
            x: Tensor of shape (batch_size, 2) with noisy data at timestep t
            t: Tensor of shape (batch_size,) with timesteps
            clip_denoised: Whether to clip denoised values (default: False for 2D landscapes)
            
        Returns:
            Tensor of shape (batch_size, 2) with denoised samples
        """
        # Convert to long for indexing
        t_index = t.long()
        
        # Normalize timestep to [0, 1] for the network
        t_normalized = t.float() / self.num_timesteps
        
        # Predict score (which is -noise / sqrt(1 - alpha_bar_t))
        predicted_score = self.score_network(x, t_normalized)
        
        # Compute predicted noise
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index].unsqueeze(1)
        predicted_noise = -predicted_score * sqrt_one_minus_alpha_cumprod_t
        
        # Compute x_0 prediction
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t_index].unsqueeze(1)
        pred_x_start = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
        
        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)
        
        # At t=0, just return predicted x_0
        nonzero_mask = (t_index != 0).float().unsqueeze(1)
        if nonzero_mask.sum() == 0:
            return pred_x_start
        
        # Compute mean of p(x_{t-1} | x_t, x_0)
        posterior_mean_coef1 = (
            torch.sqrt(self.alphas_cumprod_prev[t_index]) * self.betas[t_index]
        ).unsqueeze(1) / (1.0 - self.alphas_cumprod[t_index].unsqueeze(1))
        
        posterior_mean_coef2 = (
            torch.sqrt(self.alphas[t_index]) * (1.0 - self.alphas_cumprod_prev[t_index])
        ).unsqueeze(1) / (1.0 - self.alphas_cumprod[t_index].unsqueeze(1))
        
        posterior_mean = posterior_mean_coef1 * pred_x_start + posterior_mean_coef2 * x
        
        # Sample from posterior
        posterior_variance_t = self.posterior_variance[t_index].unsqueeze(1)
        posterior_variance_t = torch.clamp(posterior_variance_t, min=1e-20)
        
        noise = torch.randn_like(x)
        sample = posterior_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise
        
        return sample
    
    def p_sample_loop(
        self,
        shape: Tuple[int, ...],
        return_all_timesteps: bool = False
    ) -> torch.Tensor:
        """
        Generate samples by running the reverse diffusion process.
        
        Args:
            shape: Shape of samples to generate (batch_size, 2)
            return_all_timesteps: Whether to return all intermediate timesteps
            
        Returns:
            Generated samples, or list of all timesteps if return_all_timesteps=True
        """
        self.score_network.eval()
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        if return_all_timesteps:
            imgs = [x]
        
        # Reverse diffusion
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t)
            if return_all_timesteps:
                imgs.append(x)
        
        self.score_network.train()
        return torch.stack(imgs) if return_all_timesteps else x
    
    def loss(
        self,
        x_start: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the training loss.
        
        Args:
            x_start: Tensor of shape (batch_size, 2) with clean data
            noise: Optional noise tensor
            
        Returns:
            Scalar loss tensor
        """
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        
        # Sample noisy data
        x_t, noise = self.q_sample(x_start, t, noise)
        
        # Normalize timestep to [0, 1] for the network
        t_normalized = t.float() / self.num_timesteps
        
        # Predict score
        predicted_score = self.score_network(x_t, t_normalized)
        
        # Compute target score (which is -noise / sqrt(1 - alpha_bar_t))
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        target_score = -noise / (sqrt_one_minus_alpha_cumprod_t + 1e-8)
        
        # MSE loss on scores
        loss = nn.functional.mse_loss(predicted_score, target_score)
        
        return loss
    
    def get_score_at_t(
        self,
        x: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """
        Get the score function at a specific noise level t.
        
        Args:
            x: Tensor of shape (batch_size, 2) with data
            t: Timestep (integer from 0 to num_timesteps-1)
            
        Returns:
            Tensor of shape (batch_size, 2) with scores
        """
        self.score_network.eval()
        with torch.no_grad():
            t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            t_normalized = t_tensor.float() / self.num_timesteps
            score = self.score_network(x, t_normalized)
        self.score_network.train()
        return score

