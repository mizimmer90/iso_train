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
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_diff = beta_end - beta_start
        
        # Precompute constants for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        """Continuous-time beta(t) for the VP-SDE."""
        return self.beta_start + t * self.beta_diff

    def _marginal_mean_std(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Marginal mean coefficient and std for VP-SDE at time t in [0, 1].

        x(t) = mean_coeff * x0 + std * N(0, I)
        """
        # Integral_0^t beta(s) ds = beta_start * t + 0.5 * beta_diff * t^2
        log_mean_coeff = -0.5 * (self.beta_start * t + 0.5 * self.beta_diff * t * t)
        mean_coeff = torch.exp(log_mean_coeff)
        std = torch.sqrt(torch.clamp(1.0 - mean_coeff ** 2, min=1e-12))
        return mean_coeff, std
    
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

    def sde_q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from the continuous-time VP-SDE marginal.

        Args:
            x_start: Clean data
            t: Continuous timesteps in [0, 1]
            noise: Optional noise

        Returns:
            Tuple of (noisy samples, noise used)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        mean_coeff, std = self._marginal_mean_std(t)
        mean_coeff = mean_coeff.unsqueeze(1)
        std = std.unsqueeze(1)
        x_t = mean_coeff * x_start + std * noise
        return x_t, noise
    
    def _prepare_time_grid(
        self,
        t_span: Tuple[float, float],
        num_steps: Optional[int],
        num_total_steps: Optional[int]
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Build time grid and resolve dt/num_steps given a span.

        Returns:
            times (steps+1,), dt (float), steps (int)
        """
        t_start, t_end = t_span
        span = t_end - t_start

        if num_steps is None:
            if num_total_steps is not None:
                num_steps = max(1, int(math.ceil(abs(span) * num_total_steps)))
            else:
                num_steps = self.num_timesteps

        dt = span / num_steps

        times = t_start + torch.arange(num_steps + 1, device=self.device, dtype=torch.float32) * dt
        if times.numel() > 0:
            times[-1] = t_end  # ensure exact endpoint

        return times, dt, num_steps

    def sample(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        x: Optional[torch.Tensor] = None,
        t_span: Tuple[float, float] = (1.0, 0.0),
        num_steps: Optional[int] = None,
        num_total_steps: Optional[int] = None,
        mode: str = "sde",
    ) -> torch.Tensor:
        """
        Sample using the reverse process with either the SDE (stochastic) or ODE (deterministic).

        Args:
            shape: Shape to initialize noise if x is None
            x: Optional starting samples; if None, start from noise
            t_span: (t_start, t_end) for denoising
            num_steps: Explicit discretization steps
            num_total_steps: Steps per unit time span (num_steps = |t_end-t_start| * num_total_steps)
            mode: "sde" for Euler-Maruyama sampling, "ode" for probability-flow ODE
        
        Returns:
            Trajectory tensor of shape (steps+1, batch, dims)
        """
        mode = mode.lower()
        if mode not in {"sde", "ode"}:
            raise ValueError(f"Unsupported sampling mode '{mode}'. Use 'sde' or 'ode'.")

        self.score_network.eval()

        if x is None:
            if shape is None:
                raise ValueError("Either x or shape must be provided for sampling.")
            x = torch.randn(shape, device=self.device)
        else:
            shape = x.shape

        batch_size = shape[0]
        times, dt_resolved, steps = self._prepare_time_grid(t_span, num_steps, num_total_steps)
        traj = [x]

        # Shared drift, with mode-specific scaling on the score term
        score_scale = 1.0 if mode == "sde" else 0.5
        noise_scale = math.sqrt(abs(dt_resolved)) if mode == "sde" else None

        for i in range(steps):
            t = torch.full((batch_size,), times[i], device=self.device)
            beta_t = self._beta(t).unsqueeze(1)
            score = self.score_network(x, t)
            drift = -0.5 * beta_t * x - score_scale * beta_t * score

            if mode == "sde":
                diffusion = torch.sqrt(torch.clamp(beta_t, min=1e-12))
                noise = torch.randn_like(x)
                x = x + drift * dt_resolved + diffusion * noise_scale * noise
            else:
                x = x + drift * dt_resolved

            traj.append(x)

        self.score_network.train()
        return torch.stack(traj)
    
    def loss(
        self,
        x_start: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        continuous: bool = True
    ) -> torch.Tensor:
        """
        Compute the training loss.
        
        Args:
            x_start: Tensor of shape (batch_size, 2) with clean data
            noise: Optional noise tensor
            continuous: If True, use VP-SDE continuous-time loss; otherwise use discrete DDPM loss
            
        Returns:
            Scalar loss tensor
        """
        batch_size = x_start.shape[0]

        if continuous:
            if noise is None:
                noise = torch.randn_like(x_start)
            # Uniform continuous timesteps in [0, 1]
            t = torch.rand(batch_size, device=self.device)
            mean_coeff, std = self._marginal_mean_std(t)
            mean_coeff = mean_coeff.unsqueeze(1)
            std = std.unsqueeze(1)

            x_t = mean_coeff * x_start + std * noise
            predicted_score = self.score_network(x_t, t)
            target_score = -noise / (std + 1e-8)
            return nn.functional.mse_loss(predicted_score, target_score)
        else:
            # Original discrete DDPM loss
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
            x_t, noise = self.q_sample(x_start, t, noise)
            t_normalized = t.float() / self.num_timesteps
            predicted_score = self.score_network(x_t, t_normalized)
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
            target_score = -noise / (sqrt_one_minus_alpha_cumprod_t + 1e-8)
            return nn.functional.mse_loss(predicted_score, target_score)

    @torch.no_grad()
    def predict(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        continuous: bool = True
    ) -> torch.Tensor:
        """
        Estimate x_0 from a noisy sample x_t at time t.

        Args:
            x_t: Noisy sample
            t: Time; continuous in [0, 1] if continuous=True, otherwise discrete step indices
            continuous: Whether to use the continuous VP-SDE marginal or the discrete DDPM schedule
        
        Returns:
            Estimated x_0 (same shape as x_t)
        """
        if continuous:
            mean_coeff, std = self._marginal_mean_std(t)
            mean_coeff = mean_coeff.unsqueeze(1)
            std = std.unsqueeze(1)
            score = self.score_network(x_t, t)
        else:
            if t.dtype != torch.long:
                t = t.long()
            mean_coeff = self.sqrt_alphas_cumprod[t].unsqueeze(1)
            std = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
            score = self.score_network(x_t, t.float() / self.num_timesteps)

        # Target score is -noise / std, so recover noise then denoise
        noise_hat = -score * std
        x0_hat = (x_t - std * noise_hat) / (mean_coeff + 1e-8)
        return x0_hat
    
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
