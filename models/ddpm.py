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
    
    def __init__(self, dim: int, max_freq: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(self.max_freq), half)
        )
        args = time[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class MLPScoreNetwork(nn.Module):
    """MLP-based score network for 2D DDPM."""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list = [128, 128, 128],
        time_embed_dim: int = 64,
        time_embed_hidden: int = 128,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.0,
        ebm: bool = False,
    ):
        """
        Initialize the score network.
        
        Args:
            input_dim: Dimension of input data (2 for 2D)
            hidden_dims: List of hidden layer dimensions
            time_embed_dim: Dimension of sinusoidal time embeddings
            time_embed_hidden: Hidden dimension for time embedding MLP
            activation: Activation function
            dropout: Dropout rate
            ebm: Whether to make an energy based model (scalar output)
        """
        super().__init__()
        
        self.time_embed_dim = time_embed_dim
        self.time_embed_hidden = time_embed_hidden
        
        # Time embedding: sinusoidal -> MLP transformation
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_hidden),
            activation,
            nn.Linear(time_embed_hidden, time_embed_hidden),
        )
        
        # Build MLP layers (no time embedding concatenation)
        dims = [input_dim] + hidden_dims + [input_dim]
        if ebm:
            dims[-1] = 1  # Output scalar energy for EBM mode

        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        
        # Separate learned projections of time embeddings for each hidden layer
        self.time_projections = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.time_projections.append(nn.Linear(time_embed_hidden, dims[i + 1]))
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the score network.
        
        Args:
            x: Tensor of shape (batch_size, input_dim) with noisy data
            t: Tensor of shape (batch_size,) with time steps (normalized to [0, 1])
            
        Returns:
            Tensor of shape (batch_size, input_dim) with predicted scores
        """
        # Get transformed time embeddings (constant for this forward pass)
        h_t = self.time_embed(t)
        
        # Forward through network with time embeddings added at each hidden layer
        h = x
        num_hidden = len(self.time_projections)
        
        for i in range(len(self.layers)):
            h = self.layers[i](h)
            
            # Add time embedding projection for hidden layers
            if i < num_hidden:
                h = h + self.time_projections[i](h_t)
            
            # Apply activation and dropout (except for output layer)
            if i < len(self.layers) - 1:
                h = self.activation(h)
                if self.dropout is not None:
                    h = self.dropout(h)
        
        return h


def _baoa_step(score_network, x, p, t, dt, gamma=0.1, m=1.0, kBT=1.0):
    # OU parameters
    a = math.exp(-gamma * dt)
    sigma_p = math.sqrt((1.0 - a*a) * m * kBT)

    # B
    p = p + dt * (kBT * score_network(x, t))

    # A
    x = x + (dt / (2.0 * m)) * p

    # O
    eta = torch.randn_like(p)
    p = a * p + sigma_p * eta

    # A
    x = x + (dt / (2.0 * m)) * p

    return x, p

def _baoa_sample(score_network, x, t, dt, n_steps=100, gamma=0.1, m=1.0, kBT=1.0):
    if len(t.shape) == 0:
        t = torch.stack([t]*x.shape[0])
    elif t.shape[0] != x.shape[0]:
        print("t (%s) and x (%s) not same shape" % (t.shape, x.shape))
    
    p = torch.zeros_like(x)
    xtrj = [x.detach().clone()]
    for n in np.arange(n_steps):
        x,p = _baoa_step(score_network, x, p, t, dt, gamma=gamma, m=m, kBT=kBT)
        xtrj.append(x.detach().clone())
    xtrj = torch.stack(xtrj)
    return xtrj


class DDPM:
    """Denoising Diffusion Probabilistic Model."""
    
    def __init__(
        self,
        model: nn.Module,
        ebm: bool = False,
        beta_min: float = 0.2,
        beta_max: float = 20,
        beta_schedule: str = "linear",
        device: str = "cpu"
    ):
        """
        Initialize DDPM.
        
        Args:
            model: Neural network for predicting scores (or energy if ebm=True)
            ebm: If True, model outputs energy (scalar) and score is computed via gradient
            beta_min: Starting noise schedule value
            beta_max: Ending noise schedule value
            beta_schedule: Noise schedule type ("linear" or "cosine")
            device: Device to run on
        """
        self.model = model.to(device)
        self.ebm = ebm
        self.device = device
        self.beta_schedule = beta_schedule.lower()
        
        self.beta_min = beta_min
        self.beta_max = beta_max
        
    def _Beta(self, t):
        return self.beta_min * t + 0.5 *(self.beta_max - self.beta_min) * (t * t)
    
    def _beta(self,t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def _alpha(self, t, Beta=None):
        if Beta is None:
            Beta = self._Beta(t)
        return torch.exp(-0.5*Beta)

    def _sigma(self, t, alpha=None):
        if alpha is None:
            alpha = self._alpha(t)
        return torch.sqrt(1 - (alpha * alpha))
    
    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the score function.
        
        For non-EBM mode: returns model(x, t) directly.
        For EBM mode: computes gradient of energy model(x, t) with respect to x.
        
        Args:
            x: Tensor of shape (batch_size, input_dim) with noisy data
            t: Tensor of shape (batch_size,) with time steps (normalized to [0, 1])
            
        Returns:
            Tensor of shape (batch_size, input_dim) with predicted scores
        """
        if self.ebm:
            x.requires_grad_(True)
            E = self.model(x, t)
            score = torch.autograd.grad(E.sum(), x, create_graph=True)[0]
            return score
        else:
            return self.model(x, t)
    
    def q_forward(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        
        alpha = self._alpha(t)
        sigma = self._sigma(t, alpha=alpha)
        xt = alpha[:,None]*x0 + sigma[:,None]*noise
        return xt, noise
    
    def loss(self, xt, noise, t):
        """
        Compute the training loss.
        
        Args:
            xt: Noisy data at time t, shape (batch_size, input_dim)
            noise: The noise that was added, shape (batch_size, input_dim)
            t: Time steps, shape (batch_size,)
        
        Returns:
            loss = ||sigma(t)*score(xt,t) + noise||^2
        """
        sigmas = self._sigma(t)
        pred_score = self.score(xt, t)
        return nn.functional.mse_loss(sigmas[:,None]*pred_score, -noise)
    
    def _sde_step(self, xt, t, dt):
        beta_t = self._beta(t)
        score = self.score(xt, t)
        noise = torch.randn_like(xt)
        drift = (-0.5*beta_t[:,None]*xt) - (beta_t[:,None]*score)
        xt = xt + drift*dt + torch.sqrt(beta_t[:,None]*np.abs(dt))*noise
        return xt
    
    def _ode_step(self, xt, t, dt):
        beta_t = self._beta(t)
        score = self.score(xt, t)
        drift = -0.5*(beta_t[:, None]*xt + beta_t[:, None]*score)
        xt = xt + drift*dt
        return xt
    
    def sample(self, xt, tspan=(1, 0), n_steps=200, mode='sde'):
        self.model.eval()
        b = xt.shape[0]

        times = np.linspace(tspan[0], tspan[1], n_steps + 1)  # include endpoints
        traj = [xt]

        for i in range(n_steps):
            t_tmp = times[i]
            dt = times[i+1] - times[i]   # this will be NEGATIVE for 1 -> 0
            t = torch.full((b,), float(t_tmp), dtype=xt.dtype, device=xt.device)

            if mode == 'sde':
                xt = self._sde_step(xt, t, dt)
            elif mode == 'ode':
                xt = self._ode_step(xt, t, dt)
            else:
                raise ValueError("unrecognized mode")

            traj.append(xt)

        return torch.stack(traj)


    def predict(self, xt, t):
        alpha = self._alpha(t)
        sigma = self._sigma(t, alpha=alpha)
        score = self.score(xt, t)
        x0 = (xt + (sigma[:,None]*sigma[:,None ])*score)/alpha[:,None]
        return x0