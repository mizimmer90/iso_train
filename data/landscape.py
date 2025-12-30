"""
2D landscape generator with arbitrary Gaussian mixture models.

This module generates toy 2D landscapes by combining multiple Gaussian distributions.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt


class GaussianLandscape:
    """General 2D landscape with arbitrary number of Gaussian modes."""
    
    def __init__(
        self,
        centers: Union[torch.Tensor, np.ndarray],
        stds: Union[torch.Tensor, np.ndarray, float],
        weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
        device: str = "cpu"
    ):
        """
        Initialize the Gaussian mixture landscape.
        
        Args:
            centers: Tensor/array of shape (n_modes, 2) with center coordinates
            stds: Tensor/array of shape (n_modes,) with standard deviations, or float for all modes
            weights: Optional tensor/array of shape (n_modes,) with mode weights. 
                     Defaults to equal weights.
            device: Device to run computations on (default: "cpu")
        """
        self.device = device
        
        # Convert centers to tensor
        if isinstance(centers, np.ndarray):
            centers = torch.from_numpy(centers).float()
        elif not isinstance(centers, torch.Tensor):
            centers = torch.tensor(centers, dtype=torch.float32)
        
        self.centers = centers.to(device)
        self.n_modes = len(self.centers)
        
        # Convert stds to tensor
        if isinstance(stds, (float, int)):
            stds = torch.full((self.n_modes,), float(stds))
        elif isinstance(stds, np.ndarray):
            stds = torch.from_numpy(stds).float()
        elif not isinstance(stds, torch.Tensor):
            stds = torch.tensor(stds, dtype=torch.float32)
        
        if len(stds) == 1 and self.n_modes > 1:
            stds = stds.expand(self.n_modes)
        elif len(stds) != self.n_modes:
            raise ValueError(f"stds must have length {self.n_modes} or be a scalar, got {len(stds)}")
        
        self.stds = stds.to(device)
        
        # Convert weights to tensor
        if weights is None:
            weights = torch.ones(self.n_modes, device=device) / self.n_modes
        elif isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float()
        elif not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        
        if len(weights) != self.n_modes:
            raise ValueError(f"weights must have length {self.n_modes}, got {len(weights)}")
        
        self.weights = (weights / weights.sum()).to(device)  # Normalize weights
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability density at points x.
        
        Args:
            x: Tensor of shape (N, 2) representing N points in 2D space
            
        Returns:
            Tensor of shape (N,) with log probabilities
        """
        log_probs = []
        for i in range(self.n_modes):
            center = self.centers[i]
            std = self.stds[i]
            # Compute log probability for each Gaussian mode
            diff = x - center
            log_prob = -0.5 * torch.sum((diff / std) ** 2, dim=-1)
            log_prob = log_prob - 2 * np.log(std) - np.log(2 * np.pi)
            log_probs.append(log_prob)
        
        # Combine modes using log-sum-exp for numerical stability
        log_probs = torch.stack(log_probs, dim=0)  # (n_modes, N)
        log_weights = torch.log(self.weights).unsqueeze(1)  # (n_modes, 1)
        combined_log_prob = torch.logsumexp(log_probs + log_weights, dim=0)
        
        return combined_log_prob
    
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the score function (gradient of log probability) analytically.
        
        Args:
            x: Tensor of shape (N, 2) representing N points in 2D space
            
        Returns:
            Tensor of shape (N, 2) with score vectors
        """
        # Compute unnormalized probabilities for each mode
        log_probs = []
        for i in range(self.n_modes):
            center = self.centers[i]
            std = self.stds[i]
            diff = x - center
            log_prob = -0.5 * torch.sum((diff / std) ** 2, dim=-1)
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=0)  # (n_modes, N)
        log_weights = torch.log(self.weights).unsqueeze(1)  # (n_modes, 1)
        
        # Compute weighted probabilities using log-sum-exp trick
        max_log_prob = torch.max(log_probs + log_weights, dim=0, keepdim=True)[0]
        exp_log_probs = torch.exp(log_probs + log_weights - max_log_prob)
        probs = exp_log_probs / (exp_log_probs.sum(dim=0, keepdim=True) + 1e-10)
        
        # Compute score as weighted sum of individual mode scores
        # Score of each Gaussian mode: -(x - center) / std^2
        scores_per_mode = []
        for i in range(self.n_modes):
            center = self.centers[i]
            std = self.stds[i]
            score = -(x - center) / (std ** 2)
            scores_per_mode.append(score)
        
        scores_per_mode = torch.stack(scores_per_mode, dim=0)  # (n_modes, N, 2)
        # Weight by posterior probabilities
        score = torch.sum(probs.unsqueeze(-1) * scores_per_mode, dim=0)
        
        return score
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample points from the landscape using importance sampling.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tensor of shape (n_samples, 2) with sampled points
        """
        # Sample number of points per mode based on weights
        mode_counts = torch.multinomial(
            self.weights,
            num_samples=n_samples,
            replacement=True
        )
        counts = torch.bincount(mode_counts, minlength=self.n_modes)
        
        all_samples = []
        for i in range(self.n_modes):
            n = counts[i].item()
            if n > 0:
                center = self.centers[i]
                std = self.stds[i]
                # Sample from Gaussian centered at this mode
                samples = torch.randn(n, 2, device=self.device) * std + center
                all_samples.append(samples)
        
        samples = torch.cat(all_samples, dim=0) if all_samples else torch.empty(0, 2, device=self.device)
        # Shuffle to mix modes
        if len(samples) > 0:
            perm = torch.randperm(len(samples), device=self.device)
            samples = samples[perm]
        
        return samples
    
    def visualize(
        self,
        xlim: Tuple[float, float] = (-5, 5),
        ylim: Tuple[float, float] = (-5, 5),
        resolution: int = 200,
        n_samples: int = 1000,
        save_path: Optional[str] = None
    ):
        """
        Visualize the landscape with density and samples.
        
        Args:
            xlim: X-axis limits for visualization
            ylim: Y-axis limits for visualization
            resolution: Resolution for density plot
            n_samples: Number of samples to overlay
            save_path: Optional path to save the figure
        """
        # Create grid for density visualization
        x = np.linspace(xlim[0], xlim[1], resolution)
        y = np.linspace(ylim[0], ylim[1], resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1),
            device=self.device,
            dtype=torch.float32
        )
        
        # Compute log probabilities
        log_probs = self.log_prob(grid_points)
        probs = torch.exp(log_probs).cpu().numpy().reshape(X.shape)
        
        # Sample points
        samples = self.sample(n_samples).cpu().numpy()
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Density plot
        im1 = ax1.contourf(X, Y, probs, levels=50, cmap='viridis')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'{self.n_modes}-Mode Landscape Density')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax1, label='Probability Density')
        
        # Sample plot
        im2 = ax2.contourf(X, Y, probs, levels=50, cmap='viridis', alpha=0.3)
        if len(samples) > 0:
            ax2.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.5, c='red', edgecolors='black', linewidths=0.5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'{self.n_modes}-Mode Landscape with {n_samples} Samples')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def create_four_mode_landscape(
    scale: float = 2.0,
    std: Union[float, Tuple[float, float, float, float]] = 0.5,
    device: str = "cpu"
) -> GaussianLandscape:
    """
    Create a 4-mode landscape with modes in each quadrant.
    
    Args:
        scale: Distance of mode centers from origin (default: 2.0)
        std: Standard deviation(s). If float, all modes use the same std.
             If tuple of 4 floats, each mode gets its own std (default: 0.5)
        device: Device to run computations on (default: "cpu")
    
    Returns:
        GaussianLandscape instance with 4 modes
    """
    # Define centers for 4 quadrants: (x, y) coordinates
    centers = torch.tensor([
        [scale, scale],      # Quadrant I (top-right)
        [-scale, scale],     # Quadrant II (top-left)
        [-scale, -scale],    # Quadrant III (bottom-left)
        [scale, -scale],     # Quadrant IV (bottom-right)
    ], device=device, dtype=torch.float32)
    
    # Handle stds
    if isinstance(std, (float, int)):
        stds = float(std)
    elif isinstance(std, (tuple, list)) and len(std) == 4:
        stds = torch.tensor(std, dtype=torch.float32, device=device)
    else:
        raise ValueError("std must be a float or tuple/list of 4 floats")
    
    return GaussianLandscape(centers=centers, stds=stds, device=device)


# For backwards compatibility - allows FourModeLandscape(scale=2.0, std=0.5) to work
FourModeLandscape = create_four_mode_landscape
