# DDPM with Iso-Time Contrastive Learning

This project implements a Denoising Diffusion Probabilistic Model (DDPM) trained on a 2D toy landscape, with iso-time simulations used for contrastive divergence during training.

## Overview

The project explores using iso-time simulations as a contrastive divergence signal during DDPM training. Iso-time simulations take the score of a DDPM at a particular noise level and treat it like an energy landscape, performing Langevin sampling at that specified noise level.

### Key Components

1. **2D Landscape**: A toy landscape with 4 Gaussian modes (one in each quadrant)
2. **DDPM Model**: Core diffusion model for learning the score function
3. **Iso-Time Sampling**: Langevin dynamics at fixed noise levels
4. **Contrastive Learning**: Using iso-time samples for training signal

## Project Structure

```
.
├── data/
│   └── landscape.py          # 2D landscape generator with 4 Gaussian modes
├── models/
│   └── ddpm.py               # Core DDPM implementation
├── training/                 # Training scripts (to be added)
├── utils/                    # Utility functions (to be added)
├── configs/                  # Configuration files (to be added)
├── notebooks/                # Visualization notebooks (to be added)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Generate landscape visualization:
```bash
python data/landscape.py
```

## Usage

### Generate Landscape

```python
from data.landscape import create_four_mode_landscape

# Create landscape
landscape = create_four_mode_landscape(scale=2.0, std=0.5)

# Visualize
landscape.visualize(save_path="landscape.png")

# Sample data
samples = landscape.sample(n_samples=1000)
```

### Create and Use DDPM

```python
from models.ddpm import DDPM, MLPScoreNetwork

# Create score network
score_network = MLPScoreNetwork(input_dim=2, hidden_dims=[128, 256, 256, 128])

# Create DDPM (supports linear or cosine beta schedules)
ddpm = DDPM(score_network, num_timesteps=1000, beta_schedule="cosine", device="cpu")

# Compute loss for training
loss = ddpm.loss(x_start)

# Generate samples (trajectory returned; final sample at [-1])
traj = ddpm.sample((batch_size, 2), mode="sde")  # or mode="ode" for deterministic sampling
samples = traj[-1]
```

## Next Steps

- Implement iso-time Langevin sampling
- Add contrastive divergence training loop
- Create training scripts
- Add visualization utilities
- Experiment with different noise schedules and architectures
