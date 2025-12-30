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
│   └── landscape.py          # 2D landscape generator (toy systems)
├── models/
│   └── ddpm.py               # Core DDPM implementation
├── training/                 # Training scripts
├── utils/                    # Utility functions
├── configs/                  # Configuration files
├── notebooks/                # Visualization notebooks
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Local install
```bash
pip install -e .
```
