# DDPM with Iso-Time Contrastive Learning

This project implements a Denoising Diffusion Probabilistic Model (DDPM) trained on a 2D toy landscape, with iso-time simulations used for contrastive divergence during training.

## Overview

The project explores using iso-time simulations as a contrastive divergence signal during DDPM training. Iso-time simulations take the score of a DDPM at a particular noise level and treat it like an energy landscape, performing Langevin sampling at that specified noise level.

## landscapes/learned potentials

Toy landscape with 4 gaussian potentials

<img width="2084" height="881" alt="landscape_fig" src="https://github.com/user-attachments/assets/55cf4ca1-60ec-4a97-8913-211c75aad783" />

Learned score field (scores from a trained diffusion model)

https://github.com/user-attachments/assets/9aa920d9-ff8c-46b8-8ea3-d1f8ebc83759


Iso-time sampling at 2 different noise levels showing increased sampling at higher noise levels

https://github.com/user-attachments/assets/b61ac961-6371-44b4-b948-12cb3628b46f


https://github.com/user-attachments/assets/5810e9a9-7395-434d-b766-4a03be1bd4e0




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
