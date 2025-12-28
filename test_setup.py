"""
Quick test script to verify the basic setup works.
"""

import torch
from data.landscape import FourModeLandscape
from models.ddpm import DDPM, MLPScoreNetwork

def test_landscape():
    """Test landscape generation and sampling."""
    print("Testing landscape generation...")
    landscape = FourModeLandscape(scale=2.0, std=0.5)
    
    # Test sampling
    samples = landscape.sample(100)
    print(f"✓ Sampled {len(samples)} points, shape: {samples.shape}")
    assert samples.shape == (100, 2), "Sample shape mismatch"
    
    # Test log probability
    log_prob = landscape.log_prob(samples[:10])
    print(f"✓ Computed log probabilities, shape: {log_prob.shape}")
    assert log_prob.shape == (10,), "Log prob shape mismatch"
    
    # Test score
    score = landscape.score(samples[:10])
    print(f"✓ Computed scores, shape: {score.shape}")
    assert score.shape == (10, 2), "Score shape mismatch"
    
    print("✓ Landscape tests passed!\n")

def test_ddpm():
    """Test DDPM initialization and forward pass."""
    print("Testing DDPM model...")
    device = "cpu"
    
    # Create score network
    score_network = MLPScoreNetwork(input_dim=2, hidden_dims=[64, 128, 64])
    print(f"✓ Created score network")
    
    # Create DDPM
    ddpm = DDPM(score_network, num_timesteps=100, device=device)
    print(f"✓ Created DDPM with {ddpm.num_timesteps} timesteps")
    
    # Test forward pass
    batch_size = 32
    x_start = torch.randn(batch_size, 2)
    loss = ddpm.loss(x_start)
    print(f"✓ Computed loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    print("✓ DDPM tests passed!\n")

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Project Setup")
    print("=" * 50)
    print()
    
    try:
        test_landscape()
        test_ddpm()
        print("=" * 50)
        print("✓ All tests passed! Setup is correct.")
        print("=" * 50)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

