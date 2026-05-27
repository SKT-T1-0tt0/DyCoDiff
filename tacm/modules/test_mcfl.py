#!/usr/bin/env python3
"""
Simple test script for MCFL module
Run: python tacm/modules/test_mcfl.py
"""

import sys
import os

import torch
# Direct import to avoid dependency issues
sys.path.insert(0, os.path.dirname(__file__))
from mcfl import MCFL

def test_mcfl():
    """Test MCFL module"""
    print("Testing MCFL module...")
    
    # Create module
    embed_dim = 768
    num_heads = 8
    mcfl = MCFL(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
    
    # Create test data
    B = 4
    D = embed_dim
    c_text = torch.randn(B, D)
    c_image = torch.randn(B, D)
    c_audio = torch.randn(B, D)
    
    print(f"Input shapes:")
    print(f"  c_text: {c_text.shape}")
    print(f"  c_image: {c_image.shape}")
    print(f"  c_audio: {c_audio.shape}")
    
    # Forward pass
    c_fused = mcfl(c_text, c_image, c_audio)
    
    print(f"\nOutput shape: {c_fused.shape}")
    
    # Verify output shape
    assert c_fused.shape == (B, D), f"Expected ({B}, {D}), got {c_fused.shape}"
    print("\n✓ Shape check passed!")
    
    # Verify output is not all zeros
    assert not torch.allclose(c_fused, torch.zeros_like(c_fused)), "Output is all zeros!"
    print("✓ Output is not all zeros!")
    
    # Test with different batch sizes
    B2 = 2
    c_text2 = torch.randn(B2, D)
    c_image2 = torch.randn(B2, D)
    c_audio2 = torch.randn(B2, D)
    c_fused2 = mcfl(c_text2, c_image2, c_audio2)
    assert c_fused2.shape == (B2, D), f"Expected ({B2}, {D}), got {c_fused2.shape}"
    print("✓ Different batch size test passed!")
    
    # Test with None modalities
    print("\nTesting None modality support...")
    c_fused_text_only = mcfl(c_text, None, None)
    assert c_fused_text_only.shape == (B, D), f"Expected ({B}, {D}), got {c_fused_text_only.shape}"
    print("✓ Text-only test passed!")
    
    c_fused_image_only = mcfl(None, c_image, None)
    assert c_fused_image_only.shape == (B, D), f"Expected ({B}, {D}), got {c_fused_image_only.shape}"
    print("✓ Image-only test passed!")
    
    c_fused_audio_only = mcfl(None, None, c_audio)
    assert c_fused_audio_only.shape == (B, D), f"Expected ({B}, {D}), got {c_fused_audio_only.shape}"
    print("✓ Audio-only test passed!")
    
    c_fused_text_image = mcfl(c_text, c_image, None)
    assert c_fused_text_image.shape == (B, D), f"Expected ({B}, {D}), got {c_fused_text_image.shape}"
    print("✓ Text+Image test passed!")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_mcfl()
