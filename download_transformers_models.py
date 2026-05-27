#!/usr/bin/env python3
"""Helper script to download transformers models for offline use"""
import os
from transformers import CLIPTokenizer, CLIPTextModel

def download_model(model_name, local_dir):
    """Download a transformers model to local directory"""
    print(f"\nDownloading {model_name} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        print(f"Downloading tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)
        print(f"Tokenizer saved to {local_dir}")
        
        print(f"Downloading model...")
        model = CLIPTextModel.from_pretrained(model_name)
        model.save_pretrained(local_dir)
        print(f"Model saved to {local_dir}")
        
        print(f"\n✓ Successfully downloaded {model_name} to {local_dir}")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading {model_name}: {e}")
        return False

# Try to download CLIP models
models = [
    ('openai/clip-vit-large-patch14', 'tacm/modules/cache/clip-vit-large-patch14'),
    ('openai/clip-vit-base-patch32', 'tacm/modules/cache/clip-vit-base-patch32'),
]

print("=" * 60)
print("Downloading transformers models for offline use")
print("=" * 60)
print("Note: This requires network connection")
print("=" * 60)

for model_name, local_dir in models:
    if download_model(model_name, local_dir):
        print(f"\nModel {model_name} is now available locally at {local_dir}")
        break
    else:
        print(f"Failed to download {model_name}, trying next...")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
