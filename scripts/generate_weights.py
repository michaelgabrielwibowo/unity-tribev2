"""Generate the default brain weight matrix.

Run this once to create weights/default_weights.npz, which is required
for the TRIBE-Lite pipeline to function.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import tribe_lite
sys.path.insert(0, str(Path(__file__).parent.parent))

from tribe_lite.config import TribeLiteConfig
from tribe_lite.scorer.weight_matrix import create_default_weights


def main():
    """Generate and save default weight matrix."""
    config = TribeLiteConfig()
    
    print("=" * 60)
    print("TRIBE-Lite — Weight Matrix Generator")
    print("=" * 60)
    print()
    
    print(f"Configuration:")
    print(f"  Fused dimension: {config.fused_dim}-dim")
    print(f"  Video features: {config.video_features_dim}-dim")
    print(f"    - Optical flow: 10-dim (if enabled)")
    print(f"    - CLIP: 512-dim (if enabled)")
    print(f"  Audio features: {config.audio_features_dim}-dim")
    print(f"  Brain regions: 26")
    print()
    
    # Generate weights
    print(f"Creating weight matrix at {config.weight_path}...")
    W = create_default_weights(config.weight_path, config)
    
    print(f"✓ Weight matrix created")
    print(f"  Shape: {W.shape}")
    print(f"  Path:  {Path(config.weight_path).resolve()}")
    print()
    print("You can now run the pipeline:")
    print("  python scripts/run_prototype.py")
    print()


if __name__ == "__main__":
    main()
