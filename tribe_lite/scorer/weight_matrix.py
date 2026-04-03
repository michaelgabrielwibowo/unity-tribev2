"""Anatomical weight matrix definition and persistence."""

import numpy as np
from pathlib import Path

from tribe_lite.scorer.region_map import NUM_REGIONS
from tribe_lite.config import TribeLiteConfig


def init_anatomical_weights(config: TribeLiteConfig | None = None) -> np.ndarray:
    """Initialize anatomical weight matrix with heuristic values.
    
    Creates a (904, 26) matrix where each column represents a brain region's
    sensitivity profile across the 904 input features.
    
    Heuristic initialization strategy:
    - Audio encoder dims (384) get higher weights on auditory/language regions
    - CLIP semantic dims (512) get higher weights on visual regions
    - Optical flow dims (8) get higher weights on motor/attention regions
    
    Args:
        config: Configuration object. Uses defaults if None.
        
    Returns:
        Weight matrix W of shape (fused_dim, NUM_REGIONS)
    """
    if config is None:
        config = TribeLiteConfig()
    
    fused_dim = config.fused_dim
    video_dim = config.video_features_dim
    audio_dim = config.audio_features_dim
    
    # Initialize with small random values for stability
    W = np.random.randn(fused_dim, NUM_REGIONS).astype(np.float32) * 0.01
    
    # Feature dimension offsets
    of_start = 0
    of_end = 8 if config.use_optical_flow else 0
    clip_start = of_end
    clip_end = of_end + (512 if config.use_clip else 0)
    audio_start = clip_end
    audio_end = audio_start + audio_dim
    
    # Boost optical flow → motor/attention regions (1, 15, 18, 19)
    if config.use_optical_flow:
        motor_attention_regions = [1, 15, 18, 19]  # Motor, Thalamus, Sup/Inf Colliculus
        for region_idx in motor_attention_regions:
            W[of_start:of_end, region_idx] = np.random.uniform(0.3, 0.7, size=(of_end - of_start))
    
    # Boost CLIP semantics → visual regions (4, 18)
    if config.use_clip:
        visual_regions = [4, 18]  # Visual Cortex, Superior Colliculus
        for region_idx in visual_regions:
            W[clip_start:clip_end, region_idx] = np.random.uniform(0.2, 0.5, size=(clip_end - clip_start))
    
    # Boost audio semantics → auditory/language regions (3, 6, 13)
    if config.use_semantic_audio and audio_dim > 0:
        auditory_regions = [3, 6, 13]  # Auditory Cortex, ACC, Hippocampus
        for region_idx in auditory_regions:
            W[audio_start:audio_end, region_idx] = np.random.uniform(0.3, 0.6, size=(audio_end - audio_start))
    
    return W


def save_weights(W: np.ndarray, path: str | Path) -> None:
    """Save weight matrix to .npz file.
    
    Args:
        W: Weight matrix of shape (fused_dim, NUM_REGIONS)
        path: Path to save file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, weights=W, version="1.0")


def load_weights(path: str | Path) -> np.ndarray:
    """Load weight matrix from .npz file.
    
    Args:
        path: Path to .npz file
        
    Returns:
        Weight matrix W of shape (fused_dim, NUM_REGIONS)
        
    Raises:
        FileNotFoundError: If weight file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weight matrix not found at {path}")
    
    data = np.load(path)
    return data["weights"].astype(np.float32)


def create_default_weights(output_path: str | Path = "weights/default_weights.npz", 
                           config: TribeLiteConfig | None = None) -> np.ndarray:
    """Create and save default weight matrix.
    
    Args:
        output_path: Where to save the weights
        config: Configuration object
        
    Returns:
        The created weight matrix
    """
    W = init_anatomical_weights(config)
    save_weights(W, output_path)
    return W
