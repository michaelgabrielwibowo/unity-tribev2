"""Anatomical weight matrix definition and persistence.

This module provides utilities for initializing, saving, and loading the
default heuristic weight matrix. It also includes helpers that resolve the
bundled default weights path so the package works whether run from source
or after being installed via pip.
"""

import importlib.resources as resources
import shutil
from pathlib import Path

import numpy as np

from tribe_lite.scorer.region_map import NUM_REGIONS
from tribe_lite.config import TribeLiteConfig


def init_heuristic_weights(config: TribeLiteConfig | None = None) -> np.ndarray:
    """Initialize heuristic weight matrix with game-state proxy values.
    
    ⚠️  IMPORTANT: These weights are NOT neuroscience-calibrated.
    
    This creates a (fused_dim, 26) matrix where each column represents a brain 
    region's sensitivity profile. The weights are initialized using heuristic 
    rules that produce plausible-looking brain activation patterns, but they do 
    NOT reflect actual neurophysiology.
    
    The "brain scores" produced using these weights are useful for:
    - Game state signals (detecting user engagement, attention shifts)
    - Interactive entertainment (changing game dynamics based on activation)
    - Educational demonstrations of multimodal AI
    
    The "brain scores" should NOT be used for:
    - Neuroscience research or analysis
    - Medical diagnosis or monitoring
    - Any safety-critical application
    
    To use scientifically-calibrated weights, collect fMRI/EEG data paired with 
    video+audio inputs and fit W via least-squares regression (Path C in the 
    design doc).
    
    Heuristic initialization strategy:
    - Audio encoder dims (384) get higher weights on auditory/language regions
    - CLIP semantic dims (512) get higher weights on visual regions
    - Optical flow dims (10) get higher weights on motor/attention regions
    
    Args:
        config: Configuration object. Uses defaults if None.
        
    Returns:
        Heuristic weight matrix W of shape (fused_dim, NUM_REGIONS)
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
    # Derive CLIP and optical-flow block sizes from config to avoid hard-coded values
    clip_dim = config.clip_features_dim if config.use_clip else 0
    # video_features_dim is composed of optical_flow + clip (when enabled)
    optical_flow_dim = (config.video_features_dim - clip_dim) if config.use_optical_flow else 0
    of_end = of_start + optical_flow_dim
    clip_start = of_end
    clip_end = of_end + clip_dim
    audio_start = clip_end
    audio_end = audio_start + audio_dim

    # Sanity check: ensure the blocks sum exactly to fused_dim
    assert audio_end == fused_dim, (
        f"Weight init dimension mismatch: blocks sum to {audio_end} but fused_dim={fused_dim}. "
        "Check that config.video_features_dim, clip_features_dim, and audio_features_dim are consistent."
    )
    
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


def get_default_weights_path() -> Path:
    """Resolve the bundled default weights path.

    This attempts several strategies to locate the shipped default
    weights so the package works both from source and when installed.
    """
    # 1) A `weights` resource directory inside the `tribe_lite` package
    try:
        with resources.path("tribe_lite", "weights") as p:
            candidate = p / "default_weights.npz"
            if candidate.exists():
                return candidate
    except Exception:
        pass

    # 2) A direct file resource (older tooling might place the file at package root)
    try:
        with resources.path("tribe_lite", "default_weights.npz") as pfile:
            if pfile.exists():
                return pfile
    except Exception:
        pass

    # 3) Fallback to package-relative `weights/` (when running from source)
    repo_weights = Path(__file__).resolve().parents[1] / "weights" / "default_weights.npz"
    if repo_weights.exists():
        return repo_weights

    # 4) Finally, fallback to current working directory `weights/`
    return Path.cwd() / "weights" / "default_weights.npz"


def create_default_weights(output_path: str | Path = "weights/default_weights.npz", 
                           config: TribeLiteConfig | None = None) -> np.ndarray:
    """Create and save default heuristic weight matrix.
    
    ⚠️  These are NOT scientifically-calibrated weights. See init_heuristic_weights()
    for details on their limitations and appropriate use cases.
    
    Args:
        output_path: Where to save the weights
        config: Configuration object
        
    Returns:
        The created heuristic weight matrix
    """
    output_path = Path(output_path)

    # If the file already exists where requested, just load it
    if output_path.exists():
        return load_weights(output_path)

    # If there's a bundled default, copy it into place rather than regenerating
    pkg_default = get_default_weights_path()
    if pkg_default.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(pkg_default, output_path)
            return load_weights(output_path)
        except Exception:
            # Fall through to generating a fresh file if copy fails
            pass

    W = init_heuristic_weights(config)
    save_weights(W, output_path)
    return W
