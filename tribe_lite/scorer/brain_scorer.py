"""Brain region scoring engine."""

import numpy as np

from tribe_lite.scorer.region_map import REGION_MAP, NUM_REGIONS
from tribe_lite.scorer.weight_matrix import load_weights
from tribe_lite.output.schema import TribeLiteOutput
from tribe_lite.config import TribeLiteConfig


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


class BrainScorer:
    """Core scoring engine: feature vector → brain region activations.
    
    Uses a weight matrix W to project fused features into brain region space,
    then normalizes with sigmoid to get activation scores in [0, 1].
    """
    
    def __init__(self, config: TribeLiteConfig | None = None):
        """Initialize the scorer with weight matrix.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or TribeLiteConfig()
        self.W = load_weights(self.config.weight_path)
        
        # Validate dimensions
        expected_dim = self.config.fused_dim
        if self.W.shape[0] != expected_dim:
            raise ValueError(
                f"Weight matrix dimension mismatch: expected {expected_dim}, "
                f"got {self.W.shape[0]}"
            )
        if self.W.shape[1] != NUM_REGIONS:
            raise ValueError(
                f"Weight matrix region count mismatch: expected {NUM_REGIONS}, "
                f"got {self.W.shape[1]}"
            )
    
    def score(self, fused_vec: np.ndarray) -> TribeLiteOutput:
        """Score a fused feature vector and return output.
        
        Args:
            fused_vec: Fused feature vector of shape (fused_dim,)
            
        Returns:
            TribeLiteOutput with region scores and global score
        """
        # Ensure correct shape
        if fused_vec.ndim == 1:
            fused_vec = fused_vec.reshape(1, -1)
        
        # Project to region space: (1, fused_dim) @ (fused_dim, 26) → (1, 26)
        raw = fused_vec @ self.W
        raw = raw.flatten()  # (26,)
        
        # Normalize to [0, 1] with sigmoid
        region_scores_normalized = sigmoid(raw)
        
        # Compute global score as mean of all regions
        global_score = float(np.mean(region_scores_normalized))
        
        # Find top 3 most active regions
        top3_indices = np.argsort(region_scores_normalized)[-3:][::-1]
        top_regions = [REGION_MAP[idx] for idx in top3_indices]
        
        # Build full region score dictionary
        region_scores_dict = {
            REGION_MAP[idx]: float(region_scores_normalized[idx])
            for idx in range(NUM_REGIONS)
        }
        
        return TribeLiteOutput(
            timestamp=None,  # Will be set by pipeline
            global_score=global_score,
            region_scores=region_scores_dict,
            top_regions=top_regions,
            window_sec=self.config.window_sec,
        )
