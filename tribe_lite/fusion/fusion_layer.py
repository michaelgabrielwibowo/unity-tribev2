"""Fusion layer: combine multimodal features into single vector."""

import numpy as np


class FusionLayer:
    """Fuses video and audio feature vectors into a single representation.
    
    Concatenates features from different modalities with optional normalization
    to prevent one modality from dominating.
    """
    
    def __init__(self, normalize: bool = True, eps: float = 1e-8):
        """Initialize fusion layer.
        
        Args:
            normalize: Whether to L2-normalize each modality before concatenation
            eps: Small constant for numerical stability in normalization
        """
        self.normalize = normalize
        self.eps = eps
    
    def _normalize_vec(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector or batch of vectors."""
        if vec.size == 0:
            return vec
        
        norm = np.linalg.norm(vec, ord=2, axis=-1, keepdims=True)
        norm = np.maximum(norm, self.eps)  # Prevent division by zero
        return vec / norm
    
    def fuse(self, video_vec: np.ndarray, audio_vec: np.ndarray) -> np.ndarray:
        """Fuse video and audio feature vectors.
        
        Args:
            video_vec: Video features of shape (video_dim,) or (batch, video_dim)
            audio_vec: Audio features of shape (audio_dim,) or (batch, audio_dim)
            
        Returns:
            Fused features of shape (video_dim + audio_dim,) or (batch, total_dim)
        """
        # Handle scalar/empty cases
        video_empty = video_vec is None or video_vec.size == 0
        audio_empty = audio_vec is None or audio_vec.size == 0
        
        if video_empty and audio_empty:
            return np.array([], dtype=np.float32)
        
        if video_empty:
            return audio_vec.astype(np.float32)
        
        if audio_empty:
            return video_vec.astype(np.float32)
        
        # Ensure both are at least 1D
        if video_vec.ndim == 0:
            video_vec = video_vec.reshape(1)
        if audio_vec.ndim == 0:
            audio_vec = audio_vec.reshape(1)
        
        # Normalize if requested
        if self.normalize:
            video_vec = self._normalize_vec(video_vec)
            audio_vec = self._normalize_vec(audio_vec)
        
        # Concatenate along last dimension
        return np.concatenate([video_vec, audio_vec], axis=-1).astype(np.float32)
    
    @staticmethod
    def get_fused_dim(video_dim: int, audio_dim: int) -> int:
        """Calculate fused dimension size."""
        return video_dim + audio_dim
