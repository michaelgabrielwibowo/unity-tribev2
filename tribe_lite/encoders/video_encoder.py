"""Video encoder: optical flow + CLIP visual semantics."""

import numpy as np
import cv2
from typing import Optional

from tribe_lite.encoders.base_encoder import BaseEncoder
from tribe_lite.config import TribeLiteConfig


class VideoEncoder(BaseEncoder):
    """Encodes video frames into feature vectors.
    
    Two parallel paths:
    1. Optical flow motion features (10-dim: mean mag, max mag, 8-bin histogram)
    2. CLIP visual semantics (512-dim)
    
    Combined output: (522-dim) feature vector when both enabled
    """
    
    def __init__(self, config: Optional[TribeLiteConfig] = None):
        """Initialize video encoder.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        super().__init__("VideoEncoder")
        self.config = config or TribeLiteConfig()
        self._clip_model = None
        self._clip_available = False  # Local flag, not config mutation

        # Running statistics for optical flow standardization (fix 4.2)
        self._flow_n = 0
        self._flow_mean = np.zeros(10, dtype=np.float32)
        self._flow_s = np.zeros(10, dtype=np.float32)

    
    @property
    def output_dim(self) -> int:
        """Dimension of output feature vector."""
        return self.config.video_features_dim
    
    def initialize(self) -> None:
        """Load CLIP model if enabled."""
        if self.config.use_clip and not self._clip_model:
            try:
                import open_clip
                self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                    self.config.clip_model, pretrained="openai"
                )
                self._clip_model.eval()
                self._clip_available = True
                print(f"[VideoEncoder] Loaded CLIP model: {self.config.clip_model}")
            except ImportError:
                print("[VideoEncoder] Warning: open_clip not installed, skipping CLIP")
                self._clip_available = False
        
        self._initialized = True
    
    def _compute_optical_flow(self, frames: list[np.ndarray]) -> np.ndarray:
        """Compute optical flow features from frame sequence.
        
        Returns a 10-dimensional motion feature vector:
        - Mean magnitude (1)
        - Max magnitude (1)
        - Direction histogram (8 bins, normalized across 0-2π)

        Args:
            frames: List of BGR frames (uint8)

        Returns:
            10-dimensional motion feature vector
        """
        if len(frames) < 2:
            return np.zeros(10, dtype=np.float32)
        
        # Convert to grayscale
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        
        flow_magnitudes = []
        flow_angles = []
        
        # Compute flow between consecutive frames
        for i in range(len(gray_frames) - 1):
            prev = gray_frames[i]
            curr = gray_frames[i + 1]
            
            # Dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
            
            # Magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(magnitude)
            flow_angles.append(angle)
        
        if not flow_magnitudes:
            return np.zeros(10, dtype=np.float32)
        
        # Aggregate statistics
        all_mags = np.concatenate([m.flatten() for m in flow_magnitudes])
        all_angles = np.concatenate([a.flatten() for a in flow_angles])
        
        # Mean and max magnitude
        mean_mag = float(np.mean(all_mags))
        max_mag = float(np.max(all_mags))
        
        # Direction histogram (8 bins, 0-2π)
        hist, _ = np.histogram(all_angles, bins=8, range=(0, 2 * np.pi), density=True)
        
        # Combine into 10-dim vector: [mean_mag, max_mag, hist[0..7]]
        features = np.array([mean_mag, max_mag, *hist], dtype=np.float32)
        
        # Update running statistics and return standardized optical flow features
        self._update_flow_stats(features)
        return self._normalize_flow(features)

    def _update_flow_stats(self, flow_vec: np.ndarray) -> None:
        """Update running mean/std stats for flow feature normalization."""
        assert flow_vec.shape == (10,), "Optical flow vector must be 10-dim"

        self._flow_n += 1
        delta = flow_vec - self._flow_mean
        self._flow_mean += delta / self._flow_n
        delta2 = flow_vec - self._flow_mean
        self._flow_s += delta * delta2

    def _normalize_flow(self, flow_vec: np.ndarray) -> np.ndarray:
        """Standardize flow vector using running mean and std deviation."""
        if self._flow_n < 2:
            # Not enough history to compute stable standard deviation
            return flow_vec

        variance = np.maximum(self._flow_s / (self._flow_n - 1), 1e-6)
        std = np.sqrt(variance)
        return ((flow_vec - self._flow_mean) / std).astype(np.float32)

    
    def _encode_clip(self, frame: np.ndarray) -> np.ndarray:
        """Encode single frame with CLIP.
        
        Args:
            frame: BGR frame (uint8)
            
        Returns:
            512-dim CLIP embedding
        """
        if not self.config.use_clip or not self._clip_available or self._clip_model is None:
            return np.zeros(self.config.clip_features_dim, dtype=np.float32)
        
        try:
            import torch
            from PIL import Image
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Preprocess and encode
            input_tensor = self._clip_preprocess(pil_image).unsqueeze(0)
            with torch.no_grad():
                embedding = self._clip_model.encode_image(input_tensor)
            
            # Normalize and convert to numpy
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.squeeze().cpu().numpy().astype(np.float32)
            
        except Exception as e:
            print(f"[VideoEncoder] CLIP encoding error: {e}")
            return np.zeros(self.config.clip_features_dim, dtype=np.float32)
    
    def encode(self, frames: list[np.ndarray]) -> np.ndarray:
        """Encode a sequence of video frames.
        
        Args:
            frames: List of BGR frames from the time window
            
        Returns:
            Feature vector of shape (video_features_dim,)
        """
        if not frames:
            return np.zeros(self.output_dim, dtype=np.float32)
        
        features = []
        
        # Path 1: Optical flow motion features
        if self.config.use_optical_flow:
            flow_features = self._compute_optical_flow(frames)
            features.append(flow_features)
        
        # Path 2: CLIP semantic features (center frame)
        if self.config.use_clip:
            center_idx = len(frames) // 2
            clip_features = self._encode_clip(frames[center_idx])
            features.append(clip_features)
        
        # Concatenate all features
        if features:
            return np.concatenate(features).astype(np.float32)
        else:
            return np.zeros(self.output_dim, dtype=np.float32)
