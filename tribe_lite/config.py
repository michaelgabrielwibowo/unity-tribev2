"""Central configuration for TRIBE-Lite pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TribeLiteConfig:
    """Configuration parameters for the TRIBE-Lite pipeline.
    
    All tunable parameters live here - nothing is hardcoded elsewhere.
    """
    # Timing
    window_sec: float = 0.75        # Time window per inference cycle
    video_fps: int = 15             # Webcam capture rate
    audio_sample_rate: int = 16000  # Audio sampling rate

    # Video resolution (set to None to use camera default)
    video_width: int | None = None   # e.g., 640
    video_height: int | None = None  # e.g., 480
    
    # Model choices
    clip_model: str = "ViT-B-32"    # OpenCLIP model name
    whisper_model: str = "tiny.en"  # faster-whisper model size
    sem_model: str = "all-MiniLM-L6-v2"  # Sentence transformer model
    
    # Paths
    weight_matrix_path: str = "weights/default_weights.npz"
    
    # Server
    ws_port: int = 8765
    ws_host: str = "localhost"
    
    # Processing options
    normalize_output: bool = True
    use_optical_flow: bool = True
    use_clip: bool = True
    use_whisper: bool = True
    use_semantic_audio: bool = True

    # Optional temporal smoothing for brain scores
    use_score_smoothing: bool = False
    score_smoothing_alpha: float = 0.3  # EMA alpha in [0, 1]
    
    @property
    def weight_path(self) -> Path:
        """Get absolute path to weight matrix file."""
        return Path(self.weight_matrix_path)
    
    @property
    def video_features_dim(self) -> int:
        """Dimension of video feature vector."""
        dim = 0
        if self.use_optical_flow:
            dim += 10  # Motion features: mean, max, 8-bin histogram
        if self.use_clip:
            dim += 512  # CLIP embedding
        return dim
    
    @property
    def audio_features_dim(self) -> int:
        """Dimension of audio feature vector."""
        if self.use_semantic_audio:
            return 384  # MiniLM embedding
        return 0
    
    @property
    def fused_dim(self) -> int:
        """Dimension of fused feature vector."""
        return self.video_features_dim + self.audio_features_dim
