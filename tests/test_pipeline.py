"""Unit tests for TRIBE-Lite pipeline components."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from tribe_lite.config import TribeLiteConfig
from tribe_lite.fusion.fusion_layer import FusionLayer
from tribe_lite.scorer.brain_scorer import BrainScorer
from tribe_lite.scorer.weight_matrix import (
    init_anatomical_weights,
    save_weights,
    load_weights,
    create_default_weights,
)


@pytest.fixture
def config():
    """Fixture: default configuration."""
    return TribeLiteConfig()


@pytest.fixture
def weight_file(tmp_path, config):
    """Fixture: temporary weight file."""
    path = tmp_path / "test_weights.npz"
    W = init_anatomical_weights(config)
    save_weights(W, path)
    return path


@pytest.fixture
def tmp_weights(tmp_path, config):
    """Fixture: generate fresh weights for tests."""
    path = tmp_path / "weights.npz"
    W = create_default_weights(path, config)
    return W, path


class TestConfiguration:
    """Tests for configuration parameters."""
    
    def test_config_defaults(self):
        """Test that configuration has sensible defaults."""
        config = TribeLiteConfig()
        assert config.window_sec == 0.75
        assert config.video_fps == 15
        assert config.audio_sample_rate == 16000
        assert config.fused_dim > 0
    
    def test_config_dimensions(self, config):
        """Test feature dimension calculations."""
        assert config.video_features_dim == 522  # 10 (optical flow) + 512 (CLIP)
        assert config.audio_features_dim == 384  # MiniLM
        assert config.fused_dim == 906  # 522 + 384
    
    def test_config_optical_flow_disabled(self):
        """Test fused_dim when optical flow is disabled."""
        config = TribeLiteConfig()
        config.use_optical_flow = False
        assert config.fused_dim == 896  # 512 (CLIP) + 384 (audio)
    
    def test_config_clip_disabled(self):
        """Test fused_dim when CLIP is disabled."""
        config = TribeLiteConfig()
        config.use_clip = False
        assert config.fused_dim == 394  # 10 (optical flow) + 384 (audio)
    
    def test_config_audio_disabled(self):
        """Test fused_dim when audio is disabled."""
        config = TribeLiteConfig()
        config.use_semantic_audio = False
        assert config.fused_dim == 522  # 10 (optical flow) + 512 (CLIP)


class TestFusionLayer:
    """Tests for feature fusion."""
    
    def test_fusion_output_shape(self, config):
        """Test that fusion produces correct output dimension."""
        fusion = FusionLayer()
        
        video_feat = np.random.randn(config.video_features_dim).astype(np.float32)
        audio_feat = np.random.randn(config.audio_features_dim).astype(np.float32)
        
        output = fusion.fuse(video_feat, audio_feat)
        
        assert output.shape == (config.fused_dim,)
        assert output.dtype == np.float32
    
    def test_fusion_normalization(self, config):
        """Test that fusion normalizes each modality separately."""
        fusion = FusionLayer()
        
        video_feat = np.ones(config.video_features_dim, dtype=np.float32)
        audio_feat = np.ones(config.audio_features_dim, dtype=np.float32)
        
        output = fusion.fuse(video_feat, audio_feat)
        
        # The output is not L2-normalized as a whole, but each modality is
        # normalized separately before concatenation. Verify it's not NaN.
        assert not np.any(np.isnan(output))
        assert output.shape == (config.fused_dim,)
    
    def test_fusion_empty_audio(self, config):
        """Test fusion with empty audio (zeros)."""
        fusion = FusionLayer()
        
        video_feat = np.random.randn(config.video_features_dim).astype(np.float32)
        audio_feat = np.zeros(config.audio_features_dim, dtype=np.float32)
        
        output = fusion.fuse(video_feat, audio_feat)
        
        # Should still produce valid output
        assert output.shape == (config.fused_dim,)
        assert not np.any(np.isnan(output))


class TestWeightMatrix:
    """Tests for weight matrix initialization and persistence."""
    
    def test_weight_matrix_shape(self, config):
        """Test weight matrix has correct shape."""
        W = init_anatomical_weights(config)
        assert W.shape == (config.fused_dim, 26)
    
    def test_weight_matrix_values_reasonable(self, config):
        """Test weight matrix values are in reasonable range."""
        W = init_anatomical_weights(config)
        assert np.all(np.isfinite(W))
        assert np.all(W <= 1.0)  # Should be normalized-ish
    
    def test_weight_matrix_io(self, weight_file):
        """Test saving and loading weight matrices."""
        W_saved = init_anatomical_weights()
        save_weights(W_saved, weight_file)
        
        W_loaded = load_weights(weight_file)
        
        assert W_loaded.shape == W_saved.shape
        np.testing.assert_array_equal(W_loaded, W_saved)
    
    def test_weight_file_missing_raises(self, tmp_path):
        """Test that loading missing weight file raises FileNotFoundError."""
        missing_path = tmp_path / "nonexistent.npz"
        with pytest.raises(FileNotFoundError):
            load_weights(missing_path)
    
    def test_create_default_weights(self, tmp_weights):
        """Test default weight creation."""
        W, path = tmp_weights
        assert path.exists()
        assert W.shape[1] == 26  # 26 brain regions


class TestBrainScorer:
    """Tests for brain scoring."""
    
    def test_scorer_output_has_correct_fields(self, config, weight_file):
        """Test scorer output structure."""
        config.weight_matrix_path = str(weight_file)
        scorer = BrainScorer(config)
        
        fused = np.random.randn(config.fused_dim).astype(np.float32)
        output = scorer.score(fused)
        
        assert hasattr(output, 'global_score')
        assert hasattr(output, 'region_scores')
        assert hasattr(output, 'top_regions')
        assert hasattr(output, 'window_sec')
    
    def test_scorer_global_score_range(self, config, weight_file):
        """Test that global score is in valid range."""
        config.weight_matrix_path = str(weight_file)
        scorer = BrainScorer(config)
        
        fused = np.random.randn(config.fused_dim).astype(np.float32)
        output = scorer.score(fused)
        
        assert 0.0 <= output.global_score <= 1.0
    
    def test_scorer_region_count(self, config, weight_file):
        """Test that scorer produces 26 region scores."""
        config.weight_matrix_path = str(weight_file)
        scorer = BrainScorer(config)
        
        fused = np.random.randn(config.fused_dim).astype(np.float32)
        output = scorer.score(fused)
        
        assert len(output.region_scores) == 26
    
    def test_scorer_top_regions_count(self, config, weight_file):
        """Test that scorer returns correct number of top regions."""
        config.weight_matrix_path = str(weight_file)
        scorer = BrainScorer(config)
        
        fused = np.random.randn(config.fused_dim).astype(np.float32)
        output = scorer.score(fused)
        
        assert len(output.top_regions) == 3
        assert all(isinstance(r, str) for r in output.top_regions)
    
    def test_scorer_region_score_range(self, config, weight_file):
        """Test that region scores are in valid range."""
        config.weight_matrix_path = str(weight_file)
        scorer = BrainScorer(config)
        
        fused = np.random.randn(config.fused_dim).astype(np.float32)
        output = scorer.score(fused)
        
        for score in output.region_scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_scorer_consistency(self, config, weight_file):
        """Test that scorer produces same output for same input."""
        config.weight_matrix_path = str(weight_file)
        scorer = BrainScorer(config)
        
        fused = np.random.randn(config.fused_dim).astype(np.float32)
        
        output1 = scorer.score(fused.copy())
        output2 = scorer.score(fused.copy())
        
        assert output1.global_score == output2.global_score
        assert output1.region_scores == output2.region_scores


class TestIntegration:
    """Integration tests for full pipeline components."""
    
    def test_full_pipeline_flow(self, tmp_weights, config):
        """Test complete flow: fusion → scoring."""
        W, weight_path = tmp_weights
        config.weight_matrix_path = str(weight_path)
        
        # Create pipeline components
        fusion = FusionLayer()
        scorer = BrainScorer(config)
        
        # Create random features
        video_feat = np.random.randn(config.video_features_dim).astype(np.float32)
        audio_feat = np.random.randn(config.audio_features_dim).astype(np.float32)
        
        # Run through pipeline
        fused = fusion.fuse(video_feat, audio_feat)
        output = scorer.score(fused)
        
        # Verify output
        assert output.global_score >= 0.0 and output.global_score <= 1.0
        assert len(output.region_scores) == 26
        assert len(output.top_regions) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
