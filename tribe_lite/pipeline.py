"""Main pipeline orchestrator for TRIBE-Lite."""

import time
import threading
from typing import Callable, Optional
import numpy as np

from tribe_lite.config import TribeLiteConfig
from tribe_lite.output.schema import TribeLiteOutput
from tribe_lite.capture.video_capture import VideoCapture
from tribe_lite.capture.audio_capture import AudioCapture
from tribe_lite.encoders.video_encoder import VideoEncoder
from tribe_lite.encoders.audio_encoder import AudioEncoder
from tribe_lite.fusion.fusion_layer import FusionLayer
from tribe_lite.scorer.brain_scorer import BrainScorer


class TribeLitePipeline:
    """Main orchestration loop tying all TRIBE-Lite layers together.
    
    Runs a continuous inference loop:
    1. Capture video frames + audio chunk from current time window
    2. Encode video (optical flow + CLIP)
    3. Encode audio (Whisper ASR + MiniLM)
    4. Fuse multimodal features
    5. Score brain region activations
    6. Call callback with TribeLiteOutput
    
    Usage:
        def on_output(output: TribeLiteOutput):
            print(output)
        
        pipeline = TribeLitePipeline(config, on_output)
        pipeline.start()
        # ... runs until stop() is called
    """
    
    def __init__(self, config: Optional[TribeLiteConfig] = None,
                 on_output: Optional[Callable[[TribeLiteOutput], None]] = None):
        """Initialize pipeline.
        
        Args:
            config: Configuration object. Uses defaults if None.
            on_output: Callback function called with each output
        """
        self.config = config or TribeLiteConfig()
        self.on_output = on_output
        
        # Initialize components
        self.video_capture = VideoCapture(self.config)
        self.audio_capture = AudioCapture(self.config)
        self.video_encoder = VideoEncoder(self.config)
        self.audio_encoder = AudioEncoder(self.config)
        self.fusion_layer = FusionLayer(normalize=self.config.normalize_output)
        self.scorer = BrainScorer(self.config)
        
        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._iteration = 0
        self._start_time = 0.0
        self._overrun_count = 0
    
    def _initialize_models(self) -> None:
        """Load all ML models (called once at startup)."""
        print("[Pipeline] Loading models...")
        self.video_encoder.initialize()
        self.audio_encoder.initialize()
        print("[Pipeline] Models loaded")
    
    def _inference_step(self) -> Optional[TribeLiteOutput]:
        """Run one complete inference cycle.
        
        Returns:
            TribeLiteOutput or None if an error occurred
        """
        try:
            # Step 1: Capture
            video_frames = self.video_capture.get_window_frames()
            audio_chunk = self.audio_capture.get_window_audio()
            
            if not video_frames:
                return None
            
            # Step 2: Encode video
            video_features = self.video_encoder.encode(video_frames)
            
            # Step 3: Encode audio (with async transcription)
            audio_features = self.audio_encoder.encode(
                audio_chunk, 
                async_transcribe=True
            )
            
            # Step 4: Fuse
            fused_features = self.fusion_layer.fuse(video_features, audio_features)
            
            # Step 5: Score
            output = self.scorer.score(fused_features)
            
            # Set timestamp
            output.timestamp = time.time()
            
            return output
            
        except Exception as e:
            print(f"[Pipeline] Inference error: {e}")
            return None
    
    def _inference_loop(self) -> None:
        """Main inference loop running in background thread."""
        print(f"[Pipeline] Starting inference loop (window={self.config.window_sec}s)")
        
        while self._running:
            step_start = time.time()
            
            # Run inference
            output = self._inference_step()
            
            # Call callback
            if output and self.on_output:
                self.on_output(output)
            
            self._iteration += 1
            
            # Calculate sleep time to maintain window timing
            elapsed = time.time() - step_start
            sleep_time = max(0, self.config.window_sec - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Inference overran the configured window; track and notify periodically
                self._overrun_count += 1
                overrun_ms = (elapsed - self.config.window_sec) * 1000
                # Avoid spamming logs; report once every 10 overruns
                if self._overrun_count % 10 == 1:
                    print(
                        f"[Pipeline] WARNING: inference overran window by {overrun_ms:.0f}ms "
                        f"(total overruns: {self._overrun_count}). "
                        f"Consider increasing config.window_sec."
                    )
        
        print("[Pipeline] Inference loop stopped")
    
    def start(self) -> None:
        """Start the pipeline (capture + inference loop)."""
        if self._running:
            print("[Pipeline] Already running")
            return
        
        # Initialize models first
        self._initialize_models()
        
        # Start capture devices
        self.video_capture.start()
        self.audio_capture.start()
        
        # Give capture threads time to warm up
        time.sleep(0.5)
        
        # Start inference loop
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        
        print("[Pipeline] Started")
    
    def stop(self) -> None:
        """Stop the pipeline and release resources."""
        print("[Pipeline] Stopping...")
        self._running = False
        
        # Wait for inference thread
        if self._thread:
            self._thread.join(timeout=5.0)
        
        # Stop capture devices
        self.video_capture.stop()
        self.audio_capture.stop()
        
        # Cleanup encoders
        self.video_encoder.cleanup()
        self.audio_encoder.cleanup()
        
        print("[Pipeline] Stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self._running
    
    @property
    def uptime(self) -> float:
        """Seconds since pipeline started."""
        if self._start_time > 0:
            return time.time() - self._start_time
        return 0.0
    
    @property
    def iterations(self) -> int:
        """Number of inference cycles completed."""
        return self._iteration

    @property
    def overrun_count(self) -> int:
        """Number of inference cycles that exceeded window_sec."""
        return self._overrun_count
    
    def __enter__(self) -> "TribeLitePipeline":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
