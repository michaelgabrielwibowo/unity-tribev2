"""Audio capture: sounddevice ring buffer streaming."""

import numpy as np
from typing import Optional
import threading
import time
import sounddevice as sd

from tribe_lite.config import TribeLiteConfig


class AudioCapture:
    """Continuous microphone capture into a thread-safe ring buffer.
    
    Captures mono audio at configured sample rate and maintains
    a rolling buffer of the most recent samples for the current time window.
    """
    
    def __init__(self, config: Optional[TribeLiteConfig] = None):
        """Initialize audio capture.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or TribeLiteConfig()
        self._stream: Optional[sd.InputStream] = None
        self._ring_buffer: Optional[np.ndarray] = None
        self._write_idx = 0
        self._lock = threading.Lock()
        self._running = False
        self._last_sample_time = 0.0
    
    @property
    def sample_rate(self) -> int:
        """Configured sample rate."""
        return self.config.audio_sample_rate
    
    @property
    def window_samples(self) -> int:
        """Number of samples in a time window."""
        return int(self.config.window_sec * self.config.audio_sample_rate)
    
    @property
    def buffer_size(self) -> int:
        """Total ring buffer size (2x window for safety margin)."""
        return self.window_samples * 2
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                        time_info: dict, status: sd.CallbackFlags) -> None:
        """Sounddevice stream callback - called when new audio arrives.
        
        Args:
            indata: Input audio data (frames x channels)
            frames: Number of frames
            time_info: Timing information
            status: Callback status flags
        """
        if status:
            print(f"[AudioCapture] Stream status: {status}")
        
        # Convert to mono if stereo
        if indata.shape[1] > 1:
            indata = indata.mean(axis=1, keepdims=True)
        
        # Write to ring buffer
        with self._lock:
            for sample in indata.flatten():
                self._ring_buffer[self._write_idx] = sample
                self._write_idx = (self._write_idx + 1) % self.buffer_size
        
        self._last_sample_time = time.time()
    
    def start(self) -> None:
        """Start audio capture stream."""
        if self._running:
            return
        
        # Initialize ring buffer
        self._ring_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self._write_idx = 0
        
        # Start sounddevice stream
        try:
            self._stream = sd.InputStream(
                samplerate=self.config.audio_sample_rate,
                channels=1,
                callback=self._audio_callback,
                blocksize=1024,
                latency='low'
            )
            self._stream.start()
            self._running = True
            print(f"[AudioCapture] Started at {self.config.audio_sample_rate} Hz")
        except Exception as e:
            raise RuntimeError(f"Failed to start audio capture: {e}")
    
    def stop(self) -> None:
        """Stop audio capture stream."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[AudioCapture] Stopped")
    
    def get_window_audio(self) -> np.ndarray:
        """Get audio samples from the most recent time window.
        
        Returns:
            Float32 mono audio array of shape (window_samples,)
        """
        with self._lock:
            if self._ring_buffer is None:
                return np.zeros(self.window_samples, dtype=np.float32)
            
            # Calculate read position (window_samples behind write index)
            read_start = (self._write_idx - self.window_samples) % self.buffer_size
            
            # Handle wraparound
            if read_start + self.window_samples <= self.buffer_size:
                # No wraparound
                audio = self._ring_buffer[read_start:read_start + self.window_samples].copy()
            else:
                # Wraparound case
                first_part = self._ring_buffer[read_start:].copy()
                second_part = self._ring_buffer[:self.window_samples - len(first_part)].copy()
                audio = np.concatenate([first_part, second_part])
            
            return audio
    
    def get_rms_level(self) -> float:
        """Get current RMS amplitude level (for VU meter, etc.).
        
        Returns:
            RMS amplitude of current window
        """
        audio = self.get_window_audio()
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def is_healthy(self) -> bool:
        """Check if capture is running and receiving samples."""
        if not self._running:
            return False
        if self._stream is None or not self._stream.active:
            return False
        # Check if we've received samples in the last 2 seconds
        return (time.time() - self._last_sample_time) < 2.0
    
    def __enter__(self) -> "AudioCapture":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
