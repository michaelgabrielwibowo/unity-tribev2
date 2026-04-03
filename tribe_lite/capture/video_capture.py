"""Video capture: OpenCV webcam streaming."""

import cv2
import numpy as np
from typing import Optional, Generator
from collections import deque
import threading
import time

from tribe_lite.config import TribeLiteConfig


class VideoCapture:
    """Continuous webcam capture into a thread-safe frame buffer.
    
    Captures frames at configured FPS and maintains a rolling buffer
    of the most recent frames for the current time window.
    """
    
    def __init__(self, config: Optional[TribeLiteConfig] = None):
        """Initialize video capture.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or TribeLiteConfig()
        self._cap = None
        self._frame_buffer: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_frame_time = 0.0
    
    @property
    def fps(self) -> int:
        """Configured capture FPS."""
        return self.config.video_fps
    
    @property
    def window_frames(self) -> int:
        """Number of frames in a time window."""
        return int(self.config.window_sec * self.config.video_fps)
    
    def start(self) -> None:
        """Start background capture thread."""
        if self._running:
            return
        
        # Open camera
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("Failed to open webcam")
        
        # Set FPS
        self._cap.set(cv2.CAP_PROP_FPS, self.config.video_fps)
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[VideoCapture] Started at {self.config.video_fps} FPS")
    
    def stop(self) -> None:
        """Stop capture and release camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        print("[VideoCapture] Stopped")
    
    def _capture_loop(self) -> None:
        """Background capture thread main loop."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            
            # Add to buffer (oldest frames automatically dropped)
            with self._lock:
                self._frame_buffer.append(frame.copy())
                # Keep only enough frames for one window plus some margin
                max_frames = self.window_frames + 5
                while len(self._frame_buffer) > max_frames:
                    self._frame_buffer.popleft()
            
            self._last_frame_time = time.time()
            
            # Sleep to maintain target FPS
            sleep_time = (1.0 / self.config.video_fps) - 0.002  # Small buffer
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_window_frames(self) -> list[np.ndarray]:
        """Get frames from the most recent time window.
        
        Returns:
            List of BGR frames (uint8) from the current window
        """
        with self._lock:
            frames = list(self._frame_buffer)[-self.window_frames:]
        return frames
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent captured frame.
        
        Returns:
            Latest BGR frame or None if no frames captured
        """
        with self._lock:
            if self._frame_buffer:
                return self._frame_buffer[-1].copy()
        return None
    
    def is_healthy(self) -> bool:
        """Check if capture is running and receiving frames."""
        if not self._running:
            return False
        if not self._cap or not self._cap.isOpened():
            return False
        # Check if we've received a frame in the last 2 seconds
        return (time.time() - self._last_frame_time) < 2.0
    
    def __enter__(self) -> "VideoCapture":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
