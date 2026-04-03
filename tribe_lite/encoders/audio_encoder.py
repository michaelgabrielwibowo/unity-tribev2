"""Audio encoder: Whisper ASR + MiniLM semantic embedding."""

import numpy as np
from typing import Optional
from collections import deque
import threading

from tribe_lite.encoders.base_encoder import BaseEncoder
from tribe_lite.config import TribeLiteConfig


class AudioEncoder(BaseEncoder):
    """Encodes audio chunks into feature vectors.
    
    Two parallel paths:
    1. Whisper ASR transcription (text)
    2. MiniLM semantic embedding of transcript (384-dim)
    
    Output: (384-dim) feature vector, or zeros if no speech detected
    """
    
    def __init__(self, config: Optional[TribeLiteConfig] = None):
        """Initialize audio encoder.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        super().__init__("AudioEncoder")
        self.config = config or TribeLiteConfig()
        self._whisper_model = None
        self._sem_model = None
        self._transcript_buffer = deque(maxlen=5)  # Keep last 5 transcripts
        self._lock = threading.Lock()
        self._whisper_available = False  # Local flag, not config mutation
        self._semantic_audio_available = False  # Local flag, not config mutation
        # Guard: only one background transcription at a time to prevent thread buildup
        self._is_transcribing = threading.Event()
    
    @property
    def output_dim(self) -> int:
        """Dimension of output feature vector."""
        return self.config.audio_features_dim
    
    def initialize(self) -> None:
        """Load Whisper and MiniLM models."""
        # Load Whisper for ASR
        if self.config.use_whisper and not self._whisper_model:
            try:
                from faster_whisper import WhisperModel
                self._whisper_model = WhisperModel(
                    self.config.whisper_model,
                    device="cpu",
                    compute_type="int8"
                )
                self._whisper_available = True
                print(f"[AudioEncoder] Loaded Whisper model: {self.config.whisper_model}")
            except ImportError:
                print("[AudioEncoder] Warning: faster_whisper not installed, skipping ASR")
                self._whisper_available = False
        
        # Load MiniLM for semantic embedding
        if self.config.use_semantic_audio and not self._sem_model:
            try:
                from sentence_transformers import SentenceTransformer
                self._sem_model = SentenceTransformer(self.config.sem_model)
                self._semantic_audio_available = True
                print(f"[AudioEncoder] loaded MiniLM model: {self.config.sem_model}")
            except ImportError:
                print("[AudioEncoder] Warning: sentence_transformers not installed")
                self._semantic_audio_available = False
        
        self._initialized = True
    
    def _transcribe(self, audio_chunk: np.ndarray) -> str:
        """Transcribe audio chunk with Whisper.
        
        Args:
            audio_chunk: Float32 mono audio at configured sample rate
            
        Returns:
            Transcribed text string
        """
        if not self.config.use_whisper or not self._whisper_available or self._whisper_model is None:
            return ""
        
        try:
            # Ensure float32
            audio_chunk = audio_chunk.astype(np.float32)
            
            # Transcribe
            segments, info = self._whisper_model.transcribe(
                audio_chunk,
                beam_size=1,
                language="en"
            )
            
            # Combine all segments
            text = " ".join([segment.text for segment in segments]).strip()
            return text
            
        except Exception as e:
            print(f"[AudioEncoder] Whisper error: {e}")
            return ""
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text with MiniLM.
        
        Args:
            text: Text string to embed
            
        Returns:
            384-dim embedding vector, or zeros if empty
        """
        if not text.strip():
            return np.zeros(self.output_dim, dtype=np.float32)
        
        if not self.config.use_semantic_audio or not self._semantic_audio_available or self._sem_model is None:
            return np.zeros(self.output_dim, dtype=np.float32)
        
        try:
            embedding = self._sem_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"[AudioEncoder] MiniLM embedding error: {e}")
            return np.zeros(self.output_dim, dtype=np.float32)
    
    def encode(self, audio_chunk: np.ndarray, async_transcribe: bool = False) -> np.ndarray:
        """Encode an audio chunk.
        
        Args:
            audio_chunk: Float32 mono audio at configured sample rate
            async_transcribe: If True, transcribe in background thread.
                NOTE: When async_transcribe=True, the returned embedding reflects transcripts
                from previous windows. The current window's transcript will appear in the
                *next* call's embedding. This is intentional for latency management.
            
        Returns:
            Feature vector of shape (audio_features_dim,)
        """
        if not self.config.use_semantic_audio:
            return np.zeros(self.output_dim, dtype=np.float32)
        
        # Transcribe (optionally async)
        if async_transcribe and self.config.use_whisper and self._whisper_available:
            # Only spawn a new thread if no transcription is already in flight
            if not self._is_transcribing.is_set():
                self._is_transcribing.set()
                thread = threading.Thread(target=self._transcribe_and_cache, args=(audio_chunk,))
                thread.daemon = True
                thread.start()
            # Use cached transcripts for now (one-window lag by design)
            with self._lock:
                text = " ".join(list(self._transcript_buffer))
        else:
            text = self._transcribe(audio_chunk)
            if text:
                with self._lock:
                    self._transcript_buffer.append(text)
        
        # Embed the transcript
        return self._embed_text(text)
    
    def _transcribe_and_cache(self, audio_chunk: np.ndarray) -> None:
        """Background transcription worker."""
        try:
            text = self._transcribe(audio_chunk)
            if text:
                with self._lock:
                    self._transcript_buffer.append(text)
        finally:
            # Always clear the flag so the next window can spawn a thread
            self._is_transcribing.clear()
    
    def get_cached_transcript(self) -> str:
        """Get concatenated cached transcripts."""
        with self._lock:
            return " ".join(list(self._transcript_buffer))
    
    def clear_cache(self) -> None:
        """Clear transcript buffer."""
        with self._lock:
            self._transcript_buffer.clear()
