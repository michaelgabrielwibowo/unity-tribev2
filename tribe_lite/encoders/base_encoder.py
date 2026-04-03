"""Base encoder abstract class."""

from abc import ABC, abstractmethod
import numpy as np


class BaseEncoder(ABC):
    """Abstract base class for all encoders in TRIBE-Lite.
    
    All encoders must implement the encode() method which takes raw input data
    and returns a fixed-dimensional feature vector.
    """
    
    def __init__(self, name: str):
        """Initialize encoder.
        
        Args:
            name: Human-readable name for this encoder
        """
        self.name = name
        self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        """Check if encoder models are loaded and ready."""
        return self._initialized
    
    @abstractmethod
    def encode(self, data: np.ndarray | list) -> np.ndarray:
        """Encode raw input data into a feature vector.
        
        Args:
            data: Raw input data (format depends on encoder type)
            
        Returns:
            Feature vector of fixed dimension
        """
        pass
    
    def initialize(self) -> None:
        """Load models and prepare encoder for use.
        
        Override this method to load heavy models lazily.
        """
        self._initialized = True
    
    def cleanup(self) -> None:
        """Release resources (models, memory, etc.).
        
        Override this method to unload models when done.
        """
        self._initialized = False
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of the output feature vector."""
        pass
