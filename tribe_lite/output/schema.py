"""Output schema for TRIBE-Lite pipeline results."""

from dataclasses import dataclass, field
import json
import time
from typing import Any


@dataclass
class TribeLiteOutput:
    """Standardized output from the TRIBE-Lite pipeline.
    
    This schema is mirrored in Unity C# code - keep in sync!
    
    Attributes:
        timestamp: Unix timestamp when this window was processed
        global_score: Overall activation score (0.0 - 1.0)
        region_scores: Dictionary mapping region names to scores (0.0 - 1.0)
        top_regions: List of top 3 most active region names
        window_sec: Duration of the time window this covers
        metadata: Optional additional data (e.g., raw vectors, debug info)
    """
    timestamp: float = field(default_factory=time.time)
    global_score: float = 0.0
    region_scores: dict[str, float] = field(default_factory=dict)
    top_regions: list[str] = field(default_factory=list)
    window_sec: float = 0.75
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "global_score": self.global_score,
            "region_scores": self.region_scores,
            "top_regions": self.top_regions,
            "window_sec": self.window_sec,
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int | None = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TribeLiteOutput":
        """Create from dictionary (e.g., deserialized JSON)."""
        return cls(
            timestamp=data.get("timestamp", time.time()),
            global_score=data.get("global_score", 0.0),
            region_scores=data.get("region_scores", {}),
            top_regions=data.get("top_regions", []),
            window_sec=data.get("window_sec", 0.75),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "TribeLiteOutput":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        top_str = ", ".join(self.top_regions) if self.top_regions else "None"
        return (
            f"TribeLiteOutput(score={self.global_score:.2f}, "
            f"top=[{top_str}], regions={len(self.region_scores)})"
        )
