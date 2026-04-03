"""TRIBE-Lite: Real-time brain activity estimation from webcam + microphone."""

__version__ = "0.1.0"

# Lazy imports to avoid loading heavy deps on package import
def __getattr__(name):
    if name == "TribeLiteConfig":
        from tribe_lite.config import TribeLiteConfig
        return TribeLiteConfig
    elif name == "TribeLiteOutput":
        from tribe_lite.output.schema import TribeLiteOutput
        return TribeLiteOutput
    elif name == "TribeLitePipeline":
        from tribe_lite.pipeline import TribeLitePipeline
        return TribeLitePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["TribeLiteConfig", "TribeLiteOutput", "TribeLitePipeline"]
