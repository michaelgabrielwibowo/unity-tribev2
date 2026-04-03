"""Phase 1 entry point: run TRIBE-Lite prototype and print scores to terminal."""

import time
import signal
import sys

from tribe_lite.config import TribeLiteConfig
from tribe_lite.pipeline import TribeLitePipeline
from tribe_lite.output.schema import TribeLiteOutput
from tribe_lite.scorer.weight_matrix import create_default_weights


def on_output(output: TribeLiteOutput) -> None:
    """Callback: print output to terminal."""
    # Format top regions
    top_str = " | ".join(output.top_regions) if output.top_regions else "N/A"
    
    # Print single-line summary
    print(
        f"▶ [t={output.timestamp:.0f}] Score: {output.global_score:.2f} | "
        f"Top: {top_str}"
    )
    
    # Optional: print detailed region scores every 10 iterations
    if pipeline.iterations % 10 == 0:
        print("\n  Region scores:")
        sorted_regions = sorted(
            output.region_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]  # Top 5
        for region, score in sorted_regions:
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {region:30s} [{bar}] {score:.2f}")
        print()


def main():
    """Main entry point."""
    print("=" * 60)
    print("TRIBE-Lite — Phase 1 Terminal Prototype")
    print("=" * 60)
    
    # Check/create default weights
    config = TribeLiteConfig()
    if not config.weight_path.exists():
        print(f"\nCreating default weight matrix at {config.weight_path}...")
        create_default_weights(config.weight_path, config)
        print("✓ Default weights created\n")
    
    # Setup graceful shutdown
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        print("\n\nShutting down...")
        running = False
        pipeline.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start pipeline
    print(f"\nConfiguration:")
    print(f"  Window: {config.window_sec}s")
    print(f"  Video FPS: {config.video_fps}")
    print(f"  Audio sample rate: {config.audio_sample_rate} Hz")
    print(f"  Features: {config.fused_dim}-dim fused vector")
    print(f"  Regions: 26 brain areas")
    print()
    
    pipeline = TribeLitePipeline(config, on_output=on_output)
    
    try:
        pipeline.start()
        
        # Run until interrupted
        while running and pipeline.is_running:
            time.sleep(1.0)
            
            # Print status every 5 seconds
            if pipeline.iterations > 0 and pipeline.iterations % 7 == 0:
                print(
                    f"[Status] Uptime: {pipeline.uptime:.0f}s | "
                    f"Iterations: {pipeline.iterations}"
                )
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.stop()
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
