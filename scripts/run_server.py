"""Phase 2: WebSocket server for streaming brain scores to Unity.

Streams TRIBE-Lite output to connected WebSocket clients in real-time.
Connect from Unity using the TribeLiteClient.

Usage:
    python scripts/run_server.py

The server will listen on ws://localhost:8765 by default.
"""

import asyncio
import json
import signal
import sys
from typing import Set

import websockets
from websockets.server import WebSocketServerProtocol

from tribe_lite.config import TribeLiteConfig
from tribe_lite.pipeline import TribeLitePipeline
from tribe_lite.output.schema import TribeLiteOutput
from tribe_lite.scorer.weight_matrix import create_default_weights


class TribeLiteServer:
    """WebSocket server for streaming TRIBE-Lite brain scores."""
    
    def __init__(self, config: TribeLiteConfig | None = None):
        """Initialize server.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or TribeLiteConfig()
        self.pipeline: TribeLitePipeline | None = None
        self.clients: Set[WebSocketServerProtocol] = set()
        self.running = True
        # Event loop used for scheduling async tasks from background threads
        self.loop: asyncio.AbstractEventLoop | None = None
    
    async def handler(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        self.clients.add(websocket)
        print(f"[Server] Client connected: {websocket.remote_address}")
        print(f"[Server] Active clients: {len(self.clients)}")
        
        try:
            # Keep connection alive
            await websocket.wait_closed()
        except Exception as e:
            print(f"[Server] Error: {e}")
        finally:
            self.clients.discard(websocket)
            print(f"[Server] Client disconnected: {websocket.remote_address}")
            print(f"[Server] Active clients: {len(self.clients)}")
    
    def on_pipeline_output(self, output: TribeLiteOutput) -> None:
        """Callback when pipeline produces new output.
        
        Broadcasts to all connected WebSocket clients.
        
        Args:
            output: The brain score output
        """
        if not self.clients:
            return
        
        try:
            json_str = output.to_json(indent=None)  # Compact JSON
            loop = self.loop
            if loop is None:
                # No event loop available (e.g., during shutdown); skip broadcast
                return

            # Send to all clients via the main event loop (thread-safe submission)
            for client in list(self.clients):
                asyncio.run_coroutine_threadsafe(
                    self._send_to_client(client, json_str),
                    loop,
                )
        
        except Exception as e:
            print(f"[Server] Broadcast error: {e}")
    
    async def _send_to_client(self, client: WebSocketServerProtocol, data: str) -> None:
        """Send data to a single client.
        
        Args:
            client: WebSocket client
            data: JSON string to send
        """
        try:
            await client.send(data)
        except Exception as e:
            print(f"[Server] Failed to send to {client.remote_address}: {e}")
    
    def start_pipeline(self) -> None:
        """Start the TRIBE-Lite pipeline in a background thread."""
        import threading
        
        # Check/create weights if needed
        if not self.config.weight_path.exists():
            print(f"Creating default weight matrix at {self.config.weight_path}...")
            create_default_weights(self.config.weight_path, self.config)
        
        # Create pipeline with our callback
        self.pipeline = TribeLitePipeline(
            self.config,
            on_output=self.on_pipeline_output
        )
        
        # Start in background thread
        thread = threading.Thread(target=self.pipeline.start, daemon=True)
        thread.start()
        print("[Server] TRIBE-Lite pipeline started")
    
    def stop_pipeline(self) -> None:
        """Stop the TRIBE-Lite pipeline."""
        if self.pipeline:
            self.pipeline.stop()
            print("[Server] TRIBE-Lite pipeline stopped")
    
    async def serve(self) -> None:
        """Start WebSocket server."""
        # Capture the running event loop so background threads can submit tasks safely
        self.loop = asyncio.get_running_loop()
        print("=" * 60)
        print("TRIBE-Lite — Phase 2 WebSocket Server")
        print("=" * 60)
        print()
        
        print(f"Configuration:")
        print(f"  Server: ws://{self.config.ws_host}:{self.config.ws_port}")
        print(f"  Features: {self.config.fused_dim}-dim fused vector")
        print(f"  Regions: 26 brain areas")
        print()
        
        # Start pipeline
        self.start_pipeline()
        
        # Start WebSocket server
        print(f"Listening on ws://{self.config.ws_host}:{self.config.ws_port}")
        print("Waiting for client connections...")
        print()
        
        async with websockets.serve(
            self.handler,
            self.config.ws_host,
            self.config.ws_port
        ):
            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)


async def _async_main():
    """Asynchronous main entry point."""
    config = TribeLiteConfig()
    server = TribeLiteServer(config)
    
    # Setup graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        server.running = False
        server.stop_pipeline()
        # Close all client connections via the main event loop (thread-safe)
        loop = server.loop
        if loop is not None:
            for client in list(server.clients):
                asyncio.run_coroutine_threadsafe(client.close(), loop)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        server.stop_pipeline()
    
    print("Goodbye!")


def main():
    """Synchronous console entry point used by package scripts."""
    import asyncio as _asyncio
    _asyncio.run(_async_main())


if __name__ == "__main__":
    main()
