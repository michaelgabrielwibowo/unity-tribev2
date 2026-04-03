# TRIBE-Lite — Real-time Brain Activity Estimation from Webcam + Microphone

A lightweight, CPU-friendly pipeline that estimates brain region activations from audiovisual input in real time.

## Architecture

```
Webcam → VideoEncoder (Optical Flow + CLIP) ──┐
                                               ├→ FusionLayer → BrainScorer → TribeLiteOutput
Microphone → AudioEncoder (Whisper + MiniLM) ─┘
```

## Quick Start

### Phase 1: Terminal Prototype

```bash
# Install dependencies
pip install -r requirements.txt

# Run the prototype
python scripts/run_prototype.py
```

### Phase 2: Unity Integration

```bash
# Start WebSocket server
python scripts/run_server.py

# In Unity: import NativeWebSocket package and run scene with TribeLiteClient
```

## Folder Structure

```
tribe-lite/
├── tribe_lite/          # Core Python package
│   ├── capture/         # Webcam + microphone input
│   ├── encoders/        # Feature extraction (CLIP, Whisper, MiniLM)
│   ├── fusion/          # Multimodal feature fusion
│   ├── scorer/          # Brain region scoring
│   └── output/          # Output schema
├── unity_client/        # Unity C# client (Phase 2)
├── weights/             # Anatomical weight matrices
├── scripts/             # Entry points and utilities
└── tests/               # Unit tests
```

## Key Features

- **Real-time**: ~0.75s inference windows
- **CPU-only**: No GPU required
- **Modular**: Swap encoders, adjust weights, extend regions
- **Unity-ready**: WebSocket IPC for game integration

## License

MIT
