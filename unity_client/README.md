# TRIBE-Lite Unity Client

This directory contains the C# WebSocket client for connecting Unity to the TRIBE-Lite brain scoring pipeline.

## Setup

### 1. Dependencies

Add the **WebSocketSharp** package to your Unity project. Install via Package Manager:

```
Window → TextMesh Pro → Import TMP Essential Resources
```

Or install WebSocketSharp directly:
- Download from GitHub: https://github.com/sta/websocket-sharp
- Or use NuGet Package Manager (if available in your setup)

For more advanced JSON deserialization, consider using **Newtonsoft.Json (Json.NET)**:
- Package: `com.unity.nuget.newtonsoft-json`

### 2. Add to Scene

1. Create a new GameObject in your scene (e.g., "TribeLiteManager")
2. Attach the `TribeLiteClient.cs` script to it
3. Configure in the Inspector:
   - **Server URL**: `ws://localhost:8765` (or your server address)
   - **Auto Connect**: `true` (connects automatically on Start)

### 3. Start the Pipeline

Before running your Unity scene, start the TRIBE-Lite server:

```bash
# Terminal 1: Generate weights (first time only)
python scripts/generate_weights.py

# Terminal 2: Start the WebSocket server
python scripts/run_server.py
```

You should see:
```
============================================================
TRIBE-Lite — Phase 2 WebSocket Server
============================================================

Configuration:
  Server: ws://localhost:8765
  Features: 906-dim fused vector
  Regions: 26 brain areas

Listening on ws://localhost:8765
Waiting for client connections...
```

### 4. Use in Scripts

Access the latest brain scores in any of your Unity scripts:

```csharp
void Update() {
    if (TribeLiteClient.Instance != null && TribeLiteClient.Instance.IsConnected)
    {
        TribeLiteOutput output = TribeLiteClient.Instance.LastOutput;
        if (output != null)
        {
            // Use the brain score
            float score = output.global_score;  // 0.0 - 1.0
            List<string> topRegions = output.top_regions;  // Top 3 active regions
            
            Debug.Log($"Brain activation: {score:P1}");
            Debug.Log($"Top regions: {string.Join(", ", topRegions)}");
        }
    }
}
```

### 5. Events

Subscribe to connection events:

```csharp
void Start() {
    if (TribeLiteClient.Instance != null)
    {
        TribeLiteClient.Instance.OnConnected += HandleConnected;
        TribeLiteClient.Instance.OnDisconnected += HandleDisconnected;
        TribeLiteClient.Instance.OnOutputReceived += HandleNewScore;
    }
}

void HandleConnected() {
    Debug.Log("Connected to TRIBE-Lite server");
}

void HandleDisconnected(string reason) {
    Debug.LogWarning($"Disconnected: {reason}");
}

void HandleNewScore(TribeLiteOutput output) {
    Debug.Log($"New score: {output.global_score:F4}");
}
```

## Output Schema

Each WebSocket message contains:

```json
{
  "timestamp": 1712131234.567,
  "global_score": 0.73,
  "region_scores": {
    "Visual Cortex": 0.85,
    "Auditory Cortex": 0.42,
    ...
  },
  "top_regions": ["Visual Cortex", "Prefrontal Cortex", "Motor Cortex"],
  "window_sec": 0.75
}
```

## Architecture

- **TribeLiteClient**: Main singleton for connection management and message handling
- **TribeLiteOutput**: Serializable class matching the Python output schema
- **SerializableDictionary**: Wrapper for region scores (optional, for full deserialization use external JSON library)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection failed" | Check server URL, ensure `run_server.py` is running |
| "Parse error" | Ensure server is sending valid JSON; check console logs |
| WebSocket library not found | Install WebSocketSharp via Package Manager |
| Dictionary deserialization fails | Use Newtonsoft.Json for full JSON support |

## Notes

- The client uses a **Singleton pattern** (`TribeLiteClient.Instance`)
- Messages are processed on the **main thread** via a queue
- Automatic **reconnection** with exponential backoff (default: 3s delay, max 10 attempts)
- Thread-safe message handling to prevent race conditions
