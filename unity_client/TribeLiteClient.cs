using UnityEngine;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

#if UNITY_WEBGL && !UNITY_EDITOR
    using WebSocketSharp;
#else
    using WebSocketSharp;
#endif


/// <summary>
/// TRIBE-Lite client for receiving brain scores via WebSocket.
/// 
/// Usage:
///   1. Add this script to a GameObject in your scene
///   2. Configure serverUrl in the Inspector (default: ws://localhost:8765)
///   3. Access brain scores via TribeLiteClient.Instance.LastOutput
/// 
/// Example:
///   void Update() {
///       if (TribeLiteClient.Instance != null && TribeLiteClient.Instance.LastOutput != null) {
///           float score = TribeLiteClient.Instance.LastOutput.global_score;
///           Debug.Log($"Brain score: {score:F2}");
///       }
///   }
/// </summary>
public class TribeLiteClient : MonoBehaviour
{
    [Header("Connection")]
    [SerializeField]
    private string serverUrl = "ws://localhost:8765";
    
    [SerializeField]
    private bool autoConnect = true;
    
    [SerializeField]
    private float reconnectDelaySeconds = 3f;
    
    [SerializeField]
    private int maxReconnectAttempts = 10;
    
    // Singleton
    public static TribeLiteClient Instance { get; private set; }
    
    // Public events
    public event System.Action<TribeLiteOutput> OnOutputReceived;
    public event System.Action OnConnected;
    public event System.Action<string> OnDisconnected;
    
    // State
    public TribeLiteOutput LastOutput { get; private set; }
    public bool IsConnected => _ws != null && _ws.IsAlive;
    
    private WebSocket _ws;
    private int _reconnectAttempts;
    private CancellationTokenSource _reconnectCts;
    private Queue<string> _messageQueue = new Queue<string>();
    private object _messageLock = new object();
    
    private void Awake()
    {
        // Singleton setup
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);
    }
    
    private void Start()
    {
        if (autoConnect)
        {
            Connect();
        }
    }
    
    private void Update()
    {
        // Process queued messages on main thread
        lock (_messageLock)
        {
            while (_messageQueue.Count > 0)
            {
                string json = _messageQueue.Dequeue();
                ProcessMessage(json);
            }
        }
    }
    
    /// <summary>Connect to the TRIBE-Lite WebSocket server.</summary>
    public void Connect()
    {
        if (IsConnected)
        {
            Debug.LogWarning("[TribeLiteClient] Already connected");
            return;
        }
        
        try
        {
            _ws = new WebSocket(serverUrl);
            _ws.OnOpen += OnWebSocketOpen;
            _ws.OnMessage += OnWebSocketMessage;
            _ws.OnError += OnWebSocketError;
            _ws.OnClose += OnWebSocketClose;
            
            _ws.Connect();
            _reconnectAttempts = 0;
        }
        catch (Exception e)
        {
            Debug.LogError($"[TribeLiteClient] Connection failed: {e.Message}");
            ScheduleReconnect();
        }
    }
    
    /// <summary>Disconnect from the server.</summary>
    public void Disconnect()
    {
        if (_ws != null && _ws.IsAlive)
        {
            _ws.Close();
        }
    }
    
    private void OnWebSocketOpen()
    {
        Debug.Log("[TribeLiteClient] Connected to server");
        _reconnectAttempts = 0;
        OnConnected?.Invoke();
    }
    
    private void OnWebSocketMessage(string data)
    {
        // Queue message for main thread processing
        lock (_messageLock)
        {
            _messageQueue.Enqueue(data);
        }
    }
    
    private void OnWebSocketError(string error)
    {
        Debug.LogError($"[TribeLiteClient] WebSocket error: {error}");
    }
    
    private void OnWebSocketClose(CloseEventArgs e)
    {
        string reason = e?.Reason ?? "Unknown";
        Debug.LogWarning($"[TribeLiteClient] Disconnected: {reason}");
        OnDisconnected?.Invoke(reason);
        
        ScheduleReconnect();
    }
    
    private void ProcessMessage(string json)
    {
        try
        {
            LastOutput = JsonUtility.FromJson<TribeLiteOutput>(json);
            OnOutputReceived?.Invoke(LastOutput);
        }
        catch (Exception e)
        {
            Debug.LogError($"[TribeLiteClient] JSON parse error: {e.Message}");
        }
    }
    
    private void ScheduleReconnect()
    {
        if (_reconnectAttempts >= maxReconnectAttempts)
        {
            Debug.LogError($"[TribeLiteClient] Max reconnection attempts ({maxReconnectAttempts}) exceeded");
            return;
        }
        
        _reconnectAttempts++;
        Debug.Log($"[TribeLiteClient] Reconnecting in {reconnectDelaySeconds}s (attempt {_reconnectAttempts}/{maxReconnectAttempts})");
        
        _ = ReconnectAsync();
    }
    
    private async Task ReconnectAsync()
    {
        await Task.Delay((int)(reconnectDelaySeconds * 1000));
        if (!IsConnected)
        {
            Connect();
        }
    }
    
    private void OnDestroy()
    {
        Disconnect();
        if (Instance == this)
        {
            Instance = null;
        }
    }
}


/// <summary>
/// Serializable output from TRIBE-Lite.
/// Matches the Python TribeLiteOutput schema.
/// </summary>
[System.Serializable]
public class TribeLiteOutput
{
    /// <summary>Unix timestamp when this window was processed</summary>
    public float timestamp;
    
    /// <summary>Overall activation score (0.0 - 1.0)</summary>
    public float global_score;
    
    /// <summary>Scores for each of 26 brain regions (0.0 - 1.0)</summary>
    public SerializableDictionary region_scores;
    
    /// <summary>Top 3 most active regions</summary>
    public List<string> top_regions;
    
    /// <summary>Duration of the time window this covers (seconds)</summary>
    public float window_sec;
}


/// <summary>
/// Wrapper for serializing Dictionary in Unity.
/// JSON layout: {"region_name": 0.75, ...}
/// </summary>
[System.Serializable]
public class SerializableDictionary
{
    [SerializeField]
    private List<RegionScore> items = new List<RegionScore>();
    
    private Dictionary<string, float> _dict;
    
    [System.Serializable]
    private class RegionScore
    {
        public string region;
        public float score;
    }
    
    // Note: Unity's JsonUtility doesn't natively support Dictionary.
    // For full deserialization, use a third-party JSON library like Newtonsoft.Json.
    // This is a simplified placeholder.
    
    public float GetScore(string region)
    {
        foreach (var item in items)
        {
            if (item.region == region)
                return item.score;
        }
        return 0f;
    }
}
