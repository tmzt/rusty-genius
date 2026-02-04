# Ogenius API Reference

Ogenius provides an OpenAI-compatible API for seamless integration with existing tools, alongside custom endpoints for low-latency streaming and asset management.

## OpenAI Compatible Endpoints

### List Models
`GET /v1/models`

Returns a list of all models currently registered in the system (built-in + user-injected).

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen-2.5-1.5b",
      "object": "model",
      "created": 1677610602,
      "owned_by": "rusty-genius"
    }
  ]
}
```

### Chat Completions
`POST /v1/chat/completions`

*Note: Currently being implemented. Use the WebSocket interface for streaming completions.*

## Custom Endpoints

### WebSocket Chat (Streaming)
`GET /` (WebSocket upgrade on `--ws-addr`)

High-performance, low-latency streaming interface used by the Ogenius Web UI.

**Protocol:** JSON-based event stream.

## Configuration & Assets

Models are managed by the `facecrab` registry:
- **`manifest.toml`**: Static user extensions.
- **`registry.toml`**: Dynamic tracking of downloaded assets.
