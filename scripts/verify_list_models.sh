#!/bin/bash
set -e

# Start server in background
echo "🚀 Starting Ogenius server..."
./target/release/ogenius serve --addr 127.0.0.1:9090 &
SERVER_PID=$!

# Wait for server
sleep 3

# Query models
echo "📡 Querying /v1/models..."
RESPONSE=$(curl -s http://127.0.0.1:9090/v1/models)
echo "Response: $RESPONSE"

# Check if response contains "data" and "object":"list"
if [[ "$RESPONSE" == *"\"object\":\"list\""* ]]; then
    echo "✅ list_models verified successfully"
else
    echo "❌ list_models verification failed"
    kill $SERVER_PID
    exit 1
fi

# Kill server
kill $SERVER_PID
echo "🛑 Server stopped"
