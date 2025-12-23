#!/bin/bash

# Start server in background
python3.11 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 3

# Run demo
python3.11 scripts/demo_full_workflow.py

# Kill server
echo "Stopping server..."
kill $SERVER_PID
