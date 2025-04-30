#!/bin/bash
# Set environment variables for the FastAPI app
export CONNECT4_CONFIG="configs/h6_w7_c4_small_600.yaml"
export CONNECT4_CHECKPOINT="model_ckpts/h6_w7_c4_current_net_small_step30000.pth"

# Start Redis container
sudo docker run -d --name redis -p 6379:6379 redis:latest

# Start FastAPI server
uvicorn app:app --reload --port 8001