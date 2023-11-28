#!/bin/bash

# Variables
REMOTE_HOST="97.107.178.213"
REMOTE_PORT="12703"
REMOTE_USER="root"
PRIVATE_KEY_FILE="~/.ssh/gpu" 
REMOTE_DIR="/workspace/yolo/output/*" 
LOCAL_DIR="./output"

# Copy files from remote server to local directory
scp -r -i "$PRIVATE_KEY_FILE" -P "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR" "$LOCAL_DIR"
