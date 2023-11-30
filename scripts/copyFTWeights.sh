#!/bin/bash

# Variables
REMOTE_HOST="104.255.9.187"
REMOTE_PORT="14211"
REMOTE_USER="root"
PRIVATE_KEY_FILE="~/.ssh/gpu" 
REMOTE_DIR="/workspace/yolo/output/*" 
LOCAL_DIR="./output"

# Create local directory if it does not exist
if [ ! -d "$LOCAL_DIR" ]; then
    mkdir -p "$LOCAL_DIR"
fi

LATEST_FILES=$(ssh -i "$PRIVATE_KEY_FILE" -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" \
    "find '$REMOTE_DIR' -type f -printf '%T+ %p\n' | sort -r | awk 'NR>1 {print \$2}' | head -n 3")

# Copy the latest 3 files from remote server to local directory
for file in $LATEST_FILES; do
    scp -i "$PRIVATE_KEY_FILE" -P "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST:\"$file\"" "$LOCAL_DIR"
done
