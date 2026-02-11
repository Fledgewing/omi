#!/bin/bash

# Quick start script for testing the n8n forwarder setup

echo "🚀 Build and push latest"
echo "========================================"

# Build services
echo "Building services..."
docker build -f dockerfile.forwarder -t temporalise/n8n-forwarder .
docker build -t temporalise/flask-audio-saver .

echo "Pushing images to Docker Hub..."
docker push temporalise/n8n-forwarder
docker push temporalise/flask-audio-saver

echo ""
echo "✅ Services pushed successfully!"
echo ""
