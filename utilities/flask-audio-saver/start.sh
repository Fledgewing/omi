#!/bin/bash

# Quick start script for testing the n8n forwarder setup

echo "🚀 Starting Audio Saver + N8N Forwarder"
echo "========================================"

# Check if exports directory exists
if [ ! -d "./exports" ]; then
    echo "Creating exports directory..."
    mkdir -p ./exports/final
fi

# Stop existing containers if any
echo "Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Build and start services
echo "Building and starting services..."
docker build -f dockerfile.forwarder -t temporalise/n8n-forwarder .
docker build -t temporalise/flask-audio-saver .
docker-compose up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 3

# Show status
echo ""
echo "📊 Container Status:"
docker-compose ps

echo ""
echo "✅ Services started successfully!"
echo ""
echo "To view logs, run:"
echo "  docker-compose logs -f"
echo ""
echo "To stop services, run:"
echo "  docker-compose down"
