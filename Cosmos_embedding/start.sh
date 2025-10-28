#!/bin/bash

# Cosmos Embedding Admin Panel - Startup Script
# This script helps you start the FastAPI application with proper configuration

echo "🚀 Starting Cosmos Embedding Admin Panel..."
echo "=============================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "📝 Please copy .env.example to .env and configure your credentials:"
    echo "   cp .env.example .env"
    echo "   # Then edit .env with your actual values"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create uploads directory
echo "📁 Creating uploads directory..."
mkdir -p static/uploads

# Start the application
echo "🌟 Starting FastAPI application..."
echo "📖 API Documentation will be available at: http://localhost:8000/docs"
echo "🔍 Health check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="

uvicorn main:app --reload --host 0.0.0.0 --port 8000
