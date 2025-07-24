#!/bin/bash

echo "🚀 Deploying House Price Predictor to Render..."

# Check if render CLI is installed
if ! command -v render &> /dev/null; then
    echo "📦 Installing Render CLI..."
    curl -sL https://render.com/download-cli/install.sh | bash
fi

# Login to Render (if not already logged in)
echo "🔐 Logging into Render..."
render login

# Deploy the service
echo "📤 Deploying service..."
render deploy

echo "✅ Deployment initiated! Check your Render dashboard for status."
echo "🌐 Your API will be available at: https://your-service-name.onrender.com" 