#!/usr/bin/env python3
"""
Deployment script for House Price Predictor
Automates the setup and deployment process
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_dependencies():
    """Check if required files exist"""
    print("📁 Checking project files...")
    required_files = [
        "cleaned_df.csv",
        "requirements.txt",
        "train_model.py",
        "app.py",
        "api.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def train_model():
    """Train the machine learning model"""
    print("🤖 Training machine learning model...")
    
    if os.path.exists("house_price_model.pkl"):
        print("⚠️  Model already exists. Skipping training.")
        return True
    
    if not run_command("python train_model.py", "Training model"):
        return False
    
    if not os.path.exists("house_price_model.pkl"):
        print("❌ Model training failed - model file not created")
        return False
    
    print("✅ Model training completed")
    return True

def create_directories():
    """Create necessary directories"""
    print("📂 Creating directories...")
    
    directories = ["logs", "models", "data", "static"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def setup_logging():
    """Setup logging configuration"""
    print("📝 Setting up logging...")
    
    log_config = """
import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
"""
    
    try:
        with open("logging_config.py", "w") as f:
            f.write(log_config)
        print("✅ Logging configuration created")
        return True
    except Exception as e:
        print(f"❌ Failed to create logging config: {e}")
        return False

def create_startup_scripts():
    """Create startup scripts for different platforms"""
    print("🚀 Creating startup scripts...")
    
    # Windows batch script
    windows_script = """@echo off
echo Starting House Price Predictor...
echo.
echo 1. Training model...
python train_model.py
echo.
echo 2. Starting web application...
streamlit run app.py
pause
"""
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Starting House Price Predictor..."
echo ""
echo "1. Training model..."
python train_model.py
echo ""
echo "2. Starting web application..."
streamlit run app.py
"""
    
    # API startup script
    api_script = """#!/bin/bash
echo "Starting House Price Predictor API..."
echo ""
echo "1. Training model..."
python train_model.py
echo ""
echo "2. Starting API server..."
python api.py
"""
    
    try:
        # Windows
        with open("start.bat", "w") as f:
            f.write(windows_script)
        
        # Unix
        with open("start.sh", "w") as f:
            f.write(unix_script)
        os.chmod("start.sh", 0o755)
        
        # API
        with open("start_api.sh", "w") as f:
            f.write(api_script)
        os.chmod("start_api.sh", 0o755)
        
        print("✅ Startup scripts created")
        return True
    except Exception as e:
        print(f"❌ Failed to create startup scripts: {e}")
        return False

def create_dockerfile():
    """Create Dockerfile for containerization"""
    print("🐳 Creating Dockerfile...")
    
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose ports
EXPOSE 8501 5000

# Create startup script
RUN echo '#!/bin/bash\\nstreamlit run app.py --server.port 8501 --server.address 0.0.0.0' > start.sh && \\
    chmod +x start.sh

# Default command
CMD ["./start.sh"]
"""
    
    try:
        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)
        print("✅ Dockerfile created")
        return True
    except Exception as e:
        print(f"❌ Failed to create Dockerfile: {e}")
        return False

def create_docker_compose():
    """Create docker-compose.yml for easy deployment"""
    print("🐳 Creating docker-compose.yml...")
    
    compose_content = """version: '3.8'

services:
  house-price-predictor:
    build: .
    ports:
      - "8501:8501"  # Streamlit
      - "5000:5000"  # Flask API
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped

  # Optional: Add a database service
  # postgres:
  #   image: postgres:13
  #   environment:
  #     POSTGRES_DB: house_prices
  #     POSTGRES_USER: user
  #     POSTGRES_PASSWORD: password
  #   volumes:
  #     - postgres_data:/var/lib/postgresql/data
  #   ports:
  #     - "5432:5432"

# volumes:
#   postgres_data:
"""
    
    try:
        with open("docker-compose.yml", "w") as f:
            f.write(compose_content)
        print("✅ docker-compose.yml created")
        return True
    except Exception as e:
        print(f"❌ Failed to create docker-compose.yml: {e}")
        return False

def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running tests...")
    
    tests = [
        ("python -c 'import pandas; print(\"Pandas OK\")'", "Testing Pandas"),
        ("python -c 'import streamlit; print(\"Streamlit OK\")'", "Testing Streamlit"),
        ("python -c 'import flask; print(\"Flask OK\")'", "Testing Flask"),
        ("python -c 'import sklearn; print(\"Scikit-learn OK\")'", "Testing Scikit-learn"),
        ("python -c 'import plotly; print(\"Plotly OK\")'", "Testing Plotly")
    ]
    
    for command, description in tests:
        if not run_command(command, description):
            return False
    
    return True

def main():
    """Main deployment function"""
    print("🚀 House Price Predictor - Deployment Script")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Installation steps
    steps = [
        (install_dependencies, "Installing dependencies"),
        (create_directories, "Creating directories"),
        (setup_logging, "Setting up logging"),
        (create_startup_scripts, "Creating startup scripts"),
        (create_dockerfile, "Creating Dockerfile"),
        (create_docker_compose, "Creating docker-compose.yml"),
        (train_model, "Training model"),
        (run_tests, "Running tests")
    ]
    
    for step_func, description in steps:
        if not step_func():
            print(f"❌ Deployment failed at: {description}")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Deployment completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the web application: streamlit run app.py")
    print("2. Start the API server: python api.py")
    print("3. Or use Docker: docker-compose up")
    print("\n📚 For more information, see README.md")
    print("=" * 50)

if __name__ == "__main__":
    main() 