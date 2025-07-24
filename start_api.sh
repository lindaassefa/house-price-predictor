#!/bin/bash
echo "Starting House Price Predictor API..."
echo ""
echo "1. Training model..."
python train_model.py
echo ""
echo "2. Starting API server..."
python api.py
