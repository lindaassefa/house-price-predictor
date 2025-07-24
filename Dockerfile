FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose ports
EXPOSE 8501 5000

# Create startup script
RUN echo '#!/bin/bash\nstreamlit run app.py --server.port 8501 --server.address 0.0.0.0' > start.sh && \
    chmod +x start.sh

# Default command
CMD ["./start.sh"]
