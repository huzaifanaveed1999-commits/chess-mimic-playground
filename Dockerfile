# Use lightweight official Python image
FROM python:3.10-slim

# Install system dependencies (including Stockfish!)
RUN apt-get update && apt-get install -y \
    stockfish \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy and install standard requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY . .

# Expose port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Set environment variable to bind to Hugging Face's port
ENV PORT=7860

# Start server
CMD ["python", "app.py"]
