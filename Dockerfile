FROM python:3.11-slim

# Install Chrome and its dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    libx11-6 \
    libxcb1 \
    libxss1 \
    fonts-liberation \
    xdg-utils \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set Chromium path for Kaleido
ENV CHROMIUM_PATH=/usr/bin/chromium

# Run the application
CMD ["python", "standalone_engine.py"]
