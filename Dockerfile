# Build stage with all build dependencies
FROM python:3.12.3-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install build dependencies (only what's absolutely needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage with minimal dependencies
FROM python:3.12.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /usr/src/discordbot

# Copy only the installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install runtime dependencies (only what's absolutely needed)
RUN apk add --no-cache g++

# Copy the application source code (only what's needed)
COPY bot.py .
COPY src/ ./src/

# Create directories for logs and temp files
RUN mkdir -p logs temp_charts

# Command to run the application
CMD ["python3", "bot.py"]
