# Build stage with all build dependencies
FROM python:3.12.3-alpine AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install build dependencies (only what's absolutely needed)
RUN apk add --no-cache \
    gcc \
    musl-dev \
    python3-dev \
    cargo \
    libffi-dev \
    g++

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python packages to a local directory
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage with minimal dependencies
FROM python:3.12.3-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Add needed runtime dependencies (if any)
RUN apk add --no-cache \ 
    libstdc++ \
    g++

# Set the working directory
WORKDIR /usr/src/discordbot

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy only the application source code needed to run
COPY bot.py .
COPY src/ ./src/

# Create directories for logs and temp files
RUN mkdir -p logs temp_charts

# Command to run the application
CMD ["python3", "bot.py"]
