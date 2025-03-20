# Builder stage optimized for tiktoken and other dependencies
FROM python:3.13.2-alpine AS builder

# Set environment variables for more efficient builds
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAKEFLAGS="-j$(nproc)" \
    PATH="/root/.local/bin:$PATH" \
    RUSTFLAGS="-C target-feature=-crt-static" \
    CARGO_NET_GIT_FETCH_WITH_CLI=true

# Create non-root user for better security
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Install build dependencies with cleanup in the same layer
RUN apk add --no-cache \
    gcc \
    musl-dev \
    python3-dev \
    cargo \
    rust \
    libffi-dev \
    g++ \
    openssl-dev \
    git \
    && rustup update \
    && rustup default stable

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Split requirements install for better caching
# Install tiktoken separately first (the slow one)
RUN pip install --user --no-cache-dir tiktoken

# Install other requirements
RUN pip install --user --no-cache-dir -r requirements.txt \
    && find /root/.local -name "__pycache__" -type d -exec rm -rf {} +

# Runtime stage with absolute minimal dependencies
FROM python:3.13.2-alpine AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Create same non-root user as in builder
RUN addgroup -S appgroup && adduser -S appuser -G appgroup \
    && mkdir -p /home/appuser/app/logs /home/appuser/app/temp_charts \
    && chown -R appuser:appgroup /home/appuser

# Add only the necessary runtime dependencies
RUN apk add --no-cache \
    libstdc++ \
    g++

# Set the working directory
WORKDIR /home/appuser/app

# Copy Python packages from builder stage (only what's needed)
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Copy only the needed application files
COPY --chown=appuser:appgroup bot.py .
COPY --chown=appuser:appgroup src/ ./src/

# Use non-root user
USER appuser

# Command to run the application
CMD ["python3", "bot.py"]
