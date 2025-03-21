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
    git 

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install all requirements in a single layer with aggressive cleanup
RUN pip install --user --no-cache-dir -r requirements.txt && \
    find /root/.local -name "__pycache__" -type d -exec rm -rf {} + && \
    find /root/.local -name "*.pyc" -delete && \
    find /root/.local -name "*.pyo" -delete && \
    find /root/.local -name "*.so" -exec strip -s {} \; || true && \
    find /root/.local -name "*.so.*" -exec strip -s {} \; || true && \
    rm -rf /root/.cargo /root/.cache

# Runtime stage with minimal dependencies
FROM python:3.13.2-alpine AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Create non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup \
    && mkdir -p /home/appuser/app/logs /home/appuser/app/temp_charts \
    && chown -R appuser:appgroup /home/appuser

# Add only necessary runtime dependencies
RUN apk add --no-cache g++ && \
    rm -rf /var/cache/apk/* /tmp/*

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