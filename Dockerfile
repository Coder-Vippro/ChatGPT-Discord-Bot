# Builder stage optimized for dependencies
FROM python:3.13.2-alpine AS builder

# Set environment variables for more efficient builds
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAKEFLAGS="-j$(nproc)" \
    PATH="/root/.local/bin:$PATH" \
    RUSTFLAGS="-C target-feature=-crt-static" \
    CARGO_NET_GIT_FETCH_WITH_CLI=true

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
    && rm -rf /var/cache/apk/*

# Set the working directory
WORKDIR /app

# Copy only requirements file first to leverage Docker cache
COPY requirements.txt .

# Install all requirements with aggressive cleanup
RUN pip install --user --no-cache-dir -r requirements.txt && \
    find /root/.local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -type f -name "*.pyc" -delete && \
    find /root/.local -type f -name "*.pyo" -delete && \
    find /root/.local -type f -name "*.so" -exec strip -s {} \; 2>/dev/null || true && \
    find /root/.local -type f -name "*.so.*" -exec strip -s {} \; 2>/dev/null || true && \
    rm -rf /root/.cargo /root/.cache /tmp/* /var/tmp/*

# Runtime stage with minimal dependencies
FROM python:3.13.2-alpine AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Create non-root user with specific directories and install minimal runtime dependencies
RUN addgroup -S appgroup \
    && adduser -S appuser -G appgroup \
    && mkdir -p /home/appuser/app/logs /home/appuser/app/temp_charts \
    && chown -R appuser:appgroup /home/appuser \
    && chmod -R 755 /home/appuser/app \
    && rm -rf /var/cache/apk/* /tmp/*

# Set the working directory
WORKDIR /home/appuser/app

# Copy Python packages from builder stage
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Copy only the needed application files
COPY --chown=appuser:appgroup bot.py .
COPY --chown=appuser:appgroup src/ ./src/

# Define volume for persistent data
VOLUME ["/home/appuser/app/logs", "/home/appuser/app/temp_charts"]

# Use non-root user
USER appuser

# Command to run the application
CMD ["python3", "bot.py"]