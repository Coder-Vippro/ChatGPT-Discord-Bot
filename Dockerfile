# Stage 1: Build dependencies
FROM python:3.13.3-alpine AS builder

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAKEFLAGS="-j$(nproc)"

# Install required build dependencies
RUN apk add --no-cache gcc musl-dev python3-dev libffi-dev openssl-dev file binutils g++ rust cargo

WORKDIR /app

# Copy only requirements file for better caching
COPY requirements.txt .

# Install Python dependencies and clean up in a single layer
RUN pip install --user --no-cache-dir -r requirements.txt && \
    find /root/.local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -type f -name "*.py[co]" -delete && \
    find /root/.local -type f -name "*.so*" -exec strip -s {} \; 2>/dev/null || true

# Stage 2: Runtime environment
FROM python:3.13.3-alpine AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Create non-root user and necessary directories
RUN addgroup -S appgroup && \
    adduser -S appuser -G appgroup && \
    mkdir -p /home/appuser/.local/lib /home/appuser/.local/bin && \
    chown -R appuser:appgroup /home/appuser

WORKDIR /home/appuser/app

# Copy Python packages from builder stage
COPY --from=builder --chown=appuser:appgroup /root/.local/ /home/appuser/.local/

# Copy application source code
COPY --chown=appuser:appgroup bot.py .
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup logs/ ./logs/

# Use non-root user
USER appuser

# Run application
CMD ["python3", "bot.py"]
