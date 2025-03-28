# Stage 1: Build dependencies
FROM python:3.13.2-alpine AS builder

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAKEFLAGS="-j$(nproc)" \
    PATH="/root/.local/bin:$PATH"

# Install required build dependencies including file and binutils (for strip)
RUN apk add --no-cache gcc musl-dev python3-dev libffi-dev openssl-dev file binutils

# Set working directory
WORKDIR /app

# Copy only requirements file for better Docker caching
COPY requirements.txt .

# Install Python dependencies with simplified cleanup
RUN pip install --user --no-cache-dir -r requirements.txt && \
    find /root/.local -type d -name "__pycache__" -exec rm -rf {} \; 2>/dev/null || true && \
    find /root/.local -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /root/.local -type f -name "*.pyo" -delete 2>/dev/null || true && \
    find /root/.local -type f -name "*.so*" -exec strip -s {} \; 2>/dev/null || true && \
    mkdir -p /root/.local/bin

# Stage 2: Runtime environment
FROM python:3.13.2-alpine AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Create non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Set working directory
WORKDIR /home/appuser/app

# Create target directories first
RUN mkdir -p /home/appuser/.local/lib /home/appuser/.local/bin

# Copy Python packages from builder stage (only necessary folders)
COPY --from=builder --chown=appuser:appgroup /root/.local/lib /home/appuser/.local/lib/

# Fix for bin directory copy - use a safer approach
RUN mkdir -p /home/appuser/.local/bin
# Workaround to safely copy bin directory contents if they exist
RUN if [ -d "/root/.local/bin" ] && [ -n "$(ls -A /root/.local/bin 2>/dev/null)" ]; then \
    cp -r /root/.local/bin/* /home/appuser/.local/bin/ 2>/dev/null || true; \
    fi

# Copy application source code
COPY --chown=appuser:appgroup bot.py .
COPY --chown=appuser:appgroup src/ ./src/

# Use non-root user
USER appuser

# Run application
CMD ["python3", "bot.py"]