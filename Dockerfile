# Stage 1: Build dependencies
FROM python:3.13.2-alpine AS builder

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAKEFLAGS="-j$(nproc)" \
    PATH="/root/.local/bin:$PATH"

# Install only required build dependencies
RUN apk add --no-cache gcc musl-dev python3-dev libffi-dev openssl-dev

# Set working directory
WORKDIR /app

# Copy only requirements file for better Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt && \
    find /root/.local -type d -name "__pycache__" -exec rm -rf {} + && \
    find /root/.local -type f -name "*.pyc" -delete && \
    find /root/.local -type f -name "*.pyo" -delete && \
    find /root/.local -type f -name "*.so*" -exec sh -c 'file "{}" | grep -q ELF && strip -s "{}"' \;

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

# Copy Python packages from builder stage (only necessary folders)
COPY --from=builder --chown=appuser:appgroup /root/.local/lib /home/appuser/.local/lib
COPY --from=builder --chown=appuser:appgroup /root/.local/bin /home/appuser/.local/bin

# Copy application source code
COPY --chown=appuser:appgroup bot.py .
COPY --chown=appuser:appgroup src/ ./src/

# Use non-root user
USER appuser

# Run application
CMD ["python3", "bot.py"]
