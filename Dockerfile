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
RUN pip install --no-cache-dir -r requirements.txt && \
    find /usr/local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local -type f -name "*.py[co]" -delete && \
    find /usr/local -type f -name "*.so*" -exec strip -s {} \; 2>/dev/null || true

# Stage 2: Runtime environment
FROM python:3.13.3-alpine AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application source code
COPY bot.py .
COPY src/ ./src/

# Run application
CMD ["python3", "bot.py"]
