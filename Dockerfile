# Build stage with all build dependencies
FROM python:3.12.3-alpine AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies
RUN apk add --no-cache \
    curl \
    g++ \
    gcc \
    musl-dev \
    make \
    rust \
    cargo \
    build-base

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python packages with BuildKit cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Runtime stage with minimal dependencies
FROM python:3.12.3-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apk add --no-cache libstdc++

# Set the working directory
WORKDIR /usr/src/discordbot

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12.3 /usr/local/lib/python3.12.3
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application source code
COPY . .

# Command to run the application
CMD ["python3", "bot.py"]
