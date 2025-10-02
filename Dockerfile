# Stage 1: Build dependencies
FROM python:3.13.3-alpine AS builder

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MAKEFLAGS="-j$(nproc)"

# Install build dependencies
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    musl-dev \
    python3-dev \
    libffi-dev \
    openssl-dev \
    g++ \
    rust \
    cargo \
    hdf5-dev \
    openblas-dev \
    lapack-dev \
    gfortran \
    freetype-dev \
    libpng-dev \
    jpeg-dev

WORKDIR /app

# Copy only requirements file for better caching
COPY requirements.txt .

# Install Python dependencies with aggressive cleanup
RUN pip install --no-cache-dir -r requirements.txt && \
    # Remove build dependencies
    apk del .build-deps && \
    # Clean Python cache
    find /usr/local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local -type f -name "*.py[co]" -delete && \
    # Strip debug symbols from shared libraries
    find /usr/local -type f -name "*.so*" -exec strip -s {} \; 2>/dev/null || true && \
    # Remove pip cache
    rm -rf /root/.cache/pip && \
    # Remove unnecessary test files
    find /usr/local -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# Stage 2: Runtime environment
FROM python:3.13.3-alpine AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FILE_EXPIRATION_HOURS=48 \
    MAX_FILES_PER_USER=20 \
    CODE_EXECUTION_TIMEOUT=300

# Install minimal runtime dependencies and create directories in one layer
RUN apk add --no-cache \
    libstdc++ \
    libgfortran \
    openblas \
    lapack \
    hdf5 \
    freetype \
    libpng \
    libjpeg \
    tzdata \
    && mkdir -p /tmp/bot_code_interpreter/{user_files,outputs,venv} \
    && chmod -R 777 /tmp/bot_code_interpreter \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy only necessary Python packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY bot.py .
COPY src/ ./src/

# Remove unnecessary files from application
RUN find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find . -type f -name "*.py[co]" -delete

# Lightweight healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python3 -c "import sys; sys.exit(0)" || exit 1

CMD ["python3", "-u", "bot.py"]
