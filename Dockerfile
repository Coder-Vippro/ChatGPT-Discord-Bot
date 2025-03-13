# Stage 1: Build environment (có compiler và dependencies)
FROM python:3.12.3-alpine as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install necessary build dependencies
RUN apk add --no-cache \
    g++ \
    build-base \
    make

# Set working directory
WORKDIR /usr/src/discordbot

# Copy requirements and install dependencies in a virtual environment
COPY requirements.txt .
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Final lightweight image
FROM python:3.11.10-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy only the installed dependencies from builder stage
COPY --from=builder /venv /venv

# Set working directory
WORKDIR /usr/src/discordbot

# Copy the source code
COPY . .

# Use virtual environment for running the bot
CMD ["/venv/bin/python", "bot.py"]
