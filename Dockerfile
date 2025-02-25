# Use an official Python runtime as a parent image
FROM python:3.11.10-slim

# Set environment variables to reduce Python buffer and logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install curl, g++ compiler, and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    g++ \
    build-essential \
    make \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/discordbot

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Expose port (optional, only if needed for a web server)
EXPOSE 5000

# Add health check (update endpoint if needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl --fail http://localhost:5000/health || exit 1

# Copy the rest of the application source code
COPY . .

# Command to run the application
CMD ["python3", "bot.py"]
