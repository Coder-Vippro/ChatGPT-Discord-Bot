# Use an official Python runtime as a parent image
FROM python:3.11.10-slim

# Install curl and other dependencies
RUN apt-get update && apt-get install -y curl && apt-get clean

# Set the working directory in the container
WORKDIR /usr/src/discordbot

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for health check
EXPOSE 9123

# Health check command
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://localhost:9123/health || exit 1

# Copy the rest of the application source code
COPY . .

# Command to run the application
CMD ["python3", "bot.py"]