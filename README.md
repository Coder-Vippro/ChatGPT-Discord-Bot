# ChatGPT Discord Bot

![Build and Push](https://github.com/coder-vippro/ChatGPT-Discord-Bot/actions/workflows/main.yml/badge.svg)

## Overview

Welcome to **ChatGPT Discord Bot**! This bot is designed to interact with users on Discord, providing responses powered by OpenAI’s models. It can generate images based on prompts, fetch information from the web, and is containerized with Docker for easy deployment. CI/CD is managed via GitHub Actions.

## Features

- **Discord Integration**: Communicate with users directly on Discord.
- **OpenAI Responses**: Utilizes OpenAI models for intelligent responses.
- **Image Generation**: Creates images from text prompts via Runware.
- **Web Scraping**: Summarizes web content for quick info retrieval.
- **Dockerized**: Supports deployment with Docker and GHCR container images.
- **Automated CI/CD**: Integrated CI/CD through GitHub Actions.

## Prerequisites

To get started, ensure you have:

- Docker
- Python 3.12.7
- Discord Bot Token
- OpenAI API Key
- Runware API Key ([Get yours at Runware](https://runware.ai/))
- Google API Key and Custom Search Engine ID (CX)

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Coder-Vippro/ChatGPT-Discord-Bot.git
    cd ChatGPT-Discord-Bot
    ```

2. **Configure environment variables**:
    Create a `.env` file in the root directory as follows:

    ```properties
    DISCORD_TOKEN=your_discord_token
    OPENAI_API_KEY=your_openai_api_key
    RUNWARE_API_KEY=your_runware_api_key
    GOOGLE_API_KEY=your_google_api_key
    GOOGLE_CX=your_google_cx
    OPENAI_BASE_URL=https://models.inference.ai.azure.com or https://
    ```

    - **Discord Token**: Create a bot at the [Discord Developer Portal](https://discord.com/developers/applications).
    - **OpenAI API Key**: Obtain an API key at [OpenAI's API](https://platform.openai.com/signup).
    - **Runware API Key**: Register at [Runware](https://runware.ai/) for your API key.
    - **Google API Key**: Create a project on [Google Cloud Console](https://console.cloud.google.com/), enable the Custom Search API, and generate an API key.
    - **Google CX**: Set up a Custom Search Engine (CSE) at [Google Custom Search](https://cse.google.com/cse/) and retrieve the search engine ID.

3. **Deploy with Docker using GHCR Image**:
   Use the following `docker-compose.yml` to deploy the bot with GitHub Container Registry:

    ```yaml
    version: '3.8'

    services:
      bot:
        image: ghcr.io/coder-vippro/chatgpt-discord-bot:latest
        env_file:
          - .env
        restart: always
    ```

   Start the container with:
   ```bash
   docker-compose up -d
   ```

## Usage

When the bot is running, it connects to Discord using your `.env` credentials. You can then interact with it on your server.

### Example Commands

- **Generate Image**: `/generate prompt: "A futuristic cityscape"`
- **Scrape Web Content**: `/web url: "https://example.com"`
- **Search Google For Content**: `/search prompt: "latest news in vietnam"`

## Development

### Running Locally

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the bot**:
    ```bash
    python bot.py
    ```

### Running Tests

1. **Install test dependencies**:
    ```bash
    pip install pytest
    ```

2. **Run tests**:
    ```bash
    pytest tests/
    ```

## CI/CD

This project utilizes GitHub Actions for continuous integration and deployment. Workflows are defined in `.github/workflows`.

- **Main Workflow**: See [`.github/workflows/main.yml`](.github/workflows/main.yml)

## Security

For supported versions and vulnerability reporting, see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [coder-vippro](https://github.com/coder-vippro)
