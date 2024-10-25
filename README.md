# Discord Bot with OpenAI Integration

![Build and Push](https://github.com/coder-vippro/llama405bdiscord/actions/workflows/main.yml/badge.svg)

## Overview

Welcome to the **Discord Bot with OpenAI Integration** project! This bot leverages the power of OpenAI's models to interact with users on Discord, providing intelligent responses and generating images based on text prompts. The bot is containerized using Docker for seamless deployment and includes automated CI/CD pipelines.

## Features

- **Discord Integration**: Engage with users on Discord servers.
- **OpenAI Integration**: Generate responses using OpenAI's powerful models.
- **Image Generation**: Create images from text prompts using RunwayML.
- **Web Scraping**: Scrape web content and provide summarized information.
- **Dockerized**: Simplified deployment using Docker.
- **Automated CI/CD**: Continuous integration and deployment with GitHub Actions.

## Prerequisites

- Docker
- Python 3.12.7
- Discord Bot Token
- OpenAI API Key
- RunwayML API Key
- Google API Key and CX

## Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/coder-vippro/llama405bdiscord.git
    cd llama405bdiscord
    ```

2. **Create a `.env` file** with your credentials:
    ```properties
    DISCORD_TOKEN=your_discord_token
    OPENAI_API_KEY=your_openai_api_key
    RUNWARE_API_KEY=your_runware_api_key
    GOOGLE_API_KEY=your_google_api_key
    GOOGLE_CX=your_google_cx
    OPENAI_BASE_URL=https://models.inference.ai.azure.com
    ```

3. **Build the Docker image**:
    ```sh
    docker build -t llama405bdiscord .
    ```

4. **Run the Docker container**:
    ```sh
    docker run -d --name discordllm405b llama405bdiscord
    ```

## Usage

Once the bot is running, it will connect to Discord using the provided token. You can interact with it on your Discord server using various commands.

### Example Commands

- **Generate Image**: `/generate prompt: "A futuristic cityscape"`
- **Scrape Web Content**: `/web url: "https://example.com"`

## Development

### Running Locally

1. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

2. **Run the bot**:
    ```sh
    python bot.py
    ```

### Running Tests

1. **Install test dependencies**:
    ```sh
    pip install pytest
    ```

2. **Run tests**:
    ```sh
    python -m pytest tests/
    ```

## CI/CD

This project uses GitHub Actions for continuous integration and deployment. The workflows are defined in the `.github/workflows` directory.

- **Build and Push**: [`.github/workflows/githubpackage.yml1`](.github/workflows/githubpackage.yml1)
- **Main Workflow**: [`.github/workflows/main.yml`](.github/workflows/main.yml)

## Security

For information on supported versions and how to report vulnerabilities, see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [coder-vippro](https://github.com/coder-vippro)
