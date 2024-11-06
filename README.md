
# ChatGPT Discord Bot

![Build and Push](https://github.com/coder-vippro/ChatGPT-Discord-Bot/actions/workflows/main.yml/badge.svg)

## Overview

Welcome to **ChatGPT Discord Bot**! This bot is designed to interact with users on Discord, powered by OpenAI’s models. It generates responses, creates images from prompts, fetches web content, and is containerized with Docker for smooth deployment. Continuous integration and deployment (CI/CD) are managed with GitHub Actions.

## Features

- **Discord Integration**: Communicate directly with users on Discord.
- **OpenAI Responses**: Provides intelligent responses using OpenAI models.
- **Image Generation**: Generates images from prompts via Runware.
- **Web Scraping**: Fetches and summarizes content from the web.
- **Dockerized Deployment**: Ready for deployment with Docker and GHCR images.
- **Automated CI/CD**: Integrated with GitHub Actions for CI/CD.

## Prerequisites

To get started, ensure you have:

- Docker
- Python 3.12.7
- Discord Bot Token
- OpenAI API Key
- Runware API Key ([Get yours at Runware](https://runware.ai/))
- Google API Key and Custom Search Engine ID (CX)
- Mongodb url: Get from https://cloud.mongodb.com/

## Setup

### For Normal Use

1. **Option A: Deploy with Docker**  
   - Create a `.env` file in the root directory with your configuration:
     ```properties
     DISCORD_TOKEN=your_discord_token
     OPENAI_API_KEY=your_openai_api_key
     RUNWARE_API_KEY=your_runware_api_key
     GOOGLE_API_KEY=your_google_api_key
     GOOGLE_CX=your_google_cx
     OPENAI_BASE_URL=https://models.inference.ai.azure.com or https://api.openai.com/v1/models or any api else you want
     MONGODB_URI=mongodb://localhost:27017/
     ```
   - Use the following `docker-compose.yml`:
     ```yaml
     version: '3.8'

     services:
       bot:
         image: ghcr.io/coder-vippro/chatgpt-discord-bot:latest
         env_file:
           - .env
         restart: always
     ```
   - Start the bot with:
     ```bash
     docker-compose up -d
     ```

2. **Option B: Deploy Without Docker**
   - Clone the repository:
     ```bash
     git clone https://github.com/Coder-Vippro/ChatGPT-Discord-Bot.git
     cd ChatGPT-Discord-Bot
     ```
   - Create a `.env` file in the root directory with your configuration:
     ```properties
     DISCORD_TOKEN=your_discord_token
     OPENAI_API_KEY=your_openai_api_key
     RUNWARE_API_KEY=your_runware_api_key
     GOOGLE_API_KEY=your_google_api_key
     GOOGLE_CX=your_google_cx
     OPENAI_BASE_URL=https://models.inference.ai.azure.com or https://api.openai.com/v1/models or any api else you want
     MONGODB_URI=mongodb://localhost:27017/
     ```
   - Install the dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Run the bot:
     ```bash
     python3 bot.py
     ```

### For Development

1. Clone the repository:
   ```bash
   git clone https://github.com/Coder-Vippro/ChatGPT-Discord-Bot.git
   cd ChatGPT-Discord-Bot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the bot:
   ```bash
   python3 bot.py
   ```
### Running Tests

1. Install test dependencies:
   ```bash
   pip install pytest
   ```
2. Run tests:
   ```bash
   pytest tests/
   ```

   
## Usage

Once the bot is running, it connects to Discord using credentials from `.env`. Commands available include:

- **Generate Image**: `/generate prompt: "A futuristic cityscape"`
- **Scrape Web Content**: `/web url: "https://example.com"`
- **Search Google**: `/search prompt: "latest news in Vietnam"`
- **Normal chat**: `Ping the bot with a question or send a dms to the bot to start`

## CI/CD

This project uses GitHub Actions for CI/CD, with workflows in `.github/workflows`.

## Security

For supported versions and vulnerability reporting, see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [coder-vippro](https://github.com/coder-vippro)
