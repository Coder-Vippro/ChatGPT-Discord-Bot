# ChatGPT Discord Bot
![Build and Push](https://github.com/coder-vippro/ChatGPT-Discord-Bot/actions/workflows/main.yml/badge.svg)

## Overview
Welcome to **ChatGPT Discord Bot**! This bot provides a powerful AI assistant for Discord users, powered by OpenAI's latest models. It not only generates text responses but also offers a range of advanced features including image generation, data analysis, web searching, and reminders. The bot is designed for easy deployment with Docker and includes CI/CD integration via GitHub Actions.

## Features
- **Advanced AI Conversations**: Uses OpenAI's latest models (including openai/gpt-4o) for natural language interactions
- **Image Generation**: Creates custom images from text prompts using Runware's API
- **Data Analysis**: Analyzes CSV and Excel files with visualizations (distributions, correlations, box plots, etc.)
- **Code Interpretation**: Executes Python code for calculations and data processing
  - **Package Installation**: Automatically installs required Python packages for code execution
  - **Code Display**: Shows executed code, input, and output in Discord chat for transparency
  - **Secure Sandbox**: Runs code in a controlled environment with safety restrictions
- **Reminder System**: Sets timed reminders with custom timezone support
- **Web Tools**:
  - **Google Search**: Searches the web and provides relevant information
  - **Web Scraping**: Extracts and summarizes content from websites
- **PDF Analysis**: Processes and analyzes PDF documents
- **User Statistics**: Tracks token usage and model selection per user
- **Dockerized Deployment**: Ready for easy deployment with Docker
- **Automated CI/CD**: Integrated with GitHub Actions

## Prerequisites
To get started, ensure you have:
- Docker (for containerized deployment)
- Python 3.12.7
- Discord Bot Token
- OpenAI API Key
- Runware API Key ([Get yours at Runware](https://runware.ai/))
- Google API Key and Custom Search Engine ID (CX)
- MongoDB URL (Get from https://cloud.mongodb.com/)

## Setup
### For Normal Use
#### Option A: Deploy with Docker
1. Create a `.env` file in the root directory with your configuration:
   ```properties
   DISCORD_TOKEN=your_discord_token
   OPENAI_API_KEY=your_openai_api_key
   RUNWARE_API_KEY=your_runware_api_key
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CX=your_google_cx
   OPENAI_BASE_URL=https://api.openai.com/v1/models
   MONGODB_URI=mongodb://localhost:27017/
   ADMIN_ID=your_discord_user_id
   TIMEZONE=Asia/Ho_Chi_Minh
   ```

2. Use the following `docker-compose.yml`:
   ```yaml
   version: '3.8'
   services:
     bot:
       image: ghcr.io/coder-vippro/chatgpt-discord-bot:latest
       env_file:
         - .env
       restart: always
   ```

3. Start the bot with:
   ```bash
   docker-compose up -d
   ```

#### Option B: Deploy Without Docker
1. Clone the repository:
   ```bash
   git clone https://github.com/Coder-Vippro/ChatGPT-Discord-Bot.git
   cd ChatGPT-Discord-Bot
   ```

2. Create a `.env` file in the root directory with your configuration:
   ```properties
   DISCORD_TOKEN=your_discord_token
   OPENAI_API_KEY=your_openai_api_key
   RUNWARE_API_KEY=your_runware_api_key
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CX=your_google_cx
   OPENAI_BASE_URL=https://api.openai.com/v1/models
   MONGODB_URI=mongodb://localhost:27017/
   ADMIN_ID=your_discord_user_id
   TIMEZONE=Asia/Ho_Chi_Minh
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the bot:
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
Once the bot is running, it connects to Discord using credentials from `.env`. Available features include:

### Text Commands
- **Normal chat**: Ping the bot with a question or send a DM to start a conversation
- **Image Generation**: `/generate prompt: "A futuristic cityscape"`
- **Web Content**: `/web url: "https://example.com"`
- **Google Search**: `/search prompt: "latest news in Vietnam"`
- **User Statistics**: `/user_stat` - Get your token usage and model information

### Advanced Features
- **Data Analysis**: Upload CSV or Excel files for automatic analysis and visualization
- **Code Execution**: The bot can execute Python code to solve problems or create visualizations
- **Reminders**: Ask the bot to set reminders like "Remind me to check email in 30 minutes"
- **PDF Analysis**: Upload PDF documents for the bot to analyze and summarize

### Available Models
The bot supports the following models:
- openai/gpt-4o
- openai/gpt-4o-mini
- openai/gpt-5
- openai/gpt-5-nano
- openai/gpt-5-mini
- openai/gpt-5-chat
- openai/o1-preview
- openai/o1-mini
- openai/o1
- openai/o3-mini

## Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| DISCORD_TOKEN | Your Discord bot token | Required |
| OPENAI_API_KEY | Your OpenAI API key | Required |
| RUNWARE_API_KEY | Runware API key for image generation | Required |
| GOOGLE_API_KEY | Google API key for search | Required | 
| GOOGLE_CX | Google Custom Search Engine ID | Required |
| MONGODB_URI | MongoDB connection string | Required |
| ADMIN_ID | Discord user ID of the admin | Optional |
| TIMEZONE | Timezone for reminder feature | UTC |
| ENABLE_WEBHOOK_LOGGING | Enable webhook logging | False |
| LOGGING_WEBHOOK_URL | URL for webhook logging | Optional |

## CI/CD
This project uses GitHub Actions for CI/CD, with workflows in `.github/workflows`.

## Security
For supported versions and vulnerability reporting, see [SECURITY.md](SECURITY.md).

## Contributing
Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing to this project.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [coder-vippro](https://github.com/coder-vippro)
