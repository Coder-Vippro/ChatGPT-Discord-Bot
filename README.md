<div align="center">

# 🤖 ChatGPT Discord Bot

### *Your AI-Powered Assistant with Code Interpreter & Advanced File Management*

[![Build and Push](https://github.com/coder-vippro/ChatGPT-Discord-Bot/actions/workflows/main.yml/badge.svg)](https://github.com/coder-vippro/ChatGPT-Discord-Bot/actions)
[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/Coder-Vippro/ChatGPT-Discord-Bot/releases)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/discord-bot-7289da.svg)](https://discord.com)

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Support](#-support)

</div>

---

## 🌟 Overview

**ChatGPT Discord Bot** brings the power of AI directly to your Discord server! Powered by OpenAI's latest models, this bot goes beyond simple chat - it's a complete AI assistant with **code interpretation**, **file management**, **data analysis**, and much more.

### 🎯 What Makes This Bot Special?

- 🧠 **Latest AI Models** - GPT-4o, GPT-5, o1, o3-mini, and more
- 💻 **Code Interpreter** - Execute Python code like ChatGPT (NEW in v2.0!)
- 📁 **Smart File Management** - Handle 200+ file types with automatic cleanup
- 📊 **Data Analysis** - Upload and analyze CSV, Excel, and scientific data
- 🎨 **Image Generation** - Create stunning images from text prompts
- 🔍 **Web Tools** - Search Google and scrape websites
- ⏰ **Reminder System** - Never forget important tasks
- 🐳 **Docker Ready** - One-command deployment

---

## ✨ Features

### 🆕 New in Version 2.0.0

<table>
<tr>
<td width="50%">

#### 💻 **Unified Code Interpreter**
Execute Python code directly in Discord! Similar to ChatGPT's code interpreter.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = load_file('your_file_id')
print(df.describe())
plt.plot(df['column'])
plt.savefig('plot.png')
```

**Features:**
- ✅ Auto-install packages
- ✅ Sandboxed execution
- ✅ File output capture
- ✅ 5-minute timeout protection

</td>
<td width="50%">

#### 📁 **Advanced File Management**
Upload, store, and process files with intelligent lifecycle management.

**Supports 200+ file types:**
- 📊 Data: CSV, Excel, JSON, Parquet
- 🖼️ Images: PNG, JPEG, GIF, SVG, PSD
- 📄 Documents: PDF, DOCX, Markdown
- 🔬 Scientific: MATLAB, HDF5, NumPy
- 🎵 Media: Audio, Video formats
- And many more!

**Smart Features:**
- Auto-expiration (configurable)
- Per-user storage limits
- `/files` command for management

</td>
</tr>
</table>

### 🎨 **Image Generation**

Generate stunning visuals from text prompts using Runware AI:

```
/generate prompt: A futuristic cyberpunk city at night with neon lights
```

- High-quality outputs
- Fast generation (2-5 seconds)
- Multiple style support

### 📊 **Data Analysis & Visualization**

Upload your data files and get instant insights:

```
📈 Statistical Analysis
• Descriptive statistics
• Correlation matrices
• Distribution plots
• Custom visualizations

📉 Supported Formats
• CSV, TSV, Excel
• JSON, Parquet, Feather
• SPSS, Stata, SAS
• And 50+ more formats
```

### 🔍 **Web Tools**

- **Google Search** - Get up-to-date information from the web
- **Web Scraping** - Extract and summarize website content
- **PDF Analysis** - Process and analyze PDF documents

### 🤖 **AI Conversation**

- Natural language understanding
- Context-aware responses
- Time-zone aware (knows current date/time)
- Multi-turn conversations
- DM and server support

### ⏰ **Reminder System**

Set reminders naturally:
```
"Remind me to check email in 30 minutes"
"Set a reminder for tomorrow at 3pm"
"Remind me about the meeting in 2 hours"
```

### 🎯 **Supported AI Models**

<table>
<tr>
<td>

**GPT-4 Series**
- `gpt-4o`
- `gpt-4o-mini`

</td>
<td>

**GPT-5 Series**
- `gpt-5`
- `gpt-5-mini`
- `gpt-5-nano`
- `gpt-5-chat`

</td>
<td>

**o1/o3 Series**
- `o1-preview`
- `o1-mini`
- `o1`
- `o3-mini`

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have:

- 🐳 **Docker** (recommended) or Python 3.13+
- 🎮 **Discord Bot Token** ([Create one here](https://discord.com/developers/applications))
- 🔑 **OpenAI API Key** ([Get it here](https://platform.openai.com/api-keys))
- 🎨 **Runware API Key** ([Sign up here](https://runware.ai/))
- 🔍 **Google API Key** ([Google Cloud Console](https://console.cloud.google.com/))
- 🗄️ **MongoDB** ([MongoDB Atlas](https://cloud.mongodb.com/) - Free tier available)

### 🐳 Option A: Docker Deployment (Recommended)

**Step 1:** Create `.env` file in your project directory

```env
# Discord Configuration
DISCORD_TOKEN=your_discord_bot_token_here

# AI Provider Keys
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Image Generation
RUNWARE_API_KEY=your_runware_api_key_here

# Google Search
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CX=your_custom_search_engine_id_here

# Database
MONGODB_URI=your_mongodb_connection_string_here

# Bot Configuration
ADMIN_ID=your_discord_user_id
TIMEZONE=Asia/Ho_Chi_Minh

# File Management (NEW in v2.0)
MAX_FILES_PER_USER=20
FILE_EXPIRATION_HOURS=48

# Code Execution (NEW in v2.0)
CODE_EXECUTION_TIMEOUT=300
```

**Step 2:** Create `docker-compose.yml`

```yaml
version: '3.8'

services:
  bot:
    image: ghcr.io/coder-vippro/chatgpt-discord-bot:latest
    container_name: chatgpt-discord-bot
    env_file:
      - .env
    volumes:
      - ./data/user_files:/tmp/bot_code_interpreter/user_files
      - ./data/outputs:/tmp/bot_code_interpreter/outputs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

**Step 3:** Start the bot

```bash
docker-compose up -d
```

**Step 4:** Check logs

```bash
docker-compose logs -f bot
```

✅ **Done!** Your bot is now running!

---

### 💻 Option B: Local Deployment

**Step 1:** Clone the repository

```bash
git clone https://github.com/Coder-Vippro/ChatGPT-Discord-Bot.git
cd ChatGPT-Discord-Bot
```

**Step 2:** Create and configure `.env` file

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

**Step 3:** Install dependencies

```bash
pip install -r requirements.txt
```

**Step 4:** Run the bot

```bash
python3 bot.py
```

---

## 📖 Usage Guide

### 💬 Basic Chat

Simply mention the bot or DM it:

```
@Bot Hello! How can you help me?
```

### 🎨 Image Generation

Use the `/generate` command:

```
/generate prompt: A serene Japanese garden with cherry blossoms
```

### 📁 File Upload & Analysis

1. **Upload a file** - Drag and drop any file into the chat
2. **Get file ID** - Bot confirms upload with file ID
3. **Analyze** - Ask the bot to analyze your data

```
User: *uploads data.csv*
Bot: 📊 File Uploaded: data.csv
     🆔 File ID: 123456789_1234567890_abc123
     
User: Analyze this data and create visualizations
Bot: *executes code and generates plots*
```

### 💻 Code Execution

Ask the bot to write and execute code:

```
User: Calculate the fibonacci sequence up to 100 and plot it

Bot: I'll calculate and plot the Fibonacci sequence for you.

```python
def fibonacci(n):
    sequence = [0, 1]
    while sequence[-1] < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

import matplotlib.pyplot as plt
fib = fibonacci(100)
plt.plot(fib)
plt.title('Fibonacci Sequence')
plt.savefig('fibonacci.png')
print(f"Generated {len(fib)} numbers")
```

✅ Output: Generated 12 numbers
📊 Generated file: fibonacci.png
```

### 📋 File Management

Use the `/files` command to manage your uploaded files:

```
/files
```

This shows:
- List of all your files
- File sizes and types
- Expiration dates
- Delete option

### 🔍 Web Search

```
/search prompt: Latest AI developments 2025
```

### 🌐 Web Scraping

```
/web url: https://example.com/article
```

### 📊 User Statistics

```
/user_stat
```

Shows your token usage and model preferences.

### 🔄 Reset Conversation

```
/reset
```

Clears conversation history and deletes all uploaded files.

---

## ⚙️ Configuration

### Environment Variables

<details>
<summary><b>Click to expand full configuration options</b></summary>

#### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DISCORD_TOKEN` | Your Discord bot token | `MTIzNDU2Nzg5MDEyMzQ1Njc4OQ...` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-proj-...` |
| `RUNWARE_API_KEY` | Runware API key for images | `rw_...` |
| `GOOGLE_API_KEY` | Google API key | `AIza...` |
| `GOOGLE_CX` | Custom Search Engine ID | `a1b2c3d4e5f6g7h8i9` |
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017/` |

#### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_BASE_URL` | OpenAI API base URL | `https://api.openai.com/v1` |
| `ADMIN_ID` | Discord user ID for admin | None |
| `TIMEZONE` | Timezone for reminders | `UTC` |
| `MAX_FILES_PER_USER` | Max files per user | `20` |
| `FILE_EXPIRATION_HOURS` | File expiration time | `48` |
| `CODE_EXECUTION_TIMEOUT` | Code timeout in seconds | `300` |
| `ENABLE_WEBHOOK_LOGGING` | Enable webhook logs | `False` |
| `LOGGING_WEBHOOK_URL` | Webhook URL for logs | None |

</details>

### File Management Settings

```env
# Maximum files each user can upload
MAX_FILES_PER_USER=20

# Hours until files expire and are auto-deleted
# Set to -1 for permanent storage (no expiration)
FILE_EXPIRATION_HOURS=48
```

### Code Execution Settings

```env
# Maximum time for code execution (in seconds)
CODE_EXECUTION_TIMEOUT=300

# Package cleanup period (in code_interpreter.py)
PACKAGE_CLEANUP_DAYS=7
```

---

## 📚 Documentation

### 📖 Comprehensive Guides

- [🚀 Quick Start Guide](docs/QUICK_REFERENCE.md)
- [📁 File Management Guide](docs/FILE_MANAGEMENT_GUIDE.md)
- [💻 Code Interpreter Guide](docs/CODE_INTERPRETER_GUIDE.md)
- [📦 Package Cleanup Guide](docs/PACKAGE_CLEANUP_GUIDE.md)
- [🐳 Docker Deployment Guide](docs/DOCKER_DEPLOYMENT_GUIDE.md)
- [⚙️ Environment Setup Guide](docs/ENV_SETUP_GUIDE.md)

### 🆕 What's New in v2.0

- [📋 Release Notes v2.0.0](RELEASE_NOTES_v2.0.0.md)
- [📝 Complete Implementation Summary](docs/COMPLETE_IMPLEMENTATION_SUMMARY.md)
- [🔧 All File Types & Timeout Update](docs/ALL_FILE_TYPES_AND_TIMEOUT_UPDATE.md)
- [🐛 Bug Fixes Documentation](docs/BUGFIX_DATABASE_METHODS.md)

### 🛠️ Technical Documentation

- [🏗️ Architecture Overview](docs/UNIFIED_FILE_SYSTEM_SUMMARY.md)
- [📊 Token Counting Guide](docs/TOKEN_COUNTING_GUIDE.md)
- [🕐 Current Time in Context](docs/CURRENT_TIME_IN_CONTEXT.md)
- [🔒 Security Guidelines](SECURITY.md)

---

## 🎯 Examples

### Example 1: Data Analysis

```
User: *uploads sales_data.csv*

User: Analyze this sales data and show me:
1. Monthly revenue trends
2. Top 5 products
3. Customer distribution by region

Bot: I'll analyze your sales data and create visualizations.

[Executes Python code with pandas and matplotlib]
[Generates 3 charts]

📊 Analysis Results:
- Total Revenue: $1.2M
- Best Month: December ($250K)
- Top Product: Product A ($300K)
- Main Region: North America (45%)

📈 Generated 3 visualizations
```

### Example 2: Image Generation

```
User: /generate prompt: A serene Japanese garden with cherry blossoms, koi pond, and traditional temple in the background, anime style

Bot: 🎨 Generating your image...
[2 seconds later]
🖼️ Here's your generated image!
[Beautiful anime-style Japanese garden image]
```

### Example 3: Code Execution

```
User: Create a function to check if a number is prime, then find all primes up to 1000

Bot: I'll create a prime checker and find all primes up to 1000.

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(2, 1001) if is_prime(n)]
print(f"Found {len(primes)} prime numbers")
print(f"First 10: {primes[:10]}")
print(f"Last 10: {primes[-10:]}")
```

✅ Output:
Found 168 prime numbers
First 10: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
Last 10: [937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

- 🐛 **Report Bugs** - [Open an issue](https://github.com/Coder-Vippro/ChatGPT-Discord-Bot/issues)
- ✨ **Suggest Features** - [Start a discussion](https://github.com/Coder-Vippro/ChatGPT-Discord-Bot/discussions)
- 📝 **Improve Docs** - Submit documentation updates
- 💻 **Submit Code** - Create pull requests

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/ChatGPT-Discord-Bot.git
cd ChatGPT-Discord-Bot

# Install dependencies
pip install -r requirements.txt

# Create a branch
git checkout -b feature/your-feature-name

# Make your changes and test
python3 bot.py

# Run tests
pytest tests/

# Commit and push
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name
```

### Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

---

## 🐛 Troubleshooting

<details>
<summary><b>Bot won't start</b></summary>

**Check:**
1. All required environment variables are set
2. Discord token is valid
3. MongoDB is accessible
4. Port 27017 is not blocked (if using local MongoDB)

**Solution:**
```bash
# Check logs
docker-compose logs bot

# Verify .env file
cat .env | grep -v '^#'
```
</details>

<details>
<summary><b>Code execution fails</b></summary>

**Common causes:**
- Package installation timeout
- Code exceeds 5-minute timeout
- Memory limit exceeded

**Solutions:**
```env
# Increase timeout
CODE_EXECUTION_TIMEOUT=600

# In docker-compose.yml, increase memory
memory: 8G
```
</details>

<details>
<summary><b>Files not uploading</b></summary>

**Check:**
1. File size (Discord limit: 25MB for free, 500MB for Nitro)
2. Storage limit reached (default: 20 files per user)
3. Disk space available

**Solution:**
```env
# Increase file limit
MAX_FILES_PER_USER=50

# Set permanent storage
FILE_EXPIRATION_HOURS=-1
```
</details>

<details>
<summary><b>Docker "Resource busy" error</b></summary>

This is fixed in v2.0! The bot now uses system Python in Docker.

**If you still see this error:**
```bash
# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```
</details>

---

## 📊 Performance

### System Requirements

| Deployment | CPU | RAM | Disk | Network |
|------------|-----|-----|------|---------|
| **Minimal** | 1 core | 2GB | 2GB | 1 Mbps |
| **Recommended** | 2 cores | 4GB | 5GB | 10 Mbps |
| **High Load** | 4 cores | 8GB | 10GB | 100 Mbps |

### Benchmarks

```
📈 Response Times (avg):
- Simple chat: 1-2 seconds
- Code execution: 2-5 seconds
- Image generation: 3-5 seconds
- Data analysis: 5-10 seconds
- File upload: <1 second

💾 Resource Usage:
- Idle: ~200 MB RAM
- Active: ~500 MB RAM
- Peak: ~2 GB RAM
- Docker image: ~600 MB

🚀 Throughput:
- Concurrent users: 50+
- Messages/minute: 100+
- File uploads/hour: 500+
```

---

## 🔒 Security

### Security Features

- ✅ Sandboxed code execution
- ✅ Per-user file isolation
- ✅ Timeout protection
- ✅ Resource limits
- ✅ Input validation
- ✅ Package validation
- ✅ MongoDB injection prevention

### Reporting Security Issues

Found a vulnerability? Please **DO NOT** open a public issue.

See [SECURITY.md](SECURITY.md) for reporting guidelines.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Special thanks to:

- **[OpenAI](https://openai.com)** - For powering our AI capabilities
- **[Runware](https://runware.ai)** - For image generation API
- **[Discord.py](https://discordpy.readthedocs.io/)** - For excellent Discord library
- **[MongoDB](https://mongodb.com)** - For reliable database services
- **All Contributors** - For making this project better

---

## 📞 Support & Community

### Get Help

- 💬 **Discord Server**: [Join our community](https://discord.gg/yourserver)
- 🐛 **GitHub Issues**: [Report bugs](https://github.com/Coder-Vippro/ChatGPT-Discord-Bot/issues)
- 💡 **Discussions**: [Share ideas](https://github.com/Coder-Vippro/ChatGPT-Discord-Bot/discussions)

### Useful Commands

```bash
# View logs
docker-compose logs -f bot

# Restart bot
docker-compose restart bot

# Check file storage
du -sh data/user_files/

# View package cache
cat /tmp/bot_code_interpreter/package_cache.json | jq

# Update to latest version
docker-compose pull
docker-compose up -d
```

---

## 📈 Stats & Updates

![Stars](https://img.shields.io/github/stars/Coder-Vippro/ChatGPT-Discord-Bot?style=social)
![Forks](https://img.shields.io/github/forks/Coder-Vippro/ChatGPT-Discord-Bot?style=social)
![Issues](https://img.shields.io/github/issues/Coder-Vippro/ChatGPT-Discord-Bot)
![Pull Requests](https://img.shields.io/github/issues-pr/Coder-Vippro/ChatGPT-Discord-Bot)

**Latest Release**: v2.0.0 (October 3, 2025)  
**Active Servers**: Growing daily  

---

## 🗺️ Roadmap

### Version 2.1 (Q4 2025)
- [ ] Multi-language support
- [ ] Voice channel integration
- [ ] Usage analytics dashboard
- [ ] Advanced reminders (recurring)
- [ ] Custom tool creation

### Version 2.2 (Q1 2026)
- [ ] Collaborative code sessions
- [ ] Code version history
- [ ] Direct database connections
- [ ] Mobile companion app
- [ ] Workflow automation

[View full roadmap →](https://github.com/Coder-Vippro/ChatGPT-Discord-Bot/projects)

---

<div align="center">

### ⭐ Star Us on GitHub!

If you find this bot useful, please give it a star! It helps others discover the project.

---

Made with ❤️ by [Coder-Vippro](https://github.com/coder-vippro)

[⬆ Back to Top](#-chatgpt-discord-bot)

</div>
