import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Bot statuses
STATUSES = [
    "Powered by openai/gpt-4o!",
    "Generating creative text!",
    "Creating images on demand!",
    "Answering your queries with AI!",
    "Exploring AI capabilities!",
    "Crafting stories with GPT!",
    "Generating artwork with AI!",
    "Transforming ideas into text!",
    "Your personal AI assistant!",
    "Making text-based magic happen!",
    "Bringing your prompts to life!",
    "Searching the web for you!",
    "Summarizing information with AI!",
    "Discussing the latest AI trends!",
    "Innovating with neural networks!",
    "Providing image generation services!",
    "Curating knowledge with AI!",
    "Explaining concepts in simple terms!",
    "Generating visuals for your ideas!",
    "Answering coding questions!",
    "Enhancing your creativity!",
    "Crafting engaging dialogues!",
    "Bringing imagination to reality!",
    "Your AI-powered content creator!",
    "Exploring the world of AI art!",
    "Helping you learn with AI!",
    "Generating prompts for inspiration!",
    "Creating stunning visuals!",
    "Answering trivia questions!",
    "Your source for AI-generated insights!",
    "Delving into the world of machine learning!",
    "Providing data-driven answers!",
    "Crafting personalized content!",
    "Exploring creative AI solutions!",
    "Summarizing articles for you!",
    "Generating memes with AI!",
    "Transforming text into images!",
    "Enhancing your projects with AI!",
    "Creating unique characters with GPT!",
    "Exploring AI storytelling!",
    "Generating logos and designs!",
    "Helping you brainstorm ideas!",
    "Creating educational content!",
    "Your creative writing partner!",
    "Building narratives with AI!",
    "Exploring ethical AI use!",
    "Bringing concepts to life visually!",
    "Your AI companion for learning!",
    "Generating infographics!",
    "Creating art based on your prompts!",
    "Exploring AI in entertainment!",
    "Your gateway to AI innovation!",
]

# List of available models
MODEL_OPTIONS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1",
    "openai/gpt-4.1-nano",
    "openai/gpt-4.1-mini",
    "openai/gpt-5",
    "openai/gpt-5-nano",
    "openai/gpt-5-mini",
    "openai/gpt-5-chat",
    "openai/o1-preview",
    "openai/o1-mini",
    "openai/o1",
    "openai/o3-mini",
    "openai/o3",
    "openai/o4-mini"
]

# Model-specific token limits for automatic history management
MODEL_TOKEN_LIMITS = {
    "openai/o1-preview": 4000,  # Conservative limit (max 4000)
    "openai/o1-mini": 4000,
    "openai/o1": 4000,
    "openai/gpt-4o": 8000,
    "openai/gpt-4o-mini": 8000,
    "openai/gpt-4.1": 8000,
    "openai/gpt-4.1-nano": 8000,
    "openai/gpt-4.1-mini": 8000,
    "openai/o3-mini": 4000,
    "openai/o3": 4000,
    "openai/o4-mini": 4000,
    "openai/gpt-5": 4000,
    "openai/gpt-5-nano": 4000,
    "openai/gpt-5-mini": 4000,
    "openai/gpt-5-chat": 4000
}

# Default token limit for unknown models
DEFAULT_TOKEN_LIMIT = 4000

# Default model for new users
DEFAULT_MODEL = "openai/gpt-4.1"

PDF_ALLOWED_MODELS = ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-4.1","openai/gpt-4.1-nano","openai/gpt-4.1-mini"]
PDF_BATCH_SIZE = 3

# Prompt templates
WEB_SCRAPING_PROMPT = "Analyze webpage content and extract key information. Focus on relevance, cite sources when needed, stay neutral, and organize logically. Format for Discord."

NORMAL_CHAT_PROMPT = """You're ChatGPT for Discord. Be concise, helpful, safe. Reply in user's language. Use short paragraphs, bullets, minimal markdown.

Tools:
- google_search: real-time info, fact-checking, news
- scrape_webpage: extract/analyze webpage content
- execute_python_code: math, data processing, plotting (always print())
- analyze_data_file: CSV/Excel insights & visualization
- image_suite: generate/edit/upscale/create portraits
- reminders: schedule/retrieve user reminders
- web_search_multi: parallel searches for comprehensive research

Smart Usage:
- Chain tools: search→scrape→analyze for deep research
- Auto-suggest relevant tools based on user intent
- Batch operations for efficiency

Rules:
- One clarifying question if ambiguous
- Prioritize answers over details
- Cite sources: (Title – URL)
- Use execute_python_code for complex math
- Never invent sources
- Code fences for equations (no LaTeX)
- Return image URLs with brief descriptions"""

SEARCH_PROMPT = "Research Assistant with Google Search access. Synthesize search results into accurate answers. Prioritize credible sources, compare perspectives, acknowledge limitations, cite sources. Structure responses logically."

PDF_ANALYSIS_PROMPT = """PDF Analysis Assistant. Analyze content thoroughly:
- Structure clearly, highlight key info
- Connect sections, explain technical terms
- Analyze data/statistics specifically
- Simplify complex ideas when appropriate
- Respond in user's language
Focus on accuracy and relevance."""

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'logs/discord_bot.log',
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
        'discord': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
        'discord.http': {
            'handlers': ['console', 'file'],
            'level': 'WARNING',
        },
    },
}

# Webhook logging configuration
ENABLE_WEBHOOK_LOGGING = os.getenv('ENABLE_WEBHOOK_LOGGING', 'False').lower() == 'true'
LOGGING_WEBHOOK_URL = os.getenv('LOGGING_WEBHOOK_URL', '')
WEBHOOK_LOG_LEVEL = os.getenv('WEBHOOK_LOG_LEVEL', 'INFO')
WEBHOOK_APP_NAME = os.getenv('WEBHOOK_APP_NAME', 'Discord Bot')
WEBHOOK_BATCH_SIZE = int(os.getenv('WEBHOOK_BATCH_SIZE', '5'))
WEBHOOK_FLUSH_INTERVAL = int(os.getenv('WEBHOOK_FLUSH_INTERVAL', '10'))

# Map string log levels to logging module levels
LOG_LEVEL_MAP = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50,
}

# Environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
ADMIN_ID = os.getenv("ADMIN_ID")  # Add ADMIN_ID if you're using it
TIMEZONE = os.getenv("TIMEZONE", "UTC")  # Default to UTC if not specified

# Print debug information if environment variables are not found
if not DISCORD_TOKEN:
    print("WARNING: DISCORD_TOKEN not found in .env file")
if not MONGODB_URI:
    print("WARNING: MONGODB_URI not found in .env file")
if not RUNWARE_API_KEY:
    print("WARNING: RUNWARE_API_KEY not found in .env file")
if ENABLE_WEBHOOK_LOGGING and not LOGGING_WEBHOOK_URL:
    print("WARNING: Webhook logging enabled but LOGGING_WEBHOOK_URL not found in .env file")