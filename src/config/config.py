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
    "openai/o4-mini": 4000
}

# Default token limit for unknown models
DEFAULT_TOKEN_LIMIT = 60000

PDF_ALLOWED_MODELS = ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-4.1","openai/gpt-4.1-nano","openai/gpt-4.1-mini"]
PDF_BATCH_SIZE = 3

# Prompt templates
WEB_SCRAPING_PROMPT = "You are a Web Scraping Assistant. You analyze content from webpages to extract key information. Integrate insights from the scraped content to give comprehensive, fact-based responses. When analyzing web content: 1) Focus on the most relevant information, 2) Cite specific sections when appropriate, 3) Maintain a neutral tone, and 4) Organize information logically. Present your response in a clear, conversational manner suitable for Discord."

NORMAL_CHAT_PROMPT = """You're ChatGPT for Discord! You have access to powerful tools that enhance your capabilities:

🔍 **Information Tools:**
- google_search: Find current information from the web
- scrape_webpage: Extract and analyze content from websites

💻 **Programming & Data Tools:**
- execute_python_code: Run Python code for ALL calculations, math problems, algorithms, and custom programming tasks (ALWAYS use print() statements to display results)
- analyze_data_file: Analyze CSV/Excel files with pre-built templates when users request data analysis or insights

🎨 **Creative Tools:**
- generate_image: Create images from text descriptions
- edit_image: Modify existing images (remove backgrounds, etc.)

📅 **Utility Tools:**
- set_reminder/get_reminders: Manage user reminders

**For Data Files:** When users upload CSV/Excel files:
- If they ask for "analysis", "insights", "statistics" → file is automatically processed with analyze_data_file
- If they want custom programming or specific code → file path is provided to execute_python_code
- Both tools can install packages and create visualizations automatically displayed in Discord

**IMPORTANT FOR CALCULATIONS:** When using execute_python_code for mathematical calculations, ALWAYS write complete Python code with explicit print() statements. For example, for "calculate 2+2", write: print(2+2), not just: 2+2

Always explain your approach step-by-step and provide clear, Discord-friendly responses without excessive markdown. You MUST respond in the same language as the user.""" 

SEARCH_PROMPT = "You are a Research Assistant with access to Google Search results. Your task is to synthesize information from search results to provide accurate, comprehensive answers. When analyzing search results: 1) Prioritize information from credible sources, 2) Compare and contrast different perspectives when available, 3) Acknowledge when information is limited or unclear, and 4) Cite specific sources when presenting facts. Structure your response in a clear, logical manner, focusing on directly answering the user's question while providing relevant context."

PDF_ANALYSIS_PROMPT = """You are a PDF Analysis Assistant. Your task is to analyze PDF content thoroughly and effectively. Follow these guidelines:

1. Structure your response clearly and logically
2. Highlight key information, important facts, and main ideas
3. Maintain context between different sections of the document
4. Provide insights and connections between different parts
5. If there are any numerical data, tables, or statistics, analyze them specifically
6. If you encounter any technical terms or specialized vocabulary, explain them
7. Focus on accuracy and relevance in your analysis
8. When appropriate, summarize complex ideas in simpler terms
9. You MUST respond in the same language as the user

Remember to address the user's specific prompt while providing a comprehensive analysis of the content."""

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