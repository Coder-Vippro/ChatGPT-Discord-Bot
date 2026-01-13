import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== IMAGE CONFIGURATION ====================
# Load image configuration from JSON file
def load_image_config() -> dict:
    """Load image configuration from JSON file"""
    config_paths = [
        Path(__file__).parent.parent.parent / "config" / "image_config.json",
        Path(__file__).parent.parent / "config" / "image_config.json",
        Path("config/image_config.json"),
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Error loading image config from {config_path}: {e}")
    
    return {}

# Load image config once at module import
_IMAGE_CONFIG = load_image_config()

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
    "openai/o4-mini",
    "claude/claude-3-5-sonnet",
    "claude/claude-3-5-haiku",
    "claude/claude-3-opus",
]

# ==================== IMAGE GENERATION MODELS ====================
# Models are loaded from config/image_config.json
# Edit that file to add/modify image models
IMAGE_MODELS = _IMAGE_CONFIG.get("image_models", {
    "flux": {
        "model_id": "runware:101@1",
        "name": "FLUX.1",
        "description": "High-quality image generation with FLUX",
        "default_width": 1024,
        "default_height": 1024,
        "max_width": 2048,
        "max_height": 2048,
        "supports_negative_prompt": True
    }
})

# Upscale models from config
UPSCALE_MODELS = _IMAGE_CONFIG.get("upscale_models", {
    "clarity": {
        "model_id": "runware:500@1",
        "name": "Clarity",
        "supported_factors": [2, 4]
    }
})

# Background removal models from config
BACKGROUND_REMOVAL_MODELS = _IMAGE_CONFIG.get("background_removal_models", {
    "bria": {
        "model_id": "runware:110@1",
        "name": "Bria RMBG 2.0"
    }
})

# Image settings from config
IMAGE_SETTINGS = _IMAGE_CONFIG.get("settings", {
    "default_model": "flux",
    "default_upscale_model": "clarity",
    "default_background_removal_model": "bria"
})

# Default image model
DEFAULT_IMAGE_MODEL = IMAGE_SETTINGS.get("default_model", "flux")

# Default negative prompts by category
DEFAULT_NEGATIVE_PROMPTS = _IMAGE_CONFIG.get("default_negative_prompts", {
    "general": "blurry, distorted, low quality, watermark, signature, text, bad anatomy, deformed"
})

# Aspect ratios from config
ASPECT_RATIOS = _IMAGE_CONFIG.get("aspect_ratios", {
    "1:1": {"width": 1024, "height": 1024},
    "16:9": {"width": 1344, "height": 768},
    "9:16": {"width": 768, "height": 1344}
})

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
    "openai/gpt-5-chat": 4000,
    "claude/claude-3-5-sonnet": 8000,
    "claude/claude-3-5-haiku": 8000,
    "claude/claude-3-opus": 8000,
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

TOOLS:
1. google_search(query) - Web search for current info
2. scrape_webpage(url) - Extract webpage content
3. execute_python_code(code) - Run Python, packages auto-install. **FILE ACCESS: See critical instructions below!**
4. set_reminder(content, time) / get_reminders() - Manage reminders

═══════════════════════════════════════════════════════════════
⚠️ CRITICAL: FILE ACCESS IN CODE INTERPRETER
═══════════════════════════════════════════════════════════════

When users upload files, you will see a message like:
   📁 FILE UPLOADED - USE THIS FILE_ID:
   Filename: data.csv
   ⚠️ TO ACCESS THIS FILE IN CODE, YOU MUST USE:
      df = load_file('<THE_ACTUAL_FILE_ID_FROM_CONTEXT>')

**IMPORTANT: Copy the EXACT file_id from the file upload message - do NOT use examples!**

✅ CORRECT:
   df = load_file('<file_id_from_upload_message>')
   print(df.head())  # Use print() to show output!
   
⚠️ IMPORTANT: Always use print() to display results - code output is only captured via print()!

❌ WRONG - Using filename:
   df = pd.read_csv('data.csv')  # FAILS - file not found!
   
❌ WRONG - Using example file_id from prompts:
   df = load_file('example_id_from_docs')  # FAILS - use the REAL ID!

⚠️ CRITICAL: Look for the 📁 FILE UPLOADED message in this conversation and copy the EXACT file_id shown there!

═══════════════════════════════════════════════════════════════
IMAGE GENERATION & EDITING TOOLS
═══════════════════════════════════════════════════════════════

5. generate_image(prompt, model, num_images, width, height, aspect_ratio, negative_prompt, steps, cfg_scale, seed)
   Create images from text descriptions.
   
   MODELS (use model parameter):
   • "flux" - FLUX.1 (default, best quality, 1024x1024)
   • "flux-dev" - FLUX.1 Dev (more creative outputs)
   • "sdxl" - Stable Diffusion XL (detailed, high-res)
   • "realistic" - Realistic Vision (photorealistic)
   • "anime" - Anime/illustration style
   • "dreamshaper" - Creative/artistic style
   
   ASPECT RATIOS (use aspect_ratio parameter):
   • "1:1" - Square (1024x1024)
   • "16:9" - Landscape wide (1344x768)
   • "9:16" - Portrait tall (768x1344)
   • "4:3" - Landscape (1152x896)
   • "3:4" - Portrait (896x1152)
   • "3:2" - Photo landscape (1248x832)
   • "2:3" - Photo portrait (832x1248)
   • "21:9" - Ultrawide (1536x640)
   
   Examples:
   generate_image("a dragon in a forest", "flux", 1)
   generate_image({"prompt": "sunset beach", "model": "realistic", "aspect_ratio": "16:9"})
   generate_image({"prompt": "anime girl", "model": "anime", "width": 768, "height": 1024})

6. generate_image_with_refiner(prompt, model, num_images)
   Generate high-quality images using SDXL with refiner for better details.
   Best for: detailed artwork, complex scenes
   Example: generate_image_with_refiner("detailed fantasy castle", "sdxl", 1)

7. upscale_image(image_url, scale_factor, model)
   Enlarge images to higher resolution.
   
   UPSCALE MODELS:
   • "clarity" - High-quality clarity upscaling (default)
   • "ccsr" - Content-consistent super-resolution
   • "sd-latent" - SD latent space upscaling
   • "swinir" - Fast SwinIR (supports 4x)
   
   SCALE FACTORS: 2 or 4 (depending on model)
   
   Requires: User must provide an image URL first
   Example: upscale_image("https://example.com/image.jpg", 2, "clarity")

8. remove_background(image_url, model) / edit_image(image_url, "remove_background")
   Remove background from images (outputs PNG with transparency).
   
   BACKGROUND REMOVAL MODELS:
   • "bria" - Bria RMBG 2.0 (default, high quality)
   • "rembg" - RemBG 1.4 (classic, supports alpha matting)
   • "birefnet-base" - BiRefNet base model
   • "birefnet-general" - BiRefNet general purpose
   • "birefnet-portrait" - BiRefNet optimized for portraits
   
   Requires: User must provide an image URL first
   Example: remove_background("https://example.com/photo.jpg", "bria")

9. photo_maker(prompt, input_images, style, strength, num_images)
   Generate images based on reference photos (identity preservation).
   
   Parameters:
   • prompt: Text description of desired output
   • input_images: List of reference image URLs
   • style: Style to apply (default: "No style")
   • strength: Reference influence 0-100 (default: 40)
   
   Requires: User must provide reference images first
   Example: photo_maker({"prompt": "professional headshot", "input_images": ["url1", "url2"], "style": "Photographic"})

10. image_to_text(image_url)
    Generate text description/caption from an image.
    Use for: Understanding image content, accessibility, OCR-like tasks
    Example: image_to_text("https://example.com/image.jpg")

11. enhance_prompt(prompt, num_versions, max_length)
    Improve prompts for better image generation results.
    Returns multiple enhanced versions of your prompt.
    Example: enhance_prompt("cat on roof", 3, 200)

═══════════════════════════════════════════════════════════════
USAGE GUIDELINES
═══════════════════════════════════════════════════════════════

WHEN TO USE EACH TOOL:
• "create/draw/generate/make an image of X" → generate_image
• "high quality/detailed image" → generate_image_with_refiner  
• "remove/delete background" → remove_background (pass 'latest_image')
• "make image bigger/larger/upscale" → upscale_image (pass 'latest_image')
• "create image like this/based on this photo" → photo_maker (pass ['latest_image'])
• "what's in this image/describe image" → image_to_text (pass 'latest_image')
• "improve this prompt" → enhance_prompt

IMPORTANT NOTES:
• For image tools (upscale, remove_background, photo_maker, image_to_text), when user uploads an image, pass 'latest_image' as the image_url parameter - the system automatically uses their most recent uploaded image
• You don't need to extract or copy image URLs - just use 'latest_image'
• Default model is "flux" - best for general use
• Use "realistic" for photos, "anime" for illustrations
• For math/data analysis → use execute_python_code instead
• Always cite sources (Title–URL) when searching web"""

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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Claude API key

# File management settings
FILE_EXPIRATION_HOURS = int(os.getenv("FILE_EXPIRATION_HOURS", "48"))  # Hours until files expire (-1 for never)
MAX_FILES_PER_USER = int(os.getenv("MAX_FILES_PER_USER", "20"))  # Maximum files per user
CODE_EXECUTION_TIMEOUT = int(os.getenv("CODE_EXECUTION_TIMEOUT", "300"))  # Timeout for code execution in seconds (default: 5 minutes)

# Print debug information if environment variables are not found
if not DISCORD_TOKEN:
    print("WARNING: DISCORD_TOKEN not found in .env file")
if not MONGODB_URI:
    print("WARNING: MONGODB_URI not found in .env file")
if not RUNWARE_API_KEY:
    print("WARNING: RUNWARE_API_KEY not found in .env file")
if not ANTHROPIC_API_KEY:
    print("INFO: ANTHROPIC_API_KEY not found in .env file - Claude models will not be available")
if ENABLE_WEBHOOK_LOGGING and not LOGGING_WEBHOOK_URL:
    print("WARNING: Webhook logging enabled but LOGGING_WEBHOOK_URL not found in .env file")