import os
import discord
import io
import threading
import tiktoken
import asyncio
import requests
import logging
import sys
import json
import aiohttp
import tempfile
import subprocess
import re
from discord.ext import commands, tasks
from discord import app_commands
from motor.motor_asyncio import AsyncIOMotorClient
from bs4 import BeautifulSoup
from openai import OpenAI, RateLimitError
from runware import Runware, IImageInference
from collections import defaultdict
from dotenv import load_dotenv
from flask import Flask, jsonify
from PyPDF2 import PdfReader
import io

# Load environment variables
load_dotenv()

# Supported file types for text processing
supported_file_types = ['.txt', '.py', '.js', '.html', '.css', '.json', '.md', '.log', '.csv', '.xml', '.yml', '.yaml', '.ini', '.cfg', '.conf']

# Flask app for health-check
app = Flask(__name__)

# Health-check endpoint
@app.route('/health', methods=['GET'])
def health():
    """
    Checks if the bot is ready and connected to Discord.
    """
    if bot.is_closed():  # Bot is disconnected
        return jsonify(status="unhealthy", error="Bot is disconnected"), 500
    elif not bot.is_ready():  # Bot is not ready yet
        return jsonify(status="unhealthy", error="Bot is not ready"), 500
    elif bot.latency > 151:  # Bot heartbeat is blocked for more than 151 seconds
        return jsonify(status="unhealthy", error=f"Heartbeat to websocket blocked for {bot.latency:.2f} seconds"), 500
    else:
        return jsonify(status="healthy"), 200

# Run Flask server in a separate thread
def run_flask():
    """
    Starts the Flask server.
    """
    app.run(host="0.0.0.0", port=5000)

# OpenAI client initialization
client = OpenAI(
    base_url=str(os.getenv("OPENAI_BASE_URL")),
    api_key=str(os.getenv("OPENAI_API_KEY")),

)
# Admin ID for whitelist commands
ADMIN_ID = str((os.getenv("ADMIN_ID")))

# List of bot statuses
statuses = [
    "Powered by GPT-4o!",
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
    "gpt-4o",
    "gpt-4o-mini",
    "o1-preview",
    "o1-mini",
    "o1",
    "o3-mini"
]

PDF_ALLOWED_MODELS = ["gpt-4o", "gpt-4o-mini"]

# Prompt for different plugins
WEB_SCRAPING_PROMPT = "You are using the Web Scraping Plugin, gathering information from given url. Respond accurately and combine data to provide a clear, insightful summary. "
NORMAL_CHAT_PROMPT = "You're ChatGPT for Discord! You can chat, generate images, and perform searches. Craft responses that are easy to copy directly into Discord chats, without using markdown, code blocks, or extra formatting. When you solving any problems you must remember that: Let's solve this step-by-step. What information do we need to find? What operation might help us solve this? Explain your reasoning and provide the answer. For code_interpreter you must send user the code you used to run in anycase."
SEARCH_PROMPT = "You are using the Google Search Plugin, accessing information from the top 3 Google results link which is the scraped content from these 3 website. Summarize these findings clearly, adding relevant insights to answer the users question."
PDF_ANALYSIS_PROMPT = """You are a PDF Analysis Assistant. Your task is to analyze PDF content thoroughly and effectively. Follow these guidelines:

1. Structure your response clearly and logically
2. Highlight key information, important facts, and main ideas
3. Maintain context between different sections of the document
4. Provide insights and connections between different parts
5. If there are any numerical data, tables, or statistics, analyze them specifically
6. If you encounter any technical terms or specialized vocabulary, explain them
7. Focus on accuracy and relevance in your analysis
8. When appropriate, summarize complex ideas in simpler terms

Remember to address the user's specific prompt while providing a comprehensive analysis of the content."""

# Google API details
GOOGLE_API_KEY = str(os.getenv("GOOGLE_API_KEY"))  # Google API Key
GOOGLE_CX = str(os.getenv("GOOGLE_CX"))  # Search Engine ID

# Runware API key
RUNWARE_API_KEY = str(os.getenv("RUNWARE_API_KEY"))

#MongoDB URI
MONGODB_URI = str(os.getenv("MONGODB_URI"))

# Số lượng ảnh được xử lý cùng lúc cho PDF
PDF_BATCH_SIZE = 3

# Initialize Runware SDK
runware = Runware(api_key=RUNWARE_API_KEY)

# MongoDB client initialization
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client['chatgpt_discord_bot']  # Database name

# Dictionary to keep track of user requests and their cooldowns
user_requests = defaultdict(lambda: {'last_request': 0, 'queue': asyncio.Queue()})

# Dictionary to store user conversation history
user_histories = {}

# Bot token
TOKEN = str(os.getenv("DISCORD_TOKEN"))

# --- Database functions ---

async def get_history(user_id):
    user_data = await db.user_histories.find_one({'user_id': user_id})
    return user_data['history'] if user_data and 'history' in user_data else [{"role": "system", "content": NORMAL_CHAT_PROMPT}]

async def save_history(user_id, history):
    await db.user_histories.update_one(
        {'user_id': user_id},
        {'$set': {'history': history}},
        upsert=True
    )

async def get_user_model(user_id):
    user_pref = await db.user_preferences.find_one({'user_id': user_id})
    return user_pref['model'] if user_pref and 'model' in user_pref else "gpt-4o"

async def save_user_model(user_id, model):
    await db.user_preferences.update_one(
        {'user_id': user_id},
        {'$set': {'model': model}},
        upsert=True
    )

async def is_admin(user_id: int) -> bool:
    """Check if a user is an admin."""
    return str(user_id) == ADMIN_ID

async def is_user_whitelisted(user_id):
    """Check if a user is whitelisted for PDF processing."""
    if await is_admin(user_id):
        return True
    whitelist = await db.pdf_whitelist.find_one({'user_id': user_id})
    return bool(whitelist)

async def add_user_to_whitelist(user_id):
    """Add a user to the PDF processing whitelist."""
    await db.pdf_whitelist.update_one(
        {'user_id': user_id},
        {'$set': {'whitelisted': True}},
        upsert=True
    )

async def remove_user_from_whitelist(user_id):
    """Remove a user from the PDF processing whitelist."""
    result = await db.pdf_whitelist.delete_one({'user_id': user_id})
    return result.deleted_count > 0

async def add_user_to_blacklist(user_id):
    """Add a user to the bot blacklist."""
    await db.bot_blacklist.update_one(
        {'user_id': user_id},
        {'$set': {'blacklisted': True}},
        upsert=True
    )

async def remove_user_from_blacklist(user_id):
    """Remove a user from the bot blacklist."""
    result = await db.bot_blacklist.delete_one({'user_id': user_id})
    return result.deleted_count > 0

async def is_user_blacklisted(user_id):
    """Check if a user is blacklisted from using the bot."""
    if await is_admin(user_id):
        return False
    blacklist = await db.bot_blacklist.find_one({'user_id': user_id})
    return bool(blacklist)

# --- End of Database functions ---

# Intents and bot initialization
intents = discord.Intents.default()
intents.message_content = True

# Bot initialization
bot = commands.Bot(command_prefix="//quocanhvu", intents=intents, heartbeat_timeout=120)
tree = bot.tree  # For slash commands

# Function to perform a Google search and return results
def google_custom_search(query: str, num_results: int = 5, auto_scrape: bool = True) -> dict:
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": num_results
    }
    try:
        response = requests.get(search_url, params=params, timeout=30)  # Add timeout
        response.raise_for_status()  # Check for any errors in the response
        data = response.json()

        # Check if 'items' key is present in the response
        if 'items' in data:
            results = []
            
            # Try to scrape content from multiple URLs if enabled
            if auto_scrape:
                # Try up to 3 URLs for scraping
                successful_scrape = False
                urls_to_try = min(3, len(data['items']))
                
                for i in range(urls_to_try):
                    if i >= len(data['items']):
                        break
                        
                    item = data['items'][i]
                    title = item.get('title', 'No Title')
                    link = item.get('link', '')
                    snippet = item.get('snippet', 'No snippet available')
                    
                    # Add the search result without scraped content first
                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet
                    })
                    
                    # Skip non-valid URLs
                    if not link or link == 'No Link':
                        continue
                    
                    # Try to scrape content
                    try:
                        scraped_content = scrape_web_content(link)
                        # Only add if we got meaningful content
                        if scraped_content and isinstance(scraped_content, str) and len(scraped_content) > 100:
                            # Add as a separate entry to highlight the scraped content
                            results.append({
                                "title": f"Scraped content from: {title}",
                                "link": link,
                                "content": scraped_content
                            })
                            successful_scrape = True
                            # We got a successful scrape, no need to try more URLs
                            break
                    except Exception as e:
                        logging.error(f"Error scraping content from {link}: {str(e)}")
                        continue
            
            # If auto-scrape is disabled or failed for all URLs, just return regular search results
            if not auto_scrape or not successful_scrape:
                for item in data['items']:
                    title = item.get('title', 'No Title')
                    link = item.get('link', 'No Link')
                    snippet = item.get('snippet', 'No snippet available')
                    
                    # Only add if not already in results
                    if not any(r.get('link') == link for r in results):
                        results.append({
                            "title": title,
                            "link": link,
                            "snippet": snippet
                        })
                    
            return {
                "search_query": query,
                "results": results
            }
        else:
            return {
                "search_query": query,
                "results": [],
                "error": "No results found"
            }

    except requests.exceptions.RequestException as e:
        error_msg = f"Error during search request: {str(e)}"
        logging.error(error_msg)
        return {
            "search_query": query,
            "results": [],
            "error": error_msg
        }
    
# Function to scrape content from a webpage
def scrape_web_content(url: str) -> str:
    if not url:
        return "Error: No URL provided"
    
    # Ignore URLs that are unlikely to be scrapable or might cause problems
    if any(x in url.lower() for x in ['.pdf', '.zip', '.jpg', '.png', '.mp3', '.mp4', 'youtube.com', 'youtu.be']):
        return f"Skipped scraping for non-HTML content: {url}"
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.97 Safari/537.36'
        }
        
        # Use a timeout to avoid hanging on slow servers
        page = requests.get(url, headers=headers, timeout=8)
        
        # Check HTTP status code
        if page.status_code != 200:
            return f"Error: Received status code {page.status_code} for {url}"

        # Check content type to make sure we're dealing with HTML
        content_type = page.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            return f"Error: Content is not HTML. Content type: {content_type}"

        # Parse with BeautifulSoup
        soup = BeautifulSoup(page.content, "html.parser")
        
        # Remove script, style and hidden elements that aren't useful for content
        for element in soup(['script', 'style', 'meta', 'noscript', '[document]', 'head', 'img', 'header', 'footer']):
            element.extract()

        # Extract main article content if available (common patterns in websites)
        main_content = None
        
        # Try to find the main article content in common containers
        for container in ['article', 'main', '.content', '#content', '.post', '.article', '.entry-content', '.post-content']:
            if main_content:
                break
                
            if container.startswith('.') or container.startswith('#'):
                elements = soup.select(container)
            else:
                elements = soup.find_all(container)
                
            if elements:
                main_content = max(elements, key=lambda x: len(x.get_text().strip()))

        # If we found a main content container, extract text from it
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
            if text and len(text) > 10:  # Adjusted to consider shorter meaningful content
                return text
            if text and len(text) > 100:
                return text

        # Extract all paragraphs if no main content was found
        paragraphs = soup.find_all("p")
        if paragraphs:
            text_parts = []
            for p in paragraphs:
                p_text = p.get_text(strip=True)
                if p_text and len(p_text) > 10:  # Skip very short paragraphs
                    text_parts.append(p_text)
                    
            if text_parts:
                return " ".join(text_parts)
            
        # Try to get content from div tags if still no content found
        divs = soup.find_all("div")
        if divs:
            # Get text from the 10 largest divs that have meaningful content
            div_texts = []
            for d in divs:
                d_text = d.get_text(strip=True)
                if d_text and len(d_text) > 50:
                    div_texts.append((d, len(d_text)))
            
            # Sort by content length and take the top 10
            if div_texts:
                text_parts = []
                for d, _ in div_texts[:10]:
                    text_parts.append(d.get_text(separator=' ', strip=True))
                return " ".join(text_parts)
        
        # If we got here, try any text from the body
        body = soup.find('body')
        if body:
            body_text = body.get_text(separator=' ', strip=True)
            if body_text and len(body_text) > 50:
                return body_text
            
        # If all else fails
        return "No meaningful content could be extracted from the webpage."

    except requests.exceptions.RequestException as e:
        return f"Failed to scrape {url}: {str(e)}"
    except Exception as e:
        return f"An error occurred while scraping {url}: {str(e)}"

# Define tools for OpenAI API integration
def get_tools_for_model():
    """Returns the tools configuration for OpenAI API."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search Google for up-to-date information on a topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "The number of results to return (default: 3)",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "scrape_webpage",
                "description": "Scrape and extract text content from a webpage URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the webpage to scrape"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "Execute code in Python or C++ and return the output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to execute"
                        },
                        "language": {
                            "type": "string",
                            "description": "The programming language to use (python or cpp)",
                            "enum": ["python", "cpp"]
                        },
                        "input": {
                            "type": "string",
                            "description": "Optional input data for the program (for cin>>, input() functions). All inputs should be on a single line, separated by spaces",
                        }
                    },
                    "required": ["code", "language"]
                }
            }
        }
    ]
    return tools

# Map function names to their implementations
tool_functions = {
    "google_search": lambda args: google_custom_search(args["query"], args.get("num_results", 3)),
    "scrape_webpage": lambda args: scrape_web_content(args["url"]),
    "code_interpreter": lambda args: asyncio.run(execute_code(args["code"], args["language"], input_data=args.get("input", "")))
}

# Process tool calls from the model
async def process_tool_calls(model_response, messages_history):
    """Process tool calls returned by the model and add results to message history."""
    if model_response.choices[0].finish_reason == "tool_calls" and hasattr(model_response.choices[0].message, 'tool_calls'):
        # Add the model's response to messages
        messages_history.append(model_response.choices[0].message)
        
        # Process each tool call
        for tool_call in model_response.choices[0].message.tool_calls:
            if tool_call.type == "function":
                function_name = tool_call.function.name
                try:
                    # Safely parse function arguments with proper error handling
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        error_message = f"Error parsing function arguments: {str(e)}. Raw arguments: {tool_call.function.arguments}"
                        logging.error(error_message)
                        
                        # Add a dummy object with reasonable defaults for code_interpreter
                        if function_name == "code_interpreter":
                            function_args = {
                                "code": tool_call.function.arguments,
                                "language": "python"  # Default to python if we can't parse
                            }
                        else:
                            # For other functions, report the error
                            messages_history.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": error_message
                            })
                            continue
                
                    # Execute the function
                    if function_name in tool_functions:
                        try:
                            # Special handling for async code_interpreter function
                            if function_name == "code_interpreter":
                                try:
                                    # Extract code information - always ensure we have this
                                    code = function_args.get("code", "# No code provided")
                                    language = function_args.get("language", "python")
                                    input_data = function_args.get("input", "")
                                    
                                    # Always show the exact code that will be executed
                                    display_message = f"Executing {language} code:\n```{language}\n{code}\n```"
                                    if input_data:
                                        display_message += f"\nWith input:\n```\n{input_data}\n```"
                                    
                                    # Execute the code
                                    execution_result = await execute_code(
                                        code, 
                                        language, 
                                        input_data=input_data
                                    )
                                    
                                    # Combine both code display and execution results in one message
                                    combined_message = f"{display_message}\n\n{execution_result}"
                                    
                                    # Add as a single tool message
                                    messages_history.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": combined_message
                                    })
                                    
                                    # Skip rest of this iteration since we've already added the message
                                    continue
                                    
                                except Exception as e:
                                    # For code_interpreter errors, ensure we still show the code
                                    error_details = str(e)
                                    logging.error(f"Code interpreter error: {error_details}")
                                    
                                    # Extract whatever code we can
                                    code = function_args.get("code", "# Code extraction failed")
                                    language = function_args.get("language", "python")
                                    
                                    # Show both the code and the error
                                    error_message = f"Code that caused error:\n```{language}\n{code}\n```\n\nError: {error_details}"
                                    messages_history.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": error_message
                                    })
                                    continue
                            else:
                                function_response = tool_functions[function_name](function_args)
                            
                            # Make sure function_response is never empty
                            if function_response is None or function_response == "":
                                function_response = f"Function {function_name} completed successfully with no output. Return code: 0"
                            
                            # Add the function response to messages
                            messages_history.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": str(function_response)  # Ensure content is a string
                            })
                        except Exception as e:
                            # Log the error and add an error message to the history
                            error_message = f"Error executing {function_name}: {str(e)}"
                            logging.error(error_message)
                            
                            # For code_interpreter, always show the code even on error
                            if function_name == "code_interpreter":
                                code = function_args.get("code", "# Code extraction failed")
                                language = function_args.get("language", "python")
                                error_message = f"Code that caused error:\n```{language}\n{code}\n```\n\nError: {error_message}"
                                
                            messages_history.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": error_message
                            })
                    else:
                        # Function not found, add error message
                        error_message = f"Function {function_name} not found"
                        logging.error(error_message)
                        messages_history.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool", 
                            "name": function_name,
                            "content": error_message
                        })
                except Exception as e:
                    # Catch-all handler to ensure robustness
                    error_message = f"Unexpected error processing tool call: {str(e)}"
                    logging.error(error_message)
                    messages_history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name if 'function_name' in locals() else "unknown_function",
                        "content": error_message
                    })
                    
        return True, messages_history
    
    return False, messages_history

# Processes a command request with rate limiting and queuing.
async def process_request(interaction, command_func, *args):
    user_id = interaction.user.id
    now = discord.utils.utcnow().timestamp()
    last_request = user_requests[user_id]['last_request']
    
    if now - last_request < 5:
        await interaction.followup.send("You are sending requests too quickly. Please wait a moment.", ephemeral=True)
        return

    # Update last request time
    user_requests[user_id]['last_request'] = now

    # Add request to queue
    queue = user_requests[user_id]['queue']
    await queue.put((command_func, args))

    # Start processing if it's the only request in the queue
    if queue.qsize() == 1:
        await process_queue(interaction)

# Processes requests in the user's queue sequentially.
async def process_queue(interaction):
    user_id = interaction.user.id
    queue = user_requests[user_id]['queue']
    
    while not queue.empty():
        command_func, args = await queue.get()
        await command_func(interaction, *args)
        await asyncio.sleep(1)  # Optional delay between processing

def check_blacklist():
    """Decorator to check if a user is blacklisted before executing a command."""
    async def predicate(interaction: discord.Interaction):
        if await is_admin(interaction.user.id):
            return True
        if await is_user_blacklisted(interaction.user.id):
            await interaction.response.send_message("You have been blacklisted from using this bot. Please contact the admin if you think this is a mistake.", ephemeral=True)
            return False
        return True
    return app_commands.check(predicate)

# Slash command to let users choose a model and save it to the database
@tree.command(name="choose_model", description="Select the AI model to use for responses.")
@check_blacklist()
async def choose_model(interaction: discord.Interaction):
    options = [discord.SelectOption(label=model, value=model) for model in MODEL_OPTIONS]
    select_menu = discord.ui.Select(placeholder="Choose a model", options=options)

    async def select_callback(interaction: discord.Interaction):
        selected_model = select_menu.values[0]
        user_id = interaction.user.id
        
        # Save the model selection to the database
        await save_user_model(user_id, selected_model)
        await interaction.response.send_message(
            f"Model set to `{selected_model}` for your responses.", ephemeral=True
        )

    select_menu.callback = select_callback
    view = discord.ui.View()
    view.add_item(select_menu)
    await interaction.response.send_message("Choose a model:", view=view, ephemeral=True)

# Slash command for search (/search)
@tree.command(name="search", description="Search on Google and send results to AI model.")
@app_commands.describe(query="The search query")
@check_blacklist()
async def search(interaction: discord.Interaction, query: str):
    """Searches Google and sends results to the AI model."""
    await interaction.response.defer(thinking=True)
    user_id = interaction.user.id
    model = await get_user_model(user_id)
    history = await get_history(user_id)

    try:
        # Perform Google search
        search_results = google_custom_search(query)
        
        if not search_results or not search_results.get('results'):
            await interaction.followup.send("No search results found for your query.")
            return

        # Format search results for the AI model
        formatted_results = f"Search results for: {query}\n\n"
        
        for i, result in enumerate(search_results.get('results', [])):
            if 'title' in result and 'link' in result and 'snippet' in result:
                formatted_results += f"Result {i+1}:\n"
                formatted_results += f"Title: {result['title']}\n"
                formatted_results += f"Link: {result['link']}\n"
                formatted_results += f"Snippet: {result['snippet']}\n"
                if 'content' in result:
                    formatted_results += f"Content Preview: {result['content'][:500]}...\n"
                formatted_results += "\n"

        # Prepare messages for the AI model, handling system prompts appropriately
        messages = []
        if model in ["o1-mini", "o1-preview"]:
            # These models don't support system prompts
            messages = [
                {"role": "user", "content": f"Instructions: {SEARCH_PROMPT}\n\nUser query: {query}\n\n{formatted_results}"}
            ]
        else:
            messages = [
                {"role": "system", "content": SEARCH_PROMPT},
                {"role": "user", "content": f"User query: {query}\n\n{formatted_results}"}
            ]

        # Send to the AI model
        response = client.chat.completions.create(
            model=model if model in ["gpt-4o", "gpt-4o-mini"] else "gpt-4o",
            messages=messages,
            temperature=0.5
        )

        reply = response.choices[0].message.content
        
        # Add the interaction to history
        history.append({"role": "user", "content": f"Search query: {query}"})
        history.append({"role": "assistant", "content": reply})
        await save_history(user_id, history)

        # Send the response
        await interaction.followup.send(reply)

    except Exception as e:
        error_message = f"Search error: {str(e)}"
        logging.error(error_message)
        await interaction.followup.send(f"An error occurred while searching: {str(e)}")

# Slash command for web scraping (/web)
@tree.command(name="web", description="Scrape a webpage and send data to AI model.")
@app_commands.describe(url="The webpage URL to scrape")
@check_blacklist()
async def web(interaction: discord.Interaction, url: str):
    """Scrapes a webpage and sends data to the AI model."""
    await interaction.response.defer(thinking=True)
    user_id = interaction.user.id
    history = await get_history(user_id)

    try:
        content = scrape_web_content(url)
        if content.startswith("Failed"):
            await interaction.followup.send(content)
            return

        history.append({"role": "user", "content": f"Scraped content from {url}"})
        history.append({"role": "system", "content": content})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=history,
            temperature=0.3,
            top_p=0.7
        )

        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        save_history(user_id, history)

        await interaction.followup.send(reply)

    except Exception as e:
        await interaction.followup.send(f"Error: {str(e)}", ephemeral=True)

# Reset user chat history from database
@tree.command(name="reset", description="Reset the bot by clearing user data.")
@check_blacklist()
async def reset(interaction: discord.Interaction):
    """Resets the bot by clearing user data."""
    user_id = interaction.user.id
    db.user_histories.delete_one({'user_id': user_id})
    await interaction.response.send_message("Your data has been cleared and reset!", ephemeral=True)

# Slash command for user statistics (/user_stat)
@tree.command(name="user_stat", description="Get your current input token, output token, and model.")
@check_blacklist()
async def user_stat(interaction: discord.Interaction):
    """Fetches and displays the current input token, output token, and model for the user."""
    user_id = interaction.user.id
    history = await get_history(user_id)
    model = await get_user_model(user_id)

    # Handle cases where user model is not found
    if not model:
        model = "gpt-4o"  # Default model

    # Adjust model for encoding purposes
    if model in ["gpt-4o", "o1", "o1-preview", "o1-mini", "o3-mini"]:
        encoding_model = "gpt-4o"
    else:
        encoding_model = model

    # Retrieve the appropriate encoding for the selected model
    encoding = tiktoken.encoding_for_model(encoding_model)

    # Initialize token counts
    input_tokens = 0
    output_tokens = 0

    # Calculate input and output tokens
    if history:
        for item in history:
            content = item.get('content')  # Safely access 'content'

            # Handle case where content is a list or other type
            if isinstance(content, list):
                # Convert list of objects to a single string (e.g., join texts with a space)
                content = " ".join(
                    sub_item.get('text', '') for sub_item in content if isinstance(sub_item, dict)
                )

            # Ensure content is a string before processing
            if isinstance(content, str):
                token_count = len(encoding.encode(content))
                if item['role'] == 'user':
                    input_tokens += token_count
                elif item['role'] in ['assistant', 'developer']:
                    # Treat 'developer' as 'assistant' for token counting
                    output_tokens += token_count

    # Create the statistics message
    stat_message = (
        f"**User Statistics:**\n"
        f"Model: `{model}`\n"
        f"Input Tokens: `{input_tokens}`\n"
        f"Output Tokens: `{output_tokens}`\n"
    )

    # Send the response
    await interaction.response.send_message(stat_message, ephemeral=True)


# Slash command for help (/help)
@tree.command(name="help", description="Display a list of available commands.")
@check_blacklist()
async def help_command(interaction: discord.Interaction):
    """Sends a list of available commands to the user."""
    help_message = (
        "**Các lệnh có sẵn:**\n"
        "/choose_model - Chọn mô hình AI để sử dụng cho phản hồi (gpt-4o, gpt-4o-mini, o1-preview, o1-mini).\n"
        "/search `<truy vấn>` - Tìm kiếm trên Google và gửi kết quả đến mô hình AI.\n"
        "/web `<url>` - Thu thập dữ liệu từ trang web và gửi đến mô hình AI.\n"
        "/generate `<gợi ý>` - Tạo hình ảnh từ gợi ý văn bản.\n"
        "/reset - Đặt lại lịch sử trò chuyện của bạn.\n"
        "/remaining_turns - Kiểm tra số lượt trò chuyện còn lại cho mỗi mô hình.\n"
        "/user_stat - Nhận thông tin về token đầu vào, token đầu ra và mô hình hiện tại của bạn.\n"
        "/help - Hiển thị tin nhắn trợ giúp này.\n"
    )
    await interaction.response.send_message(help_message, ephemeral=True)


# Function to check if the bot should respond to a message
def should_respond_to_message(message: discord.Message) -> bool:
    """Checks if the bot should respond to the message."""
    is_bot_reply = (message.reference and 
                    message.reference.resolved and 
                    message.reference.resolved.id == 1270288366289813556)
    is_mention = bot.user.mentioned_in(message)
    is_dm = message.guild is None
    return is_bot_reply or is_mention or is_dm

# Function to send a response to the user
async def send_response(interaction: discord.Interaction, reply: str):
    """Sends the reply to the user, handling long responses."""
    if len(reply) > 2000:
        with open("response.txt", "w") as file:
            file.write(reply)
        await interaction.followup.send("The response was too long, so it has been saved to a file.", file=discord.File("response.txt"))
    else:
        await interaction.followup.send(reply)


# Event to handle incoming messages
@bot.event
async def on_message(message: discord.Message):
    """Handles incoming messages, responding to replies, mentions, and DMs."""
    if message.author == bot.user:
        return

    if should_respond_to_message(message):
        await handle_user_message(message)
    else:
        await bot.process_commands(message)

user_tasks = {}

async def handle_user_message(message: discord.Message):
    user_id = message.author.id
    if user_id not in user_tasks:
        user_tasks[user_id] = []
    task = asyncio.create_task(process_user_message(message))
    user_tasks[user_id].append(task)
    task.add_done_callback(lambda t: user_tasks[user_id].remove(t))

async def stop_user_tasks(user_id: int):
    if user_id in user_tasks:
        for task in user_tasks[user_id]:
            task.cancel()
        user_tasks[user_id] = []

async def process_user_message(message: discord.Message):
    try:
        user_id = message.author.id
        
        # Check if user is blacklisted (skip for admins)
        if not await is_admin(user_id) and await is_user_blacklisted(user_id):
            await message.channel.send("You have been blacklisted from using this bot. Please contact the admin if you think this is a mistake.")
            return
        
        # Get history and model preferences first    
        history = await get_history(user_id)
        model = await get_user_model(user_id)

        # Handle PDF files
        if message.attachments:
            for attachment in message.attachments:
                if attachment.filename.lower().endswith('.pdf'):
                    # Check if user is whitelisted (skip for admins)
                    if not await is_admin(user_id) and not await is_user_whitelisted(user_id):
                        await message.channel.send(f"You are not authorized to use PDF processing. Please contact admin (ID: {str(ADMIN_ID)}) to get whitelisted using the /whitelist_add command.")
                        return
                        
                    # Admins can use any model for PDF processing
                    if not await is_admin(user_id) and model not in PDF_ALLOWED_MODELS:
                        await message.channel.send(f"Error: PDF processing is only available with models: {', '.join(PDF_ALLOWED_MODELS)}. Please use /choose_model to switch to one of these models. Your current model: {model}")
                        return
                        
                    # Get user's prompt or use default if none provided
                    user_prompt = message.content.strip() if message.content else "Please analyze this PDF document"
                    
                    pdf_content = await attachment.read()
                    await process_pdf(message, pdf_content, user_prompt, model)
                    return

        # Handle normal messages and non-PDF attachments
        content = []
        
        # Add message content if present
        if message.content:
            content.append({"type": "text", "text": message.content})

        # Process attachments
        if message.attachments:
            for attachment in message.attachments:
                if any(attachment.filename.endswith(ext) for ext in supported_file_types):
                    file_content = await attachment.read()
                    try:
                        text_content = file_content.decode("utf-8")
                        content.append({"type": "text", "text": text_content})
                    except UnicodeDecodeError:
                        await message.channel.send("Error: The file appears to be binary data, not a text file.")
                        return
                else:
                    content.append({"type": "image_url", "image_url": {"url": attachment.url}})

        if not content:
            content.append({"type": "text", "text": "No content."})

        # Prepare current message
        current_message = {"role": "user", "content": content}
        
        try:
            # Process messages based on the model's capabilities
            messages_for_api = []
            
            # For models that don't support system prompts
            if model in ["o1-mini", "o1-preview"]:
                # Convert system messages to user instructions
                system_content = None
                history_without_system = []
                
                # Extract system message content
                for msg in history:
                    if msg["role"] == "system":
                        system_content = msg.get("content", "")
                    else:
                        history_without_system.append(msg)
                
                # Add the system content as a special user message at the beginning
                if system_content:
                    history_without_system.insert(0, {
                        "role": "user", 
                        "content": f"Instructions for you to follow in this conversation: {system_content}"
                    })
                
                # Add current message and prepare for API
                history_without_system.append(current_message)
                messages_for_api = prepare_messages_for_api(history_without_system)
            else:
                # For models that support system prompts
                history.append(current_message)
                messages_for_api = prepare_messages_for_api(history)
            
            # Determine which models should have tools available
            # o1-mini and o1-preview do not support tools
            use_tools = model in ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"]
            
            # Prepare API call parameters
            api_params = {
                "model": model,
                "messages": messages_for_api,
                "temperature": 0.3 if model in ["gpt-4o", "gpt-4o-mini"] else 1,
                "top_p": 0.7 if model in ["gpt-4o", "gpt-4o-mini"] else 1
            }
            
            # Add tools if using a supported model
            if use_tools:
                api_params["tools"] = get_tools_for_model()
            
            # Add a typing indicator to show that the bot is processing
            async with message.channel.typing():
                # Make the initial API call
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    **api_params
                )
                
                # Check if there are any tool calls to process
                if use_tools:
                    # Send a message indicating that the bot is searching for information if tool calls are present
                    if response.choices[0].finish_reason == "tool_calls":
                        await message.channel.send("🔍 Processing your request...")
                        
                    # Process any tool calls and get the updated messages
                    tool_calls_processed, updated_messages = await process_tool_calls(response, messages_for_api)
                    
                    # If tool calls were processed, make another API call with the updated messages
                    if tool_calls_processed:
                        api_params["messages"] = updated_messages
                        
                        # Make the final API call with tool results
                        response = await asyncio.to_thread(
                            client.chat.completions.create,
                            **api_params
                        )
            
            reply = response.choices[0].message.content
            
            # Store the response in history for models that support it
            if model in ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"]:
                if model in ["o1-mini", "o1-preview"]:
                    # For models without system prompt support, we keep the modified history
                    if system_content:
                        # Don't add the system instruction again to history to avoid duplication
                        modified_history = [msg for msg in history_without_system if not (msg["role"] == "user" and msg["content"].startswith("Instructions for you to follow"))]
                        modified_history.append({"role": "assistant", "content": reply})
                        await save_history(user_id, modified_history)
                else:
                    # For models with system prompt support, just append to regular history
                    history.append({"role": "assistant", "content": reply})
                    await save_history(user_id, history)
                    
            await send_response(message.channel, reply)
        except RateLimitError:
            await message.channel.send(
                "Error: Rate limit exceeded for your model. "
                "Please try again later or use /choose_model to change to any models else."
            )
        except Exception as e:
            error_message = f"Error: {str(e)}"
            logging.error(f"Error in message processing: {error_message}")
            await message.channel.send(error_message)
            
    except Exception as e:
        error_message = f"Error: {str(e)}"
        logging.error(f"Error in message handling: {error_message}")
        await message.channel.send(error_message)
    finally:
        if user_id in user_tasks:
            user_tasks[user_id] = [task for task in user_tasks[user_id] if not task.done()]

async def process_batch(model: str, user_prompt: str, batch_content: str, current_batch: int, total_batches: int, channel, max_retries=3) -> bool:
    """Process a single batch of PDF content with auto-adjustment for token limits."""
    for attempt in range(max_retries):
        try:
            # Create fresh history for each batch to avoid accumulation
            messages = [
                {"role": "user", "content": f"{user_prompt}\n\nAnalyze the following pages:\n{batch_content}"}
            ]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                max_tokens=8096
            )
            
            reply = response.choices[0].message.content
            batch_response = f"Batch {current_batch}/{total_batches}:\n{reply}"
            await send_response(channel, batch_response)
            return True
            
        except Exception as e:
            error_str = str(e)
            if "413" in error_str and attempt < max_retries - 1:
                # Split the batch content in half and try again
                content_parts = batch_content.split("\n")
                mid = len(content_parts) // 2
                batch_content = "\n".join(content_parts[:mid])
                continue
            elif attempt == max_retries - 1:
                await channel.send(f"Error processing batch {current_batch}: {str(e)}")
                return False
    return False

def count_tokens(text: str) -> int:
    """Estimate token count using a simple approximation."""
    # Rough estimate: 1 word ≈ 1.3 tokens
    return int(len(text.split()) * 1.3)

def trim_content_to_token_limit(content: str, max_tokens: int = 8096) -> str:
    """Trim content to stay within token limit while preserving the most recent content."""
    current_tokens = count_tokens(content)
    if current_tokens <= max_tokens:
        return content
        
    # Split into lines and start removing from the beginning until under limit
    lines = content.split('\n')
    while lines and count_tokens('\n'.join(lines)) > max_tokens:
        lines.pop(0)
    
    if not lines:  # If still too long, take the last part
        text = content
        while count_tokens(text) > max_tokens:
            text = text[text.find('\n', 1000):]
        return text
        
    return '\n'.join(lines)

def prepare_messages_for_api(messages, max_tokens=8096):
    """Prepare messages for API while ensuring token limit and no null content."""
    if not messages:
        return [{"role": "system", "content": NORMAL_CHAT_PROMPT}]
        
    total_tokens = 0
    prepared_messages = []
    
    # Process messages in reverse order to keep the most recent ones
    for msg in reversed(messages):
        # Ensure message has valid role and content
        if not msg or not isinstance(msg, dict):
            continue
            
        role = msg.get('role')
        content = msg.get('content')
        
        if not role or content is None:
            continue
            
        # Convert complex content to text for token counting
        if isinstance(content, list):
            text_content = ""
            for item in content:
                if not item or not isinstance(item, dict):
                    continue
                    
                item_type = item.get('type')
                if item_type == 'text' and item.get('text'):
                    text_content += item.get('text', "") + "\n"
            
            # Skip if there's no actual text content
            if not text_content:
                continue
                
            msg_tokens = count_tokens(text_content)
            if total_tokens + msg_tokens > max_tokens:
                # Trim the content
                trimmed_text = trim_content_to_token_limit(text_content, max_tokens - total_tokens)
                if trimmed_text:
                    new_content = [{"type": "text", "text": trimmed_text}]
                    # Preserve any image URLs from the original content
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'image_url' and item.get('image_url'):
                            new_content.append(item)
                    prepared_messages.insert(0, {"role": role, "content": new_content})
                break
            else:
                prepared_messages.insert(0, msg)
                total_tokens += msg_tokens
        else:
            # Handle string content
            msg_content_str = str(content) if content is not None else ""
            if not msg_content_str:  # Skip empty content
                continue
                
            msg_tokens = count_tokens(msg_content_str)
            if total_tokens + msg_tokens > max_tokens:
                # Trim the content
                trimmed_text = trim_content_to_token_limit(msg_content_str, max_tokens - total_tokens)
                if trimmed_text:
                    prepared_messages.insert(0, {"role": role, "content": trimmed_text})
                break
            else:
                prepared_messages.insert(0, {"role": role, "content": msg_content_str})
                total_tokens += msg_tokens
    
    # Ensure we have at least one message with valid content
    if not prepared_messages:
        return [{"role": "system", "content": NORMAL_CHAT_PROMPT}]
                
    return prepared_messages

async def process_pdf_batch(model: str, user_prompt: str, batch_content: str, current_batch: int, total_batches: int, channel, max_retries=3) -> bool:
    """Process a single batch of PDF content with auto-adjustment for token limits."""
    batch_size = len(batch_content.split('\n'))
    original_content = batch_content
    
    for attempt in range(max_retries):
        try:
            # Create message without history but with appropriate prompt handling
            trimmed_content = trim_content_to_token_limit(batch_content, 7000)  # Leave room for prompt
            
            messages = []
            if model in ["o1-mini", "o1-preview"]:
                # These models don't support system prompts
                messages = [
                    {"role": "user", "content": f"Instructions: {PDF_ANALYSIS_PROMPT}\n\n{user_prompt}\n\nAnalyze the following content:\n{trimmed_content}"}
                ]
            else:
                messages = [
                    {"role": "system", "content": PDF_ANALYSIS_PROMPT},
                    {"role": "user", "content": f"{user_prompt}\n\nAnalyze the following content:\n{trimmed_content}"}
                ]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1
            )
            
            reply = response.choices[0].message.content
            batch_response = f"Batch {current_batch}/{total_batches} (Pages in batch: {batch_size}):\n{reply}"
            await send_response(channel, batch_response)
            return True
            
        except Exception as e:
            error_str = str(e)
            if "413" in error_str and attempt < max_retries - 1:
                # Split the batch content in half and try again
                content_parts = batch_content.split('\n')
                mid = len(content_parts) // 2
                batch_content = '\n'.join(content_parts[:mid])
                batch_size = len(batch_content.split('\n'))
                await channel.send(f"Batch {current_batch} was too large, reducing size and retrying...")
                continue
            elif attempt == max_retries - 1:
                await channel.send(f"Error processing batch {current_batch}: {str(e)}")
                return False
            
    return False

async def process_pdf(message: discord.Message, pdf_content: bytes, user_prompt: str, model: str) -> None:
    """Process a PDF file with improved error handling and token management."""
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_file)
        pages_content = []
        
        # Extract text from PDF
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                pages_content.append({
                    "page": page_num,
                    "content": text.strip()
                })
                
        if not pages_content:
            await message.channel.send("Error: Could not extract any text from the PDF.")
            return

        # Initial batch size
        total_pages = len(pages_content)
        current_batch_size = PDF_BATCH_SIZE
        processed_pages = 0

        # Handle single-page PDF
        if total_pages == 1:
            batch_content = f"\nPDF Page 1:\n{pages_content[0]['content']}\n"
            await process_pdf_batch(
                model=model,
                user_prompt=user_prompt,
                batch_content=batch_content,
                current_batch=1,
                total_batches=1,
                channel=message.channel
            )
            return
        
        while current_batch_size > 0 and processed_pages < total_pages:
            try:
                remaining_pages = total_pages - processed_pages
                total_batches = (remaining_pages + current_batch_size - 1) // current_batch_size
                await message.channel.send(f"Processing PDF with {remaining_pages} remaining pages in {total_batches} batches...")
                
                batch_start = processed_pages
                success = True
                
                for i in range(batch_start, total_pages, current_batch_size):
                    batch = pages_content[i:i+current_batch_size]
                    batch_content = ""
                    for page_data in batch:
                        page_num = page_data["page"]
                        content = page_data["content"]
                        batch_content += f"\nPDF Page {page_num}:\n{content}\n"
                    
                    current_batch = (i - batch_start) // current_batch_size + 1
                    success = await process_pdf_batch(
                        model=model,
                        user_prompt=user_prompt,
                        batch_content=batch_content,
                        current_batch=current_batch,
                        total_batches=total_batches,
                        channel=message.channel
                    )
                    
                    if not success:
                        # If batch processing failed, reduce batch size and retry from current position
                        current_batch_size = current_batch_size // 2
                        if current_batch_size > 0:
                            await message.channel.send(f"Reducing batch size to {current_batch_size} pages and retrying from current position...")
                            break
                        else:
                            await message.channel.send("Error: Could not process PDF even with minimum batch size.")
                            return
                    else:
                        processed_pages += len(batch)
                        await asyncio.sleep(2)  # Delay between successful batches
                
                if success and processed_pages >= total_pages:
                    await message.channel.send("PDF processing completed successfully!")
                    return
                    
            except Exception as e:
                current_batch_size = current_batch_size // 2
                if current_batch_size > 0:
                    await message.channel.send(f"Error occurred. Reducing batch size to {current_batch_size} pages and retrying...")
                else:
                    await message.channel.send(f"Error processing PDF: {str(e)}")
                    return
                    
    except Exception as e:
        await message.channel.send(f"Error processing PDF: {str(e)}")
        return

# Function to get the remaining turns for each model
def trim_history(history):
    """Trims the history to avoid exceeding token limits by removing older messages first."""
    tokens_used = sum(len(str(item['content'])) for item in history)
    max_tokens_allowed = 9000
    while tokens_used > max_tokens_allowed and len(history) > 1:
        removed_item = history.pop(0)
        tokens_used -= len(str(removed_item['content']))

# Function to send response to the discord channel
async def send_response(channel: discord.TextChannel, reply: str):
    """Sends the reply to the channel, handling long responses."""
    # Safety check - ensure reply is not empty
    if not reply or not reply.strip():
        reply = "I'm sorry, I couldn't generate a proper response. Please try again."
    
    if len(reply) > 2000:
        with open("response.txt", "w", encoding="utf-8") as file:
            file.write(reply)
        await channel.send(
            "The response was too long, so it has been saved to a file.",
            file=discord.File("response.txt")
        )
    else:
        await channel.send(reply)

# Slash command for image generation (/generate)
@tree.command(name='generate', description='Generates an image from a text prompt.')
@app_commands.describe(prompt='The prompt for image generation')
@check_blacklist()
async def generate_image(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True)  # Indicate that the bot is processing
    await _generate_image_command(interaction, prompt)
async def _generate_image_command(interaction: discord.Interaction, prompt: str):
    try:
       # Create an image generation request
        request_image = IImageInference(
            positivePrompt=prompt,
            model="runware:100@1",
            numberResults=4,
            height=512,
            width=512
        )
        
        # Call the API to get the results
        images = await runware.imageInference(requestImage=request_image)

         # Check the API's return value
        if images is None:
            raise ValueError("API returned None for images")

        # Download images from URL and send as attachments
        image_files = []
        async with aiohttp.ClientSession() as session:
            for image in images:
                async with session.get(image.imageURL) as resp:
                    if resp.status == 200:
                        image_files.append(await resp.read())
                    else:
                        logging.error(f"Failed to download image: {image.imageURL} with status {resp.status}")

        # Send images as attachments
        if image_files:
            await interaction.followup.send(files=[discord.File(io.BytesIO(img), filename=f"image_{i}.png") for i, img in enumerate(image_files)])
        else:
            await interaction.followup.send("No images were generated.")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(f"Error in _generate_image_command: {error_message}")
        await interaction.followup.send(error_message)

# Slash command to add user to PDF whitelist
@tree.command(name="whitelist_add", description="Add a user to the PDF processing whitelist")
@app_commands.describe(user_id="The Discord user ID to whitelist")
async def whitelist_add(interaction: discord.Interaction, user_id: str):
    """Adds a user to the PDF processing whitelist."""
    if str(interaction.user.id) != ADMIN_ID:
        await interaction.response.send_message("You don't have permission to use this command. Only admin can use whitelist commands.", ephemeral=True)
        return
    
    try:
        user_id = int(user_id)
        if await is_admin(user_id):
            await interaction.response.send_message("Admins are automatically whitelisted and don't need to be added.", ephemeral=True)
            return
        await add_user_to_whitelist(user_id)
        await interaction.response.send_message(f"User {user_id} has been added to the PDF processing whitelist.", ephemeral=True)
    except ValueError:
        await interaction.response.send_message("Invalid user ID. Please provide a valid Discord user ID.", ephemeral=True)

@tree.command(name="whitelist_remove", description="Remove a user from the PDF processing whitelist")
@app_commands.describe(user_id="The Discord user ID to remove from whitelist")
async def whitelist_remove(interaction: discord.Interaction, user_id: str):
    """Removes a user from the PDF processing whitelist."""
    if str(interaction.user.id) != ADMIN_ID:
        await interaction.response.send_message("You don't have permission to use this command. Only admin can use whitelist commands.", ephemeral=True)
        return
    
    try:
        user_id = int(user_id)
        if await remove_user_from_whitelist(user_id):
            await interaction.response.send_message(f"User {user_id} has been removed from the PDF processing whitelist.", ephemeral=True)
        else:
            await interaction.response.send_message(f"User {user_id} was not found in the whitelist.", ephemeral=True)
    except ValueError:
        await interaction.response.send_message("Invalid user ID. Please provide a valid Discord user ID.", ephemeral=True)

@tree.command(name="blacklist_add", description="Add a user to the bot blacklist")
@app_commands.describe(user_id="The Discord user ID to blacklist")
async def blacklist_add(interaction: discord.Interaction, user_id: str):
    """Adds a user to the bot blacklist."""
    if str(interaction.user.id) != ADMIN_ID:
        await interaction.response.send_message("You don't have permission to use this command. Only admin can use blacklist commands.", ephemeral=True)
        return
    
    try:
        user_id = int(user_id)
        if await is_admin(user_id):
            await interaction.response.send_message("Cannot blacklist an admin.", ephemeral=True)
            return
        await add_user_to_blacklist(user_id)
        await interaction.response.send_message(f"User {user_id} has been added to the bot blacklist. They can no longer use any bot features.", ephemeral=True)
    except ValueError:
        await interaction.response.send_message("Invalid user ID. Please provide a valid Discord user ID.", ephemeral=True)

@tree.command(name="blacklist_remove", description="Remove a user from the bot blacklist")
@app_commands.describe(user_id="The Discord user ID to remove from blacklist")
async def blacklist_remove(interaction: discord.Interaction, user_id: str):
    """Removes a user from the bot blacklist."""
    if str(interaction.user.id) != ADMIN_ID:
        await interaction.response.send_message("You don't have permission to use this command. Only admin can use blacklist commands.", ephemeral=True)
        return
    
    try:
        user_id = int(user_id)
        if await remove_user_from_blacklist(user_id):
            await interaction.response.send_message(f"User {user_id} has been removed from the bot blacklist. They can now use bot features again.", ephemeral=True)
        else:
            await interaction.response.send_message(f"User {user_id} was not found in the blacklist.", ephemeral=True)
    except ValueError:
        await interaction.response.send_message("Invalid user ID. Please provide a valid Discord user ID.", ephemeral=True)

@tree.command(name="stop", description="Stop any process or queue of the user. Admins can stop other users' tasks by providing their ID.")
@app_commands.describe(user_id="The Discord user ID to stop tasks for (admin only)")
@check_blacklist()
async def stop(interaction: discord.Interaction, user_id: str = None):
    """Stops any process or queue of the user. Admins can stop other users' tasks by providing their ID."""
    # Defer the interaction first
    await interaction.response.defer(ephemeral=True)
    
    if user_id and not await is_admin(interaction.user.id):
        await interaction.followup.send("You don't have permission to stop other users' tasks.", ephemeral=True)
        return
    
    target_user_id = int(user_id) if user_id else interaction.user.id
    
    await stop_user_tasks(target_user_id)
    await interaction.followup.send(f"Stopped all tasks for user {target_user_id}.", ephemeral=True)

# Task to change status every minute
@tasks.loop(minutes=5)
async def change_status():
    while True:
        for status in statuses:
            await bot.change_presence(activity=discord.Game(name=status))
            await asyncio.sleep(300)  # Change every 60 seconds

# Event to run when the bot is ready
@bot.event
async def on_ready():
    """Bot startup event to sync slash commands and start status loop."""
    await tree.sync()  # Sync slash commands
    print(f"Logged in as {bot.user}")
    change_status.start() # Start the status changing loop   

# Code Interpreter functions
def sanitize_code(code, language):
    """
    Sanitize and validate code for security purposes.
    
    Args:
        code (str): The code to be sanitized.
        language (str): The programming language ('python' or 'cpp').
        
    Returns:
        tuple: (is_safe, sanitized_code or error_message)
    """
    # List of banned imports/includes and dangerous operations
    python_banned = [
        'os.system', 'subprocess', 'open(', '.open(', 'eval(', 'exec(', '__import__(',
        'importlib', '.read(', '.write(', 'shutil', '.unlink(', '.remove(', '.rmdir(',
        'socket', 'requests', 'urllib', 'curl', 'wget', '.chmod', '.chown',
        'os.path', 'pathlib', '__file__', '__builtins__._', 'file(', 'with open',
        'io.open', 'fileinput', 'tempfile', '.mktemp', '.mkstemp', '.NamedTemporaryFile',
        'shelve', 'dbm', 'sqlite3', 'pickle', 'marshal', '.loads(', '.dumps(',
        'getattr(', 'setattr(', 'delattr(', '__class__', '__bases__', '__subclasses__',
        '__globals__', '__getattribute__', '.mro(', 'ctypes', 'platform'
    ]
    
    cpp_banned = [
        'system(', 'exec', 'popen', 'fork', 'remove(', 'unlink(',
        '<fstream>', '<ofstream>', '<ifstream>', 'FILE *', 'fopen', 'fwrite',
        'fread', '<stdio.h>', '<stdlib.h>', '<unistd.h>', 'getcwd', 'opendir',
        'readdir', '<dirent.h>', '<sys/stat.h>', '<fcntl.h>',
        'freopen', 'ioctl', '<sys/socket.h>'
    ]
    
    # Allowed C++ headers
    cpp_allowed_headers = [
        '<iostream>', '<vector>', '<string>', '<algorithm>', '<cmath>', '<map>', '<unordered_map>', 
        '<set>', '<unordered_set>', '<queue>', '<stack>', '<deque>', '<list>', '<array>', 
        '<numeric>', '<utility>', '<tuple>', '<functional>', '<chrono>', '<thread>', '<future>', 
        '<mutex>', '<atomic>', '<memory>', '<limits>', '<exception>', '<stdexcept>', '<type_traits>', 
        '<random>', '<regex>', '<bitset>', '<complex>', '<initializer_list>', '<iomanip>',
        '<bits/stdc++.h>'  # Added support for bits/stdc++.h
    ]
    
    # Check if code is empty
    if not code.strip():
        return True, "Code is empty."
    
    # Determine which banned list to use
    banned_list = python_banned if language == 'python' else cpp_banned
    
    # Check for banned operations
    for banned_op in banned_list:
        if banned_op in code:
            if language == 'python':
                return False, f"Forbidden module import: {banned_op}"
            else:
                return False, f"Forbidden header include: {banned_op}"
    
    # Specific checks for Python
    if language == 'python':
        # Check for import statements with potentially dangerous modules
        import_pattern = r'^\s*(?:from\s+(\S+)\s+)?import\s+(.+)$'
        for line in code.split('\n'):
            match = re.match(import_pattern, line)
            if match:
                module = match.group(1) or match.group(2).split()[0].split('.')[0]
                if module in ['os', 'subprocess', 'sys', 'shutil', 'socket', 'requests', 'io', 
                             'pathlib', 'glob', 'fnmatch', 'fileinput', 'linecache', 
                             'pickle', 'dbm', 'sqlite3', 'ctypes', 'platform']:
                    return False, f"Forbidden module import: {module}"
        
        # Add safety header for Python
        safety_header = """
import signal
import time

def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out (exceeded 10 seconds)")

# Set a timeout of 10 seconds
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)

# Restrict __builtins__ to safe functions only
safe_builtins = {}
for k in ['abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes', 
          'chr', 'complex', 'dict', 'dir', 'divmod', 'enumerate', 'filter', 
          'float', 'format', 'frozenset', 'hash', 'hex', 'int', 'iter', 'len',
          'list', 'map', 'max', 'min', 'next', 'oct', 'ord', 'pow', 'print', 
          'range', 'repr', 'reversed', 'round', 'set', 'slice', 'sorted', 'str', 
          'sum', 'tuple', 'type', 'zip']:
    if k in __builtins__:
        safe_builtins[k] = __builtins__[k]

__builtins__ = safe_builtins

try:
"""
        # Add indentation for user code
        indented_code = "\n".join("    " + line for line in code.split("\n"))
        
        # Add exception handling and ending the try block
        safety_footer = """
except TimeoutError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    # Cancel the alarm
    signal.alarm(0)
"""
        
        code = safety_header + indented_code + safety_footer
    
    # Specific checks for C++
    if language == 'cpp':
        # Check for includes - ensure they're valid
        include_pattern = r'#\s*include\s*<(.+?)>'
        includes = re.findall(include_pattern, code)
        
        # Check if includes are in the allowed list
        for inc in includes:
            include_with_brackets = f"<{inc}>"
            if include_with_brackets in cpp_banned:
                return False, f"Forbidden header include: {inc}"
            
            # This is optional: restrict to only allowed headers
            # Uncomment if you want strict header checking
            if not any(include_with_brackets == allowed for allowed in cpp_allowed_headers):
                # Allow any inclusion that isn't explicitly banned
                pass 
                # If you want strict checking, uncomment:
                # return False, f"Header not in allowed list: {inc}"
        
        # Ensure C++ has basic structure 
        has_main = 'main(' in code or 'int main' in code or 'void main' in code
        has_iostream = '#include <iostream>' in code or '#include<iostream>' in code or '#include <bits/stdc++.h>' in code or '#include<bits/stdc++.h>' in code
        
        # Fix missing headers and namespace if needed
        if not has_iostream and ('cout' in code or 'cin' in code or 'cerr' in code):
            code = "#include <iostream>\n" + code
        
        if ('cout' in code or 'cin' in code or 'cerr' in code) and 'using namespace std' not in code:
            # Find position after includes
            lines = code.split('\n')
            last_include_index = -1
            for i, line in enumerate(lines):
                if '#include' in line:
                    last_include_index = i
            
            if last_include_index >= 0:
                lines.insert(last_include_index + 1, "using namespace std;")
            else:
                lines.insert(0, "using namespace std;")
            
            code = '\n'.join(lines)
        
        # Add main if none exists
        if not has_main:
            # Check if code has valid statements (not just function definitions)
            # For basic code without main, we wrap it in a main function
            code = """#include <bits/stdc++.h>
using namespace std;

int main() {
    // User code starts
""" + code + """
    // User code ends
    return 0;
}"""
        else:
            # Code has main, make sure it's wrapped with timeout
            code = """#include <chrono>
#include <thread>
#include <future>
#include <stdexcept>
""" + code.replace("int main(", "int userMain(").replace("void main(", "void userMain(") + """

int main() {
    // Set up a timeout for 10 seconds
    auto future = std::async(std::launch::async, []() {
        try {
            userMain();
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    });
    
    // Wait for the future to complete or timeout
    if (future.wait_for(std::chrono::seconds(10)) == std::future_status::timeout) {
        std::cerr << "Error: Code execution timed out (exceeded 10 seconds)" << std::endl;
    }
    
    return 0;
}"""
    
    # Perform syntax check for languages
    if language == 'python':
        try:
            compile(code, '<string>', 'exec')
            return True, code
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
    
    return True, code

async def execute_code(code, language, timeout=10, input_data=""):
    """
    Execute code in a sandboxed environment with strict timeout.
    
    Args:
        code (str): The code to execute.
        language (str): 'python' or 'cpp'.
        timeout (int): Maximum execution time in seconds.
        input_data (str): Optional input data for the program (for input() or cin>>).
        
    Returns:
        str: The output of the code execution.
    """
    # Import necessary modules
    import os
    import signal
    import asyncio
    import subprocess
    import tempfile
    import sys
    import logging
    
    # Validate that we have actual code to execute
    if not code or not code.strip():
        return "Error: No code provided to execute. Return code: 1"
    
    # Basic validation of language
    if language not in ["python", "cpp"]:
        return f"Error: Unsupported language '{language}'. Please use 'python' or 'cpp'. Return code: 1"
    
    # Validate and prepare input data
    if input_data and not isinstance(input_data, str):
        try:
            input_data = str(input_data)
        except Exception as e:
            return f"Error: Invalid input data - {str(e)}. Return code: 1"
    
    # Ensure input data ends with newline
    if input_data and not input_data.endswith('\n'):
        input_data += '\n'
    
    try:
        # Create temp directory for running code
        with tempfile.TemporaryDirectory() as temp_dir:
            if language == 'python':
                # Execute Python code
                file_path = os.path.join(temp_dir, 'code.py')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                try:
                    # Set process environment to restrict access to the system
                    env = {
                        'PYTHONPATH': '',  # Prevent access to installed Python modules
                        'PATH': '',  # Restrict access to system commands
                        'TEMP': temp_dir,  # Set temp directory to our controlled directory
                        'TMP': temp_dir,
                    }
                    
                    # Run the code in a subprocess with timeout
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, file_path,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE if input_data else None,
                        cwd=temp_dir,
                        env=env,
                        # Use preexec_fn only on Unix systems
                        preexec_fn=os.setpgrp if os.name != 'nt' else None
                    )
                    
                    try:
                        # Additional safety - use a shorter timeout than specified in the code
                        # to ensure our code terminates first
                        if input_data:
                            try:
                                # Send input data to the process
                                stdout, stderr = await asyncio.wait_for(
                                    proc.communicate(input_data.encode('utf-8')), 
                                    timeout=timeout
                                )
                            except Exception as e:
                                return f"Error processing input data: {str(e)}. Return code: 1"
                        else:
                            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                        
                        # Check for errors
                        if stderr:
                            stderr_content = stderr.decode('utf-8', errors='replace').strip()
                            if stderr_content:
                                return f"Error:\n```\n{stderr_content}```"
                        
                        # Return output or default message if output is empty
                        output = stdout.decode('utf-8', errors='replace').strip()
                        if output:
                            return f"Output:\n```\n{output}```"
                        else:
                            return "Output:\n```\nCode executed successfully with no output. Return code: 0\n```"
                        
                    except asyncio.TimeoutError:
                        try:
                            # Kill process differently depending on the OS
                            if os.name != 'nt':  # Unix-like systems
                                try:
                                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                                except:
                                    proc.kill()
                            else:  # Windows
                                proc.kill()
                        except:
                            pass
                        return "Code execution timed out after 10 seconds. Please optimize your code or reduce complexity."
                        
                except Exception as e:
                    return f"An error occurred during Python execution: {str(e)}"
                    
            elif language == 'cpp':
                # Execute C++ code
                src_path = os.path.join(temp_dir, 'code.cpp')
                exe_path = os.path.join(temp_dir, 'code')
                if os.name == 'nt':  # Windows
                    exe_path += '.exe'
                
                with open(src_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                try:
                    # Check if g++ is available
                    try:
                        check_proc = await asyncio.create_subprocess_exec(
                            'g++', '--version',
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        await check_proc.communicate()
                        if check_proc.returncode != 0:
                            return "Error: C++ compiler (g++) not available. Return code: 1"
                    except Exception:
                        return "Error: C++ compiler (g++) not available. Return code: 1"
                        
                    # Compile C++ code with restricted options
                    compile_proc = await asyncio.create_subprocess_exec(
                        'g++', src_path, '-o', exe_path, '-std=c++17',
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    compile_stdout, compile_stderr = await compile_proc.communicate()
                    
                    if compile_proc.returncode != 0:
                        compile_error = compile_stderr.decode('utf-8', errors='replace').strip()
                        if compile_error:
                            return f"Compilation error:\n```\n{compile_error}```"
                        else:
                            return "Compilation error: Unknown compilation failure. Return code: 1"
                    
                    # Execute the compiled program
                    try:
                        # Execute in restricted environment
                        run_proc = await asyncio.create_subprocess_exec(
                            exe_path,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE if input_data else None,
                            cwd=temp_dir,
                            # Use preexec_fn only on Unix systems
                            preexec_fn=os.setpgrp if os.name != 'nt' else None
                        )
                        
                        try:
                            # Enforce strict timeout
                            if input_data:
                                try:
                                    # Send input data to the process
                                    stdout, stderr = await asyncio.wait_for(
                                        run_proc.communicate(input_data.encode('utf-8')), 
                                        timeout=timeout
                                    )
                                except Exception as e:
                                    return f"Error processing input data for C++ program: {str(e)}. Return code: 1"
                            else:
                                stdout, stderr = await asyncio.wait_for(run_proc.communicate(), timeout=timeout)
                            
                            if stderr:
                                stderr_content = stderr.decode('utf-8', errors='replace').strip()
                                if stderr_content:
                                    return f"Runtime error:\n```\n{stderr_content}```"
                            
                            # Return output or default message if output is empty
                            output = stdout.decode('utf-8', errors='replace').strip()
                            if output:
                                return f"Output:\n```\n{output}```"
                            else:
                                return "Output:\n```\nCode executed successfully with no output. Return code: 0\n```"
                            
                        except asyncio.TimeoutError:
                            try:
                                # Kill process differently depending on the OS
                                if os.name != 'nt':  # Unix-like systems
                                    try:
                                        os.killpg(os.getpgid(run_proc.pid), signal.SIGKILL)
                                    except:
                                        run_proc.kill()
                                else:  # Windows
                                    run_proc.kill()
                            except:
                                pass
                            return "Code execution timed out after 10 seconds. Please optimize your code or reduce complexity."
                            
                    except Exception as e:
                        return f"An error occurred during C++ execution: {str(e)}"
                        
                except Exception as e:
                    return f"An error occurred: {str(e)}"
            
            # Default case for unsupported languages
            return "Unsupported language. Please use 'python' or 'cpp'."
    except Exception as e:
        # Catch-all exception handler to ensure we always return something
        error_msg = f"An unexpected error occurred: {str(e)}. Return code: 1"
        logging.error(f"Error in execute_code: {error_msg}")
        return error_msg

def extract_code_blocks(content):
    """
    Extract code blocks from the message content.
    
    Args:
        content (str): The message content.
        
    Returns:
        list: List of tuples containing (language, code).
    """
    # Regular expression to match code blocks
    # Match ```language\ncode``` pattern
    pattern = r'```(\w+)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        # If no matches found, try simpler pattern without language specifier
        pattern = r'```(.*?)```'
        simpler_matches = re.findall(pattern, content, re.DOTALL)
        if simpler_matches:
            # Try to detect language from content
            for code in simpler_matches:
                if '#include' in code and ('int main' in code or 'void main' in code):
                    matches.append(('cpp', code))
                else:
                    matches.append(('python', code))
    
    return matches

# Start Flask in a separate thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True  # Ensure it closes when the main program exits
flask_thread.start()

# Main bot startup
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    bot.run(TOKEN)
