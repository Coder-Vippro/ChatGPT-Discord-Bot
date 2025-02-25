import os
import discord
import io
import threading
import tiktoken
import asyncio
import requests
import logging
import sys
import aiohttp
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
ADMIN_ID = int(os.getenv("ADMIN_ID"))

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
NORMAL_CHAT_PROMPT = "You're ChatGPT for Discord! You can chat, generate images, and perform searches. Craft responses that are easy to copy directly into Discord chats, without using markdown, code blocks, or extra formatting. When you solving any problems you must remember that: Let's solve this step-by-step. What information do we need to find? What operation might help us solve this? Explain your reasoning and provide the answer."
SEARCH_PROMPT = "You are using the Google Search Plugin, accessing information from the top 3 Google results link which is the scraped content from these 3 website. Summarize these findings clearly, adding relevant insights to answer the users question."

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
    return user_id == ADMIN_ID

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
def google_custom_search(query: str, num_results: int = 3) -> list:
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
            for item in data['items']:
                title = item.get('title', 'No Title') # Get title or default to 'No Title'
                link = item.get('link', 'No Link')  # Get link or default to 'No Link'
                results.append(f"Title: {title}\nLink: {link}\n" + "-" * 80)
            return results
        else:
            print("No items found in the response.")
            return []  

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return [] 
    
# Function to scrape content from a webpage
def scrape_web_content(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.97 Safari/537.36'
        }
        page = requests.get(url, headers=headers, timeout=10)  # Add timeout

        # Check HTTP status code
        if page.status_code != 200:
            return f"Error: Received status code {page.status_code} for {url}"

        soup = BeautifulSoup(page.content, "html.parser")

        # Extract all paragraphs
        paragraphs = soup.find_all("p")
        if paragraphs:
            text = " ".join([p.get_text() for p in paragraphs])
            return text.strip()
        else:
            return "No content found."

    except requests.exceptions.RequestException as e:
        return f"Failed to scrape {url}: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


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

# Slash command to let users choose a model and save it to the database
@tree.command(name="choose_model", description="Select the AI model to use for responses.")
async def choose_model(interaction: discord.Interaction):
    # Check if user is blacklisted (skip for admins)
    if not await is_admin(interaction.user.id) and await is_user_blacklisted(interaction.user.id):
        await interaction.response.send_message("You have been blacklisted from using this bot. Please contact the admin if you think this is a mistake.", ephemeral=True)
        return
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
async def search(interaction: discord.Interaction, query: str):
    # Check if user is blacklisted (skip for admins)
    if not await is_admin(interaction.user.id) and await is_user_blacklisted(interaction.user.id):
        await interaction.response.send_message("You have been blacklisted from using this bot. Please contact the admin if you think this is a mistake.", ephemeral=True)
        return
    """Searches Google and sends results to the AI model."""
    await interaction.response.defer(thinking=True)
    user_id = interaction.user.id
    history = await get_history(user_id)

    history.append({"role": "user", "content": query})

    try:
        # Perform Google search
        search_results = google_custom_search(query, num_results=2)
        if not search_results:
            await interaction.followup.send("No search results found.")
            return

        # Scrape content from the first 5 links
        scraped_contents = []
        for result in search_results:
            url = result.split('\n')[1].split('Link: ')[1]
            content = scrape_web_content(url)
            scraped_contents.append(content)

        # Prepare the combined input for the AI model
        combined_input = f"{SEARCH_PROMPT}\nUser query: {query}\nScraped Contents:\n" + "\n".join(scraped_contents)

        history.append({"role": "system", "content": combined_input})

        # Send the history to the AI model
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=history,
            temperature=0.4,
            max_tokens=4096,
            top_p=1
        )

        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        save_history(user_id, history)

        # Send the final response to the user
        await interaction.followup.send(reply)

    except Exception as e:
        await interaction.followup.send(f"Error: {str(e)}", ephemeral=True)

# Slash command for web scraping (/web)
@tree.command(name="web", description="Scrape a webpage and send data to AI model.")
@app_commands.describe(url="The webpage URL to scrape")
async def web(interaction: discord.Interaction, url: str):
    # Check if user is blacklisted (skip for admins)
    if not await is_admin(interaction.user.id) and await is_user_blacklisted(interaction.user.id):
        await interaction.response.send_message("You have been blacklisted from using this bot. Please contact the admin if you think this is a mistake.", ephemeral=True)
        return
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
async def reset(interaction: discord.Interaction):
    # Check if user is blacklisted (skip for admins)
    if not await is_admin(interaction.user.id) and await is_user_blacklisted(interaction.user.id):
        await interaction.response.send_message("You have been blacklisted from using this bot. Please contact the admin if you think this is a mistake.", ephemeral=True)
        return
    """Resets the bot by clearing user data."""
    user_id = interaction.user.id
    db.user_histories.delete_one({'user_id': user_id})
    await interaction.response.send_message("Your data has been cleared and reset!", ephemeral=True)

# Slash command for user statistics (/user_stat)
@tree.command(name="user_stat", description="Get your current input token, output token, and model.")
async def user_stat(interaction: discord.Interaction):
    # Check if user is blacklisted (skip for admins)
    if not await is_admin(interaction.user.id) and await is_user_blacklisted(interaction.user.id):
        await interaction.response.send_message("You have been blacklisted from using this bot. Please contact the admin if you think this is a mistake.", ephemeral=True)
        return
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

# Function to handle user messages
async def handle_user_message(message: discord.Message):
    # Offload processing to a non-blocking task
    asyncio.create_task(process_user_message(message)) 

async def process_user_message(message: discord.Message):
    try:
        user_id = message.author.id
        
        # Admins bypass all restrictions
        is_admin_user = await is_admin(user_id)
        
        # Check if user is blacklisted (skip for admins)
        if not is_admin_user and await is_user_blacklisted(user_id):
            await message.channel.send("You have been blacklisted from using this bot. Please contact the admin if you think this is a mistake.")
            return
            
        history = await get_history(user_id)
        model = await get_user_model(user_id)

        # Handle PDF files first
        if message.attachments:
            for attachment in message.attachments:
                if attachment.filename.lower().endswith('.pdf'):
                    # Check if user is whitelisted (skip for admins)
                    if not is_admin_user and not await is_user_whitelisted(user_id):
                        await message.channel.send(f"You are not authorized to use PDF processing. Please contact admin (ID: {ADMIN_ID}) to get whitelisted using the /whitelist_add command.")
                        return
                        
                    # Admins can use any model for PDF processing
                    if not is_admin_user and model not in PDF_ALLOWED_MODELS:
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

        # Prepare current message and add to history
        current_message = {"role": "user", "content": content}
        
        try:
            # Prepare messages for API while ensuring token limit
            messages_for_api = []
            
            if model in ["gpt-4o", "gpt-4o-mini"]:
                # For these models, we keep some history but ensure token limits
                history.append(current_message)
                messages_for_api = prepare_messages_for_api(history)
            else:
                # For other models, we just use the current message
                messages_for_api = prepare_messages_for_api([current_message])

            # Prepare API call parameters
            api_params = {
                "model": model,
                "messages": messages_for_api,
                "temperature": 0.3 if model in ["gpt-4o", "gpt-4o-mini"] else 1,
                "max_tokens": 8096 if model in ["gpt-4o", "gpt-4o-mini"] else 4096,
                "top_p": 0.7 if model in ["gpt-4o", "gpt-4o-mini"] else 1
            }

            # Make the API call
            response = await asyncio.to_thread(
                client.chat.completions.create,
                **api_params
            )
            
            reply = response.choices[0].message.content
            
            # Only update history for successful calls
            if model in ["gpt-4o", "gpt-4o-mini"]:
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

def trim_content_to_token_limit(content: str, max_tokens: int = 7500) -> str:
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

def prepare_messages_for_api(messages, max_tokens=7500):
    """Prepare messages for API while ensuring token limit."""
    total_tokens = 0
    prepared_messages = []
    
    # Process messages in reverse order to keep the most recent ones
    for msg in reversed(messages):
        msg_content = msg['content']
        # Convert complex content to text for token counting
        if isinstance(msg_content, list):
            text_content = ""
            for item in msg_content:
                if item['type'] == 'text':
                    text_content += item['text'] + "\n"
            msg_tokens = count_tokens(text_content)
            if total_tokens + msg_tokens > max_tokens:
                # Trim the content
                trimmed_text = trim_content_to_token_limit(text_content, max_tokens - total_tokens)
                if trimmed_text:
                    new_content = [{"type": "text", "text": trimmed_text}]
                    # Preserve any image URLs from the original content
                    for item in msg_content:
                        if item['type'] == 'image_url':
                            new_content.append(item)
                    prepared_messages.insert(0, {"role": msg["role"], "content": new_content})
                break
            else:
                prepared_messages.insert(0, msg)
                total_tokens += msg_tokens
        else:
            msg_tokens = count_tokens(str(msg_content))
            if total_tokens + msg_tokens > max_tokens:
                # Trim the content
                trimmed_text = trim_content_to_token_limit(str(msg_content), max_tokens - total_tokens)
                if trimmed_text:
                    prepared_messages.insert(0, {"role": msg["role"], "content": trimmed_text})
                break
            else:
                prepared_messages.insert(0, msg)
                total_tokens += msg_tokens
                
    return prepared_messages

async def process_pdf_batch(model: str, user_prompt: str, batch_content: str, current_batch: int, total_batches: int, channel, max_retries=3) -> bool:
    """Process a single batch of PDF content with auto-adjustment for token limits."""
    batch_size = len(batch_content.split('\n'))
    original_content = batch_content
    
    for attempt in range(max_retries):
        try:
            # Create message without history
            trimmed_content = trim_content_to_token_limit(batch_content, 7000)  # Leave room for prompt
            messages = [
                {"role": "user", "content": f"{user_prompt}\n\nAnalyze the following content:\n{trimmed_content}"}
            ]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                max_tokens=8096
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
    if len(reply) > 2000:
        with open("response.txt", "w") as file:
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
async def generate_image(interaction: discord.Interaction, prompt: str):
    # Check if user is blacklisted (skip for admins)
    if not await is_admin(interaction.user.id) and await is_user_blacklisted(interaction.user.id):
        await interaction.response.send_message("You have been blacklisted from using this bot. Please contact the admin if you think this is a mistake.", ephemeral=True)
        return
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
    if interaction.user.id != ADMIN_ID:
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
    if interaction.user.id != ADMIN_ID:
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
    if interaction.user.id != ADMIN_ID:
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
    if interaction.user.id != ADMIN_ID:
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


# Start Flask in a separate thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True  # Ensure it closes when the main program exits
flask_thread.start()

# Main bot startup
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    bot.run(TOKEN)
