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

# Load environment variables
load_dotenv()

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
            max_tokens=4096,
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
    """Resets the bot by clearing user data."""
    user_id = interaction.user.id
    db.user_histories.delete_one({'user_id': user_id})
    await interaction.response.send_message("Your data has been cleared and reset!", ephemeral=True)

# Slash command for user statistics (/user_stat)
@tree.command(name="user_stat", description="Get your current input token, output token, and model.")
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

# Function to process user messages
async def process_user_message(message: discord.Message):
    user_id = message.author.id
    history = await get_history(user_id)
    model = await get_user_model(user_id)

    # Initialize content list for the current message
    content = []

    # Add message content if present
    if message.content:
        content.append({"type": "text", "text": message.content})

    # Supported text/code file extensions
    supported_file_types = [
        ".txt", ".json", ".py", ".cpp", ".js", ".html",
        ".css", ".xml", ".md", ".java", ".cs",
        ".rb", ".go", ".ts", ".swift", ".kt",
        ".php", ".sh", ".bat", ".pl", ".r",
        ".sql", ".yaml", ".yml", ".ini", ".cfg",
        ".tex", ".csv", ".log", ".lua", ".scala",
        ".hs", ".erl", ".ex", ".clj", ".jsx",
        ".tsx", ".vue", ".svelte", ".dart", ".m",
        ".groovy", ".ps1", ".vb", ".asp", ".aspx",
        ".jsp", ".dart", ".coffee", ".nim", ".vala",
        ".fish", ".zsh", ".csh", ".tcsh", ".mk",
        ".make", ".Dockerfile", ".env", ".graphql",
        ".twig", ".hbs", ".liquid"
    ]

    # Process attachments if any
    image_urls = []
    if message.attachments:
        attachments = message.attachments
        for attachment in attachments:
            if any(attachment.filename.endswith(ext) for ext in supported_file_types):
                file_content = await attachment.read()
                try:
                    user_message_content = file_content.decode("utf-8")
                    content.append({"type": "text", "text": user_message_content})
                except UnicodeDecodeError:
                    await message.channel.send("Error: The file appears to be binary data, not a text file.")
                    return
            else:
                image_urls.append(attachment.url)
                # Add image URLs to content
                content.append({"type": "image_url", "image_url": {"url": attachment.url}})

    # If no content was added, add a default message
    if not content and not image_urls:
        content.append({"type": "text", "text": "No content."})

    # Prepare the current message
    current_message = {"role": "user", "content": content}
    history.append(current_message)

    # Trim history before sending to OpenAI
    trim_history(history)

    # Prepare messages to send to API
    messages_to_send = history.copy()

    if model in ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"]:
        # If the model is "o1", rename "system" role to "developer"
        if model == "o1" or model == "o3-mini":
            for msg in messages_to_send:
                if msg["role"] == "system":
                    msg["role"] = "developer"
        elif model != "o1":
            for msg in messages_to_send:
                if msg["role"] == "developer":
                    msg["role"] = "system"

        # Include up to 10 previous images
        def get_last_n_images(history, n=10):
            images = []
            for msg in reversed(history):
                if msg["role"] == "user" and isinstance(msg["content"], list):
                    for part in reversed(msg["content"]):
                        if part["type"] == "image_url":
                            part["details"] = "high"
                            images.append(part)
                            if len(images) == n:
                                return images[::-1]
            return images[::-1]

        # Get the last 10 images
        latest_images = get_last_n_images(history, n=10)

        if latest_images:
            # Remove existing images from the last message
            last_message = messages_to_send[-1]
            if last_message["role"] == "user" and isinstance(last_message["content"], list):
                last_message["content"] = [
                    part for part in last_message["content"] if part["type"] != "image_url"
                ]
                last_message["content"].extend(latest_images)
            else:
                last_message["content"] = [{"type": "text", "text": last_message["content"]}]
                last_message["content"].extend(latest_images)
            messages_to_send[-1] = last_message

        # Fix the 431 error by limiting the number of images
        max_images = 10
        total_images = 0
        for msg in messages_to_send:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                image_parts = [part for part in msg["content"] if part.get("type") == "image_url"]
                total_images += len(image_parts)
        if total_images > max_images:
            images_removed = 0
            for msg in messages_to_send:
                if msg["role"] == "user" and isinstance(msg["content"], list):
                    new_content = []
                    for part in msg["content"]:
                        if part.get("type") == "image_url" and images_removed < (total_images - max_images):
                            images_removed += 1
                            continue
                        new_content.append(part)
                    msg["content"] = new_content

    else:
        # Exclude image URLs and system prompts for other models
        for msg in messages_to_send:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                msg["content"] = [
                    part for part in msg["content"] if part["type"] != "image_url"
                ]
        messages_to_send = [
            msg for msg in messages_to_send if msg.get("role") != "system"
        ]

    try:
        # Prepare API call parameters
        api_params = {
            "model": model,
            "messages": messages_to_send,
        }

        if model in ["gpt-4o", "gpt-4o-mini"]:
            api_params.update({
                "temperature": 0.3,
                "max_tokens": 8096,
                "top_p": 0.7,
            })

        # The non-blocking call, done in a background thread
        response = await asyncio.to_thread(client.chat.completions.create, **api_params)
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        await save_history(user_id, history)

        await send_response(message.channel, reply)

    # Handle rate limit errors
    except RateLimitError:
        error_message = (
            "Error: Rate limit exceeded for your model. "
            "Please try again later or use /choose_model to change to any models else."
        )
        logging.error(f"Rate limit error: {error_message}")
        await message.channel.send(error_message)

    # Handle other exceptions
    except Exception as e:
        error_message = f"Error: {str(e)}"
        logging.error(f"Error handling user message: {error_message}")
        await message.channel.send(error_message)
        db.user_histories.delete_one({'user_id': user_id})

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
