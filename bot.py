import os
import discord
import io
import sqlite3
from discord.ext import commands, tasks
from discord import app_commands
import requests
from bs4 import BeautifulSoup
import logging
import sys
from openai import OpenAI, RateLimitError
import aiohttp
from runware import Runware, IImageInference
from collections import defaultdict
import asyncio
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

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
    "o1-mini"
]

# Prompt for different plugins
WEB_SCRAPING_PROMPT = "You are using the Web Scraping Plugin, gathering information from given url. Respond accurately and combine data to provide a clear, insightful summary. "
NORMAL_CHAT_PROMPT = "You're ChatGPT for Discord! You can chat, generate images, and perform searches. Craft responses that are easy to copy directly into Discord chats, without using markdown, code blocks, or extra formatting. When you solving any problems you must remember that: Let's solve this step-by-step. What information do we need to find? What operation might help us solve this? Explain your reasoning and provide the answer."
SEARCH_PROMPT = "You are using the Google Search Plugin, accessing information from the top 3 Google results. Summarize these findings clearly, adding relevant insights to answer the users question."

# Google API details
GOOGLE_API_KEY = str(os.getenv("GOOGLE_API_KEY"))  # Google API Key
GOOGLE_CX = str(os.getenv("GOOGLE_CX"))  # Search Engine ID

# Runware API key
RUNWARE_API_KEY = str(os.getenv("RUNWARE_API_KEY"))

# Initialize Runware SDK
runware = Runware(api_key=RUNWARE_API_KEY)

# Dictionary to keep track of user requests and their cooldowns
user_requests = defaultdict(lambda: {'last_request': 0, 'queue': asyncio.Queue()})

# Dictionary to store user conversation history
user_histories = {}

# Bot token
TOKEN = str(os.getenv("DISCORD_TOKEN"))

# --- Database functions ---
def create_tables():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_histories (
            user_id INTEGER PRIMARY KEY,
            history TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id INTEGER PRIMARY KEY,
            model TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def get_history(user_id):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT history FROM user_histories WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return eval(result[0])
    return [{"role": "system", "content": NORMAL_CHAT_PROMPT}]

def save_history(user_id, history):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO user_histories (user_id, history)
        VALUES (?, ?)
    ''', (user_id, str(history)))
    conn.commit()
    conn.close()

# New function to get the user's model preference
def get_user_model(user_id):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT model FROM user_preferences WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "gpt-4o"  # Default to "gpt-4o" if no preference

# New function to save the user's model preference
def save_user_model(user_id, model):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO user_preferences (user_id, model)
        VALUES (?, ?)
    ''', (user_id, model))
    conn.commit()
    conn.close()

def initialize_db():
    db_file = 'chat_history.db'
    if not os.path.exists(db_file):
        print(f"{db_file} not found. Creating tables...")
        create_tables()
    else:
        print(f"{db_file} found. No need to create tables.")

initialize_db() # Initialize the database

# Intents and bot initialization
intents = discord.Intents.default()
intents.message_content = True

# Bot initialization
bot = commands.Bot(command_prefix="!", intents=intents, heartbeat_timeout=120)
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
        response = requests.get(search_url, params=params, timeout=15)  # Add timeout
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
        save_user_model(user_id, selected_model)
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
    history = get_history(user_id)

    history.append({"role": "user", "content": query})

    try:
        # Perform Google search
        search_results = google_custom_search(query, num_results=5)
        if not search_results:
            await interaction.followup.send("No search results found.")
            return

        # Prepare the search results for the AI model
        combined_input = f"{SEARCH_PROMPT}\nUser query: {query}\nGoogle search results:\n"
        
        # Extract URLs and prepare the message
        links = []
        for result in search_results:
            url = result.split('\n')[1].split('Link: ')[1]  # Extract URL from the result string
            links.append(url)
            combined_input += f"{result}\n"

        # Add links at the end of the combined input
        combined_input += "\nLinks:\n" + "\n".join(links)

        history.append({"role": "system", "content": combined_input})

        # Send the history to the AI model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.3,
            max_tokens=4096,
            top_p=0.7
        )

        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        save_history(user_id, history)

        # Prepare the final response including the links
        links_message = "\nLinks:\n" + "\n".join(links)
        await interaction.followup.send(reply + links_message)

    except Exception as e:
        await interaction.followup.send(f"Error: {str(e)}", ephemeral=True)

# Slash command for web scraping (/web)
@tree.command(name="web", description="Scrape a webpage and send data to AI model.")
@app_commands.describe(url="The webpage URL to scrape")
async def web(interaction: discord.Interaction, url: str):
    """Scrapes a webpage and sends data to the AI model."""
    await interaction.response.defer(thinking=True)
    user_id = interaction.user.id
    history = get_history(user_id)

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

        await send_response(interaction, reply)

    except Exception as e:
        await interaction.followup.send(f"Error: {str(e)}", ephemeral=True)

# Reset user chat history from database
@tree.command(name="reset", description="Reset the bot by clearing user data.")
async def reset(interaction: discord.Interaction):
    """Resets the bot by clearing user data."""
    user_id = interaction.user.id  # Get the user ID of the person who invoked the command
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    # Delete user data based on their user_id
    cursor.execute('DELETE FROM user_histories WHERE user_id = ?', (user_id,))
    
    # Recreate an empty history to avoid issues
    cursor.execute('INSERT INTO user_histories (user_id, history) VALUES (?, ?)', (user_id, '[]'))
    
    conn.commit()
    conn.close()
    
    await interaction.response.send_message("Your data has been cleared and reset!", ephemeral=True)
# Slash command for help (/help)
@tree.command(name="help", description="Display a list of available commands.")
async def help_command(interaction: discord.Interaction):
    """Sends a list of available commands to the user."""
    help_message = (
        "**Available Commands:**\n"
        "/choose_model - Select the AI model to use for responses (gpt-4o, gpt-4o-mini, o1-preview, o1-mini).\n"
        "/search `<query>` - Search on Google and send results to AI model.\n"
        "/web `<url>` - Scrape a webpage and send data to AI model.\n"
        "/generate `<prompt>` - Generate an image from a text prompt.\n"
        "/reset - Reset your conversation history.\n"
        "/help - Display this help message.\n"
        "**Các lệnh có sẵn:**\n"
        "/choose_model - Chọn mô hình AI để sử dụng cho phản hồi (gpt-4o, gpt-4o-mini, o1-preview, o1-mini).\n"
        "/search `<truy vấn>` - Tìm kiếm trên Google và gửi kết quả đến mô hình AI.\n"
        "/web `<url>` - Thu thập dữ liệu từ trang web và gửi đến mô hình AI.\n"
        "/generate `<gợi ý>` - Tạo hình ảnh từ gợi ý văn bản.\n"
        "/reset - Đặt lại lịch sử trò chuyện của bạn.\n"
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

async def handle_user_message(message: discord.Message):
    """Processes user messages and generates responses."""
    user_id = message.author.id
    history = get_history(user_id)
    model = get_user_model(user_id)

    # Supported text/code file extensions
    supported_file_types = [
        ".txt", ".json", ".py", ".cpp", ".js", ".html",
        ".css", ".xml", ".md", ".java", ".cs"
    ]

    # Initialize content list for the current message
    content = []

    # Add message content if present
    if message.content:
        content.append({"type": "text", "text": message.content})

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

    # If no content was added, add a default message
    if not content and not image_urls:
        content.append({"type": "text", "text": "No content."})

    # Prepare the current message
    current_message = {"role": "user", "content": content}

    # Check model and adjust behavior
    if model in ["o1-mini", "o1-preview"]:
        # Disable image support and system prompt
        user_message_text = message.content if message.content else "No content."
        history.append({"role": "user", "content": user_message_text})
    else:
        # Remove previous image messages
        history = [
            msg for msg in history
            if not (
                msg["role"] == "user" and
                isinstance(msg["content"], list) and
                any(part["type"] == "image_url" for part in msg["content"])
            )
        ]
        if image_urls:
            current_message = {"role": "user", "content": content + [
                {"type": "image_url", "image_url": {"url": url}} for url in image_urls
            ]}
            history.append(current_message)
        else:
            history.append({"role": "user", "content": user_message_text})

    # Trim history before sending to OpenAI
    trim_history(history)

    # Create a history without images for saving
    history_to_save = []
    for msg in history:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            text_parts = [part["text"] for part in msg["content"] if part["type"] == "text"]
            text_content = "\n".join(text_parts)
            history_to_save.append({"role": "user", "content": text_content})
        else:
            history_to_save.append(msg)

    # Prepare messages to send to OpenAI API
    messages_to_send = history_to_save.copy()

    if model not in ["o1-mini", "o1-preview"] and image_urls:
        latest_images = [
            {
                "type": "image_url",
                "image_url": {"url": url}
            }
            for url in image_urls
        ]
        messages_to_send.append({
            "role": "user",
            "content": latest_images
        })

    try:
        if model in ["o1-mini", "o1-preview"]:
            response = client.chat.completions.create(
                model=model,
                messages=history[1:]
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages_to_send,
                temperature=0.3,
                max_tokens=4096,
                top_p=0.7,
            )

        reply = response.choices[0].message.content
        history_to_save.append({"role": "assistant", "content": reply})
        save_history(user_id, history_to_save)

        await send_response(message.channel, reply)

    except RateLimitError:
        error_message = (
            "Error: Rate limit exceeded for o1-preview or o1-mini. "
            "Please try again later or use /choose_model to change to gpt-4o."
        )
        logging.error(f"Rate limit error: {error_message}")
        await message.channel.send(error_message)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        logging.error(f"Error handling user message: {error_message}")
        await message.channel.send(error_message)

# Function to trim the history to avoid exceeding token limits
def trim_history(history):
    """Trims the history to avoid exceeding token limits."""
    tokens_used = sum(len(str(item['content'])) for item in history)
    max_tokens_allowed = 9000
    while tokens_used > max_tokens_allowed:
        removed_item = history.pop(1)
        tokens_used -= len(str(removed_item['content']))

# Function to send a response to the channel
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

@bot.event
async def on_ready():
    """Bot startup event to sync slash commands and start status loop."""
    await tree.sync()  # Sync slash commands
    print(f"Logged in as {bot.user}")
    change_status.start()  # Start the status changing loop

# Main bot startup
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    bot.run(TOKEN)