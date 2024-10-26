import os
import discord
import io
from discord.ext import commands
from discord import app_commands
import requests
from bs4 import BeautifulSoup  # For web scraping
import logging
import sys
from openai import OpenAI
import aiohttp
from runware import Runware, IImageInference, IPromptEnhance, IImageBackgroundRemoval, IImageCaption, IImageUpscale
from collections import defaultdict
import asyncio
import aiohttp
import re
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
# OpenAI client initialization
load_dotenv()
client = OpenAI(
    base_url=str(os.getenv("OPENAI_BASE_URL")),
    api_key=str(os.getenv("OPENAI_API_KEY")),
)
WEB_SCRAPING_PROMPT = "Now you are web browsing, the data you get now is the data scraped from 5 websites. Please respond correctly and insightfully."
NORMAL_CHAT_PROMPT = "You are ChatGPT but for discord, you can generate images, search the web, and chat with me. Please respond correctly and insightfully, make the respond that user can easily copied to discord chat, you must not use markdown or anything related to that."

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

# Intents and bot initialization
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents, heartbeat_timeout=80)
tree = bot.tree  # For slash commands

def google_custom_search(query: str, num_results: int = 3) -> list:
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": num_results
    }
    try:
        response = requests.get(search_url, params=params, timeout=15)  # Thêm timeout
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        data = response.json()

        # Kiểm tra xem có trường 'items' không
        if 'items' in data:
            results = []
            for item in data['items']:
                title = item.get('title', 'No Title')  # Lấy tiêu đề
                link = item.get('link', 'No Link')  # Lấy liên kết
                results.append(f"Title: {title}\nLink: {link}\n" + "-" * 80)
            return results
        else:
            print("No items found in the response.")
            return []  # Trả về danh sách rỗng

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return []  # Trả về danh sách rỗng trong trường hợp có lỗi
    
# Function to scrape content from a webpage
def scrape_web_content(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        page = requests.get(url, headers=headers, timeout=10)  # Thêm timeout

        # Kiểm tra mã trạng thái HTTP
        if page.status_code != 200:
            return f"Error: Received status code {page.status_code} for {url}"

        soup = BeautifulSoup(page.content, "html.parser")

        # Trích xuất nội dung chính
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs[:5]])  # Lấy 5 đoạn đầu tiên
        return text.strip() if text else "No content found."
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

@bot.event
async def on_ready():
    """Bot startup event to sync slash commands."""
    await tree.sync()  # Sync slash commands to the server
    print(f"Logged in as {bot.user}")

@tree.command(name="search", description="Search on Google and send results to AI model.")
@app_commands.describe(query="The search query")
async def search(interaction: discord.Interaction, query: str):
    user_id = interaction.user.id
    if user_id not in user_histories:
        user_histories[user_id] = []

    user_histories[user_id].append({"role": "user", "content": query})

    try:
        await interaction.response.defer(thinking=True)  # Acknowledge the command

        # Perform Google search
        search_results = google_custom_search(query, num_results=3)
        if not search_results:
            await interaction.followup.send("No search results found.")
            return

        # Scrape content from the first few links
        scraped_content = []
        for link in search_results:  # Loop through the links directly
            content = scrape_web_content(link)  # Scrape the content of each link
            title = link  # Use link as title; adjust as needed for your use case
            scraped_content.append(f"**{title}**\n{link}\n{content}\n")

        combined_input = f"User query: {query}\nGoogle search results with content:\n{''.join(scraped_content)}"
        user_histories[user_id].append({"role": "system", "content": combined_input})

        # Send the query and results to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=user_histories[user_id],
            temperature=0.3,
            max_tokens=4096,
            top_p=0.7
        )

        reply = response.choices[0].message.content
        user_histories[user_id].append({"role": "assistant", "content": reply})

        # Check if the reply is too long for Discord
        if len(reply) > 2000:
            with open("response.txt", "w") as file:
                file.write(reply)
            await interaction.followup.send("The response was too long, so it has been saved to a file.", file=discord.File("response.txt"))
        else:
            await interaction.followup.send(reply)

    except Exception as e:
        await interaction.followup.send(f"Error: {str(e)}", ephemeral=True)
        
# Slash command: /web (Scrape web data and send to AI model)
@tree.command(name="web", description="Scrape a webpage and send data to AI model.")
@app_commands.describe(url="The webpage URL to scrape")
async def web(interaction: discord.Interaction, url: str):
    user_id = interaction.user.id
    if user_id not in user_histories:
        user_histories[user_id] = []

    try:
        await interaction.response.defer(thinking=True)  # Acknowledge the command

        # Scrape the provided URL
        content = scrape_web_content(url)

        if content.startswith("Failed"):
            await interaction.followup.send(content)  # If scraping failed, send error
            return

        user_histories[user_id].append({"role": "user", "content": f"Scraped content from {url}"})
        user_histories[user_id].append({"role": "system", "content": content})

        # Send the scraped content to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=user_histories[user_id],
            temperature=0.3,
            max_tokens=4096,
            top_p=0.7
        )

        reply = response.choices[0].message.content
        user_histories[user_id].append({"role": "assistant", "content": reply})

        # Check if the reply is too long for Discord
        if len(reply) > 2000:
            with open("response.txt", "w") as file:
                file.write(reply)
            await interaction.followup.send("The response was too long, so it has been saved to a file.", file=discord.File("response.txt"))
        else:
            await interaction.followup.send(reply)

    except Exception as e:
        await interaction.followup.send(f"Error: {str(e)}", ephemeral=True)

# Reset command to clear user histories and resync slash commands
@tree.command(name="reset", description="Reset the bot by clearing all user data and commands.")
async def reset(interaction: discord.Interaction):
    user_histories.clear()  # Clear user conversation histories
    await interaction.response.send_message("All user data cleared and commands reset!", ephemeral=True)

@bot.event
async def on_message(message: discord.Message):
    # Skip messages sent by the bot itself
    if message.author == bot.user:
        return

    # Check if the message is a reply to the specific bot message, a mention of the bot, or sent in DMs
    is_bot_reply = False
    if message.reference and message.reference.resolved:
        # Check if the referenced message ID is the one we are targeting
        referenced_message = message.reference.resolved
        if referenced_message.id == 1270288366289813556:
            is_bot_reply = True

    is_mention = bot.user.mentioned_in(message)
    is_dm = message.guild is None

    # Only respond if the user replies to the specific bot message, mentions the bot, or sends a DM
    if is_bot_reply or is_mention or is_dm:
        await handle_user_message(message)
    else:
        await bot.process_commands(message)  # Process slash commands

async def handle_user_message(message: discord.Message):
    user_id = message.author.id
    
    # Initialize history if not present
    if user_id not in user_histories:
        # Add the system prompt at the start of the history
        user_histories[user_id] = [{"role": "system", "content": NORMAL_CHAT_PROMPT}]

    # Check for image attachments and combine them with the message content
    if message.attachments:  # If there's an image
        image_url = message.attachments[0].url
        if message.content:  # If there's also text
            user_histories[user_id].append({
                "role": "user", 
                "content": [
                    {"type": "text", "text": message.content},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            })
        else:  # Only image, no text
            user_histories[user_id].append({
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Here's an image."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            })
    else:  # Handle regular text messages
        user_histories[user_id].append({"role": "user", "content": message.content})

    # Limit history to avoid token overflow
    def trim_history(history):
        tokens_used = sum(len(str(item['content'])) for item in history)
        max_tokens_allowed = 9000  # Keeping some buffer to avoid overflow
        
        while tokens_used > max_tokens_allowed:
            removed_item = history.pop(1)  # Remove the oldest user or assistant message, keep the system prompt
            tokens_used -= len(str(removed_item['content']))
    
    # Trim the history if needed
    trim_history(user_histories[user_id])

    try:
        # Prepare and send the message to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=user_histories[user_id],
            temperature=0.3,
            max_tokens=4096,  # Adjusted max_tokens to avoid overuse
            top_p=0.7
        )
        
        # Get the assistant's reply
        reply = response.choices[0].message.content
        
        # Add assistant's reply to history
        user_histories[user_id].append({"role": "assistant", "content": reply})
        
        # Check if the reply is too long for Discord
        if len(reply) > 2000:
            with open("response.txt", "w") as file:
                file.write(reply)
            await message.channel.send("The response was too long, so it has been saved to a file.", file=discord.File("response.txt"))
        else:
            await message.channel.send(reply)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        logging.error(f"Error handling user message: {error_message}")
        await message.channel.send(error_message)

# Slash command for image generation
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

# Main bot startup
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    bot.run(TOKEN)