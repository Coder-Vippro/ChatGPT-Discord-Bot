import os
import sys
import discord
import logging
import asyncio
from discord.ext import commands, tasks

# Import configuration
from src.config.config import (
    DISCORD_TOKEN, MONGODB_URI, RUNWARE_API_KEY, STATUSES
)

# Import database handler
from src.database.db_handler import DatabaseHandler

# Import the message handler
from src.module.message_handler import MessageHandler

# Import various utility modules
from src.utils.image_utils import ImageGenerator

async def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Check if required environment variables are set
    if not DISCORD_TOKEN:
        logging.error("DISCORD_TOKEN environment variable not set")
        return
    
    if not MONGODB_URI:
        logging.error("MONGODB_URI environment variable not set")
        return
    
    if not RUNWARE_API_KEY:
        logging.warning("RUNWARE_API_KEY environment variable not set - image generation will not work")
    
    # Initialize the OpenAI client
    try:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI()
    except ImportError:
        logging.error("Failed to import OpenAI. Make sure it's installed: pip install openai")
        return
    except Exception as e:
        logging.error(f"Error initializing OpenAI client: {e}")
        return
        
    # Initialize image generator if API key is available
    image_generator = None
    if RUNWARE_API_KEY:
        try:
            image_generator = ImageGenerator(RUNWARE_API_KEY)
        except Exception as e:
            logging.error(f"Error initializing image generator: {e}")
    
    # Set up Discord intents
    intents = discord.Intents.default()
    intents.message_content = True
    
    # Initialize the bot
    bot = commands.Bot(command_prefix="//quocanhvu", intents=intents, heartbeat_timeout=120)
    
    # Initialize database handler
    db_handler = DatabaseHandler(MONGODB_URI)
    
    # Create a function to change bot status periodically
    @tasks.loop(minutes=5)
    async def change_status():
        """Change bot status every 5 minutes"""
        for status in STATUSES:
            await bot.change_presence(activity=discord.Game(name=status))
            await asyncio.sleep(300)  # Change every 5 minutes
    
    # Event handler when the bot is ready
    @bot.event
    async def on_ready():
        """Bot startup event to sync slash commands and start status loop."""
        await bot.tree.sync()  # Sync slash commands
        logging.info(f"Logged in as {bot.user}")
        
        # Start the status changing loop
        change_status.start()
    
    # Initialize message handler
    message_handler = MessageHandler(bot, db_handler, openai_client, image_generator)
    
    # Set up slash commands
    from src.commands.commands import setup_commands
    setup_commands(bot, db_handler, openai_client, image_generator)
    
    # Run the bot with the Discord token
    await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    # Use asyncio.run to properly run the async main function
    asyncio.run(main())
