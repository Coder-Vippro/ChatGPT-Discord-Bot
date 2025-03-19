import os
import sys
import discord
import logging
import asyncio
import signal
import traceback
import time
import logging.config
from discord.ext import commands, tasks
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from discord import app_commands

# Import configuration
from src.config.config import (
    DISCORD_TOKEN, MONGODB_URI, RUNWARE_API_KEY, STATUSES, 
    LOGGING_CONFIG, ENABLE_WEBHOOK_LOGGING, LOGGING_WEBHOOK_URL, 
    WEBHOOK_LOG_LEVEL, WEBHOOK_APP_NAME, WEBHOOK_BATCH_SIZE, 
    WEBHOOK_FLUSH_INTERVAL, LOG_LEVEL_MAP
)

# Import webhook logger
from src.utils.webhook_logger import webhook_log_manager, webhook_logger

# Import database handler
from src.database.db_handler import DatabaseHandler

# Import the message handler
from src.module.message_handler import MessageHandler

# Import various utility modules
from src.utils.image_utils import ImageGenerator

# Global shutdown flag
shutdown_flag = asyncio.Event()

# Load environment variables
load_dotenv()

# Configure logging with more detail, rotation, and webhook integration
def setup_logging():
    # Apply the dictionary config
    try:
        logging.config.dictConfig(LOGGING_CONFIG)
        logging.info("Configured logging from dictionary configuration")
    except Exception as e:
        # Fall back to basic configuration
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        
        # File handler with rotation (keep 5 files of 5MB each)
        try:
            from logging.handlers import RotatingFileHandler
            os.makedirs('logs', exist_ok=True)
            file_handler = RotatingFileHandler(
                'logs/discord_bot.log', 
                maxBytes=5*1024*1024,  # 5MB
                backupCount=5
            )
            file_handler.setFormatter(log_formatter)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # Fall back to basic logging if file logging fails
            logging.basicConfig(
                level=logging.INFO, 
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                stream=sys.stdout
            )
            logging.warning(f"Could not set up file logging: {str(e)}")

    # Set up webhook logging if enabled
    if ENABLE_WEBHOOK_LOGGING and LOGGING_WEBHOOK_URL:
        try:
            # Convert string log level to int using our mapping
            log_level = LOG_LEVEL_MAP.get(WEBHOOK_LOG_LEVEL.upper(), logging.INFO)
            
            # Set up webhook logging
            webhook_log_manager.setup_webhook_logging(
                webhook_url=LOGGING_WEBHOOK_URL,
                app_name=WEBHOOK_APP_NAME,
                level=log_level,
                loggers=None,  # Use root logger
                batch_size=WEBHOOK_BATCH_SIZE,
                flush_interval=WEBHOOK_FLUSH_INTERVAL
            )
            logging.info(f"Webhook logging enabled at level {WEBHOOK_LOG_LEVEL}")
        except Exception as e:
            logging.error(f"Failed to set up webhook logging: {str(e)}")

# Create a function to change bot status periodically
async def change_status_loop(bot):
    """Change bot status every 5 minutes"""
    while not shutdown_flag.is_set():
        for status in STATUSES:
            await bot.change_presence(activity=discord.Game(name=status))
            try:
                # Wait but be interruptible
                await asyncio.wait_for(shutdown_flag.wait(), timeout=300)
                if shutdown_flag.is_set():
                    break
            except asyncio.TimeoutError:
                # Normal timeout, continue to next status
                continue

async def main():
    # Set up logging
    setup_logging()
    
    # Check if required environment variables are set
    missing_vars = []
    if not DISCORD_TOKEN:
        missing_vars.append("DISCORD_TOKEN")
    
    if not MONGODB_URI:
        missing_vars.append("MONGODB_URI")
    
    if missing_vars:
        logging.error(f"The following required environment variables are not set: {', '.join(missing_vars)}")
        return
    
    if not RUNWARE_API_KEY:
        logging.warning("RUNWARE_API_KEY environment variable not set - image generation will not work")
    
    # Initialize the OpenAI client
    try:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI()
        logging.info("OpenAI client initialized successfully")
    except ImportError:
        logging.error("Failed to import OpenAI. Make sure it's installed: pip install openai")
        return
    except Exception as e:
        logging.error(f"Error initializing OpenAI client: {e}")
        return
    
    # Global references to objects that need cleanup
    message_handler = None
    db_handler = None
    
    try:
        # Initialize image generator if API key is available
        image_generator = None
        if RUNWARE_API_KEY:
            try:
                image_generator = ImageGenerator(RUNWARE_API_KEY)
                logging.info("Image generator initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing image generator: {e}")
        
        # Set up Discord intents
        intents = discord.Intents.default()
        intents.message_content = True
        
        # Initialize the bot with command prefixes and more robust timeout settings
        bot = commands.Bot(
            command_prefix="//quocanhvu", 
            intents=intents, 
            heartbeat_timeout=120,
            max_messages=10000  # Cache more messages to improve experience
        )
        
        # Initialize database handler
        db_handler = DatabaseHandler(MONGODB_URI)
        
        # Create database indexes for performance
        await db_handler.create_indexes()
        logging.info("Database indexes created")
        
        # Khởi tạo collection reminders
        await db_handler.ensure_reminders_collection()
        
        # Event handler when the bot is ready
        @bot.event
        async def on_ready():
            """Bot startup event to sync slash commands and start status loop."""
            await bot.tree.sync()  # Sync slash commands
            bot_info = f"Logged in as {bot.user} (ID: {bot.user.id})"
            logging.info("=" * len(bot_info))
            logging.info(bot_info)
            logging.info(f"Connected to {len(bot.guilds)} guilds")
            logging.info("=" * len(bot_info))
            
            # Start the status changing task
            asyncio.create_task(change_status_loop(bot))
        
        # Handle general errors to prevent crashes
        @bot.event
        async def on_error(event, *args, **kwargs):
            error_msg = traceback.format_exc()
            logging.error(f"Discord event error in {event}:\n{error_msg}")
        
        @bot.event
        async def on_command_error(ctx, error):
            if isinstance(error, commands.CommandNotFound):
                return
            
            error_msg = str(error)
            trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            logging.error(f"Command error: {error_msg}\n{trace}")
            await ctx.send(f"Error: {error_msg}")
        
        # Initialize message handler
        message_handler = MessageHandler(bot, db_handler, openai_client, image_generator)
        
        # Set up slash commands
        from src.commands.commands import setup_commands
        setup_commands(bot, db_handler, openai_client, image_generator)
        
        # Handle shutdown signals
        loop = asyncio.get_running_loop()
        
        # Signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda sig=sig: asyncio.create_task(shutdown(sig, loop, bot, db_handler, message_handler))
                )
            except (NotImplementedError, RuntimeError):
                # Windows doesn't support SIGTERM or add_signal_handler
                # Use fallback for Windows
                pass  
        
        logging.info("Starting bot...")
        await bot.start(DISCORD_TOKEN)
        
    except Exception as e:
        error_msg = traceback.format_exc()
        logging.critical(f"Fatal error in main function: {str(e)}\n{error_msg}")
        
        # Clean up resources if initialization failed halfway
        await cleanup_resources(bot=None, db_handler=db_handler, message_handler=message_handler)
        
async def shutdown(sig, loop, bot, db_handler, message_handler):
    """Handle graceful shutdown of the bot"""
    logging.info(f"Received signal {sig.name}. Starting graceful shutdown...")
    
    # Set shutdown flag to stop ongoing tasks
    shutdown_flag.set()
    
    # Give running tasks a moment to detect shutdown flag
    await asyncio.sleep(1)
    
    # Start cleanup
    await cleanup_resources(bot, db_handler, message_handler)
    
    # Stop the event loop
    loop.stop()

async def cleanup_resources(bot, db_handler, message_handler):
    """Clean up all resources during shutdown"""
    try:
        # Close the bot connection
        if bot is not None:
            logging.info("Closing bot connection...")
            await bot.close()
        
        # Close message handler resources
        if message_handler is not None:
            logging.info("Closing message handler resources...")
            await message_handler.close()
        
        # Close database connection
        if db_handler is not None:
            logging.info("Closing database connection...")
            await db_handler.close()
        
        # Clean up webhook logging
        if ENABLE_WEBHOOK_LOGGING and LOGGING_WEBHOOK_URL:
            logging.info("Cleaning up webhook logging...")
            webhook_log_manager.cleanup()
            
        logging.info("Cleanup completed successfully")
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        # Use asyncio.run to properly run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped via keyboard interrupt")
    except Exception as e:
        logging.critical(f"Unhandled exception in main thread: {str(e)}")
        traceback.print_exc()
    finally:
        logging.info("Bot shut down completely")
